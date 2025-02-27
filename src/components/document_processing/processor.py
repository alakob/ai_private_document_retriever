"""
Document processing component for handling file ingestion and chunking.
"""

import os
import logging
from typing import List, Optional, Union
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from datetime import datetime
import asyncio
from concurrent.futures import ProcessPoolExecutor
import hashlib
from sqlalchemy import text, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager
import asyncpg
import json
import numpy as np

from ...config.processor import ProcessorConfig
from ...services.database import create_async_db_engine, create_async_session_maker
from ...services.embeddings import get_embeddings
from ...models.documents import Base, DocumentModel, DocumentChunk
from ...utils.text import sanitize_text
from ...utils.monitoring import ProcessingMonitor

logger = logging.getLogger(__name__)

def calculate_file_checksum(file_path: str) -> str:
    """Calculate SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files efficiently
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

async def check_duplicate_document(session: AsyncSession, checksum: str) -> Optional[DocumentModel]:
    """Check if document with given checksum already exists."""
    try:
        result = await session.execute(
            select(DocumentModel).where(DocumentModel.checksum == checksum)
        )
        return result.scalar_one_or_none()
    except Exception as e:
        logger.error(f"Error checking for duplicate document: {str(e)}")
        return None

class DocumentProcessor:
    """Handles document processing and storage."""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.text_splitter = CharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.embeddings = get_embeddings()
        self.monitor = ProcessingMonitor()
        
        # Initialize process pool for CPU-bound operations
        self.process_pool = ProcessPoolExecutor(
            max_workers=config.max_workers
        )

        # Create database engine and session maker
        self.engine = create_async_db_engine(
            connection_string=self.config.postgres_config.sqlalchemy_url,
            pool_size=self.config.db_pool_size
        )
        self.async_session = create_async_session_maker(self.engine)
        
        # Add semaphore for concurrent operations
        self.semaphore = asyncio.Semaphore(config.max_workers)

    async def initialize_database(self):
        """Initialize database tables and extensions."""
        try:
            async with self.engine.begin() as conn:
                # Create pgvector extension if it doesn't exist
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                
                # Create all tables
                await conn.run_sync(Base.metadata.create_all)
                
            logger.info("✓ Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise

    async def process_directory(self, directory: Path) -> List[DocumentModel]:
        """Process all documents in a directory."""
        try:
            if not directory.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")
            
            start_time = datetime.now()
            self.stats = {
                'total_docs': 0,
                'total_chunks': 0,
                'failed_docs': 0,
                'skipped_docs': 0,
                'embedding_tokens': 0
            }
            
            logger.info(f"Starting document processing from {directory}")
            
            files = [f for f in directory.glob('*.*') 
                    if f.suffix.lower() in ['.pdf', '.txt', '.doc', '.docx']]
            
            if not files:
                logger.warning(f"No supported documents found in {directory}")
                return []

            documents = []
            
            # Initialize pool with vector support
            async def init_connection(conn):
                try:
                    # Register the vector type codec
                    await conn.set_type_codec(
                        'vector',
                        encoder=lambda v: np.array(v, dtype=np.float32).tobytes(),
                        decoder=lambda v: np.frombuffer(v, dtype=np.float32),
                        format='binary'
                    )
                except Exception as e:
                    logger.error(f"Error setting up vector codec: {str(e)}")
                    raise

            pool = await asyncpg.create_pool(
                self.config.postgres_config.asyncpg_dsn,
                min_size=2,
                max_size=self.config.db_pool_size,
                init=init_connection
            )

            def validate_embedding(embedding) -> Optional[np.ndarray]:
                """Validate and format embedding vector."""
                try:
                    # Convert to numpy array if not already
                    if not isinstance(embedding, np.ndarray):
                        embedding = np.array(embedding, dtype=np.float32)
                    
                    # Check dimensions
                    if embedding.size == 0:
                        logger.warning("Empty embedding vector")
                        return None
                        
                    # Ensure vector is 1-dimensional
                    if len(embedding.shape) > 1:
                        embedding = embedding.flatten()
                        
                    # Check if dimensions are too large
                    if embedding.size > 16000:
                        logger.warning(f"Embedding dimension too large: {embedding.size}")
                        return None
                        
                    # Check if dimensions match expected OpenAI dimensions
                    if embedding.size != 1536:
                        logger.warning(f"Unexpected embedding dimension: {embedding.size}, expected 1536")
                        # Try to pad or truncate to correct dimensions
                        if embedding.size < 1536:
                            # Pad with zeros
                            padded = np.zeros(1536, dtype=np.float32)
                            padded[:embedding.size] = embedding
                            embedding = padded
                        else:
                            # Truncate
                            embedding = embedding[:1536]
                        
                    # Ensure correct data type
                    if embedding.dtype != np.float32:
                        embedding = embedding.astype(np.float32)
                        
                    # Normalize the vector to unit length for cosine similarity
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                        
                    return embedding
                except Exception as e:
                    logger.error(f"Error validating embedding: {str(e)}")
                    return None

            try:
                async with pool.acquire() as conn:
                    for file_path in files:
                        try:
                            logger.info(f"Processing {file_path}")
                            
                            # Check for duplicate using raw SQL
                            checksum = calculate_file_checksum(str(file_path))
                            duplicate = await conn.fetchrow(
                                "SELECT id, filename, created_at FROM documents WHERE checksum = $1",
                                checksum
                            )
                            
                            if duplicate:
                                logger.info(
                                    f"Skipping duplicate document: {file_path}\n"
                                    f"  └─ Original file: {duplicate['filename']}\n"
                                    f"  └─ First processed: {duplicate['created_at']}\n"
                                    f"  └─ Checksum: {checksum[:8]}..."
                                )
                                self.stats['skipped_docs'] += 1
                                continue

                            # Start a transaction for document insertion
                            async with conn.transaction():
                                # Insert document
                                metadata = json.dumps({
                                    'file_type': file_path.suffix.lower(),
                                    'file_path': str(file_path),
                                    'processed_at': datetime.now().isoformat()
                                })

                                doc_id = await conn.fetchval("""
                                    INSERT INTO documents (filename, checksum, doc_metadata, created_at, upload_date)
                                    VALUES ($1, $2, $3, $4, $5)
                                    RETURNING id
                                """, str(file_path), checksum, metadata, datetime.now(), datetime.now())

                            # Process chunks in batches
                            text = await self._extract_text(file_path)
                            if not text:
                                logger.warning(f"No text extracted from {file_path}")
                                continue

                            chunks = self._split_text(text)
                            if not chunks:
                                logger.warning(f"No chunks created from {file_path}")
                                continue

                            # Process chunks in batches
                            total_chunks_processed = 0
                            
                            for i in range(0, len(chunks), self.config.batch_size):
                                batch = chunks[i:i + self.config.batch_size]
                                
                                # Generate embeddings for batch
                                embeddings = await self.embeddings.aembed_documents(batch)
                                if not embeddings:
                                    logger.warning(f"No embeddings generated for batch {i//self.config.batch_size}")
                                    continue

                                # Pre-validate all embeddings before starting transaction
                                valid_pairs = []
                                for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                                    validated_embedding = validate_embedding(embedding)
                                    if validated_embedding is not None:
                                        valid_pairs.append((chunk, validated_embedding, i + j))
                                
                                if not valid_pairs:
                                    logger.warning(f"No valid embeddings in batch {i//self.config.batch_size}")
                                    continue
                                
                                # Prepare chunk records for insertion
                                chunk_records = []
                                for chunk, embedding, chunk_index in valid_pairs:
                                    chunk_records.append((
                                        doc_id,
                                        sanitize_text(chunk),
                                        chunk_index,
                                        len(chunk),
                                        embedding.tolist(),
                                        1,
                                        json.dumps({}),
                                        datetime.now()
                                    ))
                                
                                # Insert all valid chunks in a single transaction
                                try:
                                    async with conn.transaction():
                                        for record in chunk_records:
                                            await conn.execute("""
                                                INSERT INTO document_chunks 
                                                (document_id, content, chunk_index, chunk_size, embedding, page_number, chunk_metadata, created_at)
                                                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                                            """, *record)
                                        
                                        # Update stats only if transaction succeeds
                                        total_chunks_processed += len(chunk_records)
                                        self.stats['embedding_tokens'] += sum(len(chunk.split()) for chunk, _, _ in valid_pairs)
                                        logger.info(f"Successfully inserted {len(chunk_records)} chunks in batch {i//self.config.batch_size}")
                                
                                except Exception as e:
                                    logger.error(f"Failed to insert batch {i//self.config.batch_size}: {str(e)}")
                                    # Continue with next batch
                            
                            if total_chunks_processed > 0:
                                self.stats['total_docs'] += 1
                                self.stats['total_chunks'] += total_chunks_processed
                                logger.info(f"Successfully processed {file_path} with {total_chunks_processed} chunks")

                                # Fetch the complete document record
                                document = await conn.fetchrow(
                                    "SELECT * FROM documents WHERE id = $1",
                                    doc_id
                                )
                                documents.append(document)
                            else:
                                # If no chunks were processed, delete the document
                                async with conn.transaction():
                                    await conn.execute("DELETE FROM documents WHERE id = $1", doc_id)
                                logger.warning(f"Deleted document {file_path} as no chunks were successfully processed")
                                self.stats['failed_docs'] += 1

                        except Exception as e:
                            self.stats['failed_docs'] += 1
                            logger.error(f"Error processing {file_path}: {str(e)}")
                            continue  # Move to next file

            finally:
                await pool.close()

            # Calculate and log statistics
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            logger.info("\n" + "="*50)
            logger.info("Document Processing Summary:")
            logger.info(f"Total documents found: {len(files)}")
            logger.info(f"Successfully processed: {self.stats['total_docs']}")
            logger.info(f"Skipped (duplicates): {self.stats['skipped_docs']}")
            logger.info(f"Failed: {self.stats['failed_docs']}")
            logger.info(f"Total chunks created: {self.stats['total_chunks']}")
            logger.info(f"Total processing time: {processing_time:.2f} seconds")
            logger.info(f"Embedding tokens processed: {self.stats['embedding_tokens']}")
            logger.info("="*50 + "\n")
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to process directory: {str(e)}")
            raise

    async def _extract_text(self, file_path: Path) -> Optional[str]:
        """Extract text from document based on file type."""
        try:
            if file_path.suffix.lower() == '.pdf':
                from pypdf import PdfReader
                reader = PdfReader(str(file_path))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            elif file_path.suffix.lower() == '.txt':
                return file_path.read_text()
            else:
                logger.warning(f"Unsupported file type: {file_path.suffix}")
                return None
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {str(e)}")
            return None

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        # Simple character-based splitting for now
        chunks = []
        current_pos = 0
        text_length = len(text)
        
        while current_pos < text_length:
            chunk_end = min(current_pos + self.config.chunk_size, text_length)
            
            # Adjust chunk end to nearest sentence or paragraph break
            if chunk_end < text_length:
                for separator in ['\n\n', '\n', '. ', ' ']:
                    next_break = text.rfind(separator, current_pos, chunk_end + 100)
                    if next_break != -1:
                        chunk_end = next_break + len(separator)
                        break
            
            chunks.append(text[current_pos:chunk_end].strip())
            current_pos = max(chunk_end - self.config.chunk_overlap, current_pos + 1)
        
        return [chunk for chunk in chunks if chunk]

    async def cleanup(self):
        """Cleanup resources."""
        if self.engine:
            await self.engine.dispose()

    async def process_file(self, file_path: Union[str, Path]) -> Optional[List[Document]]:
        """Process file with duplicate checking"""
        try:
            # Check for duplicate document
            existing_doc = await self.check_duplicate_document(str(file_path))
            if existing_doc:
                logger.info(
                    f"Document {file_path} already exists with same content "
                    f"(checksum: {existing_doc.checksum})"
                )
                return None

            # Continue with normal processing if not a duplicate
            return await super().process_file(file_path)

        except IntegrityError as e:
            if "uq_document_checksum" in str(e):
                logger.warning(f"Duplicate document detected: {file_path}")
                return None
            raise
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

    async def check_duplicate_document(self, file_path: str) -> Optional[DocumentModel]:
        """Check if document already exists using checksum"""
        try:
            checksum = calculate_file_checksum(file_path)
            async with self.async_session() as session:
                result = await session.execute(
                    select(DocumentModel).where(DocumentModel.checksum == checksum)
                )
                return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error checking duplicate document: {str(e)}")
            return None

    @staticmethod
    def calculate_file_checksum(file_path: str) -> str:
        """Calculate SHA-256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest() 