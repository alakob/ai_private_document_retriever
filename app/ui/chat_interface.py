import os
from typing import List, Dict, Optional, Any, Tuple, Callable, Union, AsyncGenerator, Type
import time
import logging
from datetime import datetime, timedelta
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.memory import BaseMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from sqlalchemy import select, func
from sqlalchemy.sql import text
from dotenv import load_dotenv
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import trim_messages
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
import tiktoken
import asyncio
import re
from app.config import prompt_config
from pgvector.sqlalchemy import Vector
import numpy as np
import shutil
from pathlib import Path

# Define custom exceptions for security and circuit breaking
class SecurityError(Exception):
    """Raised when a security violation is detected"""
    pass

class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open and requests are blocked"""
    pass

# Circuit breaker implementation
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold  # Number of failures before opening circuit
        self.reset_timeout = reset_timeout  # Seconds to wait before attempting reset
        self.failure_count = 0  # Current failure count
        self.last_failure_time = None  # Timestamp of last failure
        self.state = "CLOSED"  # Circuit state: CLOSED, OPEN, HALF-OPEN
        
    def check_state(self):
        """Check if circuit is open and block requests if needed"""
        if self.state == "OPEN":
            # Check if reset timeout has elapsed
            if self.last_failure_time and time.time() - self.last_failure_time > self.reset_timeout:
                # Try half-open state to test if system has recovered
                self.state = "HALF-OPEN"
                logging.info("Circuit breaker moving to HALF-OPEN state")
            else:
                # Circuit still open, block request
                raise CircuitOpenError(f"Circuit breaker is OPEN until {datetime.fromtimestamp(self.last_failure_time + self.reset_timeout)}")
        
    def record_success(self):
        """Record successful operation"""
        if self.state == "HALF-OPEN":
            # If successful in half-open state, reset circuit
            self.state = "CLOSED"
            self.failure_count = 0
            logging.info("Circuit breaker reset to CLOSED state")
    
    def record_failure(self):
        """Record failed operation and potentially open circuit"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            if self.state != "OPEN":
                self.state = "OPEN"
                logging.warning(f"Circuit breaker OPENED after {self.failure_count} failures")

# Initialize the circuit breaker
CIRCUIT_BREAKER = CircuitBreaker(failure_threshold=3, reset_timeout=60)
import logging
import plotly.graph_objects as go
from pyvis.network import Network
import json
import tempfile
import random
from utils.vector_store_visualization import (
    VectorStoreDataAcquisition,
    VectorStoreVisualization,
    visualize_vector_store
)
from sqlalchemy.ext.asyncio import AsyncSession
import hashlib
from datetime import datetime
import time

from app.config import chat_config, db_config, processor_config, PostgresConfig, ProcessorConfig

from app.core.document_rag_loader import (
    DocumentModel,
    DocumentChunk,
    PostgresConfig,
    async_sessionmaker,
    create_async_engine,
    OpenAIEmbeddings,
    DocumentProcessor,
    ProcessorConfig,
    calculate_file_checksum,
    check_duplicate_document,
    sanitize_text
)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityError(Exception):
    """Raised when potential security issue is detected"""
    def __init__(self, message: str, payload: Optional[Dict] = None):
        super().__init__(message)
        self.payload = payload or {}
        logger.security(f"Security violation: {message}", extra=self.payload)

# Rate limiter using token bucket algorithm
class MermaidRateLimiter:
    def __init__(self, rate: int = 5, period: float = 60.0):
        self.tokens = rate
        self.rate = rate
        self.last_check = time.monotonic()
        self.period = period
        self.metrics = {
            'total_requests': 0,
            'rate_limited': 0,
            'avg_response_time': 0.0
        }

    async def acquire(self):
        start_time = time.monotonic()
        self.metrics['total_requests'] += 1
        now = time.monotonic()
        elapsed = now - self.last_check
        
        if elapsed > self.period:
            self.tokens = self.rate
            self.last_check = now
        
        if self.tokens < 1:
            self.metrics['rate_limited'] += 1
            raise RateLimitExceeded("Too many diagram generation requests")
        
        self.tokens -= 1
        try:
            return True
        finally:
            self.metrics['avg_response_time'] = (
                (self.metrics['avg_response_time'] * (self.metrics['total_requests'] - 1)) + 
                (time.monotonic() - start_time)
            ) / self.metrics['total_requests']

    def get_metrics(self) -> Dict[str, Any]:
        return {
            **self.metrics,
            'current_tokens': self.tokens,
            'last_reset': self.last_check
        }

RATE_LIMITER = MermaidRateLimiter(rate=5, period=60)  # 5 requests/minute

class CircuitBreaker:
    def __init__(self, max_failures=3, reset_timeout=30):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure = 0

    def check_state(self):
        if self.failure_count >= self.max_failures:
            if time.monotonic() - self.last_failure > self.reset_timeout:
                self.reset()
                return True
            raise CircuitOpenError("Service unavailable - circuit open")
        return True

    def record_failure(self):
        self.failure_count += 1
        self.last_failure = time.monotonic()

    def reset(self):
        self.failure_count = 0

CIRCUIT_BREAKER = CircuitBreaker()

class PostgresVectorRetriever(BaseRetriever, BaseModel):
    """Custom retriever for PostgreSQL vector store"""
    
    session_maker: Callable = Field(..., description="Async session maker for PostgreSQL")
    embeddings: Any = Field(..., description="Embeddings model")
    k: int = Field(default=chat_config.retriever_k, description="Number of documents to retrieve")
    score_threshold: float = Field(default=chat_config.retriever_score_threshold, description="Minimum similarity score threshold")
    
    class Config:
        arbitrary_types_allowed = True
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for a query using vector similarity search."""
        try:
            # Generate embedding for query
            query_embedding = await self.embeddings.aembed_query(query)
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.astype(np.float32).tolist()
            
            # Convert embedding to PostgreSQL vector string format
            query_embedding_str = f"[{','.join(map(str, query_embedding))}]"
            
            print(f"✓ Generated embedding for query: {query[:50]}...")
            
            async with self.session_maker() as session:
                # Ensure pgvector extension is installed
                await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                
                # SQL query for similarity search using cosine distance
                sql = text("""
                    WITH similarity_scores AS (
                        SELECT 
                            dc.content,
                            dc.chunk_index,
                            dc.chunk_size,
                            dc.page_number,
                            dm.filename,
                            dc.chunk_metadata,
                            1 - (dc.embedding <=> CAST(:embedding AS vector)) as similarity_score
                        FROM document_chunks dc
                        JOIN documents dm ON dc.document_id = dm.id
                        WHERE 1 - (dc.embedding <=> CAST(:embedding AS vector)) > :threshold
                        ORDER BY similarity_score DESC
                        LIMIT :limit
                    )
                    SELECT * FROM similarity_scores;
                """)
                
                # Execute search with parameters
                result = await session.execute(
                    sql,
                    {
                        "embedding": query_embedding_str,
                        "threshold": self.score_threshold,
                        "limit": self.k
                    }
                )
                
                # Process results into Documents
                documents = []
                for row in result:
                    doc = Document(
                        page_content=row.content,
                        metadata={
                            'source': row.filename,
                            'page': row.page_number,
                            'chunk_index': row.chunk_index,
                            'chunk_size': row.chunk_size,
                            'similarity_score': row.similarity_score,
                            **(row.chunk_metadata or {})
                        }
                    )
                    documents.append(doc)
                
                print(f"✓ Found {len(documents)} matching documents")
                return documents
                
        except Exception as e:
            print("\n=== Document Retrieval Error ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Traceback:")
            import traceback
            print(traceback.format_exc())
            return []
    
    def _get_relevant_documents(
        self, 
        query: str,
        *,
        runnable_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Synchronous method to get relevant documents"""
        raise NotImplementedError("This retriever only supports async operations")

class ChatInterface:
    """Handles chat interface and conversation"""
    
    def __init__(self):
        self.documents_dir = Path(chat_config.documents_dir)
        self.documents_dir.mkdir(exist_ok=True)
        
        # Create PostgreSQL connection string
        self.postgres_config = PostgresConfig(
            connection_string=db_config.connection_string,
            pre_delete_collection=False,
            drop_existing=False
        )
        
        # Create async engine and session
        self.engine = create_async_engine(
            self.postgres_config.connection_string,
            echo=False,
            pool_size=5,
            max_overflow=10
        )
        self.async_session = async_sessionmaker(self.engine, expire_on_commit=False)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-ada-002"
        )
        
        # Initialize retriever
        self.retriever = PostgresVectorRetriever(
            session_maker=self.async_session,
            embeddings=self.embeddings,
            k=chat_config.retriever_k,
            score_threshold=chat_config.retriever_score_threshold
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=chat_config.model,
            temperature=chat_config.temperature,
            max_tokens=chat_config.max_tokens,
            streaming=chat_config.streaming
        )
        
        # Initialize chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True
        )
        
        # Initialize system message
        self.system_message = SystemMessage(
            content="You are a helpful assistant that answers questions based on the provided documents."
        )
        
        # Define prompt templates
        document_prompt = PromptTemplate(
            input_variables=["page_content"],
            template=prompt_config.document_template
        )

        qa_prompt = ChatPromptTemplate.from_template(prompt_config.qa_template)

        # Initialize conversation chain with correct parameters and prompts
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            verbose=True,
            return_source_documents=True,
            chain_type="stuff",
            combine_docs_chain_kwargs={
                "document_prompt": document_prompt,
                "document_variable_name": "context",
                "prompt": qa_prompt
            }
        )
        
        # Initialize conversation history
        self.messages = [self.system_message]
        
        # Initialize tokenizer and set max tokens
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.max_token_limit = 4000  # Adjust based on your needs
        
        # Add debug logging
        self.debug = True
        
        # Test database connection at initialization
        if self.debug:
            asyncio.create_task(self.test_db_connection())
    
    def _count_tokens(self, messages: list) -> int:
        """Count tokens in concatenated message content"""
        try:
            text_content = " ".join([
                str(msg.content) 
                for msg in messages 
                if hasattr(msg, 'content') and msg.content is not None
            ])
            return len(self.tokenizer.encode(text_content))
        except Exception as e:
            print(f"Token counting error: {str(e)}")
            return 0
    
    def _trim_history(self):
        """Trim conversation history using LangChain's utilities"""
        try:
            # Validate messages
            validated_messages = [
                msg for msg in self.messages 
                if isinstance(msg, BaseMessage) and hasattr(msg, 'content')
            ]
            
            # Trim messages
            self.messages = trim_messages(
                messages=validated_messages,
                max_tokens=self.max_token_limit,
                token_counter=self._count_tokens,
                strategy="last"  # Keep most recent messages
            )
        except Exception as e:
            print(f"History trimming error: {str(e)}")
            # Keep only system message and last few messages if trimming fails
            self.messages = [self.system_message] + self.messages[-4:]
    
    async def chat(self, message: str, history: List[Dict[str, str]]) -> Dict[str, str]:
        """Process chat message and return response"""
        try:
            if self.debug:
                print(f"\nQuestion: {message}")
                print(f"History: {history}")
            
            # Convert history to messages format
            if not history:
                self.messages = [self.system_message]
            
            # Add user message
            user_message = HumanMessage(content=message)
            self.messages.append(user_message)
            
            # Trim history if needed
            self._trim_history()
            
            # Format chat history as tuples
            chat_history = []
            messages = self.messages[1:]  # Skip system message
            for i in range(0, len(messages)-1, 2):
                if i+1 < len(messages):
                    human_msg = messages[i]
                    ai_msg = messages[i+1]
                    if isinstance(human_msg, HumanMessage) and isinstance(ai_msg, AIMessage):
                        chat_history.append((human_msg.content, ai_msg.content))
            
            # Invoke chain
            result = await self.conversation_chain.ainvoke({
                "question": message,
                "chat_history": chat_history
            })
            
            # Create assistant message
            assistant_message = AIMessage(content=result["answer"])
            self.messages.append(assistant_message)
            
            # Format response with sources
            response = result["answer"]
            if "source_documents" in result and result["source_documents"]:
                unique_sources = {
                    doc.metadata['source']: doc.metadata.get('page', 'N/A') 
                    for doc in result["source_documents"]
                }
                sources = [
                    f"Source: {source} (Page: {page})" 
                    for source, page in unique_sources.items()
                ]
                response = f"{response}\n\nSources:\n" + "\n".join(sources)
            
            # Add debug logging for retrieved documents
            if self.debug and "source_documents" in result:
                print("\nRetrieved Documents:")
                for doc in result["source_documents"]:
                    print(f"Source: {doc.metadata['source']}")
                    print(f"Content: {doc.page_content[:200]}...")
            
            # Return in OpenAI message format
            return {"role": "assistant", "content": response}
            
        except Exception as e:
            import traceback
            print(f"Error: {str(e)}")
            print(traceback.format_exc())
            return {"role": "assistant", "content": f"Error processing question: {str(e)}"}
    
    async def cleanup(self):
        """Cleanup resources properly"""
        try:
            # Close any active sessions
            if hasattr(self, 'async_session'):
                async with self.async_session() as session:
                    await session.close()
            
            # Dispose of the engine
            if hasattr(self, 'engine'):
                await self.engine.dispose()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def test_db_connection(self):
        """Test database connection and document retrieval"""
        try:
            async with self.async_session() as session:
                # Try to get one document
                result = await session.execute(
                    select(DocumentModel).limit(1)
                )
                doc = result.scalar_one_or_none()
                if doc:
                    print("\n=== 🔍 Database Connection Test ===")
                    print("✅ Successfully connected to database")
                    print(f"📄 Found document: {doc.filename}")
                    
                    # Test vector retrieval
                    test_query = "test query"
                    docs = await self.retriever._aget_relevant_documents(
                        test_query
                    )
                    print(f"✅ Vector retrieval test: found {len(docs)} documents")
                    return True
                else:
                    print("⚠️ No documents found in database")
                    return False
        except Exception as e:
            print("\n=== ❌ Database Connection Error ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            print("Traceback:")
            print(traceback.format_exc())
            return False

    async def handle_file_upload(self, files: List[str]) -> str:
        """Handle initial file upload to temporary storage"""
        try:
            logger.info(f"\n📤 Received {len(files)} files")
            
            html_output = ["<div>"]
            html_output.append(f"<b>📤 Received {len(files)} files</b><br>")
            
            successful_uploads = []
            for file_path in files:
                try:
                    # Convert to Path object
                    source_path = Path(file_path)
                    
                    # Create destination path in documents directory
                    dest_path = self.documents_dir / source_path.name
                    
                    # Handle duplicate filenames
                    counter = 1
                    while dest_path.exists():
                        stem = source_path.stem
                        suffix = source_path.suffix
                        new_name = f"{stem}_{counter}{suffix}"
                        dest_path = self.documents_dir / new_name
                        counter += 1
                    
                    logger.info(f"Moving {source_path} to {dest_path}")
                    
                    # Move file to documents directory
                    shutil.move(str(source_path), str(dest_path))
                    
                    successful_uploads.append(dest_path.name)
                    html_output.append(f"✅ Successfully moved <span style='color: green;'>{dest_path.name}</span><br>")
                    logger.info(f"✅ Successfully moved {dest_path.name}")
                    
                except Exception as e:
                    html_output.append(f"❌ Error processing <span style='color: red;'>{file_path}</span>: {str(e)}<br>")
                    logger.error(f"❌ Error processing {file_path}: {str(e)}")
                    continue

            if successful_uploads:
                html_output.append("<br><b>Successfully uploaded files:</b><br>")
                for f in successful_uploads:
                    html_output.append(f"- <span style='color: green;'>{f}</span><br>")
                html_output.append("<br><b>Click 'Process Documents' to proceed with document processing.</b>")
            else:
                html_output.append("<br><span style='color: red;'>No files were successfully uploaded</span>")
            
            html_output.append("</div>")
            return "".join(html_output)
            
        except Exception as e:
            logger.error(f"❌ Upload error: {str(e)}", exc_info=True)
            return f"<div style='color: red;'>Error uploading files: {str(e)}</div>"

    def calculate_file_checksum(self, file_path: str) -> str:
        """
        Calculate SHA-256 checksum for a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hexadecimal digest of the SHA-256 hash
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                # Read the file in chunks to handle large files efficiently
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"Error calculating checksum for {file_path}: {e}")
            return None

    async def process_single_file(self, file_path: Path, processor_config: ProcessorConfig) -> Optional[List[Document]]:
        """Process a single file with document processor."""
        try:
            print(f"\n=== Processing File: {file_path.name} ===")
            print(f"Checksum: {self.calculate_file_checksum(str(file_path))}")
            
            # Initialize processor for embeddings
            processor = DocumentProcessor(processor_config)
            
            # Load and process the document to get chunks and embeddings
            documents = await processor._load_document(str(file_path))
            if not documents:
                print(f"⚠️ No content extracted from {file_path.name}")
                return None
            
            # Process in batches like document_rag_loader
            all_chunks = []
            
            # Calculate checksum once for the file
            checksum = self.calculate_file_checksum(str(file_path))
            
            # Check for duplicate file first
            async with self.async_session() as session:
                result = await session.execute(
                    select(DocumentModel).where(DocumentModel.checksum == checksum)
                )
                if result.scalar_one_or_none():
                    print(f"⏩ Document {file_path.name} already exists with same checksum")
                    return None
            
            # Process document in batches
            for i in range(0, len(documents), processor_config.batch_size):
                batch = documents[i:i + processor_config.batch_size]
                
                # Get text content
                texts = [doc.page_content for doc in batch]
                
                # Get embeddings
                embeddings = await processor._get_embeddings_batch(texts)
                
                # Validate chunks before database operations
                valid_chunks = []
                valid_embeddings = []
                
                for idx, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                    content = sanitize_text(chunk.page_content)
                    if not content:
                        print(f"Skipping chunk {idx}: Empty or invalid content")
                        continue
                    
                    chunk.page_content = content
                    valid_chunks.append(chunk)
                    valid_embeddings.append(embedding)
                
                if not valid_chunks:
                    print(f"No valid chunks found in batch")
                    continue
                
                # Process database operations in a single transaction
                async with self.async_session() as session:
                    try:
                        async with session.begin():
                            # Final duplicate check before insert
                            result = await session.execute(
                                select(DocumentModel).where(DocumentModel.checksum == checksum)
                            )
                            if result.scalar_one_or_none():
                                print(f"⚠️ Race condition detected! Document was inserted by another process")
                                return None
                            
                            # Create document record if this is the first batch
                            if not all_chunks:  # Only create document on first batch
                                doc = DocumentModel(
                                    filename=file_path.name,
                                    checksum=checksum,
                                    doc_metadata={
                                        'source': file_path.name,
                                        'file_path': str(file_path),
                                        'processed_at': datetime.now().isoformat()
                                    }
                                )
                                session.add(doc)
                                await session.flush()
                                doc_id = doc.id
                            
                            # Create chunk records
                            for idx, (chunk, embedding) in enumerate(zip(valid_chunks, valid_embeddings)):
                                chunk_record = DocumentChunk(
                                    document_id=doc_id,
                                    content=chunk.page_content,
                                    embedding=embedding,
                                    chunk_index=len(all_chunks) + idx,  # Global index across all batches
                                    chunk_size=len(chunk.page_content),
                                    page_number=chunk.metadata.get('page', None),
                                    section_title=chunk.metadata.get('section_title', None),
                                    chunk_metadata={
                                        **chunk.metadata,
                                        'processed_at': datetime.now().isoformat(),
                                        'embedding_model': 'text-embedding-ada-002',
                                        'batch_index': i // processor_config.batch_size
                                    }
                                )
                                session.add(chunk_record)
                        
                        print(f"✅ Stored batch with {len(valid_chunks)} chunks")
                        all_chunks.extend(valid_chunks)
                        
                    except Exception as e:
                        print(f"❌ Transaction error: {str(e)}")
                        raise
            
            if all_chunks:
                print(f"✅ Successfully processed {file_path.name} with {len(all_chunks)} total chunks")
            return all_chunks

        except Exception as e:
            print(f"❌ Error processing file {file_path}: {str(e)}")
            logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
            raise

    async def process_documents(self, loader_type="Docling (Enhanced)") -> str:
        """Process the uploaded documents with progress tracking"""
        try:
            logger.info("Starting document processing...")
            print("\n=== Document Processing Started ===")
            
            # Initialize database tables if needed
            async with self.async_session() as session:
                # Create extensions if they don't exist
                await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                await session.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto;"))
                await session.commit()
                
                # Create index if it doesn't exist
                try:
                    await session.execute(
                        text("CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_checksum ON documents (checksum);")
                    )
                    await session.commit()
                    logger.info("Tables and indices created successfully")
                except Exception as e:
                    logger.warning(f"Index creation warning (can be ignored if already exists): {str(e)}")
            
            # Get all files in documents directory
            all_files = list(self.documents_dir.glob('*.*'))
            total_files = len(all_files)
            print(f"Found {total_files} files to process")
            
            # Process files with checksum verification
            total_chunks = 0
            processed_files = []
            skipped_files = []
            error_files = []
            
            # First, get all existing checksums in a separate session
            existing_checksums = {}
            existing_filenames = {}
            
            async with self.async_session() as session:
                result = await session.execute(select(DocumentModel))
                existing_docs = result.scalars().all()
                
                for doc in existing_docs:
                    existing_checksums[doc.checksum] = doc.id
                    existing_filenames[doc.filename] = doc.checksum
            
            print(f"\n=== Existing Documents in Database ===")
            print(f"Found {len(existing_checksums)} documents in database")
            print("All checksums in database:")
            for checksum, doc_id in existing_checksums.items():
                print(f"ID: {doc_id} | Checksum: {checksum[:8]}...")
            
            # Create processor config
            postgres_config = PostgresConfig(
                connection_string=self.postgres_config.connection_string,
                pre_delete_collection=False,
                drop_existing=False
            )
            
            # Determine which loader to use based on selection
            use_docling = False
            use_mistral = False
            loader_name = "LangChain"
            
            if loader_type == "Docling (Enhanced)":
                use_docling = True
                loader_name = "Docling (Enhanced)"
            elif loader_type == "Mistral OCR":
                use_mistral = True
                loader_name = "Mistral OCR"
                
            # Log the selected loader
            logger.info(f"Using document loader: {loader_name}")
            print(f"\n=== 📚 Document Loader: {loader_name} ===\n")
                
            processor_config = ProcessorConfig(
                chunk_size=500,
                chunk_overlap=50,
                vector_store_type="postgres",
                postgres_config=postgres_config,
                batch_size=50,
                max_workers=1,
                db_pool_size=5,
                openai_api_key=OPENAI_API_KEY,
                # Set loader options based on user selection
                use_docling=use_docling,
                docling_artifacts_path=os.getenv('DOCLING_ARTIFACTS_PATH'),
                docling_enable_remote=os.getenv('DOCLING_ENABLE_REMOTE', '').lower() in ('true', '1', 'yes'),
                docling_use_cache=os.getenv('DOCLING_USE_CACHE', 'true').lower() in ('true', '1', 'yes'),
                use_mistral=use_mistral,
                mistral_api_key=os.getenv('MISTRAL_API_KEY'),
                mistral_ocr_model=os.getenv('MISTRAL_OCR_MODEL', 'mistral-ocr-latest'),
                mistral_include_images=os.getenv('MISTRAL_INCLUDE_IMAGES', '').lower() in ('true', '1', 'yes')
            )
            
            # Process each file
            for file_path in all_files:
                try:
                    # Calculate checksum
                    file_checksum = calculate_file_checksum(str(file_path))
                    print(f"\n===== Checking file: {file_path.name} =====")
                    print(f"Calculated checksum: {file_checksum}")
                    
                    # Check if file exists in database by checksum
                    checksum_exists = file_checksum in existing_checksums
                    print(f"Checksum exists in database? {checksum_exists}")
                    
                    if checksum_exists:
                        print(f"⏩ Skipping {file_path.name} - found matching checksum in database")
                        logger.info(f"Skipping {file_path.name} - already exists with checksum {file_checksum}")
                        skipped_files.append(file_path.name)
                        continue
                    
                    # Process file
                    print(f"🔄 Processing {file_path.name}...")
                    logger.info(f"Processing {file_path.name}")
                    chunks = await self.process_single_file(file_path, processor_config)
                    
                    if chunks:
                        total_chunks += len(chunks)
                        processed_files.append(file_path.name)
                        print(f"✅ Processed {file_path.name} - created {len(chunks)} chunks")
                        
                        # Update our local cache
                        existing_checksums[file_checksum] = True
                        existing_filenames[file_path.name] = file_checksum
                    
                except Exception as e:
                    print(f"❌ Error processing {file_path.name}: {str(e)}")
                    logger.error(f"Error processing {file_path.name}: {str(e)}")
                    error_files.append(file_path.name)
                    continue
            
            # Create status message with HTML formatting
            status_message = []
            status_message.append("<div style='padding: 10px; background-color: #1a1a1a; border-radius: 5px;'>")
            status_message.append("<h3 style='text-align: center; color: #ffffff;'>=== Processing Summary ===</h3>")
            status_message.append(f"<p><b>Total files:</b> {len(all_files)}</p>")
            
            if processed_files:
                status_message.append(f"<p><b>Successfully processed files:</b> {len(processed_files)}</p>")
                status_message.append("<ul>")
                for file in processed_files:
                    status_message.append(f"<li style='color: #00cc00;'>{file}</li>")
                status_message.append("</ul>")
                status_message.append(f"<p><b>Total chunks created:</b> {total_chunks}</p>")
            
            if skipped_files:
                status_message.append(f"<p><b>Skipped files:</b> {len(skipped_files)}</p>")
                status_message.append("<ul>")
                for file in skipped_files:
                    status_message.append(f"<li style='color: orange;'>{file}</li>")
                status_message.append("</ul>")
            
            if error_files:
                status_message.append(f"<p><b>Files with errors:</b> {len(error_files)}</p>")
                status_message.append("<ul>")
                for file in error_files:
                    status_message.append(f"<li style='color: #ff3333;'>{file}</li>")
                status_message.append("</ul>")
                status_message.append("<p>Check logs for detailed error information.</p>")
            
            status_message.append("</div>")
            return "".join(status_message)
            
        except Exception as e:
            error_message = f"<div style='color: red; padding: 10px;'>Error processing documents: {str(e)}</div>"
            logger.error(error_message, exc_info=True)
            return error_message

    async def stream_chat(self, message: str, history: List[Dict[str, str]]) -> AsyncGenerator[Dict[str, str], None]:
        """Process chat message and stream response"""
        try:
            # Add debug flag to track streaming
            streaming_debug = True
            
            if streaming_debug:
                print(f"\n🔄 Starting streaming response for: {message}")
            
            if self.debug:
                print(f"\n📝 Question: {message}")
                print(f"📚 History length: {len(history) if history else 0}")
            
            # Convert history to messages format if needed
            if not self.messages or len(self.messages) <= 1:  # Only system message or empty
                self.messages = [self.system_message]
                
                # Convert history to messages if available
                if history:
                    for h_msg, a_msg in history:
                        self.messages.append(HumanMessage(content=h_msg))
                        self.messages.append(AIMessage(content=a_msg))
            
            # Add user message
            user_message = HumanMessage(content=message)
            self.messages.append(user_message)
            
            # Trim history if needed
            self._trim_history()
            
            # Format chat history as tuples for the prompt
            chat_history = []
            messages = self.messages[1:]  # Skip system message
            for i in range(0, len(messages)-1, 2):
                if i+1 < len(messages):
                    human_msg = messages[i]
                    ai_msg = messages[i+1]
                    if isinstance(human_msg, HumanMessage) and isinstance(ai_msg, AIMessage):
                        chat_history.append((human_msg.content, ai_msg.content))
            
            # Prepare for streaming - get relevant documents
            docs = await self.retriever._aget_relevant_documents(message)
            
            if self.debug:
                print("\n📄 Retrieved Documents:")
                for i, doc in enumerate(docs):
                    print(f"  📑 Doc {i+1}: {doc.metadata['source']} (Page: {doc.metadata.get('page', 'N/A')})")
                    print(f"  📝 Content: {doc.page_content[:100]}...")
            
            # Format documents for context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create a streaming LLM
            streaming_llm = ChatOpenAI(
                model=chat_config.model,
                temperature=chat_config.temperature,
                max_tokens=chat_config.max_tokens,
                streaming=True
            )
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_template(prompt_config.qa_template)
            
            # Format chat history for prompt
            formatted_chat_history = ""
            for human, ai in chat_history:
                formatted_chat_history += f"Human: {human}\nAI: {ai}\n\n"
            
            # Create the chain
            chain = prompt | streaming_llm
            
            # Start streaming response
            full_response = ""
            
            # Invoke the chain with streaming
            async for chunk in chain.astream({
                "context": context,
                "chat_history": formatted_chat_history,
                "question": message
            }):
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    full_response += content
                    
                    if streaming_debug and len(content) > 0:
                        print(f"🔤 Streaming chunk: {content[:20]}...")
                    
                    # Yield the current state of the response
                    yield {"role": "assistant", "content": full_response}
            
            if streaming_debug:
                print(f"✅ Streaming complete, final length: {len(full_response)}")
            
            # After streaming is complete, add sources information
            if docs:
                unique_sources = {
                    doc.metadata['source']: doc.metadata.get('page', 'N/A') 
                    for doc in docs
                }
                sources = [
                    f"Source: {source} (Page: {page})" 
                    for source, page in unique_sources.items()
                ]
                final_response = f"{full_response}\n\nSources:\n" + "\n".join(sources)
                
                # Yield the final response with sources
                yield {"role": "assistant", "content": final_response}
                
                # Create assistant message for history
                assistant_message = AIMessage(content=final_response)
                self.messages.append(assistant_message)
            else:
                # Create assistant message for history
                assistant_message = AIMessage(content=full_response)
                self.messages.append(assistant_message)
                
                # Yield the final response
                yield {"role": "assistant", "content": full_response}
            
        except Exception as e:
            import traceback
            print(f"❌ Error: {str(e)}")
            print(traceback.format_exc())
            yield {"role": "assistant", "content": f"Error processing question: {str(e)}"}

    async def check_database_integrity(self):
        """Check database tables for data integrity"""
        try:
            async with self.async_session() as session:
                # Check documents table
                print("\n=== Checking Documents Table ===")
                result = await session.execute(select(DocumentModel))
                documents = result.scalars().all()
                print(f"Total documents: {len(documents)}")
                
                for doc in documents:
                    print(f"\nDocument ID: {doc.id}")
                    print(f"Filename: {doc.filename}")
                    print(f"Checksum: {doc.checksum}")
                    print(f"Metadata: {doc.doc_metadata}")
                    
                    # Check associated chunks
                    chunk_result = await session.execute(
                        select(DocumentChunk)
                        .where(DocumentChunk.document_id == doc.id)
                    )
                    chunks = chunk_result.scalars().all()
                    print(f"Number of chunks: {len(chunks)}")
                    
                    if chunks:
                        # Check first and last chunk
                        print("\nFirst chunk details:")
                        print(f"Content length: {len(chunks[0].content)}")
                        print(f"Embedding dimension: {len(chunks[0].embedding)}")
                        print(f"Metadata: {chunks[0].chunk_metadata}")
                        
                        print("\nLast chunk details:")
                        print(f"Content length: {len(chunks[-1].content)}")
                        print(f"Embedding dimension: {len(chunks[-1].embedding)}")
                        print(f"Metadata: {chunks[-1].chunk_metadata}")
                    
                # Check for orphaned chunks
                print("\n=== Checking for Orphaned Chunks ===")
                orphaned_result = await session.execute(
                    select(DocumentChunk)
                    .outerjoin(DocumentModel)
                    .where(DocumentModel.id == None)
                )
                orphaned_chunks = orphaned_result.scalars().all()
                if orphaned_chunks:
                    print(f"Warning: Found {len(orphaned_chunks)} orphaned chunks!")
                else:
                    print("No orphaned chunks found.")
                
                return True
                
        except Exception as e:
            print(f"Error checking database: {str(e)}")
            logger.error(f"Database check error: {str(e)}", exc_info=True)
            return False

async def main():
    """Main function to run the chat interface"""
    chat_interface = None
    try:
        chat_interface = ChatInterface()
        
        with gr.Blocks() as demo:
            gr.Markdown("""
                <div style="text-align: center; margin-bottom: 20px;">
                    <h1>DocuSeek AI</h1>
                    <p>Ask questions about your documents. The system will search through the document collection and provide relevant answers.</p>
                </div>
            """)
            
            # Add global CSS for document accordion only
            gr.HTML("""
            <style>
            /* Target document accordion by ID and make everything inside it white */
            #document-accordion,
            #document-accordion > div,
            #document-accordion > div > div,
            #document-accordion > div > div > div,
            #document-accordion [class*="bg-gray"],
            #document-accordion [data-testid="accordion-content"],
            #document-accordion [data-testid="accordion-content"] > div {
                background-color: white !important;
            }
            
            /* Force white background on any potentially gray backgrounds */
            #document-accordion .bg-gray-50,
            #document-accordion .bg-gray-100,
            #document-accordion .bg-gray-200,
            #document-accordion .prose,
            #document-accordion .contain,
            #document-accordion .block {
                background-color: white !important;
            }
            
            /* Target document accordion with direct child selectors to be thorough */
            #document-accordion > * {
                background-color: white !important;
            }
            
            /* Overlay to force white background */
            #document-accordion::before {
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: white;
                z-index: -1;
            }
            </style>
            """, visible=False)
            
            # Document Upload Accordion with white background
            with gr.Accordion("Document Upload/Processing", open=False):
                upload_button = gr.UploadButton(
                    "Click to Upload Files",
                    variant="huggingface",
                    size="sm",
                    file_types=[".pdf", ".txt", ".doc", ".docx"],
                    file_count="multiple"
                )
                
                # Custom CSS to make the radio buttons display horizontally and centrally
                # Also set the background color to white specifically for this accordion
                gr.HTML("""
                <style>
                /* Change background color of document upload accordion to white */
                .white-bg {
                    background-color: white !important;
                }
                
                /* Target the accordion content area specifically */
                .white-bg > .grow {
                    background-color: white !important;
                }
                
                /* Set all children elements to have white background too */
                .white-bg > div,
                .white-bg > div > div,
                .white-bg .gradio-box,
                .white-bg .contain,
                .white-bg .prose,
                .white-bg .gap,
                .white-bg .block,
                .white-bg .relative,
                .white-bg .border,
                .white-bg .border-gray-200 {
                    background-color: white !important;
                }
                
                /* Direct targeting of the content below the header */
                .white-bg [data-testid="accordion-content"] {
                    background-color: white !important;
                }
                
                /* Target any row, column, and container elements within the accordion */
                .white-bg .gr-form,
                .white-bg .gr-box,
                .white-bg .gr-row,
                .white-bg .gr-column,
                .white-bg .gr-panel,
                .white-bg .gr-padded,
                .white-bg .bg-gray-50,
                .white-bg .bg-gray-100,
                .white-bg .bg-gray-200 {
                    background-color: white !important;
                }
                
                /* Fix spacing in radio button group */
                .gradio-container .compact-radio {
                    /* max-width: 500px; */ /* Removed to allow full width */
                    margin: 0 auto;
                }
                /* Center the label */
                .gradio-container .compact-radio > .label-wrap {
                    text-align: center;
                    justify-content: center;
                    margin-bottom: 8px; /* Reduce space between label and options */
                }
                /* Make options horizontal and centered */
                .gradio-container .compact-radio > .options {
                    display: flex;
                    flex-direction: row;
                    justify-content: center;
                    gap: 15px;
                    flex-wrap: nowrap;
                }
                /* Make each option more compact and match button height */
                .gradio-container .compact-radio > .options > .wrap {
                    flex: 0 0 auto;
                    min-width: 0;
                    margin: 0;
                }
                /* Style the radio button options to match button height */
                .gradio-container .compact-radio .gradio-radio {
                    height: 32px !important; /* Match sm button height */
                    padding: 0 12px !important; /* Match button padding */
                    border-radius: 4px !important; /* Match button border radius */
                    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important; /* Light shadow like buttons */
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                }
                /* Make radio buttons smaller */
                .gradio-container .compact-radio input[type="radio"] {
                    transform: scale(0.8) !important; /* Make radio circles smaller */
                    margin-right: 4px !important; /* Reduce space between radio and label */
                }
                /* Reduce the label font size */
                .gradio-container .compact-radio label {
                    font-size: 0.9rem !important; /* Smaller text */
                    font-weight: normal !important;
                    white-space: nowrap !important;
                }
                </style>
                """, visible=False)
                
                # Use direct container without extra box for cleaner layout
                loader_type = gr.Radio(
                    label="Document Loader",
                    choices=["Docling (Enhanced)", "LangChain", "Mistral OCR"],
                    value="Docling (Enhanced)",
                    interactive=True,
                    info="Select the document loader to use for processing",
                    elem_classes="compact-radio"
                )
                
                with gr.Row(equal_height=True, elem_classes="center-items"):
                    process_btn = gr.Button("Process Documents", variant="primary", size="sm")
                    clear_btn = gr.Button("Clear", variant="stop", size="sm")
                
                progress_box = gr.HTML(
                    label="Status",
                    value="<div style='padding: 10px;'>Upload files and click Process to begin...</div>",
                    elem_id="progress-box"
                )
                
            # Add custom CSS for the progress box
            gr.Markdown("""
                <style>
                    /* Style for the progress box */
                    #progress-box {
                        min-height: 250px;
                        max-height: 500px;
                        overflow-y: auto;
                        background-color: #f5f5f5;  /* Changed from #1a1a1a (black) to light gray */
                        border-radius: 4px;
                        margin-top: 10px;
                        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        color: #333;  /* Ensuring text color is dark for readability on light background */
                    }
                    
                    /* Style for lists in the progress box */
                    #progress-box ul {
                        margin-top: 5px;
                        margin-bottom: 15px;
                        padding-left: 25px;
                    }
                    
                    #progress-box li {
                        margin-bottom: 3px;
                    }
                    
                    /* Make sure the progress bar is visible */
                    .progress-container {
                        width: 100% !important;
                        height: 20px !important;
                        background-color: #333 !important;
                        border-radius: 10px !important;
                        margin: 10px 0 !important;
                        overflow: hidden !important;
                        display: block !important;
                    }
                    
                    .progress-bar {
                        height: 100% !important;
                        background-color: #ff5500 !important;
                        border-radius: 10px !important;
                        transition: width 0.3s ease !important;
                        display: block !important;
                    }
                    
                    /* Ensure Gradio's progress bar is visible */
                    .gradio-container .progress {
                        display: block !important;
                        visibility: visible !important;
                        opacity: 1 !important;
                    }
                </style>
            """)

            # Create a container for the chat interface
            with gr.Column():
                # Add a state component to store chat history for other components to access
                chat_history_state = gr.State([])
                chat = gr.Chatbot(type="messages")
                msg = gr.Textbox(label="Message")
                with gr.Row():
                    submit = gr.Button("Submit", variant="huggingface", size="sm")
                    clear = gr.Button("Clear", variant="stop", size="sm")

                # Add example questions
                gr.Examples(
                    examples=chat_config.example_questions,
                    inputs=msg
                )

            # Vector Visualization Accordion
            with gr.Accordion("Vector Store Visualization", open=False):
                with gr.Row():
                    visualize_btn = gr.Button("Generate Visualization", variant="huggingface", size="sm")
                    plot_status = gr.Textbox(
                        label="Visualization Status",
                        placeholder="Click Generate Visualization to begin...",
                        interactive=False
                    )
                
                # Center the plot using a container with CSS
                with gr.Column(elem_classes="center-plot"):
                    # Plotly output component
                    plot_output = gr.Plot(
                        label="Vector Store Visualization",
                        show_label=True,
                        container=True,
                        elem_classes="plot-container"
                    )

            # Add custom CSS for centering
            gr.Markdown("""
                <style>
                    /* Center the plot container */
                    .center-plot {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        width: 100%;
                        min-height: 700px;  /* Add minimum height */
                    }
                    
                    /* Style the plot container */
                    .plot-container {
                        width: 100%;
                        max-width: 1200px;
                        margin: 0 auto;
                    }
                    
                    /* Center the Plotly plot itself */
                    .plot-container > div {
                        display: flex;
                        justify-content: center;
                        width: 100%;
                    }
                    
                    /* Ensure the SVG is centered */
                    .js-plotly-plot {
                        margin: 0 auto !important;
                    }
                    
                    /* Add responsive behavior */
                    @media (max-width: 768px) {
                        .plot-container {
                            max-width: 100%;
                            padding: 10px;
                        }
                    }
                </style>
            """)
            
            # Chat History Conceptual Diagram Accordion
            with gr.Accordion("Chat History Conceptual Diagram", open=False, elem_id="chat-diagram-accordion"):
                with gr.Row():
                    create_diagram_btn = gr.Button("Generate Conceptual Diagram", variant="primary", size="sm")
                    diagram_status = gr.Textbox(
                        label="Diagram Status",
                        placeholder="Click Generate Conceptual Diagram to begin...",
                        interactive=False
                    )
                
                # Previously had renderer selection here, now using streamlit_mermaid styling by default
                
                # Container for the diagram output
                with gr.Column(elem_classes="center-diagram"):
                    # Create the HTML component for diagram output
                    diagram_output = gr.HTML(
                        label="Chat Conceptual Diagram",
                        show_label=True,
                        container=True,
                        elem_classes="diagram-container"
                    )
            
            # Add custom CSS for the diagram container
            gr.HTML("""
                <style>
                    /* Center the diagram container */
                    .center-diagram {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        width: 100%;
                        min-height: 500px;
                    }
                    
                    /* Style the diagram container */
                    .diagram-container {
                        width: 100%;
                        max-width: 1000px;
                        margin: 0 auto;
                    }
                </style>
            """, visible=False)

            # Event handlers
            async def chat_response(message, history, history_state):
                history = history or []
                history_state = history_state or []
                
                # Add user message to history immediately using messages format
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": ""})
                
                # Update the history state
                history_state = history.copy()
                
                # Create a generator for streaming responses
                try:
                    # First yield the history with empty response to show user message immediately
                    yield history, history_state
                    
                    # Get the full response (non-streaming first to debug)
                    full_response = ""
                    
                    # Stream the response
                    async for partial_response in chat_interface.stream_chat(message, history[:-1]):
                        # Update the last assistant message with the current content
                        full_response = partial_response['content']
                        history[-1] = {"role": "assistant", "content": full_response}
                        history_state[-1] = {"role": "assistant", "content": full_response}
                        
                        # Yield the updated history and state
                        yield history, history_state
                    
                except Exception as e:
                    logger.error(f"Error in chat response: {str(e)}", exc_info=True)
                    history[-1] = {"role": "assistant", "content": f"Error: {str(e)}"}
                    yield history

            async def process_documents(loader_type):
                try:
                    logger.info("Starting document processing...")
                    print("\n=== Document Processing Started ===")
                    
                    # Show loader type in progress box
                    status_html = f"<div style='padding: 10px;'><b>\ud83d\udcda Document Loader:</b> {loader_type}<br/><br/>Processing documents...</div>"
                    yield status_html
                    
                    # Fix: Create config objects first
                    postgres_config = PostgresConfig(
                        connection_string=chat_interface.postgres_config.connection_string,
                        pre_delete_collection=False,
                        drop_existing=False
                    )
                    
                    # Determine which loader to use based on selection
                    use_docling = False
                    use_mistral = False
                    loader_name = "LangChain"
                    
                    if loader_type == "Docling (Enhanced)":
                        use_docling = True
                        loader_name = "Docling (Enhanced)"
                    elif loader_type == "Mistral OCR":
                        use_mistral = True
                        loader_name = "Mistral OCR"
                        
                    # Log the selected loader
                    logger.info(f"Using document loader: {loader_name}")
                    print(f"\n=== 📚 Document Loader: {loader_name} ===\n")
                    
                    processor_config = ProcessorConfig(
                        chunk_size=500,
                        chunk_overlap=50,
                        vector_store_type="postgres",
                        postgres_config=postgres_config,
                        batch_size=50,
                        max_workers=1,
                        db_pool_size=5,
                        openai_api_key=OPENAI_API_KEY,
                        # Set loader options based on user selection
                        use_docling=use_docling,
                        docling_artifacts_path=os.getenv('DOCLING_ARTIFACTS_PATH'),
                        docling_enable_remote=os.getenv('DOCLING_ENABLE_REMOTE', '').lower() in ('true', '1', 'yes'),
                        docling_use_cache=os.getenv('DOCLING_USE_CACHE', 'true').lower() in ('true', '1', 'yes'),
                        use_mistral=use_mistral,
                        mistral_api_key=os.getenv('MISTRAL_API_KEY'),
                        mistral_ocr_model=os.getenv('MISTRAL_OCR_MODEL', 'mistral-ocr-latest'),
                        mistral_include_images=os.getenv('MISTRAL_INCLUDE_IMAGES', '').lower() in ('true', '1', 'yes')
                    )
                    
                    # Initialize document processor with config object
                    processor = DocumentProcessor(config=processor_config)
                    
                    # Initialize database
                    await processor.initialize_database()
                    
                    # Get all files in documents directory
                    all_files = list(chat_interface.documents_dir.glob('*.*'))
                    total_files = len(all_files)
                    print(f"Found {total_files} files to process")
                    
                    # Process files with checksum verification
                    total_chunks = 0
                    processed_files = []
                    skipped_files = []
                    
                    # First, get all existing checksums in a separate session
                    existing_checksums = {}
                    existing_filenames = {}
                    
                    async with chat_interface.async_session() as session:
                        # Get all existing documents
                        result = await session.execute(select(DocumentModel))
                        existing_docs = result.scalars().all()
                        
                        for doc in existing_docs:
                            existing_checksums[doc.checksum] = doc.id
                            existing_filenames[doc.filename] = doc.checksum
                    
                    print(f"\n=== Existing Documents in Database ===")
                    print(f"Found {len(existing_checksums)} documents in database")
                    
                    # Now process each file individually without an outer session
                    for file_path in all_files:
                        try:
                            # Calculate checksum
                            file_checksum = chat_interface.calculate_file_checksum(str(file_path))
                            print(f"\n===== Checking file: {file_path.name} =====")
                            print(f"Calculated checksum: {file_checksum}")
                            
                            # Check if file exists in database by checksum
                            checksum_exists = file_checksum in existing_checksums
                            print(f"Checksum exists in database? {checksum_exists}")
                            
                            if checksum_exists:
                                print(f"⏩ Skipping {file_path.name} - found matching checksum in database")
                                logger.info(f"Skipping {file_path.name} - already exists with checksum {file_checksum}")
                                skipped_files.append(file_path.name)
                                continue
                            
                            # Also check by filename as a fallback
                            if file_path.name in existing_filenames:
                                db_checksum = existing_filenames[file_path.name]
                                print(f"⚠️ File {file_path.name} exists in DB but with different checksum!")
                                print(f"  DB checksum: {db_checksum[:8]}...")
                                print(f"  File checksum: {file_checksum[:8]}...")
                                # Skip if filename matches but checksum doesn't
                                skipped_files.append(file_path.name)
                                continue
                            
                            # Process file - let the processor handle its own session
                            print(f"🔄 Processing {file_path.name}...")
                            logger.info(f"Processing {file_path.name}")
                            chunks = await chat_interface.process_single_file(file_path, processor_config)
                            
                            if chunks:
                                total_chunks += len(chunks)
                                processed_files.append(file_path.name)
                                print(f"✅ Processed {file_path.name} - created {len(chunks)} chunks")
                                
                                # Update our local cache of checksums and filenames
                                existing_checksums[file_checksum] = True
                                existing_filenames[file_path.name] = file_checksum
                            
                        except Exception as e:
                            print(f"❌ Error processing {file_path.name}: {str(e)}")
                            logger.error(f"Error processing {file_path.name}: {str(e)}")
                            continue
                    
                    # Create HTML status message with loader info
                    html_status = f"<div style='padding: 10px;'>"
                    html_status += f"<h3>✅ Processing Complete</h3>"
                    
                    # Add document loader information
                    html_status += f"<p><b>📚 Document Loader:</b> {loader_type}</p>"
                    
                    # Add loader-specific configuration details
                    if use_docling:
                        html_status += f"<p><b>Docling Configuration:</b></p>"
                        html_status += "<ul>"
                        html_status += f"<li>Remote Processing: {os.getenv('DOCLING_ENABLE_REMOTE', 'false').lower() in ('true', '1', 'yes')}</li>"
                        html_status += f"<li>Cache Enabled: {os.getenv('DOCLING_USE_CACHE', 'true').lower() in ('true', '1', 'yes')}</li>"
                        html_status += "</ul>"
                    elif use_mistral:
                        html_status += f"<p><b>Mistral OCR Configuration:</b></p>"
                        html_status += "<ul>"
                        html_status += f"<li>Model: {os.getenv('MISTRAL_OCR_MODEL', 'mistral-ocr-latest')}</li>"
                        html_status += f"<li>Include Images: {os.getenv('MISTRAL_INCLUDE_IMAGES', 'false').lower() in ('true', '1', 'yes')}</li>"
                        html_status += "</ul>"
                    
                    # Add processing statistics
                    html_status += f"<p><b>Processing Summary:</b></p>"
                    html_status += "<ul>"
                    html_status += f"<li>Total files: {len(all_files)}</li>"
                    html_status += f"<li>Processed files: {len(processed_files)}</li>"
                    html_status += f"<li>Skipped files: {len(skipped_files)}</li>"
                    html_status += f"<li>Total chunks created: {total_chunks}</li>"
                    html_status += "</ul>"
                    
                    # List processed files if any
                    if processed_files:
                        html_status += f"<p><b>Files Processed:</b></p>"
                        html_status += "<ul>"
                        for file in processed_files:
                            html_status += f"<li>{file}</li>"
                        html_status += "</ul>"
                    
                    html_status += "</div>"
                    
                    # Also print to console
                    print(f"\n=== Processing Summary ===\n")
                    print(f"Document Loader: {loader_type}")
                    print(f"Total files: {len(all_files)}")
                    print(f"Processed files: {len(processed_files)}")
                    print(f"Skipped files: {len(skipped_files)}")
                    print(f"Total chunks created: {total_chunks}")
                    
                    # Use yield instead of return since we're in an async generator function
                    yield html_status
                    
                except Exception as e:
                    error_message = f"<div style='color: red; padding: 10px;'>Error processing documents with {loader_type} loader: {str(e)}</div>"
                    logger.error(error_message, exc_info=True)
                    yield error_message

            # The generate_chat_diagram function is now implemented further below
                    
            async def generate_visualization():
                try:
                    plot_status.value = "Generating visualization... This may take a while."
                    
                    # Create async session
                    async with chat_interface.async_session() as session:
                        # Generate visualization
                        fig = await visualize_vector_store(session)
                        
                        return (
                            fig,
                            "Visualization generated successfully! Hover over points to see document details."
                        )
                except Exception as e:
                    error_msg = f"Error generating visualization: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    return (
                        None,
                        error_msg
                    )

            def clear_upload():
                return "<div style='padding: 10px;'>Upload files and click Process to begin...</div>"

            # Connect the chat components with proper Gradio streaming configuration
            submit.click(
                chat_response,
                inputs=[msg, chat, chat_history_state],
                outputs=[chat, chat_history_state],
                api_name="submit",
            ).then(
                lambda: "",  # Clear the message box after submission
                None,
                msg
            )

            # Also enable streaming for the Enter key
            msg.submit(
                chat_response,
                inputs=[msg, chat, chat_history_state],
                outputs=[chat, chat_history_state],
                api_name="submit_msg",
            ).then(
                lambda: "",  # Clear the message box after submission
                None,
                msg
            )

            # Clear both the chat display and history state when clearing
            clear.click(lambda: (None, []), None, [chat, chat_history_state])

            # Connect file upload components
            upload_button.upload(
                fn=chat_interface.handle_file_upload,
                inputs=upload_button,
                outputs=progress_box,
            )
            
            process_btn.click(
                fn=chat_interface.process_documents,
                inputs=loader_type,
                outputs=progress_box,
                show_progress=True,
            )
            
            clear_btn.click(
                fn=clear_upload,
                inputs=None,
                outputs=progress_box,
            )
            
            # Connect visualization button
            visualize_btn.click(
                fn=generate_visualization,
                inputs=None,
                outputs=[plot_output, plot_status],
            )
            
            # Define function to generate conceptual diagram from chat history using Mermaid
            async def generate_chat_diagram(history: List[Dict[str, str]]) -> Tuple[Optional[str], str]:
                """Render Mermaid diagrams with streamlit_mermaid styling from LLM output with essential error handling"""
                try:
                    if not history:
                        return None, "No chat history available"

                    # Extract and format chat history
                    chat_history = '\n'.join([f"{msg['role']}: {msg['content']}" for msg in history])
                    
                    # Get Mermaid code from LLM with metrics tracking
                    start_time = time.time()
                    success = False
                    try:
                        # Call LLM to generate Mermaid diagram
                        response = await chat_interface.llm.ainvoke(f"Create Mermaid diagram from:\n{chat_history}")
                        success = True
                    except Exception as e:
                        # Log the error and return user-friendly message
                        logging.error(f"LLM invocation failed: {str(e)}")
                        return None, f"Error generating diagram: {str(e)}"
                    finally:
                        # Record metrics
                        elapsed_time = time.time() - start_time
                        logging.info(f"Diagram generation took {elapsed_time:.2f}s (success={success})")
                    # Extract Mermaid code from response with better error handling
                    try:
                        # Log the COMPLETE raw LLM response for debugging
                        logging.info(f"==========RAW LLM RESPONSE==========\n{response.content}\n==========END RAW LLM RESPONSE==========\n")
                        
                        # First try to extract Mermaid code from a code block (multiple formats)
                        mermaid_match = re.search(r'```mermaid\n(.*?)\n```', response.content, re.DOTALL)
                        if not mermaid_match:
                            # Try alternative formats
                            mermaid_match = re.search(r'```\n(.*?)\n```', response.content, re.DOTALL)
                        if not mermaid_match:
                            # Try without newlines after backticks
                            mermaid_match = re.search(r'```mermaid(.+?)```', response.content, re.DOTALL)
                        if not mermaid_match:
                            mermaid_match = re.search(r'```(.+?)```', response.content, re.DOTALL)
                            
                        if mermaid_match:
                            mermaid_code = mermaid_match.group(1)
                            # Log the raw extracted Mermaid code without any modifications
                            logging.info(f"==========RAW MERMAID CODE FROM LLM==========\n{mermaid_code}\n==========END RAW MERMAID CODE==========\n")
                        else:
                            # Use the full response if no Mermaid code block found
                            mermaid_code = response.content
                            logging.warning(f"No Mermaid code block found, using full response as Mermaid code\n")
                            
                        # Security validation of the Mermaid code
                        # Check for potential XSS or injection attacks
                        dangerous_patterns = [
                            r'<script', r'javascript:', r'onerror=', r'onclick=', 
                            r'onload=', r'eval\(', r'document\.', r'window\.'
                        ]
                        
                        for pattern in dangerous_patterns:
                            if re.search(pattern, mermaid_code, re.IGNORECASE):
                                error_msg = f"Security violation detected in Mermaid code: {pattern}"
                                logging.error(error_msg)
                                return None, error_msg
                                
                    except Exception as e:
                        logging.error(f"Error extracting Mermaid code: {str(e)}")
                        return None, f"Error processing diagram: {str(e)}"

                    # Apply minimal needed fixes to make Mermaid syntax work while preserving content
                    try:
                        # Create a guaranteed-valid fallback diagram with modern Mermaid syntax
                        fallback_code = (
                            "flowchart TD\n" +
                            "    A[\"Chat History\"] -->|\"Analyzed by\"| B[\"LLM\"]\n" +
                            "    B -->|\"Generates\"| C[\"Mermaid Diagram\"]\n" +
                            "    C -->|\"Displayed in\"| D[\"User Interface\"]\n"
                        )
                        
                        # Only use fallback if absolutely no mermaid code was found
                        if not mermaid_code or mermaid_code.strip() == "":
                            logging.warning("No valid mermaid code found, using fallback diagram")
                            mermaid_code = fallback_code
                        else:
                            logging.info("Applying minimal fixes to Mermaid syntax while preserving content")
                            
                            # Remove leading/trailing whitespace
                            mermaid_code = mermaid_code.strip()
                            
                            # Log the original code
                            logging.info(f"ORIGINAL MERMAID CODE:\n{mermaid_code}\n")
                            
                            # Make sure diagram starts with a valid directive using modern syntax
                            if not re.match(r'^(graph|flowchart|sequenceDiagram|classDiagram|stateDiagram|gantt|pie|er)', mermaid_code, re.IGNORECASE):
                                logging.info("Adding flowchart TD directive as the diagram has no directive")
                                mermaid_code = "flowchart TD\n" + mermaid_code
                            # Convert older 'graph TD' syntax to modern 'flowchart TD'
                            elif re.match(r'^graph\s+(TD|TB|BT|RL|LR)', mermaid_code, re.IGNORECASE):
                                logging.info("Converting from graph to flowchart directive for better compatibility")
                                mermaid_code = re.sub(r'^graph\s+(TD|TB|BT|RL|LR)', r'flowchart \1', mermaid_code, flags=re.IGNORECASE)
                            
                            # Ensure line breaks are consistent
                            mermaid_code = re.sub(r'\r\n', '\n', mermaid_code)
                            
                            # Fix common syntax issues with minimal impact
                            # 1. Ensure nodes have proper connections
                            lines = mermaid_code.split('\n')
                            fixed_lines = []
                            for line in lines:
                                line = line.rstrip()
                                if line and not line.isspace():
                                    # Quote node text for better handling of spaces and special characters
                                    # Look for node patterns like A[Text] or A(Text) and add quotes around the text
                                    line = re.sub(r'(\w+)\[(.*?)\]', r'\1["\2"]', line)
                                    line = re.sub(r'(\w+)\((.*?)\)', r'\1("\2")', line)
                                    # Also handle edge labels like -->|text| by adding quotes around the text
                                    line = re.sub(r'(-->|---)\|(.*?)\|', r'\1|"\2"|', line)
                                    fixed_lines.append(line)
                            
                            # Join back with proper line endings
                            mermaid_code = '\n'.join(fixed_lines)
                            
                            # Log the fixed code
                            logging.info(f"MINIMALLY FIXED MERMAID CODE:\n{mermaid_code}\n")
                        
                    except Exception as e:
                        logging.error(f"Error preparing Mermaid code: {str(e)}")
                        # Absolute fallback for syntax errors with modern syntax and quoted text
                        mermaid_code = "flowchart TD\n    A[\"Syntax Error\"] -->|\"in\"| B[\"Diagram\"]\n    B -->|\"Using\"| C[\"Fallback Diagram\"]"

                    # Define streamlit_mermaid styling with improved responsive layout
                    config = {
                        "css": ".mermaid { background: #f0f2f6; padding: 1rem; width: 100%; max-width: 100%; overflow-x: visible; } svg { max-width: 100% !important; height: auto !important; }",
                        "js": "document.body.style.background = '#f0f2f6';" 
                    }

                    # Enhanced HTML template with streamlit_mermaid styling
                    html_template = '''
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <!-- Use a very stable version of Mermaid (8.8.0) which has excellent backwards compatibility -->
                        <script src="https://cdn.jsdelivr.net/npm/mermaid@8.8.0/dist/mermaid.min.js"></script>
                        <style>
                            body { margin: 0; padding: 0; overflow-x: hidden; background-color: #ffffff; }
                            .container { width: 100%; max-width: 100%; min-height: 500px; overflow-x: hidden; background-color: #ffffff; }
                            .mermaid { font-family: sans-serif; width: 100%; max-width: 100%; overflow: visible; text-align: center; background-color: #ffffff; padding: 0; }
                            .error-message { color: #d32f2f; background-color: #ffebee; padding: 10px; border-radius: 4px; margin-bottom: 15px; display: none; }
                            svg { width: 100% !important; max-width: 100% !important; min-height: 500px !important; background-color: #ffffff !important; }
                            /* Force normal text size in all diagram elements */
                            .node rect, .node circle, .node ellipse, .node polygon, .node path { stroke-width: 1px !important; }
                            .node text, .edgeLabel text { font-size: 14px !important; }
                            .edgePath { stroke-width: 1px !important; }
                            /* Prevent diagram from being zoomed in */
                            .label { font-size: 14px !important; }
                            /* Additional styling for better diagram presentation */
                            .mermaid > svg { min-height: 500px; }
                            pre.mermaid { background: #ffffff; padding: 0; margin: 0; }
                            /* Download button styling */
                            .download-buttons { display: flex; justify-content: center; margin-top: 15px; gap: 10px; }
                            .download-btn { 
                                background-color: #4CAF50; 
                                color: white; 
                                border: none; 
                                padding: 8px 16px; 
                                text-align: center; 
                                text-decoration: none; 
                                display: inline-block; 
                                font-size: 14px; 
                                margin: 4px 2px; 
                                cursor: pointer; 
                                border-radius: 4px;
                            }
                            .download-btn:hover { background-color: #45a049; }
                            /* CSS_PLACEHOLDER */
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <div class="error-message" id="mermaid-error">Error rendering diagram. Please check syntax.</div>
                            <pre class="mermaid">
%%MERMAID_CODE%%
                            </pre>
                            <div class="download-buttons">
                                <button class="download-btn" id="download-svg-btn" onclick="downloadAsSvg()">Download SVG</button>
                                <button class="download-btn" id="download-png-btn" onclick="downloadAsPng()">Download PNG</button>
                            </div>
                        </div>
                        <script>
                            // Configure Mermaid with responsive settings and better error handling
                            mermaid.initialize({ 
                                startOnLoad: false, // We'll manually render
                                securityLevel: 'loose',
                                theme: 'default',
                                logLevel: 1,
                                flowchart: { 
                                    htmlLabels: true, 
                                    useMaxWidth: true,
                                    curve: 'monotoneX',
                                    diagramPadding: 20,
                                    nodeSpacing: 60,
                                    rankSpacing: 100,
                                    rankDir: 'TB', // Top to Bottom layout instead of Left to Right
                                    ranker: 'tight-tree' // Better for hierarchical diagrams
                                },
                                sequence: { useMaxWidth: true },
                                gantt: { useMaxWidth: true },
                                pie: { useMaxWidth: true },
                                fontFamily: 'sans-serif',
                                fontSize: 14,
                                fontWeight: 400
                            });
                            
                            // Manual rendering with enhanced error handling and responsive layout
                            document.addEventListener('DOMContentLoaded', function() {
                                try {
                                    // Pre-validate diagram syntax
                                    try {
                                        var element = document.querySelector('.mermaid');
                                        var graphDefinition = element.textContent;
                                        mermaid.parse(graphDefinition);
                                    } catch (syntaxError) {
                                        console.error('Mermaid syntax validation error:', syntaxError);
                                        document.getElementById('mermaid-error').style.display = 'block';
                                        document.getElementById('mermaid-error').textContent = 'Diagram syntax error: ' + syntaxError.message;
                                    }
                                    
                                    // Initialize with error handling
                                    mermaid.init(undefined, document.querySelectorAll('.mermaid'));
                                    
                                    // Ensure SVG fits container width and fix zoom issues
                                    setTimeout(function() {
                                        const svgElements = document.querySelectorAll('svg');
                                        if (svgElements.length === 0) {
                                            console.warn('No SVG elements found after render');
                                            return;
                                        }
                                        
                                        svgElements.forEach(function(svg) {
                                            // Fix zoom issues and ensure diagram uses full canvas
                                            const bbox = svg.getBBox();
                                            // Add padding to viewBox for better layout
                                            const padding = 20;
                                            svg.setAttribute('viewBox', `${bbox.x - padding} ${bbox.y - padding} ${bbox.width + padding*2} ${bbox.height + padding*2}`);
                                            // Ensure background is pure white
                                            svg.style.backgroundColor = '#ffffff';
                                            svg.style.padding = '0';
                                            svg.style.margin = '0 auto';
                                            
                                            // Set responsive styling
                                            svg.setAttribute('width', '100%');
                                            svg.style.maxWidth = '100%';
                                            svg.style.height = 'auto';
                                            svg.style.display = 'block';
                                            
                                            // Add transform for proper scaling
                                            svg.style.transformOrigin = 'center';
                                            svg.style.margin = '0 auto';
                                            
                                            // Remove any fixed dimensions from child elements
                                            const gElements = svg.querySelectorAll('g');
                                            gElements.forEach(function(g) {
                                                if (g.hasAttribute('width')) g.removeAttribute('width');
                                                if (g.hasAttribute('height')) g.removeAttribute('height');
                                                if (g.hasAttribute('transform')) {
                                                    // Keep important transforms but fix scaling issues
                                                    const transform = g.getAttribute('transform');
                                                    if (transform.includes('scale') && !transform.includes('translate')) {
                                                        g.setAttribute('transform', 'scale(1)');
                                                    }
                                                }
                                            });
                                        });
                                    }, 200);
                                } catch (error) {
                                    console.error('Mermaid rendering error:', error);
                                    document.getElementById('mermaid-error').style.display = 'block';
                                    document.getElementById('mermaid-error').textContent = 'Error rendering diagram: ' + error.message;
                                }
                            });
                            /* Add download functions to the global scope */
                            function downloadAsSvg() {
                                const svg = document.querySelector('svg');
                                if (!svg) {
                                    console.error('No SVG found for download');
                                    return;
                                }
                                
                                try {
                                    // Clone and prepare SVG for download
                                    const svgClone = svg.cloneNode(true);
                                    svgClone.setAttribute('width', svg.getBoundingClientRect().width);
                                    svgClone.setAttribute('height', svg.getBoundingClientRect().height);
                                    svgClone.style.background = '#ffffff';
                                    
                                    // Create downloadable SVG data
                                    const svgData = new XMLSerializer().serializeToString(svgClone);
                                    const svgBlob = new Blob([svgData], {type: 'image/svg+xml;charset=utf-8'});
                                    const svgUrl = URL.createObjectURL(svgBlob);
                                    
                                    // Create download link and trigger download
                                    const downloadLink = document.createElement('a');
                                    downloadLink.href = svgUrl;
                                    downloadLink.download = 'mermaid_diagram_' + new Date().toISOString().split('T')[0] + '.svg';
                                    document.body.appendChild(downloadLink);
                                    downloadLink.click();
                                    document.body.removeChild(downloadLink);
                                    URL.revokeObjectURL(svgUrl);
                                } catch (e) {
                                    console.error('SVG download error:', e);
                                }
                            }
                            
                            function downloadAsPng() {
                                const svg = document.querySelector('svg');
                                if (!svg) {
                                    console.error('No SVG found for download');
                                    return;
                                }
                                
                                try {
                                    // Get the dimensions of the SVG
                                    const bbox = svg.getBoundingClientRect();
                                    const width = bbox.width;
                                    const height = bbox.height;
                                    
                                    // Create a canvas with 2x resolution for better quality
                                    const canvas = document.createElement('canvas');
                                    canvas.width = width * 2;
                                    canvas.height = height * 2;
                                    const ctx = canvas.getContext('2d');
                                    
                                    // Set white background
                                    ctx.fillStyle = '#FFFFFF';
                                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                                    
                                    // Scale for higher resolution
                                    ctx.scale(2, 2);
                                    
                                    // Convert SVG to a data URL
                                    const svgData = new XMLSerializer().serializeToString(svg);
                                    const svgImage = new Image();
                                    
                                    // Use a data URL with base64 encoding
                                    const svgBase64 = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgData)));
                                    
                                    // When the image loads, draw it to the canvas and create download link
                                    svgImage.onload = function() {
                                        ctx.drawImage(svgImage, 0, 0);
                                        
                                        try {
                                            // Get the PNG data URL
                                            const pngUrl = canvas.toDataURL('image/png');
                                            
                                            // Create and trigger download
                                            const downloadLink = document.createElement('a');
                                            downloadLink.href = pngUrl;
                                            downloadLink.download = 'mermaid_diagram_' + new Date().toISOString().split('T')[0] + '.png';
                                            document.body.appendChild(downloadLink);
                                            downloadLink.click();
                                            document.body.removeChild(downloadLink);
                                        } catch (e) {
                                            console.error('PNG creation error:', e);
                                        }
                                    };
                                    
                                    // Handle loading errors
                                    svgImage.onerror = function(e) {
                                        console.error('Error loading SVG into image:', e);
                                    };
                                    
                                    // Start loading the image
                                    svgImage.src = svgBase64;
                                } catch (e) {
                                    console.error('PNG download error:', e);
                                }
                            }
                            /* JS_PLACEHOLDER */
                        </script>
                    </body>
                    </html>
                    '''

                    # Replace placeholders with actual content
                    # First substitute the mermaid code, then the CSS and JS
                    final_html = html_template.replace('%%MERMAID_CODE%%', mermaid_code)
                    final_html = final_html.replace('/* CSS_PLACEHOLDER */', config['css'])
                    final_html = final_html.replace('/* JS_PLACEHOLDER */', config['js'])

                    # Safe HTML embedding with additional security measures
                    escaped_html = final_html.replace('"', '&quot;').replace("'", '&#39;')
                    
                    # Log the final Mermaid code to confirm it's properly inserted
                    # Extract the code from between the mermaid pre tags (updated from div to pre)
                    mermaid_in_html = re.search(r'<pre class="mermaid">\s*([\s\S]*?)\s*</pre>', final_html)
                    if mermaid_in_html:
                        extracted_code = mermaid_in_html.group(1).strip()
                        logging.info(f"FINAL MERMAID CODE IN HTML: \n{extracted_code}\n")
                    else:
                        logging.warning("Failed to extract Mermaid code from HTML for logging")
                    
                    # Create the iframe HTML in a simpler way to avoid f-string syntax issues
                    iframe_html = '<iframe '
                    iframe_html += f'srcdoc="{escaped_html}" '
                    iframe_html += 'style="width:100%;height:700px;border:none;overflow:hidden;background-color:white;" '
                    iframe_html += 'sandbox="allow-scripts allow-popups allow-downloads allow-same-origin" '
                    iframe_html += 'referrerpolicy="no-referrer" '
                    
                    # Add onload JavaScript as a separate string
                    onload_js = '''
                    onload="
                        this.style.height = Math.max(500, this.contentWindow.document.body.scrollHeight + 50) + 'px';
                        this.contentWindow.document.body.style.overflow = 'hidden';
                        this.contentWindow.document.body.style.background = '#ffffff';
                        
                        // Setup direct download buttons with inline functions
                        const downloadSvgBtn = this.contentWindow.document.getElementById('download-svg-btn');
                        const downloadPngBtn = this.contentWindow.document.getElementById('download-png-btn');
                        
                        const svgElements = this.contentWindow.document.querySelectorAll('svg');
                        for (const svg of svgElements) {
                            svg.setAttribute('width', '100%');
                            svg.style.maxWidth = '100%';
                            svg.style.minHeight = '500px';
                            svg.style.height = 'auto';
                            svg.style.background = '#ffffff';
                            svg.style.display = 'block';
                            svg.style.margin = '0 auto';
                            
                            // Add padding to viewBox for better layout
                            try {
                                // Only try to get bbox if the SVG is rendered
                                if (svg.getBoundingClientRect().width > 0) {
                                    const bbox = svg.getBBox();
                                    const padding = 20;
                                    svg.setAttribute('viewBox', (bbox.x - padding) + ' ' + (bbox.y - padding) + ' ' + (bbox.width + padding*2) + ' ' + (bbox.height + padding*2));
                                } else {
                                    // Fallback for SVGs that aren't properly rendered yet
                                    svg.setAttribute('viewBox', '0 0 800 600');
                                }
                                } catch (e) {
                                    console.log('Error getting bbox:', e);
                                    // Safe fallback viewBox
                                    svg.setAttribute('viewBox', '0 0 800 600');
                                }
                            svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
                            
                            // Remove width/height restrictions
                            const gElements = svg.querySelectorAll('g');
                            gElements.forEach(function(g) {
                                if (g.hasAttribute('width')) g.removeAttribute('width');
                                if (g.hasAttribute('height')) g.removeAttribute('height');
                                if (g.hasAttribute('transform') && g.getAttribute('transform').includes('scale(0.')) {
                                    g.setAttribute('transform', 'scale(1)');
                                }
                            });
                        }
                    "
                    '''
                    
                    # Complete the iframe HTML
                    iframe_html += onload_js + '></iframe>'
                    
                    # Success message with timestamp for auditability
                    success_msg = f"Diagram generated successfully at {datetime.now().strftime('%H:%M:%S')}"
                    logging.info(success_msg)
                    
                    return iframe_html, success_msg

                except Exception as e:
                    error_msg = f"Diagram error: {str(e)}"
                    logger.error(error_msg)
                    return None, error_msg

            # Connect conceptual diagram button
            create_diagram_btn.click(
                fn=generate_chat_diagram,
                inputs=[chat_history_state],
                outputs=[diagram_output, diagram_status],
            )

        await demo.launch(
            server_name=chat_config.server_name, # Use config value
            server_port=chat_config.server_port,
            share=chat_config.share,
            # Add other parameters like share=True if needed
        )
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise
    finally:
        if chat_interface:
            await chat_interface.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application failed: {str(e)}") 