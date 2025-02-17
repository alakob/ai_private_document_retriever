import os
import logging
from typing import List, Optional, Union, Literal, Generator, Dict
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, Docx2txtLoader, UnstructuredPowerPointLoader, UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma, PGVector
from dataclasses import dataclass
import numpy as np
from contextlib import contextmanager
import sqlalchemy
from sqlalchemy import (
    create_engine, 
    Column, 
    String, 
    Integer, 
    Float, 
    JSON, 
    ForeignKey, 
    text, 
    func,
    DateTime  # Add this import
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import json
from sqlalchemy.dialects.postgresql import ARRAY, JSONB  # Add JSONB import
from datetime import datetime
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial, wraps
from tqdm.asyncio import tqdm_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.dialects.postgresql import insert
from pgvector.sqlalchemy import Vector
from sqlalchemy.sql import select
import time
from collections import defaultdict
import multiprocessing as mp
import psutil
import random
import hashlib

from dotenv import load_dotenv
from models import Base, DocumentModel, DocumentChunk
from config import (
    PostgresConfig, 
    ProcessorConfig,
    RateLimitConfig,
    ResourceConfig
)

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Add these lines to control SQLAlchemy logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.dialects').setLevel(logging.WARNING)


@contextmanager
def create_pgvector_store(
    docs: List[Document], 
    embeddings: OpenAIEmbeddings, 
    config: PostgresConfig
) -> Generator[PGVector, None, None]:
    """Context manager for PGVector to ensure proper connection handling."""
    try:
        store = PGVector.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=config.collection_name,
            connection_string=config.connection_string,
            pre_delete_collection=config.pre_delete_collection,
            embedding_dimension=config.embedding_dimension,
            collection_metadata={"metadata_field_type": "jsonb"}
        )
        yield store
    finally:
        if 'store' in locals():
            if hasattr(store, '_conn'):
                store._conn.close()
            if hasattr(store, '_engine'):
                store._engine.dispose()

# Add custom exceptions
class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""
    pass

class FileLoadError(DocumentProcessingError):
    """Raised when a file cannot be loaded."""
    pass

class EmbeddingError(DocumentProcessingError):
    """Raised when embeddings cannot be generated."""
    pass

class DatabaseError(DocumentProcessingError):
    """Raised when database operations fail."""
    pass

# Update utility functions with better error handling
async def create_vector_index(conn) -> None:
    """Create vector index for document chunks."""
    try:
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS embedding_idx 
            ON document_chunks 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """))
    except Exception as e:
        logger.error(f"Failed to create vector index: {str(e)}")
        raise DatabaseError(f"Vector index creation failed: {str(e)}") from e

async def get_embeddings_with_backoff(
    embeddings: OpenAIEmbeddings,
    texts: List[str],
    max_retries: int = 3,
    base_delay: float = 1.0
) -> List[List[float]]:
    """Get embeddings with exponential backoff retry."""
    if not texts:
        raise ValueError("No texts provided for embedding generation")
        
    for attempt in range(max_retries):
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: embeddings.embed_documents(texts)
            )
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to get embeddings after {max_retries} attempts: {str(e)}")
                raise EmbeddingError(f"Embedding generation failed after {max_retries} attempts") from e
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Embedding attempt {attempt + 1} failed, retrying in {delay}s")
            await asyncio.sleep(delay)

def create_chunk_records(
    doc_id: int,
    chunks: List[Document],
    embeddings: List[List[float]]
) -> List[Dict]:
    """Create chunk records for database insertion."""
    return [
        {
            'document_id': doc_id,
            'content': chunk.page_content,
            'embedding': embedding,
            'chunk_index': idx,
            'chunk_size': len(chunk.page_content),
            'page_number': chunk.metadata.get('page', None),
            'section_title': chunk.metadata.get('section_title', None),
            'chunk_metadata': {
                **chunk.metadata,
                'processed_at': datetime.now().isoformat(),
                'embedding_model': 'text-embedding-ada-002'
            }
        }
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

def create_document_metadata(file_path: str) -> Dict:
    """Create metadata for document record with checksum."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        return {
            'file_type': Path(file_path).suffix.lower(),
            'created_at': datetime.now().isoformat(),
            'file_size': os.path.getsize(file_path)
        }
    except Exception as e:
        logger.error(f"Failed to create document metadata for {file_path}: {str(e)}")
        raise DocumentProcessingError(f"Metadata creation failed: {str(e)}") from e

# Add retry decorator for resilient operations
def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exceptions: tuple = (Exception,)
) -> callable:
    """Decorator for retrying operations with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                        raise
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s")
                    await asyncio.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

# Add validation decorator
def validate_input(func):
    """Decorator for validating input parameters."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Get function's type hints
        hints = func.__annotations__
        
        # Validate each parameter
        for param_name, param_type in hints.items():
            if param_name == 'return':
                continue
            if param_name in kwargs:
                value = kwargs[param_name]
                if not isinstance(value, param_type):
                    raise ValueError(f"Parameter {param_name} must be of type {param_type}")
        
        return await func(*args, **kwargs)
    return wrapper

# Add connection pooling configuration
POOL_CONFIG = {
    'pool_size': 20,
    'max_overflow': 10,
    'pool_timeout': 30,
    'pool_recycle': 1800,
    'pool_pre_ping': True
}

# Add batch processing configuration
BATCH_CONFIG = {
    'chunk_batch_size': 1000,
    'embedding_batch_size': 100,
    'db_batch_size': 500,
    'max_concurrent_files': os.cpu_count(),  # Use number of CPU cores
    'process_pool_workers': max(os.cpu_count() - 1, 1),  # Leave one core free
}

# Add caching decorator for expensive operations
def cache_result(ttl_seconds: int = 3600):
    """Cache decorator with TTL."""
    cache = {}
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            current_time = time.time()
            
            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < ttl_seconds:
                    return result
                    
            result = await func(*args, **kwargs)
            cache[key] = (result, current_time)
            return result
        return wrapper
    return decorator

# Add this function at module level
def extract_page_content(batch: List[Document]) -> List[str]:
    """Extract page content from a batch of documents."""
    return [doc.page_content for doc in batch]

# Add a monitoring class
class ProcessingMonitor:
    def __init__(self):
        self.start_times = defaultdict(dict)
        self.end_times = defaultdict(dict)
        self.active_tasks = 0
        self._lock = asyncio.Lock()
    
    async def start_task(self, file_path: str):
        async with self._lock:
            self.start_times[file_path] = time.time()
            self.active_tasks += 1
            logger.info(
                f"Started processing {file_path}. "
                f"Active tasks: {self.active_tasks}"
            )
    
    async def end_task(self, file_path: str):
        async with self._lock:
            self.end_times[file_path] = time.time()
            self.active_tasks -= 1
            duration = self.end_times[file_path] - self.start_times[file_path]
            logger.info(
                f"Finished processing {file_path} in {duration:.2f}s. "
                f"Active tasks: {self.active_tasks}"
            )
    
    def get_statistics(self) -> Dict:
        """Get processing statistics with default values."""
        total_files = len(self.end_times)
        if total_files == 0:
            return {
                "total_files": 0,
                "avg_duration": 0.0,
                "max_duration": 0.0,
                "min_duration": 0.0,
                "total_duration": 0.0
            }
            
        durations = [
            self.end_times[f] - self.start_times[f] 
            for f in self.end_times.keys()
        ]
        
        return {
            "total_files": total_files,
            "avg_duration": sum(durations) / len(durations),
            "max_duration": max(durations),
            "min_duration": min(durations),
            "total_duration": max(self.end_times.values()) - min(self.start_times.values())
        }

    def get_summary(self) -> str:
        """Get a formatted summary of processing statistics."""
        stats = self.get_statistics()
        return (
            f"Processing Statistics:\n"
            f"Total files: {stats['total_files']}\n"
            f"Average processing time: {stats['avg_duration']:.2f}s\n"
            f"Maximum processing time: {stats['max_duration']:.2f}s\n"
            f"Minimum processing time: {stats['min_duration']:.2f}s\n"
            f"Total duration: {stats['total_duration']:.2f}s"
        )

# Update the process_document_parallel function
async def process_document_parallel(
    file_path: Path,
    processor: 'DocumentProcessor',
    semaphore: asyncio.Semaphore,
    monitor: ProcessingMonitor
) -> Optional[List[Document]]:
    """Process a single document with error handling and monitoring."""
    try:
        async with semaphore:
            await monitor.start_task(str(file_path))
            result = await processor.process_file(file_path)
            await monitor.end_task(str(file_path))
            return result
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {str(e)}")
        await monitor.end_task(str(file_path))
        return None

# Add monitoring classes
class ResourceMonitor:
    def __init__(self, config: ResourceConfig):
        self.config = config
        self.peak_memory = 0
        self.peak_cpu = 0
        
    def check_resources(self) -> bool:
        """Monitor and manage resource usage."""
        try:
            process = psutil.Process()
            current_memory = process.memory_info().rss / (1024 * 1024)  # MB
            current_cpu = process.cpu_percent()
            
            self.peak_memory = max(self.peak_memory, current_memory)
            self.peak_cpu = max(self.peak_cpu, current_cpu)
            
            # Check system memory
            system_memory = psutil.virtual_memory()
            free_memory_mb = system_memory.available / (1024 * 1024)
            
            if (current_memory > self.config.memory_threshold_mb or 
                free_memory_mb < self.config.min_free_memory_mb):
                logger.warning(
                    f"Memory usage high: {current_memory:.1f}MB, "
                    f"Free: {free_memory_mb:.1f}MB"
                )
                import gc
                gc.collect()
                return False
                
            return True
        except Exception as e:
            logger.error(f"Resource monitoring error: {str(e)}")
            return True  # Continue on monitoring error

class EnhancedRateLimiter:
    """Enhanced rate limiter with token bucket and concurrent request limiting."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests = []
        self.concurrent_requests = 0
        self.lock = asyncio.Lock()
        self.last_request_time = time.time()
        
    async def wait_if_needed(self):
        """Implement enhanced rate limiting with backoff."""
        async with self.lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            self.requests = [t for t in self.requests if now - t < 60]
            
            # Check rate limit
            if len(self.requests) >= self.config.max_requests_per_minute:
                wait_time = 60 - (now - self.requests[0])
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    
            # Check concurrent requests
            while self.concurrent_requests >= self.config.max_concurrent_requests:
                await asyncio.sleep(self.config.min_delay)
            
            # Add jitter to prevent thundering herd
            jitter = random.uniform(0, self.config.jitter)
            
            # Ensure minimum delay between requests
            time_since_last = now - self.last_request_time
            if time_since_last < self.config.min_delay:
                await asyncio.sleep(self.config.min_delay - time_since_last + jitter)
            
            self.requests.append(now)
            self.concurrent_requests += 1
            self.last_request_time = time.time()
    
    async def release(self):
        """Release a concurrent request slot."""
        async with self.lock:
            self.concurrent_requests = max(0, self.concurrent_requests - 1)

    async def backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = min(
            self.config.max_delay,
            self.config.initial_delay * (self.config.backoff_factor ** attempt)
        )
        return delay + random.uniform(0, self.config.jitter)

class AdaptiveBatcher:
    def __init__(self, initial_batch_size: int = 100):
        self.batch_size = initial_batch_size
        self.processing_times = []
        self.min_batch_size = 10
        self.max_batch_size = 200
        self.target_time = 5.0  # seconds
        
    def adjust_batch_size(self, last_processing_time: float) -> int:
        """Dynamically adjust batch size based on processing time."""
        self.processing_times.append(last_processing_time)
        
        # Keep last 5 processing times
        if len(self.processing_times) > 5:
            self.processing_times = self.processing_times[-5:]
            
        avg_time = sum(self.processing_times) / len(self.processing_times)
        
        # Adjust batch size
        if avg_time > self.target_time * 1.2:  # Too slow
            self.batch_size = max(
                self.min_batch_size, 
                int(self.batch_size * 0.8)
            )
        elif avg_time < self.target_time * 0.8:  # Too fast
            self.batch_size = min(
                self.max_batch_size, 
                int(self.batch_size * 1.2)
            )
            
        return self.batch_size

# Add text sanitization
def sanitize_text(text: str) -> str:
    """Clean text for PostgreSQL storage with validation."""
    if not text or not isinstance(text, str):
        return ""
        
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Replace problematic characters
    text = ''.join(
        char for char in text 
        if ord(char) >= 32 or char in '\n\r\t'
    )
    
    # Validate content
    text = text.strip()
    if not text:
        return ""
    
    # Limit length if needed
    max_length = 1_000_000  # Adjust based on your PostgreSQL config
    if len(text) > max_length:
        text = text[:max_length]
        logger.warning(f"Text truncated to {max_length} characters")
    
    return text

# Add checksum function
def calculate_file_checksum(file_path: str) -> str:
    """Calculate SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files efficiently
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# Add function to check for duplicate files
async def check_duplicate_file(session: AsyncSession, file_path: str) -> Optional[DocumentModel]:
    """Check if file already exists in database using checksum."""
    try:
        checksum = calculate_file_checksum(file_path)
        result = await session.execute(
            select(DocumentModel).where(DocumentModel.checksum == checksum)
        )
        return result.scalar_one_or_none()
    except Exception as e:
        logger.error(f"Error checking duplicate file {file_path}: {str(e)}")
        return None

class DocumentProcessor:
    def __init__(self, config: ProcessorConfig):
        """Initialize with performance optimizations."""
        if not isinstance(config, ProcessorConfig):
            raise ValueError("config must be an instance of ProcessorConfig")
        if not config.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self.config = config
        self.text_splitter = CharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=config.openai_api_key,
            model="text-embedding-ada-002"
        )
        
        self.vector_store = None
        
        # Add connection pooling
        if config.vector_store_type == "postgres":
            if not config.postgres_config:
                raise ValueError("PostgreSQL configuration is required")
            
            # Create engine with pool configuration
            self.engine = create_async_engine(
                config.postgres_config.connection_string.replace('postgresql://', 'postgresql+asyncpg://'),
                **POOL_CONFIG
            )
            
            # Create session factory without pool_size
            self.async_session = async_sessionmaker(
                bind=self.engine,
                expire_on_commit=False,
                class_=AsyncSession
            )

        # Add semaphore for concurrent operations
        self.semaphore = asyncio.Semaphore(BATCH_CONFIG['max_concurrent_files'])
        
        # Add process pool for CPU-bound operations
        self.process_pool = ProcessPoolExecutor(
            max_workers=BATCH_CONFIG['process_pool_workers'],
            mp_context=mp.get_context('spawn')  # Ensure proper process isolation
        )
        
        # Add thread pool for I/O operations
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_workers)

        self.file_loaders = {
            '.pdf': lambda path: PyPDFLoader(path).load(),
            '.txt': lambda path: TextLoader(path).load(),
            '.docx': lambda path: Docx2txtLoader(path).load(),
            '.doc': lambda path: Docx2txtLoader(path).load(),
            '.ppt': lambda path: UnstructuredPowerPointLoader(path).load(),
            '.pptx': lambda path: UnstructuredPowerPointLoader(path).load(),
        }

        # Add new components
        self.rate_limiter = EnhancedRateLimiter(config.rate_limit_config)
        self.resource_monitor = ResourceMonitor(ResourceConfig())
        self.adaptive_batcher = AdaptiveBatcher()

    async def initialize_database(self) -> None:
        """Create database tables if they don't exist."""
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
                await conn.run_sync(Base.metadata.drop_all)
                await conn.run_sync(Base.metadata.create_all)
                await create_vector_index(conn)
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise DatabaseError("Failed to initialize database") from e

    async def _load_document(self, file_path: str) -> List[Document]:
        """Load document using appropriate loader based on file extension."""
        file_ext = Path(file_path).suffix.lower()
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_ext not in self.file_loaders:
            logger.warning(f"Unsupported file type: {file_ext}")
            return []
            
        try:
            documents = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                self.file_loaders[file_ext],
                file_path
            )
            return documents
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise FileLoadError(f"Failed to load file {file_path}") from e

    def _split_text_in_process(self, text: str) -> List[str]:
        """Split text in a separate process with validation."""
        if not text or not isinstance(text, str):
            return []

        try:
            chunks = self.text_splitter.split_text(text)
            # Validate chunks
            valid_chunks = [
                chunk for chunk in chunks 
                if chunk and chunk.strip()
            ]
            
            if len(valid_chunks) < len(chunks):
                logger.warning(
                    f"Removed {len(chunks) - len(valid_chunks)} empty chunks "
                    f"from text splitting"
                )
            
            return valid_chunks
        
        except Exception as e:
            logger.error(f"Error in text splitting: {str(e)}")
            return []

    async def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings with improved rate limiting and batching."""
        if not texts:
            return []
        
        # Split into smaller batches
        batch_size = self.config.rate_limit_config.batch_size
        batches = [
            texts[i:i + batch_size] 
            for i in range(0, len(texts), batch_size)
        ]
        
        results = []
        for attempt in range(self.config.rate_limit_config.max_retries):
            try:
                for batch in batches:
                    try:
                        # Wait for rate limit
                        await self.rate_limiter.wait_if_needed()
                        
                        # Get embeddings for batch
                        batch_embeddings = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self.embeddings.embed_documents(batch)
                        )
                        
                        results.extend(batch_embeddings)
                        
                    finally:
                        # Always release the concurrent request slot
                        await self.rate_limiter.release()
                        
                return results
                
            except Exception as e:
                if attempt == self.config.rate_limit_config.max_retries - 1:
                    raise EmbeddingError(
                        f"Failed to get embeddings after "
                        f"{self.config.rate_limit_config.max_retries} attempts"
                    ) from e
                    
                delay = await self.rate_limiter.backoff_delay(attempt)
                logger.warning(
                    f"Embedding attempt {attempt + 1} failed, "
                    f"retrying in {delay:.2f}s: {str(e)}"
                )
                await asyncio.sleep(delay)

    async def _store_chunks_batch(
        self, 
        chunks: List[Document], 
        embeddings: List[List[float]], 
        file_path: str
    ) -> None:
        """Store chunks with improved validation, error handling, and duplicate checking."""
        if not chunks or not embeddings:
            return
            
        try:
            batch_start_time = time.time()
            async with self.async_session() as session:
                async with session.begin():
                    # Check for duplicate file using checksum
                    duplicate_doc = await check_duplicate_file(session, file_path)
                    if duplicate_doc:
                        logger.warning(
                            f"Document {file_path} already exists with same checksum, "
                            f"original file: {duplicate_doc.filename}"
                        )
                        return
                    
                    # Validate chunks before processing
                    valid_chunks = []
                    valid_embeddings = []
                    
                    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        content = sanitize_text(chunk.page_content)
                        if not content:
                            logger.warning(
                                f"Skipping chunk {idx} from {file_path}: Empty or invalid content"
                            )
                            continue
                            
                        chunk.page_content = content
                        valid_chunks.append(chunk)
                        valid_embeddings.append(embedding)
                    
                    if not valid_chunks:
                        logger.warning(f"No valid chunks found in {file_path}")
                        return
                    
                    # Calculate checksum for the file
                    checksum = calculate_file_checksum(file_path)
                    
                    # Insert new document with checksum
                    doc = DocumentModel(
                        filename=str(file_path),
                        doc_metadata=create_document_metadata(file_path),
                        checksum=checksum
                    )
                    session.add(doc)
                    await session.flush()

                    # Create sanitized chunk records
                    chunk_records = [
                        {
                            'document_id': doc.id,
                            'content': chunk.page_content,
                            'embedding': embedding,
                            'chunk_index': idx,
                            'chunk_size': len(chunk.page_content),
                            'page_number': chunk.metadata.get('page', None),
                            'section_title': chunk.metadata.get('section_title', None),
                            'chunk_metadata': {
                                **chunk.metadata,
                                'processed_at': datetime.now().isoformat(),
                                'embedding_model': 'text-embedding-ada-002'
                            }
                        }
                        for idx, (chunk, embedding) in enumerate(zip(valid_chunks, valid_embeddings))
                    ]
                    
                    # Use upsert instead of insert
                    stmt = insert(DocumentChunk).values(chunk_records)
                    stmt = stmt.on_conflict_do_nothing()
                    await session.execute(stmt)
                    await session.commit()
                    
                    logger.info(
                        f"Stored {len(chunk_records)} valid chunks from {file_path} "
                        f"(skipped {len(chunks) - len(valid_chunks)} invalid chunks)"
                    )
                    
                    batch_processing_time = time.time() - batch_start_time
                    self.adaptive_batcher.adjust_batch_size(batch_processing_time)
                    
        except Exception as e:
            logger.error(f"Failed to store batch for {file_path}: {str(e)}")
            raise DatabaseError(f"Failed to store document chunks") from e

    async def process_file(self, file_path: Union[str, Path]) -> List[Document]:
        """Process file with concurrent operations."""
        async with self.semaphore:  # Limit concurrent file processing
            file_path = str(file_path)
            
            if not file_path:
                raise ValueError("file_path cannot be empty")
                
            try:
                logger.info(f"Processing file: {file_path}")
                
                documents = await self._load_document(file_path)
                        
                if not documents:
                    return []

                all_chunks = []
                # Process in optimal batch sizes
                for i in range(0, len(documents), BATCH_CONFIG['chunk_batch_size']):
                    batch = documents[i:i + BATCH_CONFIG['chunk_batch_size']]
                    
                    # Process text splitting in process pool
                    texts = await asyncio.get_event_loop().run_in_executor(
                        self.process_pool,
                        extract_page_content,
                        batch
                    )
                    
                    if not all(isinstance(text, str) for text in texts):
                        continue
                        
                    # Get embeddings with optimized batching
                    embeddings = await self._get_embeddings_batch(texts)
                    
                    if self.config.vector_store_type == "postgres":
                        await self._store_chunks_batch(batch, embeddings, file_path)
                    
                    all_chunks.extend(batch)

                return all_chunks

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                raise

    async def process_directory(self, directory_path: Union[str, Path]) -> List[Document]:
        """Process directory with duplicate prevention using checksums."""
        try:
            start_time = time.time()
            directory_path = Path(directory_path)
            
            # Get unique files to process
            files_to_process = set()
            for ext in ['.pdf', '.txt', '.doc', '.docx']:
                files_to_process.update(directory_path.glob(f'*{ext}'))
            
            if not files_to_process:
                logger.warning(f"No supported files found in {directory_path}")
                return []
            
            # Check for existing files with detailed logging
            async with self.async_session() as session:
                existing_docs = await session.execute(
                    select(DocumentModel.filename, DocumentModel.created_at, 
                           DocumentModel.checksum,
                           func.count(DocumentChunk.id).label('chunk_count'))
                    .outerjoin(DocumentChunk)
                    .group_by(DocumentModel.filename, DocumentModel.created_at, 
                             DocumentModel.checksum)
                )
                existing_docs = existing_docs.fetchall()
                
                # Log existing documents details
                if existing_docs:
                    logger.info("Documents already in database:")
                    for doc in existing_docs:
                        logger.info(
                            f"- {doc.filename}\n"
                            f"  Processed at: {doc.created_at}\n"
                            f"  Checksum: {doc.checksum}\n"
                            f"  Chunks: {doc.chunk_count}"
                        )
                
                # Create dict of existing checksums
                existing_checksums = {
                    calculate_file_checksum(str(f)): str(f)
                    for f in files_to_process
                }
                
                # Filter out files with matching checksums
                new_files = [
                    f for f in files_to_process 
                    if not any(doc.checksum == calculate_file_checksum(str(f)) 
                              for doc in existing_docs)
                ]
                
                skipped_files = [
                    f for f in files_to_process 
                    if any(doc.checksum == calculate_file_checksum(str(f)) 
                          for doc in existing_docs)
                ]
            
            logger.info(
                f"\nProcessing Summary:\n"
                f"Total files found: {len(files_to_process)}\n"
                f"Already processed: {len(skipped_files)}\n"
                f"New files to process: {len(new_files)}\n"
            )
            
            if skipped_files:
                logger.info("\nSkipping previously processed files:")
                for file in skipped_files:
                    logger.info(f"- {file}")
            
            if new_files:
                logger.info("\nStarting processing of new files:")
                for file in new_files:
                    logger.info(f"- {file}")
            
            # Create process pool for true parallelism
            successful_docs = []
            file_times = {}  # Track processing time per file
            
            with ProcessPoolExecutor(max_workers=BATCH_CONFIG['process_pool_workers']) as pool:
                # Create tasks for parallel processing with timing
                futures_with_start_times = {
                    pool.submit(
                        process_single_document,
                        str(file_path),
                        self.config
                    ): (str(file_path), time.time())
                    for file_path in new_files
                }
                
                # Monitor progress
                for future in as_completed(futures_with_start_times.keys()):
                    file_path, start_time_file = futures_with_start_times[future]
                    try:
                        result = future.result()
                        end_time_file = time.time()
                        processing_time = end_time_file - start_time_file
                        
                        if result:
                            successful_docs.extend(result)
                            file_times[file_path] = processing_time
                            logger.info(
                                f"Completed processing {file_path}, "
                                f"pid: {os.getpid()}, "
                                f"time: {processing_time:.2f}s, "
                                f"chunks: {len(result)}"
                            )
                    except Exception as e:
                        logger.error(f"Process failed for {file_path}: {str(e)}")
            
            # Calculate accurate statistics
            total_time = time.time() - start_time
            processed_files = len(file_times)
            
            # Add checks for zero processed files
            if processed_files > 0:
                avg_time = sum(file_times.values()) / processed_files
                max_time = max(file_times.values())
                min_time = min(file_times.values())
                total_processing_time = sum(file_times.values())
                parallel_efficiency = total_processing_time / total_time if total_time > 0 else 0
                avg_chunks_per_file = len(successful_docs) / processed_files
            else:
                avg_time = max_time = min_time = total_processing_time = parallel_efficiency = 0
                avg_chunks_per_file = 0

            logger.info(
                f"\nProcessing Statistics:\n"
                f"Total files processed: {processed_files} of {len(new_files)}\n"
                f"Total chunks created: {len(successful_docs)}\n"
                f"Total processing time: {total_processing_time:.2f}s\n"
                f"Average time per file: {avg_time:.2f}s\n"
                f"Maximum time per file: {max_time:.2f}s\n"
                f"Minimum time per file: {min_time:.2f}s\n"
                f"Total wall clock time: {total_time:.2f}s\n"
                f"Parallel efficiency: {parallel_efficiency:.2f}x\n"
                f"Average chunks per file: {avg_chunks_per_file:.1f}\n"
                f"Successfully processed: {len(successful_docs)} chunks from {processed_files} files"
            )
            
            return successful_docs
            
        except Exception as e:
            logger.error(f"Error in directory processing: {str(e)}")
            raise
        finally:
            # Clean up pools
            self.process_pool.shutdown(wait=True)
            self.thread_pool.shutdown(wait=True)

    @validate_input
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search using the vector store."""
        # Add parameter validation
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
            
        if not query.strip():
            raise ValueError("Empty search query")
            
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Process documents first.")
        
        try:
            # Add timeout for search operation
            async def _search():
                return await asyncio.wait_for(
                    self.vector_store.similarity_search(query, k=k),
                    timeout=30.0
                )
                
            return asyncio.run(_search())
        except asyncio.TimeoutError:
            logger.error("Similarity search timed out")
            raise DocumentProcessingError("Search operation timed out") from None
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            raise DocumentProcessingError("Failed to perform similarity search") from e

    def get_document_metadata(self, documents: List[Document]) -> List[dict]:
        """
        Extract metadata from processed documents.
        
        Args:
            documents: List of documents to extract metadata from
            
        Returns:
            List of metadata dictionaries
        """
        metadata_list = []
        for doc in documents:
            metadata = {
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 0),
                'chunk_size': len(doc.page_content),
                'total_chunks': len(documents)
            }
            metadata_list.append(metadata)
        return metadata_list

    async def load_pdf_safely(self, file_path: str) -> List[Document]:
        """Load PDF with enhanced error handling."""
        try:
            # Try primary loader
            loader = PyPDFLoader(file_path)
            documents = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                loader.load
            )
            if documents:
                return documents
            
            # If no documents, try alternative loader
            loader = UnstructuredPDFLoader(file_path)
            documents = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                loader.load
            )
            return documents
        
        except Exception as e:
            if "wrong pointing object" in str(e):
                logger.warning(f"PDF parsing issue with {file_path}, trying alternative loader")
                try:
                    loader = UnstructuredPDFLoader(file_path)
                    return await asyncio.get_event_loop().run_in_executor(
                        self.thread_pool,
                        loader.load
                    )
                except Exception as e2:
                    logger.error(f"Both PDF loaders failed for {file_path}: {str(e2)}")
                    raise FileLoadError(f"Failed to load PDF {file_path}") from e2
            raise FileLoadError(f"Failed to load file {file_path}") from e

# Add this function at module level
def process_single_document(file_path: str, config: ProcessorConfig) -> Optional[List[Document]]:
    """Process a single document in a separate process."""
    try:
        # Log process information
        process = psutil.Process()
        logger.info(
            f"Processing {file_path} in process {os.getpid()} "
            f"CPU: {process.cpu_percent()}% MEM: {process.memory_info().rss / 1024 / 1024:.1f}MB"
        )
        
        # Create processor instance for this process
        processor = DocumentProcessor(config)
        
        # Process the document
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(processor.process_file(file_path))
            return result
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error processing {file_path} in process {os.getpid()}: {str(e)}")
        return None

async def async_main():
    """Async main function."""
    try:
        postgres_config = PostgresConfig(
            connection_string="postgresql+asyncpg://postgres:1%40SSongou2@192.168.1.185:5432/ragSystem",
            pre_delete_collection=True,
            drop_existing=True
        )
        
        config = ProcessorConfig(
            vector_store_type="postgres",
            postgres_config=postgres_config,
            batch_size=100,
            max_workers=40,
            db_pool_size=50
        )
        
        processor = DocumentProcessor(config)
        await processor.initialize_database()
        
        dir_path = "documents"
        if os.path.exists(dir_path):
            docs = await processor.process_directory(dir_path)
            logger.info(f"Processed directory into {len(docs)} chunks")
            
    except Exception as e:
        logger.error(f"Error in async_main: {str(e)}")
        raise
    finally:
        if hasattr(processor, 'engine'):
            await processor.engine.dispose()

if __name__ == "__main__":
    asyncio.run(async_main())