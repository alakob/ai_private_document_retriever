import os
import logging
import base64
from typing import List, Optional, Union, Literal, Generator, Dict, Any, Tuple, Callable
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader, Docx2txtLoader, UnstructuredPowerPointLoader, UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma, PGVector
from dataclasses import dataclass
import importlib.util
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
import argparse
from sqlalchemy.schema import DropSchema

from dotenv import load_dotenv
from app.models import Base, DocumentModel, DocumentChunk
from app.config import (
    PostgresConfig, 
    ProcessorConfig,
    RateLimitConfig,
    ResourceConfig,
    processor_config, 
    embedding_config, 
    chunking_config, 
    file_processing_config, 
    index_config,
    db_config  # Add this import
)
from rich.console import Console

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

class DoclingNotInstalledError(ImportError):
    """Raised when docling is not installed."""
    pass

class MistralNotInstalledError(ImportError):
    """Raised when mistral client is not installed."""
    pass

# Add document loader helper functions
def is_docling_installed() -> bool:
    """Check if docling is installed."""
    return importlib.util.find_spec("docling") is not None

def is_mistral_installed() -> bool:
    """Check if mistral client is installed."""
    return importlib.util.find_spec("mistralai") is not None

class DoclingLoader:
    """Document loader using docling for enhanced document conversion.
    
    This loader uses docling to convert documents to structured text with better
    preservation of tables, lists, and other document elements.
    
    Attributes:
        artifacts_path: Optional path to pre-downloaded docling models
        enable_remote_services: Whether to allow docling to use remote services
        use_cache: Whether to use cached conversions
    """
    
    def __init__(self, 
                artifacts_path: Optional[str] = None,
                enable_remote_services: bool = False,
                use_cache: bool = True):
        """Initialize the DoclingLoader.
        
        Args:
            artifacts_path: Path to pre-downloaded models (optional)
            enable_remote_services: Whether to allow external API calls
            use_cache: Whether to cache conversion results
        
        Raises:
            DoclingNotInstalledError: If docling is not installed
        """
        if not is_docling_installed():
            raise DoclingNotInstalledError(
                "Docling is not installed. Please install it with: "
                "pip install docling"
            )
            
        self.artifacts_path = artifacts_path
        self.enable_remote_services = enable_remote_services
        self.use_cache = use_cache
        self._converter = None
    
    @property
    def converter(self):
        """Lazy initialization of the DocumentConverter."""
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import PdfFormatOption
                
                # Configure pipeline options
                pipeline_options = {}
                if self.artifacts_path:
                    pipeline_options["artifacts_path"] = self.artifacts_path
                if self.enable_remote_services:
                    pipeline_options["enable_remote_services"] = True
                    
                pdf_pipeline_options = PdfPipelineOptions(**pipeline_options)
                
                # Create converter with specific format options
                format_options = {
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
                }
                
                self._converter = DocumentConverter(
                    format_options=format_options,
                    use_cache=self.use_cache
                )
                
            except ImportError as e:
                raise DoclingNotInstalledError(
                    f"Error importing docling components: {e}"
                )
        return self._converter
    
    def load(self, file_path: str) -> List[Document]:
        """Load a document using docling.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of LangChain Document objects
            
        Raises:
            FileLoadError: If the document cannot be loaded
        """
        try:
            logger.info(f"Loading document with docling: {file_path}")
            # Convert the document
            result = self.converter.convert(file_path)
            
            # Extract document content as markdown
            content = result.document.export_to_markdown()
            
            # Extract metadata
            metadata = {
                "source": file_path,
                "title": result.document.title if hasattr(result.document, "title") else Path(file_path).stem,
                "doc_type": Path(file_path).suffix.lower().lstrip("."),
                "processor": "docling",
                "page_count": len(result.document.blocks) if hasattr(result.document, "blocks") else 1
            }
            
            # Create a LangChain Document
            document = Document(page_content=content, metadata=metadata)
            
            return [document]
            
        except Exception as e:
            error_msg = f"Failed to load document with docling: {str(e)}"
            logger.error(error_msg)
            raise FileLoadError(error_msg) from e

class MistralOCRLoader:
    """Document loader using Mistral AI's OCR API for enhanced document conversion.
    
    This loader leverages Mistral AI's OCR capabilities to extract text from documents
    while preserving structure, hierarchy, and formatting elements like headers, tables, and lists.
    
    Attributes:
        api_key: Mistral API key for authentication
        include_image_base64: Whether to include base64-encoded images in the response
        model: Mistral OCR model name to use
    """
    
    def __init__(self,
                api_key: str,
                model: str = "mistral-ocr-latest",
                include_image_base64: bool = False):
        """Initialize the MistralOCRLoader.
        
        Args:
            api_key: Mistral API key
            model: OCR model to use, defaults to "mistral-ocr-latest"
            include_image_base64: Whether to include images in response
            
        Raises:
            MistralNotInstalledError: If mistral package is not installed
        """
        if not is_mistral_installed():
            raise MistralNotInstalledError(
                "Mistral package is not installed. Please install it with: "
                "pip install mistralai"
            )
            
        self.api_key = api_key
        self.model = model
        self.include_image_base64 = include_image_base64
        self._client = None
        
    @property
    def client(self):
        """Lazy initialization of the Mistral client."""
        if self._client is None:
            from mistralai import Mistral
            self._client = Mistral(api_key=self.api_key)
        return self._client
    
    async def load(self, file_path: str) -> List[Document]:
        """Load a document using Mistral OCR API.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of LangChain Document objects
            
        Raises:
            FileLoadError: If the document cannot be processed
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Get file name for metadata
            file_name = os.path.basename(file_path)
            
            # Process file based on whether it's local or a URL
            ocr_response = await self._process_document(file_path)
            
            # Convert OCR response to Document objects
            documents = self._convert_to_documents(ocr_response, file_name)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document with Mistral OCR: {str(e)}")
            raise FileLoadError(f"Failed to load {file_path} with Mistral OCR: {str(e)}")
    
    async def _process_document(self, file_path: str):
        """Process a document with Mistral OCR API."""
        # We need to run this in a thread pool since the Mistral client is synchronous
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._process_document_sync, file_path)
    
    def _process_document_sync(self, file_path: str):
        """Synchronous method to process a document with OCR."""
        # Determine if we're processing a local file or URL
        if file_path.startswith(('http://', 'https://')):
            # For URLs
            return self.client.ocr.process(
                model=self.model,
                document={
                    "type": "document_url",
                    "document_url": file_path
                },
                include_image_base64=self.include_image_base64
            )
        else:
            # For local files, we need to open and read the file
            with open(file_path, "rb") as f:
                file_content = f.read()
                
            return self.client.ocr.process(
                model=self.model,
                document={
                    "type": "document_base64",
                    "document_base64": base64.b64encode(file_content).decode("utf-8"),
                    "filename": os.path.basename(file_path)
                },
                include_image_base64=self.include_image_base64
            )
    
    def _convert_to_documents(self, ocr_response, file_name: str) -> List[Document]:
        """Convert OCR response to LangChain Document objects."""
        documents = []
        
        # Extract page content from the OCR response
        if hasattr(ocr_response, 'pages') and ocr_response.pages:
            for i, page in enumerate(ocr_response.pages):
                # Create metadata
                metadata = {
                    "source": file_name,
                    "page": i + 1,  # 1-indexed page numbers
                    "total_pages": len(ocr_response.pages),
                    "file_path": file_name
                }
                
                # Get page content (either markdown or text)
                if hasattr(page, 'markdown') and page.markdown:
                    content = page.markdown
                elif hasattr(page, 'text') and page.text:
                    content = page.text
                else:
                    # Skip pages with no content
                    continue
                    
                # Create Document object
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
        
        return documents

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
                raise
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

def calculate_checksum(content: bytes) -> str:
    """
    Calculate SHA-256 checksum from file content.
    
    Args:
        content: File content as bytes
        
    Returns:
        SHA-256 checksum as hexadecimal string
    """
    if not content:
        return ""
        
    sha256_hash = hashlib.sha256()
    sha256_hash.update(content)
    return sha256_hash.hexdigest()

# Add this function to check for duplicate documents
async def check_duplicate_document(session: AsyncSession, checksum: str) -> Optional[DocumentModel]:
    """Check if document with given checksum already exists."""
    try:
        # Use FOR UPDATE SKIP LOCKED to prevent race conditions in concurrent processing
        result = await session.execute(
            select(DocumentModel)
            .where(DocumentModel.checksum == checksum)
            .with_for_update(skip_locked=True)
        )
        return result.scalar_one_or_none()
    except Exception as e:
        logger.error(f"Error checking for duplicate document: {str(e)}")
        return None

# Add this function after the other utility functions but before the DocumentProcessor class
async def reset_database(engine) -> None:
    """Reset the database by dropping and recreating all tables."""
    try:
        async with engine.begin() as conn:
            # Drop vector extension objects first
            await conn.execute(text('DROP EXTENSION IF EXISTS vector CASCADE;'))
            
            # Drop all tables
            await conn.run_sync(Base.metadata.drop_all)
            
            # Recreate vector extension
            await conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
            
            # Recreate all tables
            await conn.run_sync(Base.metadata.create_all)
            
            # Recreate vector index
            await create_vector_index(conn)
            
        logger.info("Database reset successfully")
    except Exception as e:
        logger.error(f"Failed to reset database: {str(e)}")
        raise DatabaseError(f"Database reset failed: {str(e)}") from e

class DocumentProcessor:
    def __init__(self, config: ProcessorConfig):
        """Initialize with performance optimizations."""
        if not isinstance(config, ProcessorConfig):
            raise ValueError("config must be an instance of ProcessorConfig")
        if not config.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunking_config.chunk_size,
            chunk_overlap=chunking_config.chunk_overlap
        )
        
        # Use the cached embedding service instead of directly instantiating OpenAIEmbeddings
        from app.services.embedding_service import get_embedding_service
        # We'll retrieve the actual service instance when needed asynchronously
        self.embedding_service = None
        
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

        # Set up document loaders based on configuration
        self._setup_document_loaders()

        # Add new components
        self.rate_limiter = EnhancedRateLimiter(config.rate_limit_config)
        self.resource_monitor = ResourceMonitor(ResourceConfig())
        self.adaptive_batcher = AdaptiveBatcher()
        
    def _setup_document_loaders(self):
        """Set up document loaders based on configuration"""
        # Default loaders using LangChain
        self.file_loaders = {
            '.pdf': lambda path: PyPDFLoader(path).load(),
            '.txt': lambda path: TextLoader(path).load(),
            '.docx': lambda path: Docx2txtLoader(path).load(),
            '.doc': lambda path: Docx2txtLoader(path).load(),
            '.ppt': lambda path: UnstructuredPowerPointLoader(path).load(),
            '.pptx': lambda path: UnstructuredPowerPointLoader(path).load(),
        }
        
        # Try to set up docling if requested and available
        if getattr(self.config, 'use_docling', False):
            try:
                # Check if docling is installed
                if is_docling_installed():
                    # Create docling loader instance with configuration
                    docling_loader = DoclingLoader(
                        artifacts_path=getattr(self.config, 'docling_artifacts_path', None),
                        enable_remote_services=getattr(self.config, 'docling_enable_remote', False),
                        use_cache=getattr(self.config, 'docling_use_cache', True)
                    )
                    
                    # Replace existing loaders with docling for supported formats
                    supported_formats = ['.pdf', '.docx', '.doc', '.ppt', '.pptx']
                    for fmt in supported_formats:
                        self.file_loaders[fmt] = lambda path, loader=docling_loader: loader.load(path)
                    
                    logger.info("Docling document loader configured successfully")
                else:
                    logger.warning("Docling requested but not installed. Using default loaders.")
            except Exception as e:
                logger.warning(f"Failed to initialize docling loader: {str(e)}. Using default loaders.")
        
        # Try to set up Mistral OCR loader if requested and available
        elif getattr(self.config, 'use_mistral', False):
            try:
                # Check if Mistral API key is provided
                mistral_api_key = getattr(self.config, 'mistral_api_key', None)
                if not mistral_api_key:
                    logger.warning("Mistral OCR requested but no API key provided. Using default loaders.")
                    return
                
                # Check if mistral client is installed
                if is_mistral_installed():
                    # Create Mistral OCR loader instance with configuration
                    mistral_loader = MistralOCRLoader(
                        api_key=mistral_api_key,
                        model=getattr(self.config, 'mistral_ocr_model', 'mistral-ocr-latest'),
                        include_image_base64=getattr(self.config, 'mistral_include_images', False)
                    )
                    
                    # Replace existing loaders with Mistral OCR for supported formats
                    supported_formats = ['.pdf', '.docx', '.doc', '.ppt', '.pptx', '.jpg', '.jpeg', '.png']
                    for fmt in supported_formats:
                        self.file_loaders[fmt] = lambda path, loader=mistral_loader: loader.load(path)
                    
                    logger.info("Mistral OCR document loader configured successfully")
                else:
                    logger.warning("Mistral OCR requested but mistralai package not installed. Using default loaders.")
            except Exception as e:
                logger.warning(f"Failed to initialize Mistral OCR loader: {str(e)}. Using default loaders.")

    async def initialize_database(self, drop_all: bool = False) -> None:
        """
        Create database tables if they don't exist.
        
        Args:
            drop_all: If True, drop and recreate all tables
        """
        try:
            async with self.engine.begin() as conn:
                # First create the extension
                await conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
                
                # Drop all tables if requested
                if drop_all:
                    logger.warning("Dropping all tables and recreating schema...")
                    await conn.run_sync(Base.metadata.drop_all)
                    logger.info("All tables dropped successfully")
                
                # Create all tables
                await conn.run_sync(Base.metadata.create_all)
                logger.info("Tables created successfully")
                
                # Create vector index for document chunks
                await create_vector_index(conn)
                
                # Only handle duplicates if we're not dropping all tables
                if not drop_all:
                    # Check if we need to handle duplicates before creating the index
                    table_name = DocumentModel.__tablename__
                    
                    # First check if the index already exists
                    index_exists = await conn.execute(text(f"""
                        SELECT 1 FROM pg_indexes 
                        WHERE indexname = 'document_checksum_idx'
                    """))
                    
                    if not index_exists.scalar():
                        # Check for duplicates
                        duplicates = await conn.execute(text(f"""
                            SELECT checksum, COUNT(*) 
                            FROM {table_name} 
                            GROUP BY checksum 
                            HAVING COUNT(*) > 1
                        """))
                        
                        duplicate_rows = duplicates.fetchall()
                        
                        if duplicate_rows:
                            logger.warning(f"Found {len(duplicate_rows)} duplicate checksums in the database")
                            
                            # Instead of deleting, mark duplicates in a way that preserves foreign key integrity
                            for checksum, count in duplicate_rows:
                                logger.info(f"Found {count} documents with checksum {checksum[:8]}...")
                                
                                # Get all document IDs with this checksum
                                doc_ids = await conn.execute(text(f"""
                                    SELECT id FROM {table_name}
                                    WHERE checksum = :checksum
                                    ORDER BY id
                                """), {"checksum": checksum})
                                
                                doc_ids = [row[0] for row in doc_ids.fetchall()]
                                
                                if len(doc_ids) > 1:
                                    # Keep the first one, mark others as duplicates
                                    original_id = doc_ids[0]
                                    duplicate_ids = doc_ids[1:]
                                    
                                    logger.info(f"Marking {len(duplicate_ids)} documents as duplicates of document ID {original_id}")
                                    
                                    # Update duplicate documents to indicate they're duplicates
                                    for dup_id in duplicate_ids:
                                        await conn.execute(text(f"""
                                            UPDATE {table_name}
                                            SET doc_metadata = doc_metadata || 
                                                '{{"is_duplicate": true, "original_document_id": {original_id}}}'::jsonb
                                            WHERE id = :dup_id
                                        """), {"dup_id": dup_id})
                    
                        # Now try to create the unique index with a WHERE clause to exclude duplicates
                        try:
                            await conn.execute(text(f"""
                                CREATE UNIQUE INDEX IF NOT EXISTS document_checksum_idx 
                                ON {table_name} (checksum)
                                WHERE (doc_metadata->>'is_duplicate')::boolean IS NOT TRUE;
                            """))
                            logger.info(f"Created conditional unique index on {table_name}(checksum)")
                        except Exception as idx_error:
                            logger.error(f"Failed to create index after handling duplicates: {str(idx_error)}")
                            # Fall back to application-level duplicate prevention
                            logger.warning("Will rely on application-level duplicate prevention")
                    else:
                        logger.info("Unique index on checksum already exists")
                
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

    async def _get_embeddings_batch(self, texts: List[str], no_cache: bool = False) -> List[List[float]]:
        """Get embeddings with improved rate limiting and batching, using cache where available."""
        if not texts:
            return []
        
        # Initialize embedding service if not done yet
        if self.embedding_service is None:
            from app.services.embedding_service import get_embedding_service
            self.embedding_service = await get_embedding_service()
            logger.info("Initialized cached embedding service for document processing")
        
        results = []
        try:
            # Get embeddings with caching through the embedding service
            logger.info(f"Generating embeddings for {len(texts)} text chunks (with caching)")
            results = await self.embedding_service.get_embeddings_batch(texts, no_cache)
            logger.info(f"Successfully generated embeddings for {len(results)} text chunks")
            return results
            
        except Exception as e:
            # We'll still use the retry and backoff strategy
            logger.warning(f"Error generating embeddings: {str(e)}")
            raise EmbeddingError(f"Failed to get embeddings: {str(e)}") from e

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
                # Calculate checksum first
                checksum = calculate_file_checksum(file_path)
                
                # Check for duplicate file OUTSIDE the transaction to prevent race conditions
                duplicate_doc = await check_duplicate_document(session, checksum)
                
                if duplicate_doc:
                    logger.warning(
                        f"Document {file_path} already exists with same checksum, "
                        f"original file: {duplicate_doc.filename}"
                    )
                    return
                
                # Start transaction
                async with session.begin():
                    # Double-check for duplicate within transaction
                    duplicate_doc = await check_duplicate_document(session, checksum)
                    if duplicate_doc:
                        logger.warning(
                            f"Race condition detected! Document {file_path} was inserted by another process, "
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
                    
                    # Insert new document
                    doc = DocumentModel(
                        filename=str(file_path),
                        doc_metadata=create_document_metadata(file_path),
                        checksum=checksum
                    )
                    session.add(doc)
                    await session.flush()  # Get the document ID

                    # Create chunk records with explicit DocumentChunk model
                    for idx, (chunk, embedding) in enumerate(zip(valid_chunks, valid_embeddings)):
                        chunk_record = DocumentChunk(
                            document_id=doc.id,
                            content=chunk.page_content,
                            embedding=embedding,
                            chunk_index=idx,
                            chunk_size=len(chunk.page_content),
                            page_number=chunk.metadata.get('page', None),
                            section_title=chunk.metadata.get('section_title', None),
                            chunk_metadata={
                                **chunk.metadata,
                                'processed_at': datetime.now().isoformat(),
                                'embedding_model': 'text-embedding-ada-002'
                            }
                        )
                        session.add(chunk_record)

                    # Commit the transaction
                    await session.commit()
                    
                    logger.info(
                        f"Successfully stored document and {len(valid_chunks)} chunks for {file_path}"
                    )
                    
        except Exception as e:
            logger.error(f"Failed to store chunks for {file_path}: {str(e)}")
            raise DatabaseError(f"Failed to store document chunks: {str(e)}") from e

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
            
            # Pre-calculate checksums for all files
            file_checksums = {}
            for file_path in files_to_process:
                file_checksums[str(file_path)] = calculate_file_checksum(str(file_path))
            
            # Check for existing files in a single database call
            async with self.async_session() as session:
                existing_checksums = await session.execute(
                    select(DocumentModel.checksum)
                )
                existing_checksums_set = {row[0] for row in existing_checksums.fetchall()}
                
                # Filter out files with matching checksums
                new_files = [
                    f for f in files_to_process 
                    if file_checksums[str(f)] not in existing_checksums_set
                ]
                
                skipped_files = [
                    f for f in files_to_process 
                    if file_checksums[str(f)] in existing_checksums_set
                ]
            
            # After checking for duplicates
            if skipped_files:
                logger.info("\nDuplicate Files Details:")
                logger.info(f"Total duplicates found: {len(skipped_files)}")
                logger.info("Skipping the following files (already processed):")
                for file in skipped_files:
                    async with self.async_session() as session:
                        checksum = calculate_file_checksum(str(file))
                        existing_doc = await check_duplicate_document(session, checksum)
                        if existing_doc:
                            logger.info(
                                f"- {file}\n"
                                f"  └─ Original file: {existing_doc.filename}\n"
                                f"  └─ First processed: {existing_doc.created_at}\n"
                                f"  └─ Checksum: {checksum[:8]}..."
                            )

            # Update the processing summary to be more prominent
            logger.info(
                f"\n{'='*50}\n"
                f"PROCESSING SUMMARY\n"
                f"{'='*50}\n"
                f"Total files found: {len(files_to_process)}\n"
                f"Duplicate files skipped: {len(skipped_files)} "
                f"({(len(skipped_files)/len(files_to_process)*100):.1f}% of total)\n"
                f"New files to process: {len(new_files)}\n"
                f"{'='*50}"
            )
            
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
                        processed_chunks = future.result()
                        end_time_file = time.time()
                        processing_time = end_time_file - start_time_file
                        
                        if processed_chunks:
                            # Now store the processed chunks in the main process
                            await self._store_processed_chunks(processed_chunks)
                            
                            # Add to successful docs for reporting
                            successful_docs.extend([item['chunk'] for item in processed_chunks])
                            file_times[file_path] = processing_time
                            logger.info(
                                f"Completed processing {file_path}, "
                                f"pid: {os.getpid()}, "
                                f"time: {processing_time:.2f}s, "
                                f"chunks: {len(processed_chunks)}"
                            )
                    except Exception as e:
                        logger.error(f"Process failed for {file_path}: {str(e)}")
            
            # Calculate accurate statistics
            total_time = time.time() - start_time
            processed_files = len(file_times)
            skipped_count = len(skipped_files)
            
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

            # Update the final statistics to include percentage of duplicates
            logger.info(
                f"\n{'='*50}\n"
                f"FINAL PROCESSING STATISTICS\n"
                f"{'='*50}\n"
                f"Total files found: {len(files_to_process)}\n"
                f"Files skipped (duplicates): {skipped_count} "
                f"({(skipped_count/len(files_to_process)*100):.1f}% of total)\n"
                f"Files processed: {processed_files}\n"
                f"Total chunks created: {len(successful_docs)}\n"
                f"Total processing time: {total_processing_time:.2f}s\n"
                f"Average time per file: {avg_time:.2f}s\n"
                f"Maximum time per file: {max_time:.2f}s\n"
                f"Minimum time per file: {min_time:.2f}s\n"
                f"Total wall clock time: {total_time:.2f}s\n"
                f"Parallel efficiency: {parallel_efficiency:.2f}x\n"
                f"Average chunks per file: {avg_chunks_per_file:.1f}\n"
                f"Successfully processed: {len(successful_docs)} chunks from {processed_files} files\n"
                f"{'='*50}"
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

    async def _load_and_process_file(self, file_path: str) -> List[dict]:
        """Load and process a file without storing in database."""
        try:
            logger.info(f"Processing file: {file_path}")
            
            documents = await self._load_document(file_path)
                    
            if not documents:
                return []

            all_processed_chunks = []
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
                
                # Create processed chunk data
                for idx, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                    content = sanitize_text(chunk.page_content)
                    if not content:
                        continue
                        
                    chunk.page_content = content
                    all_processed_chunks.append({
                        'chunk': chunk,
                        'embedding': embedding,
                        'file_path': file_path,
                        'chunk_index': idx
                    })

            return all_processed_chunks
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

    async def _store_processed_chunks(self, processed_chunks: List[dict]) -> None:
        """Store processed chunks in the database."""
        if not processed_chunks:
            return
            
        # Group chunks by file path
        chunks_by_file = {}
        for item in processed_chunks:
            file_path = item['file_path']
            if file_path not in chunks_by_file:
                chunks_by_file[file_path] = []
            chunks_by_file[file_path].append(item)
        
        # Process each file's chunks
        for file_path, items in chunks_by_file.items():
            chunks = [item['chunk'] for item in items]
            embeddings = [item['embedding'] for item in items]
            
            try:
                # Create a new session for each file to avoid transaction conflicts
                async with self.async_session() as session:
                    # Calculate checksum for duplicate prevention
                    file_content = await self._get_file_content(file_path)
                    checksum = calculate_checksum(file_content)
                    
                    # Check if document already exists
                    existing_doc = await check_duplicate_document(session, checksum)
                    if existing_doc:
                        logger.info(f"Document {file_path} already exists with ID {existing_doc.id}")
                        return
                    
                    # Create document record - use filename instead of filepath
                    document = DocumentModel(
                        filename=str(file_path),  # Use the full path as filename
                        checksum=checksum,
                        doc_metadata={"source": "file", 
                                     "original_path": file_path,
                                     "processed_at": datetime.now().isoformat()}
                    )
                    session.add(document)
                    await session.flush()  # Get the document ID
                    
                    # Create chunk records
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        metadata = chunk.metadata.copy() if hasattr(chunk, 'metadata') else {}
                        metadata.update({
                            "chunk_index": i,
                            "source_file": file_path,
                            "processed_at": datetime.now().isoformat()
                        })
                        
                        # Make sure we're using the correct model name
                        chunk_record = DocumentChunk(  # Changed from DocumentChunkModel
                            document_id=document.id,
                            content=chunk.page_content,
                            chunk_metadata=metadata,  # Changed from metadata
                            embedding=embedding
                        )
                        session.add(chunk_record)
                    
                    # Commit the transaction
                    await session.commit()
                    logger.info(f"Successfully stored {len(chunks)} chunks for {file_path}")
                    
            except Exception as e:
                logger.error(f"Failed to store chunks for {file_path}: {str(e)}")

    async def _get_file_content(self, file_path: str) -> bytes:
        """
        Read file content as bytes for checksum calculation.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as bytes
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            async with aiofiles.open(file_path, 'rb') as f:
                return await f.read()
        except Exception as e:
            logger.error(f"Failed to read file content from {file_path}: {str(e)}")
            raise FileLoadError(f"Failed to read file content: {str(e)}") from e

def process_single_document(file_path: str, config: ProcessorConfig) -> Optional[List[dict]]:
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
            # Make sure we're calling a method that exists on the class
            result = loop.run_until_complete(processor._load_and_process_file(file_path))
            
            # Properly dispose of the engine while the loop is still running
            if hasattr(processor, 'engine'):
                loop.run_until_complete(processor.engine.dispose())
                
            return result
        finally:
            # Close the loop after all async operations are complete
            loop.close()
            
    except Exception as e:
        logger.error(f"Error processing {file_path} in process {os.getpid()}: {str(e)}")
        return None

# Update the async_main function to handle command line arguments
async def async_main(directory: str = None, reset_db: bool = False, use_docling: bool = False, use_mistral: bool = False):
    """Main asynchronous function for document processing."""
    # Update processor config based on arguments
    processor_config.use_docling = use_docling
    processor_config.use_mistral = use_mistral
    
    # Use default directory from config if not provided
    if directory is None:
        directory = processor_config.documents_dir

    logger.info(f"Starting document processing for directory: {directory}")
    logger.info(f"Database connection string: {db_config.connection_string}")
    logger.info(f"Docling enabled: {use_docling}")
    logger.info(f"Mistral OCR enabled: {use_mistral}")

    try:
        # Initialize the database engine
        engine = create_async_engine(db_config.connection_string)
        
        # Ensure database tables are created
        async with engine.begin() as conn:
            logger.info("Creating database tables if they do not exist...")
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables checked/created successfully.")
            
        # Optionally reset the database
        if reset_db:
            logger.warning("Resetting the database...")
            await reset_database(engine)
            logger.info("Database reset complete.")
            # Recreate tables after reset
            async with engine.begin() as conn:
                logger.info("Recreating database tables after reset...")
                await conn.run_sync(Base.metadata.create_all)
                logger.info("Database tables recreated successfully.")
                
        # Initialize the DocumentProcessor
        processor = DocumentProcessor(processor_config)
        
        # Process the directory
        logger.info(f"Processing directory: {directory}")
        await processor.process_directory(directory)
        
        logger.info("Document processing finished successfully.")
        
    except Exception as e:
        logger.error(f"Error during document processing: {e}", exc_info=True)
        # Optionally, re-raise the exception if needed
        # raise e
    finally:
        # Clean up resources if necessary
        if 'engine' in locals():
            await engine.dispose()
        logger.info("Database engine disposed.")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Document RAG Loader")
    parser.add_argument(
        "--directory", "-d", 
        type=str, 
        help="Directory containing documents to process"
    )
    parser.add_argument(
        "--reset-db", "-r", 
        action="store_true", 
        help="Reset database before processing"
    )
    parser.add_argument(
        "--use-docling", "-dl", 
        action="store_true", 
        help="Use docling for document loading instead of default loaders"
    )
    
    args = parser.parse_args()
    
    asyncio.run(async_main(
        directory=args.directory,
        reset_db=args.reset_db,
        use_docling=args.use_docling
    ))