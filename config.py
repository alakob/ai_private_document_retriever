from dataclasses import dataclass
from typing import Optional, Literal, List
import os
from dotenv import load_dotenv
import psutil

load_dotenv()

@dataclass
class RateLimitConfig:
    """Enhanced configuration for rate limiting."""
    max_requests_per_minute: int = 150
    max_concurrent_requests: int = 20  # New: limit concurrent requests
    batch_size: int = 50  # Reduced from 100
    min_delay: float = 0.1  # Minimum delay between requests
    max_delay: float = 30.0  # Maximum delay for backoff
    initial_delay: float = 1.0  # Starting delay for backoff
    backoff_factor: float = 2.0  # Exponential backoff multiplier
    jitter: float = 0.1  # Add randomness to prevent thundering herd
    max_retries: int = 3  # Add max_retries attribute
    retry_delay: float = 5.0  # Add retry_delay attribute

@dataclass
class ResourceConfig:
    """Configuration for resource monitoring."""
    memory_threshold_mb: int = 1000
    cpu_threshold_percent: float = 80.0
    min_free_memory_mb: int = 500

@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL connection."""
    connection_string: str
    collection_name: str = "documents"
    pre_delete_collection: bool = False
    embedding_dimension: int = 1536
    drop_existing: bool = True

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str = "text-embedding-ada-002"
    embedding_dimension: int = 1536
    batch_size: int = 100
    max_retries: int = 3
    timeout: int = 60
    max_tokens_per_batch: int = 8000

@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    chunk_size: int = int(os.getenv('CHUNK_SIZE', 1000))
    chunk_overlap: int = int(os.getenv('CHUNK_OVERLAP', 200))
    length_function: str = "token_count"  # or "character_count"

@dataclass
class FileProcessingConfig:
    """Configuration for file processing."""
    supported_extensions: List[str] = None
    max_file_size_mb: int = 50
    min_chunk_size: int = 50
    max_chunk_size: int = 2000
    
    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = [
                ".pdf", ".txt", ".md", ".docx", ".csv", 
                ".json", ".html", ".htm", ".xml"
            ]

@dataclass
class IndexConfig:
    """Configuration for vector indexes."""
    faiss_index_type: str = "Flat"  # or "HNSW", "IVF", etc.
    chroma_distance_function: str = "cosine"  # or "l2", "ip"
    postgres_index_type: str = "ivfflat"  # or "hnsw"
    postgres_lists: int = 100
    postgres_probes: int = 10

@dataclass
class ProcessorConfig:
    """Configuration for document processing."""
    chunk_size: int = int(os.getenv('CHUNK_SIZE', 1000))
    chunk_overlap: int = int(os.getenv('CHUNK_OVERLAP', 200))
    vector_store_type: Literal["postgres", "faiss", "chroma"] = "postgres"
    postgres_config: Optional[PostgresConfig] = None
    persist_directory: Optional[str] = None
    batch_size: int = 10
    max_workers: int = 4
    db_pool_size: int = 5
    openai_api_key: str = os.getenv('OPENAI_API_KEY')
    rate_limit_config: RateLimitConfig = RateLimitConfig()
    resource_config: ResourceConfig = ResourceConfig()
    embedding_config: EmbeddingConfig = EmbeddingConfig()
    chunking_config: ChunkingConfig = None
    file_processing_config: FileProcessingConfig = FileProcessingConfig()
    index_config: IndexConfig = IndexConfig()

    def __post_init__(self):
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        if self.chunking_config is None:
            self.chunking_config = ChunkingConfig(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

@dataclass
class ChatConfig:
    """Configuration for chat interface."""
    model: str = os.getenv('MODEL', 'gpt-4o-mini')
    server_name: str = os.getenv('SERVER_NAME', '127.0.0.1')
    share: bool = True
    inbrowser: bool = True
    documents_dir: str = os.getenv('KNOWLEDGE_BASE_DIR', 'documents')
    retriever_k: int = 4
    retriever_score_threshold: float = 0.5
    temperature: float = 0.7
    max_tokens: int = 1000
    streaming: bool = True
    example_questions: List[str] = None
    
    def __post_init__(self):
        if self.example_questions is None:
            self.example_questions = [
                "What are the main topics covered in the documents?",
                "Can you summarize the key points about machine learning?",
                "What are the best practices mentioned in the documents?"
            ]

@dataclass
class DatabaseConfig:
    """Configuration for database connection."""
    host: str = os.getenv('POSTGRES_HOST', 'localhost')
    port: int = int(os.getenv('POSTGRES_PORT', 5432))
    user: str = os.getenv('POSTGRES_USER', 'postgres')
    password: str = os.getenv('POSTGRES_PASSWORD', '')
    database: str = os.getenv('POSTGRES_DB', 'ragSystem')
    
    @property
    def connection_string(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class VectorSearchConfig:
    """Configuration for vector similarity search."""
    top_k: int = 5
    score_threshold: float = 0.7
    pool_size: int = 5
    max_overflow: int = 10
    max_retries: int = 3
    base_delay: float = 1.0
    embedding_model: str = "text-embedding-ada-002"
    
    # Add a method to create a connection string from database config
    @classmethod
    def get_connection_string(cls, db_config: DatabaseConfig) -> str:
        return db_config.connection_string

def get_optimal_workers() -> int:
    """Determine optimal number of workers based on system resources."""
    try:
        cpu_count = os.cpu_count() or 1
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Allocate 1 worker per 2GB of RAM, up to CPU count
        workers = min(
            cpu_count,
            max(1, int(memory_gb / 2)),
            10  # Maximum workers limit
        )
        return workers
    except Exception:
        return 4  # Default fallback

# Create instances of configs for easy import
embedding_config = EmbeddingConfig()
chunking_config = ChunkingConfig(
    chunk_size=int(os.getenv('CHUNK_SIZE', 1000)),
    chunk_overlap=int(os.getenv('CHUNK_OVERLAP', 200))
)
file_processing_config = FileProcessingConfig()
index_config = IndexConfig()

chat_config = ChatConfig()
db_config = DatabaseConfig()
processor_config = ProcessorConfig(
    chunk_size=int(os.getenv('CHUNK_SIZE', 1000)),
    chunk_overlap=int(os.getenv('CHUNK_OVERLAP', 200)),
    max_workers=get_optimal_workers(),
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    embedding_config=embedding_config,
    chunking_config=chunking_config,
    file_processing_config=file_processing_config,
    index_config=index_config
)

# Create instances of configs for easy import
vector_search_config = VectorSearchConfig() 