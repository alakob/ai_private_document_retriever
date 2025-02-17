from dataclasses import dataclass
from typing import Optional, Literal
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
class ProcessorConfig:
    """Configuration for document processing."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    vector_store_type: Literal["postgres", "faiss", "chroma"] = "postgres"
    postgres_config: Optional[PostgresConfig] = None
    persist_directory: Optional[str] = None
    batch_size: int = 10
    max_workers: int = 4
    db_pool_size: int = 5
    openai_api_key: str = os.getenv('OPENAI_API_KEY')
    rate_limit_config: RateLimitConfig = RateLimitConfig()
    resource_config: ResourceConfig = ResourceConfig()

    def __post_init__(self):
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        # Validate other fields...

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