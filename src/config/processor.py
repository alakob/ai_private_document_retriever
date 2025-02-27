"""
Document processor configuration settings.
"""

from dataclasses import dataclass
from typing import Optional, Literal
import os
from .database import PostgresConfig
from .rate_limiter import RateLimitConfig
from .resource import ResourceConfig

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