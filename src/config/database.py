"""
Database configuration settings.
"""

from dataclasses import dataclass
from sqlalchemy import Column, Integer, String, ForeignKey, JSON, DateTime, UniqueConstraint, CheckConstraint
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import ARRAY, FLOAT
from pgvector.sqlalchemy import Vector
from datetime import datetime
from typing import Optional

Base = declarative_base()

class DocumentModel(Base):
    """Model for storing document metadata"""
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    checksum = Column(String, nullable=False, unique=True)
    doc_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to chunks
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    # Add unique constraint on checksum
    __table_args__ = (
        UniqueConstraint('checksum', name='uq_document_checksum'),
    )

class DocumentChunk(Base):
    """Model for storing document chunks with embeddings"""
    __tablename__ = 'document_chunks'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'))
    content = Column(String, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_size = Column(Integer)
    page_number = Column(Integer)
    chunk_metadata = Column(JSON, nullable=True)
    embedding = Column(Vector(1536))  # OpenAI's standard embedding dimension
    
    # Add check constraint for non-empty content
    __table_args__ = (
        CheckConstraint('length(trim(content)) > 0', name='content_not_empty'),
    )
    
    # Relationship to parent document
    document = relationship("DocumentModel", back_populates="chunks")

@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL database."""
    connection_string: str
    pre_delete_collection: bool = False
    drop_existing: bool = False
    collection_name: str = "documents"
    embedding_dimension: int = 1536
    pool_size: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 1800
    pool_pre_ping: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.connection_string:
            raise ValueError("connection_string is required")
            
    @property
    def asyncpg_dsn(self) -> str:
        """Get the connection string in asyncpg format."""
        return self.connection_string.replace('postgresql+asyncpg://', 'postgresql://')

    @property
    def sqlalchemy_url(self) -> str:
        """Get the connection string in SQLAlchemy format."""
        if not self.connection_string.startswith('postgresql+asyncpg://'):
            return f"postgresql+asyncpg://{self.connection_string.split('://', 1)[1]}"
        return self.connection_string

__all__ = ['DocumentModel', 'DocumentChunk', 'PostgresConfig', 'Base'] 