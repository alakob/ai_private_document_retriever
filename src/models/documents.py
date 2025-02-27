"""
Database models for document storage.
"""

from sqlalchemy import Column, String, Integer, Float, JSON, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from datetime import datetime

Base = declarative_base()

class DocumentModel(Base):
    """Model for storing document metadata."""
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    doc_metadata = Column(JSONB, nullable=True)
    checksum = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    chunks = relationship(
        "DocumentChunk", 
        back_populates="document", 
        cascade="all, delete-orphan",
        lazy="selectin"  # Use eager loading
    )

class DocumentChunk(Base):
    """Model for storing document chunks with embeddings."""
    __tablename__ = 'document_chunks'

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'))
    content = Column(String, nullable=False)
    embedding = Column(Vector(1536))  # OpenAI embedding dimension
    chunk_index = Column(Integer)
    chunk_size = Column(Integer)
    page_number = Column(Integer, nullable=True)
    section_title = Column(String, nullable=True)
    chunk_metadata = Column(JSONB, nullable=True)
    document = relationship("DocumentModel", back_populates="chunks") 