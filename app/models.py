from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, String, Integer, ForeignKey, DateTime, UniqueConstraint, CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from datetime import datetime

Base = declarative_base()

class DocumentModel(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String, unique=True)
    doc_metadata = Column(JSONB)
    checksum = Column(String(64))  # Add checksum column for SHA-256
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    created_at = Column(DateTime, default=datetime.utcnow)
    upload_date = Column(DateTime, default=datetime.utcnow)

class DocumentChunk(Base):
    __tablename__ = 'document_chunks'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'))
    content = Column(String, nullable=False)
    embedding = Column(Vector(1536))
    chunk_metadata = Column(JSONB)
    
    # Core metadata fields as columns for efficient querying
    chunk_index = Column(Integer)
    chunk_size = Column(Integer)
    page_number = Column(Integer, nullable=True)
    section_title = Column(String, nullable=True)
    
    document = relationship("DocumentModel", back_populates="chunks")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        # Add unique constraint for document_id + chunk_index
        UniqueConstraint('document_id', 'chunk_index', name='uix_doc_chunk'),
        CheckConstraint("content != ''", name='content_not_empty')  # Add content validation
    ) 