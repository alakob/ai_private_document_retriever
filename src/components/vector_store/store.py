"""
Vector store component for managing document vectors.
"""

import logging
from typing import List, Optional, Dict, Any
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from langchain.schema import Document

from ...models.documents import DocumentModel, DocumentChunk
from ...services.embeddings import get_embeddings
from ...utils.text import sanitize_text

logger = logging.getLogger(__name__)

class PostgresVectorStore:
    """Manages vector storage in PostgreSQL."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.embeddings = get_embeddings()
    
    async def initialize(self):
        """Initialize vector store and create necessary extensions."""
        try:
            async with self.session.begin():
                # Create pgvector extension
                await self.session.execute(
                    text("CREATE EXTENSION IF NOT EXISTS vector;")
                )
                
                # Create vector index
                await self.session.execute(text("""
                    CREATE INDEX IF NOT EXISTS embedding_idx 
                    ON document_chunks 
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """))
                
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    async def add_documents(
        self,
        documents: List[Document],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Add documents to vector store."""
        try:
            document_ids = []
            async with self.session.begin():
                for doc in documents:
                    # Create document record
                    document = DocumentModel(
                        filename=doc.metadata.get('source', 'unknown'),
                        doc_metadata=metadata or {},
                        checksum=doc.metadata.get('checksum', '')
                    )
                    self.session.add(document)
                    await self.session.flush()
                    
                    # Get embeddings for content
                    content = sanitize_text(doc.page_content)
                    if not content:
                        continue
                        
                    embedding = await self.embeddings.aembed_query(content)
                    
                    # Create chunk record
                    chunk = DocumentChunk(
                        document_id=document.id,
                        content=content,
                        embedding=embedding,
                        chunk_index=doc.metadata.get('chunk_index', 0),
                        chunk_size=len(content),
                        page_number=doc.metadata.get('page'),
                        section_title=doc.metadata.get('section_title'),
                        chunk_metadata=doc.metadata
                    )
                    self.session.add(chunk)
                    document_ids.append(str(document.id))
                    
            return document_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise

    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        threshold: float = 0.5
    ) -> List[Document]:
        """Search for similar documents."""
        try:
            # Generate query embedding
            query_embedding = await self.embeddings.aembed_query(query)
            
            # Convert to PostgreSQL vector format
            query_embedding_str = f"[{','.join(map(str, query_embedding))}]"
            
            # Execute search
            async with self.session.begin():
                result = await self.session.execute(text("""
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
                """), {
                    "embedding": query_embedding_str,
                    "threshold": threshold,
                    "limit": k
                })
                
                # Convert to Documents
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
                
                return documents
                
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {str(e)}")
            raise 