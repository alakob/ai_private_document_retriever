"""
Vector retrieval service for PostgreSQL.
"""

from typing import List, Callable, Optional, Any
from pydantic import BaseModel, Field
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
import numpy as np
from sqlalchemy import text

class PostgresVectorRetriever(BaseRetriever, BaseModel):
    """Custom retriever for PostgreSQL vector store"""
    
    session_maker: Callable = Field(..., description="Async session maker for PostgreSQL")
    embeddings: Any = Field(..., description="Embeddings model")
    k: int = Field(default=4, description="Number of documents to retrieve")
    score_threshold: float = Field(default=0.5, description="Minimum similarity score threshold")
    
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
                
                # SQL query for similarity search
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