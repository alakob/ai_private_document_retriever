"""
Search component for similarity search functionality.
"""

import logging
from typing import List, Optional
from langchain.schema import Document
from sqlalchemy.ext.asyncio import AsyncSession

from ..vector_store.manager import VectorStoreManager
from ...utils.monitoring import ProcessingMonitor

logger = logging.getLogger(__name__)

class DocumentSearcher:
    """Handles document similarity search."""
    
    def __init__(self, session: AsyncSession):
        self.vector_store = VectorStoreManager(session)
        self.monitor = ProcessingMonitor()
    
    async def initialize(self):
        """Initialize searcher."""
        await self.vector_store.initialize()
    
    async def search(
        self,
        query: str,
        k: int = 4,
        threshold: float = 0.5
    ) -> List[Document]:
        """Perform similarity search."""
        try:
            await self.monitor.start_task(f"search_{query[:50]}")
            
            # Validate inputs
            if not query.strip():
                raise ValueError("Search query cannot be empty")
            
            if k < 1:
                raise ValueError("k must be positive")
            
            if not 0 <= threshold <= 1:
                raise ValueError("threshold must be between 0 and 1")
            
            # Perform search
            results = await self.vector_store.search(
                query=query,
                k=k,
                threshold=threshold
            )
            
            logger.info(f"Found {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
        finally:
            await self.monitor.end_task(f"search_{query[:50]}")

    def format_results(self, results: List[Document]) -> str:
        """Format search results for display."""
        if not results:
            return "No matching documents found."
            
        formatted = ["Search Results:"]
        for i, doc in enumerate(results, 1):
            score = doc.metadata.get('similarity_score', 0)
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            
            formatted.extend([
                f"\n{i}. Score: {score:.3f}",
                f"Source: {source} (Page: {page})",
                f"Content: {doc.page_content[:200]}..."
            ])
        
        return "\n".join(formatted) 