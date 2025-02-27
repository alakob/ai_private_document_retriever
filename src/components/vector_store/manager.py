"""
Vector store manager for coordinating vector operations.
"""

import logging
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from langchain.schema import Document

from .store import PostgresVectorStore
from ...utils.monitoring import ProcessingMonitor

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages vector store operations with monitoring."""
    
    def __init__(self, session: AsyncSession):
        self.store = PostgresVectorStore(session)
        self.monitor = ProcessingMonitor()
    
    async def initialize(self):
        """Initialize vector store."""
        await self.store.initialize()
    
    async def add_documents(
        self,
        documents: List[Document],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Add documents with monitoring."""
        try:
            await self.monitor.start_task("add_documents")
            doc_ids = await self.store.add_documents(documents, metadata)
            return doc_ids
        finally:
            await self.monitor.end_task("add_documents")
    
    async def search(
        self,
        query: str,
        k: int = 4,
        threshold: float = 0.5
    ) -> List[Document]:
        """Search documents with monitoring."""
        try:
            await self.monitor.start_task("search")
            results = await self.store.similarity_search(query, k, threshold)
            return results
        finally:
            await self.monitor.end_task("search") 