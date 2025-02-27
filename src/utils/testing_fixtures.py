"""
Test fixtures for components.
"""

import pytest
import asyncio
from typing import AsyncGenerator, Generator
from pathlib import Path
import tempfile
import shutil

from ..config.database import PostgresConfig
from ..config.processor import ProcessorConfig
from ..services.database import create_async_db_engine, create_async_session_maker
from ..components.document_processing.processor import DocumentProcessor
from ..components.vector_store.manager import VectorStoreManager
from ..components.search.searcher import DocumentSearcher

@pytest.fixture
async def test_engine():
    """Fixture for test database engine."""
    engine = create_async_db_engine(
        "postgresql+asyncpg://postgres:test@localhost:5432/test_db"
    )
    yield engine
    await engine.dispose()

@pytest.fixture
async def test_session_maker(test_engine):
    """Fixture for test session maker."""
    return create_async_session_maker(test_engine)

@pytest.fixture
async def processor_config():
    """Fixture for processor configuration."""
    postgres_config = PostgresConfig(
        connection_string="postgresql+asyncpg://postgres:test@localhost:5432/test_db"
    )
    return ProcessorConfig(
        chunk_size=100,
        chunk_overlap=20,
        batch_size=5,
        postgres_config=postgres_config
    )

@pytest.fixture
async def document_processor(processor_config):
    """Fixture for document processor."""
    processor = DocumentProcessor(processor_config)
    await processor.initialize_database()
    return processor

@pytest.fixture
async def vector_store_manager(test_session_maker):
    """Fixture for vector store manager."""
    async with test_session_maker() as session:
        manager = VectorStoreManager(session)
        await manager.initialize()
        return manager

@pytest.fixture
async def document_searcher(test_session_maker):
    """Fixture for document searcher."""
    async with test_session_maker() as session:
        searcher = DocumentSearcher(session)
        await searcher.initialize()
        return searcher

@pytest.fixture
def temp_dir():
    """Fixture for temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def sample_documents(temp_dir):
    """Fixture for sample test documents."""
    docs = [
        ("test1.txt", "This is a test document."),
        ("test2.txt", "Another test document with different content."),
        ("test3.txt", "A third document for testing purposes.")
    ]
    
    created_files = []
    for filename, content in docs:
        file_path = temp_dir / filename
        file_path.write_text(content)
        created_files.append(file_path)
    
    yield created_files
    
    for file_path in created_files:
        if file_path.exists():
            file_path.unlink() 