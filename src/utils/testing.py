"""
Testing utilities for components.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
import pytest

from ..config.database import PostgresConfig
from ..models.documents import Base, DocumentModel, DocumentChunk
from ..utils.file_handler import FileHandler

logger = logging.getLogger(__name__)

class TestDatabaseManager:
    """Manages test database setup and teardown."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine = None
        self.session_maker = None
    
    async def setup(self):
        """Set up test database."""
        self.engine = create_async_engine(
            self.connection_string,
            echo=False
        )
        
        self.session_maker = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
            class_=AsyncSession
        )
        
        async with self.engine.begin() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
    
    async def teardown(self):
        """Clean up test database."""
        if self.engine:
            await self.engine.dispose()

class TestFileManager:
    """Manages test file creation and cleanup."""
    
    def __init__(self, test_dir: str = "test_documents"):
        self.test_dir = Path(test_dir)
        self.file_handler = FileHandler(test_dir)
        self.created_files: List[Path] = []
    
    def setup(self):
        """Set up test directory."""
        self.test_dir.mkdir(exist_ok=True)
    
    def create_test_file(
        self,
        content: str,
        filename: str,
        extension: str = ".txt"
    ) -> Path:
        """Create a test file with content."""
        file_path = self.test_dir / f"{filename}{extension}"
        file_path.write_text(content)
        self.created_files.append(file_path)
        return file_path
    
    def cleanup(self):
        """Clean up test files."""
        for file_path in self.created_files:
            try:
                file_path.unlink()
            except Exception as e:
                logger.error(f"Failed to remove test file {file_path}: {str(e)}")
        
        try:
            self.test_dir.rmdir()
        except Exception as e:
            logger.error(f"Failed to remove test directory: {str(e)}")

@pytest.fixture
async def test_db():
    """Fixture for test database."""
    db_manager = TestDatabaseManager(
        "postgresql+asyncpg://postgres:test@localhost:5432/test_db"
    )
    await db_manager.setup()
    yield db_manager.session_maker
    await db_manager.teardown()

@pytest.fixture
def test_files():
    """Fixture for test files."""
    file_manager = TestFileManager()
    file_manager.setup()
    yield file_manager
    file_manager.cleanup() 