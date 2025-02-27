"""
Database service handling PostgreSQL connections and operations.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
import logging

logger = logging.getLogger(__name__)

# Add connection pooling configuration
POOL_CONFIG = {
    'pool_size': 20,
    'max_overflow': 10,
    'pool_timeout': 30,
    'pool_recycle': 1800,
    'pool_pre_ping': True
}

def create_async_db_engine(connection_string: str, pool_size: int = 5, max_overflow: int = 10):
    """Create async database engine with connection pooling."""
    return create_async_engine(
        connection_string,
        pool_size=pool_size,
        max_overflow=max_overflow,
        echo=False
    )

def create_async_session_maker(engine) -> async_sessionmaker[AsyncSession]:
    """Create async session maker."""
    return async_sessionmaker(
        engine,
        expire_on_commit=False,
        class_=AsyncSession
    ) 