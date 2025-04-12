"""
System Management API Router

This module provides endpoints for system management:
- Get system information (database status, document statistics)
- Reset the database (with authentication)
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Security, Query, Header
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from pydantic import BaseModel, Field
import os
import time
import psutil

from app.routers.document_routes import get_db
from app.models import DocumentModel, DocumentChunk
from app.config import db_config

# Pydantic models for request/response schemas
class SystemStats(BaseModel):
    """System statistics model"""
    document_count: int
    chunk_count: int
    avg_chunks_per_document: float
    database_size_mb: float
    embedding_dimension: int
    uptime_seconds: float

class DatabaseSize(BaseModel):
    """Database size information model"""
    table_name: str
    row_count: int
    size_mb: float
    index_size_mb: float

class SystemInfo(BaseModel):
    """System information response model"""
    stats: SystemStats
    database: List[DatabaseSize]
    config: Dict[str, Any]
    version: str = "1.0.0"

# API key security
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("ADMIN_API_KEY", "dev_api_key_not_secure")  # Default key for development
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Create router
router = APIRouter(
    prefix="/system",
    tags=["system"],
    responses={404: {"description": "Not found"}},
)

# Start time for uptime calculation
START_TIME = time.time()

async def validate_api_key(api_key: str = Security(api_key_header)):
    """Validate API key for protected endpoints"""
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": API_KEY_NAME},
        )
    return api_key

@router.get("/info", response_model=SystemInfo)
async def get_system_info(db: AsyncSession = Depends(get_db)):
    """
    Get system information
    
    Returns detailed statistics about the system, including document counts, 
    database size, and configuration information.
    """
    try:
        # Get document count
        doc_count_result = await db.execute(select(func.count()).select_from(DocumentModel))
        document_count = doc_count_result.scalar() or 0
        
        # Get chunk count
        chunk_count_result = await db.execute(select(func.count()).select_from(DocumentChunk))
        chunk_count = chunk_count_result.scalar() or 0
        
        # Calculate average chunks per document
        avg_chunks = chunk_count / document_count if document_count > 0 else 0
        
        # Get database size information
        db_size_query = """
        SELECT
            table_name,
            pg_total_relation_size(quote_ident(table_name)) as total_bytes,
            pg_indexes_size(quote_ident(table_name)) as index_bytes,
            pg_relation_size(quote_ident(table_name)) as table_bytes,
            reltuples::bigint as row_count
        FROM information_schema.tables
        JOIN pg_class ON pg_class.relname = table_name
        WHERE table_schema = 'public'
        ORDER BY total_bytes DESC;
        """
        
        # Execute raw SQL
        result = await db.execute(text(db_size_query))
        db_size_info = []
        total_db_size = 0
        
        for row in result:
            table_size_mb = row.table_bytes / (1024 * 1024)
            index_size_mb = row.index_bytes / (1024 * 1024)
            total_size_mb = row.total_bytes / (1024 * 1024)
            total_db_size += total_size_mb
            
            db_size_info.append({
                "table_name": row.table_name,
                "row_count": row.row_count,
                "size_mb": table_size_mb,
                "index_size_mb": index_size_mb
            })
        
        # Calculate uptime
        uptime = time.time() - START_TIME
        
        # Create filtered config dict (remove sensitive information)
        from app.config import chunking_config
        
        safe_config = {
            "embedding_dimension": 1536,
            "chunk_size": chunking_config.chunk_size,  # Use the correct config object for chunk_size
            "pool_size": db_config.pool_size,
            "max_overflow": db_config.max_overflow,
            "pool_timeout": db_config.pool_timeout,
            "pool_recycle": db_config.pool_recycle
        }
        
        # Assemble response
        return {
            "stats": {
                "document_count": document_count,
                "chunk_count": chunk_count,
                "avg_chunks_per_document": avg_chunks,
                "database_size_mb": total_db_size,
                "embedding_dimension": 1536,
                "uptime_seconds": uptime
            },
            "database": db_size_info,
            "config": safe_config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving system info: {str(e)}")

@router.post("/reset", status_code=204)
async def reset_database(
    reset_vector_index: bool = Query(True, description="Whether to reset vector indexes"),
    confirm: bool = Query(False, description="Confirmation flag for database reset"),
    api_key: str = Depends(validate_api_key),
    db: AsyncSession = Depends(get_db)
):
    """
    Reset the database
    
    WARNING: This endpoint will delete all data in the database.
    Requires API key authentication and confirmation flag.
    """
    if not confirm:
        raise HTTPException(
            status_code=400, 
            detail="Confirmation required. Set confirm=true to proceed with database reset."
        )
    
    try:
        # Import reset function and use it
        from app.core.document_rag_loader import reset_database
        from sqlalchemy.ext.asyncio import create_async_engine
        
        # Create synchronous engine instance for reset operation
        # The reset_database function expects a sync engine
        sync_connection_string = db_config.connection_string.replace(
            'postgresql+asyncpg://', 'postgresql://'
        )
        
        engine = create_async_engine(db_config.connection_string)
        await reset_database(engine)
        
        # Additional index creation if needed
        if reset_vector_index:
            from app.core.document_rag_loader import create_vector_index
            # Execute raw SQL through the async session
            await create_vector_index(db)
        
        return None  # 204 No Content
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting database: {str(e)}")
