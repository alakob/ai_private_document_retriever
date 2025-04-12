"""
Search API Router

This module provides endpoints for search operations:
- Perform vector similarity search on documents
- Get cache information
- Clear the embedding cache
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field, validator

from app.routers.document_routes import get_db
from app.services.vector_similarity_search import VectorSimilaritySearch
from app.config import db_config

# Pydantic models for request/response schemas
class SearchQuery(BaseModel):
    """Search query model"""
    query: str = Field(..., description="The search query text")
    top_k: int = Field(5, description="Number of results to return", ge=1, le=100)
    threshold: float = Field(0.7, description="Similarity threshold (0-1)", ge=0, le=1)
    no_cache: bool = Field(False, description="Whether to skip the embedding cache")

class SearchResult(BaseModel):
    """Individual search result model"""
    document_id: int
    document_name: str
    chunk_id: int
    content: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    score: float
    metadata: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    """Search response model"""
    results: List[SearchResult]
    total_results: int
    query: str
    elapsed_time_ms: float

class CacheInfo(BaseModel):
    """Cache information model"""
    size: int
    hit_count: int
    miss_count: int
    hit_rate: float
    entries: List[Dict[str, Any]]

# Create router
router = APIRouter(
    prefix="/search",
    tags=["search"],
    responses={404: {"description": "Not found"}},
)

# Helper to initialize the search service
async def get_search_service():
    """Get the vector similarity search service"""
    search_service = VectorSimilaritySearch(
        connection_string=db_config.connection_string
    )
    try:
        yield search_service
    finally:
        await search_service.cleanup()

@router.post("/", response_model=SearchResponse)
async def search_documents(
    search_params: SearchQuery,
    search_service: VectorSimilaritySearch = Depends(get_search_service),
    db: AsyncSession = Depends(get_db)
):
    """
    Perform vector similarity search on documents
    
    This endpoint searches through document chunks using vector embeddings
    to find the most semantically similar content to the provided query.
    """
    try:
        # Perform the search
        start_time = __import__('time').time()
        
        # Set parameters on the search service instance
        search_service.top_k = search_params.top_k
        search_service.score_threshold = search_params.threshold
        
        # The search method only accepts the query directly
        results = await search_service.search(query=search_params.query)
        elapsed_time = (__import__('time').time() - start_time) * 1000  # Convert to ms
        
        # Format results
        formatted_results = []
        for idx, result in enumerate(results, 1):
            # Extract the document_id and chunk_id from the metadata or chunk_index
            # The SearchResult from vector_similarity_search.py has keys:
            # content, filename, page, chunk_index, chunk_size, metadata, similarity_score
            metadata = result.get('metadata', {}) or {}  # Ensure metadata is always a dict
            
            # Extract document_id from metadata or set a placeholder value
            document_id = metadata.get('document_id')
            if document_id is None:
                # Try to extract from chunk_metadata fields
                chunk_metadata = result.get('chunk_metadata', {}) or {}
                document_id = chunk_metadata.get('document_id', 0)
            
            # Ensure document_id is an integer
            try:
                document_id = int(document_id) if document_id is not None else idx
            except (ValueError, TypeError):
                document_id = idx
            
            # Generate a chunk_id - use the result index if nothing else is available
            chunk_id = None
            try:
                # First try getting from metadata
                if metadata and 'chunk_id' in metadata:
                    chunk_id = int(metadata['chunk_id'])
                # Then try from chunk_metadata
                elif result.get('chunk_metadata') and 'chunk_id' in result['chunk_metadata']:
                    chunk_id = int(result['chunk_metadata']['chunk_id'])
                # Then try chunk_index
                elif result.get('chunk_index') is not None:
                    chunk_id = int(result['chunk_index'])
            except (ValueError, TypeError):
                pass
            
            # If all else fails, use the result index
            if chunk_id is None:
                chunk_id = idx
            
            formatted_results.append({
                "document_id": document_id,
                "document_name": result.get('filename', 'Unknown'),
                "chunk_id": chunk_id,  # This is now guaranteed to be an integer
                "content": result.get('content', ''),
                "page_number": result.get('page'),
                "section_title": metadata.get('section_title'),
                "score": result.get('similarity_score', 0.0),
                "metadata": metadata
            })
        
        return {
            "results": formatted_results,
            "total_results": len(formatted_results),
            "query": search_params.query,
            "elapsed_time_ms": elapsed_time
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@router.get("/cache-info", response_model=CacheInfo)
async def get_cache_info(
    search_service: VectorSimilaritySearch = Depends(get_search_service)
):
    """
    Get information about the embedding cache
    
    Returns statistics and entries in the embedding cache.
    """
    try:
        # The get_cache_info method doesn't return a value, it only prints to console
        # We need to format our own response based on the cache data
        
        # Initialize default values
        cache_entries = []
        hit_count = 0
        miss_count = 0
        cache_size = 0
        
        if search_service.use_redis and search_service.redis_client:
            # Handle Redis cache
            try:
                # Get all embedding keys
                keys = search_service.redis_client.keys("embedding:*")
                cache_size = len(keys)
                
                # Get metadata for displaying recent entries
                meta_keys = search_service.redis_client.keys("meta:*")
                
                # Get up to 10 most recent entries
                for key in meta_keys[:10]:
                    try:
                        meta_json = search_service.redis_client.get(key)
                        if meta_json:
                            meta = json.loads(meta_json)
                            cache_entries.append({
                                "query": meta.get("query", "Unknown"),
                                "timestamp": meta.get("timestamp", 0),
                                "key": key.decode('utf-8') if isinstance(key, bytes) else key
                            })
                    except Exception:
                        continue
                        
                # Get hit/miss statistics if available
                hit_key = "stats:cache_hits"
                miss_key = "stats:cache_misses"
                hit_count = int(search_service.redis_client.get(hit_key) or 0)
                miss_count = int(search_service.redis_client.get(miss_key) or 0)
            except Exception as e:
                # If Redis operations fail, we still want to return a valid response
                pass
        else:
            # Handle file-based cache
            cache_size = len(search_service.embedding_cache)
            
            # Get recent entries
            entries_with_time = []
            for key, data in search_service.embedding_cache.items():
                entries_with_time.append((key, data.get('timestamp', 0), data.get('query', 'Unknown')))
            
            # Sort by timestamp (descending) and take top 10
            entries_with_time.sort(key=lambda x: x[1], reverse=True)
            for key, timestamp, query in entries_with_time[:10]:
                cache_entries.append({
                    "query": query,
                    "timestamp": timestamp,
                    "key": key
                })
        
        # Calculate hit rate
        total_requests = hit_count + miss_count
        hit_rate = hit_count / total_requests if total_requests > 0 else 0.0
        
        # Format response according to CacheInfo model
        return {
            "size": cache_size,
            "hit_count": hit_count,
            "miss_count": miss_count,
            "hit_rate": hit_rate,
            "entries": cache_entries
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache info error: {str(e)}")

@router.delete("/cache", status_code=204)
async def clear_cache(
    search_service: VectorSimilaritySearch = Depends(get_search_service)
):
    """
    Clear the embedding cache
    
    Removes all entries from the embedding cache.
    """
    try:
        search_service.clear_embedding_cache()
        return None  # 204 No Content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear error: {str(e)}")
