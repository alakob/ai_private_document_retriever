"""
Embedding service with caching support.

This module provides a unified interface for embedding generation with caching,
whether file-based or Redis-based. It abstracts the caching logic away from
other components.
"""

import os
import json
import time
import pickle
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
import redis

from langchain_openai import OpenAIEmbeddings
from rich import print as rprint

from app.config import embedding_config, redis_config

class CachedEmbeddingService:
    """A service for generating and caching embeddings."""
    
    def __init__(self, model_name: str = None, cache_dir: str = None, use_redis: bool = None):
        """
        Initialize the embedding service with caching support.
        
        Args:
            model_name: The name of the embedding model to use (default: from config)
            cache_dir: Directory to store cache file (default: ~/.cache/ai_document_retriever)
            use_redis: Whether to use Redis for caching (default: from config)
        """
        self.model_name = model_name or embedding_config.model_name
        self.embedding_model = OpenAIEmbeddings(model=self.model_name)
        
        # Cache settings
        self.use_redis = use_redis if use_redis is not None else redis_config.use_redis_cache
        self.redis_client = None
        self.embedding_cache = {}
        
        # Set up cache directory and file
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "ai_document_retriever"
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "embedding_cache.pkl"
        
        # Initialize caching
        if self.use_redis:
            self._init_redis_cache()
        else:
            self._load_cache()
            
        logging.info(f"Initialized embedding service with {'Redis' if self.use_redis else 'file-based'} caching")
            
    def _init_redis_cache(self) -> None:
        """Initialize Redis connection for caching."""
        try:
            self.redis_client = redis.Redis(
                host=redis_config.host,
                port=redis_config.port,
                db=redis_config.db,
                password=redis_config.password if redis_config.password else None,
                decode_responses=False  # We need binary data for embeddings
            )
            # Test connection
            self.redis_client.ping()
            logging.info(f"Redis cache connection established successfully at {redis_config.host}:{redis_config.port}")
        except redis.exceptions.ConnectionError as e:
            logging.warning(f"Redis connection failed: {str(e)} - Falling back to file-based caching")
            self.use_redis = False
            self._load_cache()
        except Exception as e:
            logging.warning(f"Redis initialization error: {str(e)}")
            self.use_redis = False
            self._load_cache()
            
    def _load_cache(self) -> None:
        """Load the embedding cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logging.info(f"Loaded {len(self.embedding_cache)} entries from file cache at {self.cache_file}")
            except Exception as e:
                logging.warning(f"Could not load cache file: {str(e)}")
                self.embedding_cache = {}
        else:
            logging.info("No existing cache file found, starting with empty cache")
            self.embedding_cache = {}
            
    def _save_cache(self) -> None:
        """Save the embedding cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logging.info(f"Saved embedding cache to {self.cache_file}")
        except Exception as e:
            logging.error(f"Error saving cache: {str(e)}")
            
    def _generate_cache_key(self, text: str) -> str:
        """Generate a cache key for the given text."""
        # Create a deterministic hash of the text for caching
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    async def _get_redis_cache(self, key: str) -> Optional[List[float]]:
        """
        Retrieve embedding from Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            List of float values for embedding or None if not found
        """
        if not self.redis_client:
            return None
            
        try:
            # Get embedding data
            embedding_data = self.redis_client.get(f"embedding:{key}")
            if not embedding_data:
                return None
                
            # Get metadata
            meta_json = self.redis_client.get(f"meta:{key}")
            if meta_json:
                # Update metadata access time
                meta = json.loads(meta_json)
                meta["last_accessed"] = time.time()
                self.redis_client.set(f"meta:{key}", json.dumps(meta))
                
                # Add expiry if not already set
                if redis_config.cache_expiry > 0:
                    self.redis_client.expire(f"embedding:{key}", redis_config.cache_expiry)
                    self.redis_client.expire(f"meta:{key}", redis_config.cache_expiry)
                
            # Deserialize the embedding
            embedding = pickle.loads(embedding_data)
            return embedding
        except Exception as e:
            logging.warning(f"Redis cache retrieval error: {str(e)}")
            return None
            
    async def _set_redis_cache(self, key: str, embedding: List[float], text: str) -> None:
        """
        Store embedding in Redis cache.
        
        Args:
            key: Cache key
            embedding: The embedding vector to cache
            text: The original text for reference
        """
        if not self.redis_client:
            return
            
        try:
            # Serialize the embedding
            embedding_data = pickle.dumps(embedding)
            
            # Store the embedding
            self.redis_client.set(f"embedding:{key}", embedding_data)
            
            # Store metadata
            meta = {
                "timestamp": time.time(),
                "last_accessed": time.time(),
                "model": self.model_name,
                "query": text[:100] + "..." if len(text) > 100 else text
            }
            self.redis_client.set(f"meta:{key}", json.dumps(meta))
            
            # Set expiry if configured
            if redis_config.cache_expiry > 0:
                self.redis_client.expire(f"embedding:{key}", redis_config.cache_expiry)
                self.redis_client.expire(f"meta:{key}", redis_config.cache_expiry)
                
        except Exception as e:
            logging.warning(f"Redis cache storage error: {str(e)}")
    
    async def get_embedding(self, text: str, no_cache: bool = False) -> List[float]:
        """
        Get embedding for text, using cache if available.
        
        Args:
            text: Text to generate embedding for
            no_cache: If True, bypass cache and generate fresh embedding
            
        Returns:
            Embedding vector as list of floats
        """
        if not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
            
        # Generate cache key
        cache_key = self._generate_cache_key(text)
        
        # Check if no_cache flag is set
        if no_cache:
            embedding = await self.embedding_model.aembed_query(text)
            return embedding
            
        # Try to get from cache
        if self.use_redis:
            # Try Redis cache
            cached_data = await self._get_redis_cache(cache_key)
            if cached_data:
                logging.info(f"Cache HIT: Using cached embedding from Redis for text '{text[:30]}...'")
                return cached_data
        else:
            # Try file cache
            if cache_key in self.embedding_cache:
                logging.info(f"Cache HIT: Using cached embedding from file cache for text '{text[:30]}...'")
                # Update last accessed time
                self.embedding_cache[cache_key]["last_accessed"] = time.time()
                return self.embedding_cache[cache_key]["embedding"]
        
        # Generate new embedding if not in cache
        logging.info(f"Cache MISS: Generating new embedding for text '{text[:30]}...'")
        embedding = await self.embedding_model.aembed_query(text)
        
        # Store in appropriate cache
        if self.use_redis:
            await self._set_redis_cache(cache_key, embedding, text)
            logging.info(f"Stored new embedding in Redis cache with key {cache_key[:8]}")
        else:
            self.embedding_cache[cache_key] = {
                "embedding": embedding,
                "timestamp": time.time(),
                "last_accessed": time.time(),
                "model": self.model_name,
                "query": text[:100] + "..." if len(text) > 100 else text
            }
            self._save_cache()
            logging.info(f"Stored new embedding in file cache with key {cache_key[:8]}")
            
        return embedding
        
    async def get_embeddings_batch(self, texts: List[str], no_cache: bool = False) -> List[List[float]]:
        """
        Get embeddings for multiple texts, using cache where available.
        
        Args:
            texts: List of texts to generate embeddings for
            no_cache: If True, bypass cache and generate fresh embeddings
            
        Returns:
            List of embedding vectors
        """
        cache_type = "Redis" if self.use_redis else "file"
        logging.info(f"Processing batch of {len(texts)} embeddings using {cache_type} cache (no_cache={no_cache})")
        
        cache_hits = 0
        cache_misses = 0
        results = []
        
        for text in texts:
            # Generate cache key to check if it's in cache before calling get_embedding
            cache_key = self._generate_cache_key(text)
            in_cache = False
            
            if not no_cache:
                if self.use_redis:
                    cached_data = await self._get_redis_cache(cache_key)
                    in_cache = cached_data is not None
                else:
                    in_cache = cache_key in self.embedding_cache
            
            # Get the embedding (will use cache if available)
            embedding = await self.get_embedding(text, no_cache)
            results.append(embedding)
            
            # Update stats
            if in_cache:
                cache_hits += 1
            else:
                cache_misses += 1
        
        # Log cache efficiency
        if len(texts) > 0:
            hit_rate = cache_hits / len(texts) * 100
            logging.info(f"Embedding batch completed: {cache_hits} cache hits, {cache_misses} cache misses ({hit_rate:.1f}% hit rate)")
            
        return results
        
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self.use_redis and self.redis_client:
            try:
                # Get cache size before clearing
                keys = self.redis_client.keys("embedding:*")
                cache_size = len(keys)
                
                # Delete all embedding keys
                if keys:
                    self.redis_client.delete(*keys)
                # Delete all metadata keys
                meta_keys = self.redis_client.keys("meta:*")
                if meta_keys:
                    self.redis_client.delete(*meta_keys)
                    
                logging.info(f"Cleared Redis embedding cache ({cache_size} entries)")
            except Exception as e:
                logging.error(f"Error clearing Redis cache: {str(e)}")
        else:
            # Clear file-based cache
            cache_size = len(self.embedding_cache)
            self.embedding_cache.clear()
            self._save_cache()  # Save empty cache to disk
            logging.info(f"Cleared file embedding cache ({cache_size} entries)")

# Singleton instance for shared use
_embedding_service = None

async def get_embedding_service() -> CachedEmbeddingService:
    """Get or create the shared embedding service instance asynchronously."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = CachedEmbeddingService()
        # Initialize Redis connection asynchronously if needed
        if _embedding_service.use_redis and not _embedding_service.redis_client:
            try:
                _embedding_service.redis_client = redis.Redis(
                    host=redis_config.host,
                    port=redis_config.port,
                    db=redis_config.db,
                    password=redis_config.password,
                    decode_responses=False
                )
                logging.info("Redis connection initialized for embedding service")
            except Exception as e:
                logging.error(f"Failed to connect to Redis: {str(e)}")
                logging.info("Falling back to file-based cache")
                _embedding_service.use_redis = False
    return _embedding_service

def get_embedding_service_instance() -> CachedEmbeddingService:
    """Get or create the shared embedding service instance synchronously.
    
    This is meant to be used during app initialization where async
    functions cannot be awaited.
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = CachedEmbeddingService()
        # Initialize Redis connection if needed
        if _embedding_service.use_redis:
            try:
                _embedding_service.redis_client = redis.Redis(
                    host=redis_config.host,
                    port=redis_config.port,
                    db=redis_config.db,
                    password=redis_config.password,
                    decode_responses=False
                )
                logging.info("Redis connection initialized for embedding service")
            except Exception as e:
                logging.error(f"Failed to connect to Redis: {str(e)}")
                logging.info("Falling back to file-based cache")
                _embedding_service.use_redis = False
    return _embedding_service