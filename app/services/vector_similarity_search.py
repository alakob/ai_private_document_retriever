import asyncio
import os
import hashlib
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, TypedDict, Union, Any
import numpy as np
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.sql import text
from langchain_openai import OpenAIEmbeddings
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from functools import wraps
import time

# Import Redis conditionally to avoid errors if not installed
try:
    import redis
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
from app.core.document_rag_loader import (
    calculate_file_checksum,
    calculate_checksum
)

# Import config
from app.config import db_config, vector_search_config, embedding_config, redis_config

# Load environment variables for API keys and configuration
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Define exceptions locally since they're used in this file
class SearchError(Exception):
    """Base exception for search-related errors."""
    pass

class EmbeddingError(SearchError):
    """Raised when there's an error generating embeddings."""
    pass

class DatabaseError(SearchError):
    """Raised when there's an error with database operations."""
    pass

# Add type definitions
class SearchResult(dict):
    """Type definition for search results."""
    def __init__(self, content: str, filename: str, page: Optional[int], 
                chunk_index: int, chunk_size: int, metadata: Dict, 
                similarity_score: float):
        super().__init__()
        self['content'] = content
        self['filename'] = filename
        self['page'] = page
        self['chunk_index'] = chunk_index
        self['chunk_size'] = chunk_size
        self['metadata'] = metadata
        self['similarity_score'] = similarity_score

# Add utility functions
def format_embedding_vector(embedding: Union[List[float], np.ndarray]) -> str:
    """Convert embedding to PostgreSQL vector format."""
    if isinstance(embedding, np.ndarray):
        embedding = embedding.astype(np.float32).tolist()
    return f"[{','.join(map(str, embedding))}]"

# Add retry decorator
def retry_with_backoff(max_retries: int = vector_search_config.max_retries, base_delay: float = vector_search_config.base_delay):
    """Decorator for retrying operations with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt == max_retries - 1:
                        rprint(f"[red]Failed after {max_retries} attempts: {str(e)}[/red]")
                        raise
                    delay = base_delay * (2 ** attempt)
                    rprint(f"[yellow]Attempt {attempt + 1} failed, retrying in {delay}s[/yellow]")
                    await asyncio.sleep(delay)
            raise last_error
        return wrapper
    return decorator

class VectorSimilaritySearch:
    """
    A class for performing vector similarity search using PostgreSQL with pgvector extension.
    
    This class handles:
    - Asynchronous database connections
    - Vector embedding generation using OpenAI
    - Similarity search using cosine distance
    - Result formatting and display
    
    Attributes:
        connection_string (str): PostgreSQL connection URL
        top_k (int): Number of results to return
        score_threshold (float): Minimum similarity score threshold
    """
    
    def __init__(
        self,
        connection_string: str = None,
        top_k: int = vector_search_config.top_k,
        score_threshold: float = vector_search_config.score_threshold,
        pool_size: int = vector_search_config.pool_size,
        max_overflow: int = vector_search_config.max_overflow
    ):
        """
        Initialize the vector similarity search system.
        
        Args:
            connection_string: PostgreSQL connection URL
            top_k: Number of most similar documents to return
            score_threshold: Minimum similarity score (0-1) for matches
        """
        # Use provided connection string or get from config
        if connection_string is None:
            connection_string = db_config.connection_string
            
        # Initialize async database engine with connection pool
        self.engine = create_async_engine(
            connection_string,
            pool_size=pool_size,
            max_overflow=max_overflow,
            echo=False
        )
        
        # Create async session factory
        self.async_session = async_sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            class_=AsyncSession
        )
        
        # Initialize OpenAI embeddings model
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=embedding_config.model_name
        )
        
        # Set search parameters
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.console = Console()
        
        # Initialize embedding cache
        self.cache_dir = Path(os.path.expanduser("~/.cache/ai_document_retriever"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "embedding_cache.pkl"
        self.cache_expiry = vector_search_config.cache_expiry  # 24 hours in seconds
        
        # Initialize Redis client if enabled
        self.use_redis = vector_search_config.use_redis_cache and REDIS_AVAILABLE
        self.redis_client = None
        if self.use_redis:
            try:
                # Only create Redis client if Redis is available
                if not REDIS_AVAILABLE:
                    rprint(f"[bold yellow]Warning: Redis package not installed. Install with 'pip install redis'[/bold yellow]")
                    rprint(f"[yellow]Falling back to file-based cache[/yellow]")
                    self.use_redis = False
                else:
                    # Create Redis client with password only if password is provided
                    if redis_config.password:
                        self.redis_client = redis.Redis(
                            host=redis_config.host,
                            port=redis_config.port,
                            db=redis_config.db,
                            password=redis_config.password,
                            socket_timeout=redis_config.timeout,
                            decode_responses=False  # Keep binary data as is
                        )
                    else:
                        self.redis_client = redis.Redis(
                            host=redis_config.host,
                            port=redis_config.port,
                            db=redis_config.db,
                            socket_timeout=redis_config.timeout,
                            decode_responses=False  # Keep binary data as is
                        )
                    # Test connection
                    self.redis_client.ping()
                    rprint(f"[bold green]✓ Connected to Redis cache at {redis_config.host}:{redis_config.port}[/bold green]")
            except Exception as e:
                rprint(f"[bold yellow]Warning: Redis connection failed: {str(e)}[/bold yellow]")
                rprint(f"[yellow]Falling back to file-based cache[/yellow]")
                self.use_redis = False
                self.redis_client = None
        
        # Load file-based cache if not using Redis
        if not self.use_redis:
            self.embedding_cache = self._load_cache()
        else:
            self.embedding_cache = {}  # Empty dict as we're using Redis

    def _load_cache(self) -> Dict:
        """Load embedding cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    if len(cache) > 0:
                        rprint(f"[bold green]✓ Loaded embedding cache with {len(cache)} entries[/bold green]")
                        # Show some cached queries as examples
                        sample_size = min(3, len(cache))
                        sample_keys = list(cache.keys())[:sample_size]
                        rprint("[green]Sample cached queries:[/green]")
                        for i, key in enumerate(sample_keys):
                            query = cache[key].get('query', 'Unknown')
                            timestamp = cache[key].get('timestamp', 0)
                            date_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
                            rprint(f"  {i+1}. [cyan]\"{query}\"[/cyan] (cached on {date_str})")
                    else:
                        rprint("[yellow]Embedding cache exists but is empty[/yellow]")
                    return cache
            except Exception as e:
                rprint(f"[yellow]Could not load embedding cache: {str(e)}[/yellow]")
        else:
            rprint("[yellow]No embedding cache found - will create new cache[/yellow]")
        return {}
        
    def _save_cache(self) -> None:
        """Save embedding cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            # No need to print this message every time - it's too verbose
            # Only show it if explicitly managing the cache
        except Exception as e:
            rprint(f"[bold red]Error: Could not save embedding cache: {str(e)}[/bold red]")
    
    async def _get_redis_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cache data from Redis."""
        try:
            if not self.redis_client or not self.use_redis:
                return None
                
            # Get cached embedding data
            try:
                cached_data = self.redis_client.get(f"embedding:{cache_key}")
                if not cached_data:
                    return None
                    
                # Deserialize embedding data
                return pickle.loads(cached_data)
            except RedisError as re:
                rprint(f"[yellow]Redis connection error: {str(re)}. Falling back to API.[/yellow]")
                return None
        except Exception as e:
            rprint(f"[yellow]Redis get error: {str(e)}. Falling back to API.[/yellow]")
            return None
            
    async def _set_redis_cache(self, cache_key: str, data: Dict[str, Any]) -> bool:
        """Set cache data in Redis with expiry."""
        try:
            if not self.redis_client or not self.use_redis:
                return False
                
            try:
                # Serialize and store embedding data
                serialized = pickle.dumps(data)
                self.redis_client.setex(
                    f"embedding:{cache_key}",
                    self.cache_expiry,
                    serialized
                )
                
                # Store metadata separately for debugging/monitoring
                self.redis_client.setex(
                    f"meta:{cache_key}",
                    self.cache_expiry,
                    json.dumps({
                        'query': data.get('query', ''),
                        'timestamp': data.get('timestamp', 0),
                        'cache_source': 'redis'
                    })
                )
                return True
            except RedisError as re:
                rprint(f"[yellow]Redis connection error: {str(re)}. Falling back to file cache.[/yellow]")
                return False
        except Exception as e:
            rprint(f"[yellow]Redis set error: {str(e)}. Falling back to file cache.[/yellow]")
            return False
    
    @retry_with_backoff()
    async def _generate_embedding(self, query: str) -> List[float]:
        """Generate embedding for query text with retry logic and caching."""
        try:
            # Create a cache key from the query
            query_normalized = query.strip().lower()
            cache_key = hashlib.md5(query_normalized.encode()).hexdigest()
            current_time = time.time()
            
            # Check Redis cache first if enabled
            cached_data = None
            if self.use_redis:
                cached_data = await self._get_redis_cache(cache_key)
                if cached_data:
                    age_minutes = (current_time - cached_data.get('timestamp', 0)) / 60
                    rprint(f"[bold cyan]✓ USING REDIS CACHED EMBEDDING[/bold cyan] for query: {query[:50]}...")
                    rprint(f"  [cyan]Cache age:[/cyan] {age_minutes:.1f} minutes")
                    rprint(f"  [cyan]Cache key:[/cyan] {cache_key}")
                    return cached_data.get('embedding', [])
            else:
                # Check file-based cache
                cached_data = self.embedding_cache.get(cache_key)
                if cached_data and (current_time - cached_data['timestamp'] < self.cache_expiry):
                    # Use cached embedding if it exists and isn't expired
                    age_minutes = (current_time - cached_data['timestamp']) / 60
                    rprint(f"[bold cyan]✓ USING FILE CACHED EMBEDDING[/bold cyan] for query: {query[:50]}...")
                    rprint(f"  [cyan]Cache age:[/cyan] {age_minutes:.1f} minutes")
                    rprint(f"  [cyan]Cache key:[/cyan] {cache_key}")
                    return cached_data['embedding']
            
            # Generate new embedding from OpenAI API
            rprint(f"[bold yellow]⚠ GENERATING NEW EMBEDDING[/bold yellow] for query: {query[:50]}...")
            rprint(f"  [yellow]Reason:[/yellow] {'Cache expired' if cached_data else 'Not in cache'}")
            rprint(f"  [yellow]Cache key:[/yellow] {cache_key}")
            
            embedding = await self.embeddings.aembed_query(query)
            
            # Create cache data
            cache_data = {
                'embedding': embedding,
                'timestamp': current_time,
                'query': query[:100]  # Store original query for reference
            }
            
            # Store in appropriate cache
            if self.use_redis:
                await self._set_redis_cache(cache_key, cache_data)
                rprint(f"[green]✓ Saved embedding to Redis cache[/green]")
            else:
                # Store in file-based cache
                self.embedding_cache[cache_key] = cache_data
                self._save_cache()
                rprint(f"[green]✓ Saved embedding to file cache[/green]")
            
            return embedding
        except Exception as e:
            rprint(f"[red]Error generating embedding: {str(e)}[/red]")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}") from e

    async def _execute_similarity_search(
        self, 
        embedding: str,
        threshold: float,
        limit: int
    ) -> List[SearchResult]:
        """Execute similarity search query."""
        try:
            async with self.async_session() as session:
                # Ensure pgvector extension is installed
                await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                
                # SQL query for similarity search using cosine distance
                # Uses pgvector's <=> operator for cosine distance calculation
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
                        "embedding": embedding,
                        "threshold": threshold,
                        "limit": limit
                    }
                )
                
                # Process and format results
                return [
                    SearchResult(
                        content=row.content,
                        filename=row.filename,
                        page=row.page_number,
                        chunk_index=row.chunk_index,
                        chunk_size=row.chunk_size,
                        metadata=row.chunk_metadata,
                        similarity_score=row.similarity_score
                    )
                    for row in result
                ]
                
        except Exception as e:
            rprint(f"[red]Database error: {str(e)}[/red]")
            raise DatabaseError(f"Search query failed: {str(e)}") from e

    async def search(self, query: str) -> List[SearchResult]:
        """
        Perform vector similarity search for the given query.
        
        Process:
        1. Generate embedding for the query text
        2. Convert embedding to PostgreSQL vector format
        3. Execute similarity search using cosine distance
        4. Process and return matching documents
        
        Args:
            query: The search query text
            
        Returns:
            List of dictionaries containing matching documents and their metadata
        
        Raises:
            Exception: If there's an error in embedding generation or database query
        """
        if not query.strip():
            raise ValueError("Empty search query")
            
        try:
            # Generate embedding for query
            query_embedding = await self._generate_embedding(query)
            query_embedding_str = format_embedding_vector(query_embedding)
            
            rprint(f"[green]✓ Generated embedding for query:[/green] {query[:50]}...")
            
            # Execute search
            results = await self._execute_similarity_search(
                embedding=query_embedding_str,
                threshold=self.score_threshold,
                limit=self.top_k
            )
            
            rprint(f"[green]✓ Found {len(results)} matching documents[/green]")
            return results
                
        except SearchError:
            raise
        except Exception as e:
            rprint(f"[red]Error performing similarity search: {str(e)}[/red]")
            raise SearchError(f"Search operation failed: {str(e)}") from e

    def display_results(self, results: List[SearchResult]) -> None:
        """
        Display search results in a formatted table.
        
        Args:
            results: List of SearchResult objects
        """
        table = Table(
            title="Similarity Search Results",
            show_header=True,
            header_style="bold magenta"
        )
        
        # Configure table columns
        table.add_column("Score", justify="right", style="cyan", no_wrap=True)
        table.add_column("Source", style="green")
        table.add_column("Page", justify="right", style="blue")
        table.add_column("Content", style="white", width=60)
        
        # Add results to table
        for result in results:
            table.add_row(
                f"{result['similarity_score']:.3f}",
                result['filename'],
                str(result['page'] or 'N/A'),
                result['content'][:200] + "..."
            )
        
        self.console.print(table)

    def clear_embedding_cache(self) -> None:
        """Clear the embedding cache to free memory and force fresh embeddings."""
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
                    
                rprint(f"[bold green]✓ Cleared Redis embedding cache ({cache_size} entries)[/bold green]")
                rprint(f"  [green]Cache location:[/green] Redis at {redis_config.host}:{redis_config.port}")
            except Exception as e:
                rprint(f"[bold red]Error clearing Redis cache: {str(e)}[/bold red]")
        else:
            # Clear file-based cache
            cache_size = len(self.embedding_cache)
            self.embedding_cache.clear()
            self._save_cache()  # Save empty cache to disk
            rprint(f"[bold green]✓ Cleared file embedding cache ({cache_size} entries)[/bold green]")
            rprint(f"  [green]Cache location:[/green] {self.cache_file}")
        
    def get_cache_info(self) -> None:
        """Display information about the current cache."""
        if self.use_redis and self.redis_client:
            try:
                # Get all embedding keys
                keys = self.redis_client.keys("embedding:*")
                cache_size = len(keys)
                
                if cache_size == 0:
                    rprint("[yellow]Redis embedding cache is empty[/yellow]")
                    return
                    
                rprint(f"[bold green]Redis Embedding Cache Info:[/bold green]")
                rprint(f"  • [green]Total entries:[/green] {cache_size}")
                rprint(f"  • [green]Cache location:[/green] Redis at {redis_config.host}:{redis_config.port}")
                rprint(f"  • [green]Cache type:[/green] Redis (high-performance)")
                
                # Get metadata for displaying recent entries
                meta_keys = self.redis_client.keys("meta:*")
                entries = []
                
                # Get up to 10 most recent entries
                for key in meta_keys[:10]:
                    try:
                        meta_json = self.redis_client.get(key)
                        if meta_json:
                            meta = json.loads(meta_json)
                            entries.append((key, meta))
                    except Exception:
                        continue
                
                # Sort by timestamp (newest first)
                entries.sort(key=lambda x: x[1].get('timestamp', 0), reverse=True)
                
                # Show entries
                if entries:
                    rprint("[bold green]Recent Redis Cache Entries:[/bold green]")
                    for i, (key, meta) in enumerate(entries[:10]):
                        timestamp = meta.get('timestamp', 0)
                        date_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
                        query = meta.get('query', 'Unknown query')
                        rprint(f"  {i+1}. [cyan]\"{query}\"[/cyan] (cached on {date_str})")
            except Exception as e:
                rprint(f"[bold red]Error accessing Redis cache: {str(e)}[/bold red]")
                rprint("[yellow]Falling back to file cache info[/yellow]")
                self._show_file_cache_info()
        else:
            # Show file-based cache info
            self._show_file_cache_info()
    
    def _show_file_cache_info(self) -> None:
        """Display information about the file-based cache."""
        cache_size = len(self.embedding_cache)
        if cache_size == 0:
            rprint("[yellow]File embedding cache is empty[/yellow]")
            return
            
        rprint(f"[bold green]File Embedding Cache Info:[/bold green]")
        rprint(f"  • [green]Total entries:[/green] {cache_size}")
        rprint(f"  • [green]Cache location:[/green] {self.cache_file}")
        rprint(f"  • [green]Cache type:[/green] File-based (pickle)")
        
        # Sort by timestamp (newest first)
        sorted_items = sorted(
            self.embedding_cache.items(), 
            key=lambda x: x[1]['timestamp'], 
            reverse=True
        )
        
        # Show up to 10 most recent items
        if sorted_items:
            rprint("[bold green]Recent File Cache Entries:[/bold green]")
            for i, (key, data) in enumerate(sorted_items[:10]):
                timestamp = data.get('timestamp', 0)
                date_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
                query = data.get('query', 'Unknown query')
                rprint(f"  {i+1}. [cyan]\"{query}\"[/cyan] (cached on {date_str})")
        
    async def cleanup(self, clear_cache: bool = False) -> None:
        """Clean up resources.
        
        Args:
            clear_cache: If True, also clear the embedding cache (default: False)
        """
        # Only clear the cache if explicitly requested
        if clear_cache:
            self.clear_embedding_cache()
            
        # Always dispose of the database engine
        await self.engine.dispose()

async def main(query: str = None, top_k: int = None, threshold: float = None, no_cache: bool = False) -> None:
    """
    Main function to run the interactive search interface.
    
    Args:
        query: Optional query string to search for directly (skips interactive prompt if provided)
        top_k: Optional number of results to return (overrides config)
        threshold: Optional similarity threshold (overrides config)
        no_cache: If True, embedding caching will be disabled (default: False)
    """
    searcher = None
    try:
        # Display welcome banner
        console = Console()
        console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        console.print("[bold green]                 VECTOR SIMILARITY SEARCH[/bold green]")
        console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        console.print("\n[yellow]Search your document collection using natural language queries.[/yellow]")
        console.print("[yellow]This tool uses AI embeddings to find the most relevant content.[/yellow]\n")
        
        # Display system information
        console.print("[bold]System Information:[/bold]")
        console.print(f"  • [cyan]Database:[/cyan] {db_config.host}:{db_config.port}/{db_config.database}")
        console.print(f"  • [cyan]Embedding Model:[/cyan] {embedding_config.model_name}")
        console.print(f"  • [cyan]Results Returned:[/cyan] {vector_search_config.top_k}")
        console.print(f"  • [cyan]Similarity Threshold:[/cyan] {vector_search_config.score_threshold}")
        console.print(f"  • [cyan]Embedding Cache:[/cyan] {'Disabled' if no_cache else 'Enabled'}")
        
        # Display usage tips
        console.print("\n[bold]Usage Tips:[/bold]")
        console.print("  • [green]Be specific[/green] in your queries for better results")
        console.print("  • [green]Use natural language[/green] rather than keywords")
        console.print("  • [green]Type 'q'[/green] at any time to exit the program")
        console.print("  • [green]Try asking about specific topics[/green] in your documents\n")
        
        console.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
        
        # Initialize searcher with overrides if provided
        console.print("\n[bold]Initializing search engine...[/bold]")
        searcher = VectorSimilaritySearch(
            connection_string=db_config.connection_string,
            top_k=top_k if top_k is not None else vector_search_config.top_k,
            score_threshold=threshold if threshold is not None else vector_search_config.score_threshold
        )
        
        # Disable cache if requested
        if no_cache:
            if searcher.use_redis:
                # For Redis we can just skip using the cache in the _generate_embedding method
                searcher.use_redis = False
                rprint("[yellow]Redis cache disabled for this session[/yellow]")
            # Also disable file cache
            searcher.embedding_cache = {}
            searcher.cache_expiry = 0
        console.print("[bold green]✓[/bold green] Search engine ready!\n")
        
        # If direct query was provided, skip interactive mode
        if query:
            console.print(f"\n[bold]Running direct query:[/bold] \"{query}\"")
            console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            
            try:
                # Show searching indicator
                with console.status("[bold green]Searching documents...[/bold green]"):
                    # Perform search and display results
                    results = await searcher.search(query)
                
                if results:
                    console.print(f"\n[bold green]Found {len(results)} relevant documents[/bold green]")
                    searcher.display_results(results)
                else:
                    console.print("\n[yellow]No relevant documents found.[/yellow]")
                    console.print("[yellow]Tips: Be more specific or use different terminology.[/yellow]")
                # Exit after direct query
                return
            except SearchError as e:
                console.print(f"\n[bold red]Search error:[/bold red] {str(e)}")
                return
            except Exception as e:
                console.print(f"\n[bold red]Unexpected error:[/bold red] {str(e)}")
                return
        else:
            # Example queries for interactive mode
            console.print("[bold]Example Queries:[/bold]")
            example_queries = [
                "What are the main topics covered in the documents?",
                "Explain the concept of machine learning in simple terms",
                "What are the best practices for software development?",
                "Summarize the key points about data analysis"
            ]
            for i, example_query in enumerate(example_queries, 1):
                console.print(f"  {i}. [italic cyan]\"{example_query}\"[/italic cyan]")
            
            console.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
            
            # Main search loop for interactive mode
            while True:
                # Get search query from user
                user_query = input("\n[bold]Enter your search query[/bold] (or 'q' to quit): ")
                if user_query.lower() == 'q':
                    break
                
                if not user_query.strip():
                    console.print("[yellow]Please enter a valid query[/yellow]")
                    continue
                
                try:
                    # Show searching indicator
                    with console.status("[bold green]Searching documents...[/bold green]"):
                        # Perform search and display results
                        results = await searcher.search(user_query)
                    
                    if results:
                        console.print(f"\n[bold green]Found {len(results)} relevant documents[/bold green]")
                        searcher.display_results(results)
                    else:
                        console.print("\n[yellow]No relevant documents found. Try a different query.[/yellow]")
                        console.print("[yellow]Tips: Be more specific or use different terminology.[/yellow]")
                except SearchError as e:
                    console.print(f"\n[bold red]Search error:[/bold red] {str(e)}")
                except Exception as e:
                    console.print(f"\n[bold red]Unexpected error:[/bold red] {str(e)}")

    
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Search terminated by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {str(e)}")
    finally:
        if searcher:
            # Don't clear cache during normal cleanup
            await searcher.cleanup(clear_cache=False)
        console.print("\n[green]Search session ended. Thank you for using Vector Similarity Search![/green]")

if __name__ == "__main__":
    asyncio.run(main()) 