import asyncio
import os
from typing import List, Dict, Optional, TypedDict, Union
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

# Load environment variables for API keys and configuration
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Add custom exceptions
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
class SearchResult(TypedDict):
    """Type definition for search results."""
    content: str
    filename: str
    page: Optional[int]
    chunk_index: int
    chunk_size: int
    metadata: Dict
    similarity_score: float

# Add utility functions
def format_embedding_vector(embedding: Union[List[float], np.ndarray]) -> str:
    """Convert embedding to PostgreSQL vector format."""
    if isinstance(embedding, np.ndarray):
        embedding = embedding.astype(np.float32).tolist()
    return f"[{','.join(map(str, embedding))}]"

# Add retry decorator
def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
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
        connection_string: str,
        top_k: int = 5,
        score_threshold: float = 0.7,
        pool_size: int = 5,
        max_overflow: int = 10
    ):
        """
        Initialize the vector similarity search system.
        
        Args:
            connection_string: PostgreSQL connection URL
            top_k: Number of most similar documents to return
            score_threshold: Minimum similarity score (0-1) for matches
        """
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
            model="text-embedding-ada-002"
        )
        
        # Set search parameters
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.console = Console()

    @retry_with_backoff()
    async def _generate_embedding(self, query: str) -> List[float]:
        """Generate embedding for query text with retry logic."""
        try:
            embedding = await self.embeddings.aembed_query(query)
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
            results: List of search result dictionaries
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

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.engine.dispose()

async def main() -> None:
    """
    Main function to run the interactive search interface.
    """
    searcher = None
    try:
        searcher = VectorSimilaritySearch(
            connection_string="postgresql+asyncpg://postgres:1%40SSongou2@192.168.1.185:5432/ragSystem",
            top_k=5,
            score_threshold=0.5
        )
        
        while True:
            # Get search query from user
            query = input("\nEnter your search query (or 'q' to quit): ")
            if query.lower() == 'q':
                break
            
            try:
                # Perform search and display results
                results = await searcher.search(query)
                if results:
                    rprint(f"\n[green]Found {len(results)} relevant documents[/green]")
                    searcher.display_results(results)
                else:
                    rprint("[yellow]No relevant documents found[/yellow]")
            except SearchError as e:
                rprint(f"[red]Search error: {str(e)}[/red]")
            except Exception as e:
                rprint(f"[red]Unexpected error: {str(e)}[/red]")
    
    except KeyboardInterrupt:
        rprint("\n[yellow]Search terminated by user[/yellow]")
    except Exception as e:
        rprint(f"[red]Fatal error: {str(e)}[/red]")
    finally:
        if searcher:
            await searcher.cleanup()
        rprint("[green]Search session ended[/green]")

if __name__ == "__main__":
    asyncio.run(main()) 