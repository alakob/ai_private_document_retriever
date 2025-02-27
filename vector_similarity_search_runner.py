#!/usr/bin/env python3
"""
Vector Similarity Search Runner

This script provides an interactive interface for searching document embeddings
using cosine similarity in a PostgreSQL vector store.

Usage:
    python vector_similarity_search_runner.py [--threshold THRESHOLD] [--top_k TOP_K]

Options:
    --threshold THRESHOLD    Minimum similarity score threshold (default: 0.5)
    --top_k TOP_K            Number of results to return (default: 5)
"""

import asyncio
import argparse
import os
import sys
from dotenv import load_dotenv

# Ensure src directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import search components
from src.services.vector_similarity_adapter import (
    VectorSimilaritySearch,
    SearchError,
    ConfigurationError,
    logger
)
from rich import print as rprint

async def main(threshold: float = 0.5, top_k: int = 5) -> None:
    """
    Main function to run the interactive search interface.
    
    Args:
        threshold: Minimum similarity score threshold
        top_k: Number of results to return
    """
    # Load environment variables
    load_dotenv()
    
    # Validate environment variables
    required_vars = ['POSTGRES_HOST', 'POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DB', 'OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        rprint(f"[red]Error: Missing required environment variables: {', '.join(missing_vars)}[/red]")
        rprint("[yellow]Please check your .env file or set these variables in your environment.[/yellow]")
        return
    
    searcher = None
    try:
        logger.info("Initializing vector similarity search")
        # Create searcher using environment variables for connection
        searcher = VectorSimilaritySearch(
            top_k=top_k,
            score_threshold=threshold
        )
        
        logger.info(f"Search initialized with threshold={threshold}, top_k={top_k}")
        rprint("[green]Vector similarity search initialized successfully[/green]")
        rprint(f"[blue]Using threshold: {threshold} and returning top {top_k} results[/blue]")
        rprint("[yellow]Enter your search queries below. Type 'q' to quit.[/yellow]")
        
        while True:
            # Get search query from user
            query = input("\nEnter your search query (or 'q' to quit): ")
            if query.lower() == 'q':
                logger.info("User requested to quit")
                break
            
            logger.info(f"Processing search query: {query[:50]}...")
            try:
                # Perform search and display results
                results = await searcher.search(query)
                if results:
                    logger.info(f"Found {len(results)} matching documents")
                    rprint(f"\n[green]Found {len(results)} relevant documents[/green]")
                    searcher.display_results(results)
                else:
                    logger.info("No matching documents found")
                    rprint("[yellow]No relevant documents found[/yellow]")
            except SearchError as e:
                logger.error(f"Search error: {str(e)}")
                rprint(f"[red]Search error: {str(e)}[/red]")
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                rprint(f"[red]Unexpected error: {str(e)}[/red]")
    
    except KeyboardInterrupt:
        logger.info("Search terminated by user (KeyboardInterrupt)")
        rprint("\n[yellow]Search terminated by user[/yellow]")
    except ConfigurationError as e:
        logger.error(f"Configuration error: {str(e)}")
        rprint(f"[red]Configuration error: {str(e)}[/red]")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        rprint(f"[red]Fatal error: {str(e)}[/red]")
    finally:
        if searcher:
            logger.info("Cleaning up resources")
            await searcher.cleanup()
        logger.info("Search session ended")
        rprint("[green]Search session ended[/green]")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Search document embeddings using vector similarity")
    parser.add_argument(
        "--threshold", 
        type=float,
        default=0.5,
        help="Minimum similarity score threshold (default: 0.5)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    args = parser.parse_args()
    
    # Run the main function
    asyncio.run(main(args.threshold, args.top_k)) 