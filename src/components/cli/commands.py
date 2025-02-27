"""
CLI commands for document processing and search.
"""

import click
import asyncio
from pathlib import Path
from typing import List, Optional
import logging

from ...services.database import create_async_db_engine, create_async_session_maker
from ...components.document_processing.processor import DocumentProcessor
from ...components.search.searcher import DocumentSearcher
from ...config.processor import ProcessorConfig
from ...config.database import PostgresConfig
from ...utils.file_handler import FileHandler

logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Document processing and search CLI."""
    pass

@cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('--chunk-size', default=1000, help='Size of text chunks')
@click.option('--chunk-overlap', default=200, help='Overlap between chunks')
@click.option('--batch-size', default=10, help='Batch size for processing')
def process(directory: str, chunk_size: int, chunk_overlap: int, batch_size: int):
    """Process documents in a directory."""
    async def _process():
        try:
            # Initialize configuration
            postgres_config = PostgresConfig(
                connection_string="postgresql+asyncpg://postgres:1%40SSongou2@192.168.1.185:5432/ragSystem"
            )
            
            processor_config = ProcessorConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                batch_size=batch_size,
                postgres_config=postgres_config
            )
            
            # Initialize processor
            processor = DocumentProcessor(processor_config)
            await processor.initialize_database()
            
            # Process directory
            directory_path = Path(directory)
            docs = await processor.process_directory(directory_path)
            
            click.echo(f"Successfully processed {len(docs)} documents")
            
        except Exception as e:
            click.echo(f"Error: {str(e)}", err=True)
            raise click.Abort()
    
    asyncio.run(_process())

@cli.command()
@click.argument('query')
@click.option('--k', default=4, help='Number of results to return')
@click.option('--threshold', default=0.5, help='Similarity threshold')
def search(query: str, k: int, threshold: float):
    """Search for documents similar to query."""
    async def _search():
        try:
            # Initialize database
            engine = create_async_db_engine(
                "postgresql+asyncpg://postgres:1%40SSongou2@192.168.1.185:5432/ragSystem"
            )
            session_maker = create_async_session_maker(engine)
            
            # Initialize searcher
            async with session_maker() as session:
                searcher = DocumentSearcher(session)
                await searcher.initialize()
                
                # Perform search
                results = await searcher.search(query, k, threshold)
                
                # Format and display results
                formatted_results = searcher.format_results(results)
                click.echo(formatted_results)
                
        except Exception as e:
            click.echo(f"Error: {str(e)}", err=True)
            raise click.Abort()
        finally:
            await engine.dispose()
    
    asyncio.run(_search())

@cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('--max-age-days', default=30, help='Maximum age of files to keep')
def cleanup(directory: str, max_age_days: int):
    """Clean up old files from directory."""
    try:
        file_handler = FileHandler(directory)
        file_handler.cleanup_old_files(max_age_days)
        click.echo(f"Successfully cleaned up files older than {max_age_days} days")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

if __name__ == '__main__':
    cli() 