"""
Document RAG System Runner.

This script uses the DocumentRAGAdapter to process documents and store them in a vector database.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv
from src.services.document_rag_adapter import DocumentRAGAdapter, async_main_document

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main function to run the document RAG system."""
    try:
        # Get database configuration from environment variables
        postgres_host = os.getenv('POSTGRES_HOST', 'localhost')
        postgres_port = os.getenv('POSTGRES_PORT', '5432')
        postgres_user = os.getenv('POSTGRES_USER', 'postgres')
        postgres_password = os.getenv('POSTGRES_PASSWORD', 'postgres')
        postgres_db = os.getenv('POSTGRES_DB', 'vectordb')
        
        # Construct connection string from environment variables
        connection_string = f"postgresql+asyncpg://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
        
        # Get other configuration from environment variables
        vector_store_type = os.getenv('VECTOR_STORE_TYPE', 'postgres')
        directory_path = os.getenv('KNOWLEDGE_BASE_DIR', 'documents')
        
        # Run the document RAG system
        await async_main_document(
            directory_path=directory_path,
            connection_string=connection_string,
            vector_store_type=vector_store_type
        )
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 