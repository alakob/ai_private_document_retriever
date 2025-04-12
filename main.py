#!/usr/bin/env python3
"""
AI Private Document Retriever - Main Application Entry Point

This script provides a central entry point to the application,
with options to run either the document processor or the chat interface.
"""

import os
import argparse
from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create a FastAPI app for health checks and API endpoints
app = FastAPI(
    title="AI Private Document Retriever",
    description="API for document storage, retrieval, and question answering using AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware for frontend access
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint to provide basic information or redirect to docs."""
    return {
        "app": "AI Private Document Retriever",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker container monitoring."""
    return {"status": "healthy"}

# Include all API routers
from app.routers import api_router
app.include_router(api_router, prefix="/api/v1")

def main():
    """Parse command line arguments and run the appropriate component."""
    parser = argparse.ArgumentParser(
        description="AI Private Document Retriever"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Document processing command
    process_parser = subparsers.add_parser(
        "process", help="Process documents and store in vector database"
    )
    process_parser.add_argument(
        "--dir", 
        type=str, 
        default="documents",
        help="Directory containing documents to process"
    )
    process_parser.add_argument(
        "--reset-db", 
        action="store_true",
        help="Reset the database before processing (warning: deletes existing data)"
    )
    process_parser.add_argument(
        "--use-docling", 
        action="store_true",
        help="Use Docling for enhanced document conversion"
    )
    process_parser.add_argument(
        "--use-mistral", 
        action="store_true",
        help="Use Mistral OCR API for document processing and text extraction"
    )
    
    # Chat interface command
    chat_parser = subparsers.add_parser(
        "chat", help="Start the chat interface"
    )
    chat_parser.add_argument(
        "--share", 
        action="store_true",
        help="Create a shareable link"
    )
    
    # Vector search command
    search_parser = subparsers.add_parser(
        "search", help="Run vector similarity search interactive CLI"
    )
    search_parser.add_argument(
        "--query", 
        type=str,
        help="Optional query to search for (if not provided, will enter interactive mode)"
    )
    search_parser.add_argument(
        "--top-k", 
        type=int,
        default=5,
        help="Number of results to return"
    )
    search_parser.add_argument(
        "--threshold", 
        type=float,
        default=0.7,
        help="Similarity threshold (0-1)"
    )
    search_parser.add_argument(
        "--no-cache", 
        action="store_true",
        help="Disable embedding cache to always generate fresh embeddings"
    )
    search_parser.add_argument(
        "--cache-info", 
        action="store_true",
        help="Display information about the embedding cache"
    )
    search_parser.add_argument(
        "--clear-cache", 
        action="store_true",
        help="Clear the embedding cache"
    )
    
    # Visualization command
    viz_parser = subparsers.add_parser(
        "visualize", help="Generate vector store visualization"
    )
    viz_parser.add_argument(
        "--output", 
        type=str,
        default="visualizations/vector_visualization.html",
        help="Output HTML file for visualization (saved to visualizations directory)"
    )
    viz_parser.add_argument(
        "--perplexity", 
        type=int,
        default=30,
        help="TSNE perplexity parameter"
    )
    
    # API server command
    api_parser = subparsers.add_parser(
        "api", help="Start the API server with health check endpoint"
    )
    api_parser.add_argument(
        "--host", 
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    api_parser.add_argument(
        "--port", 
        type=int,
        default=8000,
        help="Port to bind the server to"
    )
    
    args = parser.parse_args()
    
    # Process the command
    if args.command == "process":
        from app.core.document_rag_loader import async_main
        import asyncio
        # Create a config dictionary to pass to async_main
        config_dict = {
            "directory": args.dir,
            "reset_db": args.reset_db,
            "use_docling": args.use_docling,
            "use_mistral": args.use_mistral
        }
        asyncio.run(async_main(**config_dict))
    elif args.command == "chat":
        from app.ui.chat_interface import main as chat_main
        from app.config import chat_config
        
        # Update share setting in config based on argument
        if args.share:
            chat_config.share = True
            
        # Run the async chat interface
        import asyncio
        asyncio.run(chat_main())
    elif args.command == "api":
        # Start the FastAPI server
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.command == "search":
        from app.services.vector_similarity_search import main as search_main
        import asyncio
        
        # Check if we need to perform special cache operations
        if args.cache_info or args.clear_cache:
            from app.services.vector_similarity_search import VectorSimilaritySearch
            from app.config import db_config
            
            async def manage_cache():
                # Initialize the searcher
                searcher = VectorSimilaritySearch(connection_string=db_config.connection_string)
                
                try:
                    # Display cache info if requested
                    if args.cache_info:
                        searcher.get_cache_info()
                    
                    # Clear cache if requested
                    if args.clear_cache:
                        searcher.clear_embedding_cache()
                finally:
                    # Clean up resources - only clear cache if we were explicitly asked to
                    await searcher.cleanup(clear_cache=args.clear_cache)
            
            # Run the cache management function
            asyncio.run(manage_cache())
        else:
            # Normal search operation
            search_kwargs = {}
            if args.query:
                search_kwargs["query"] = args.query
            if args.top_k:
                search_kwargs["top_k"] = args.top_k
            if args.threshold:
                search_kwargs["threshold"] = args.threshold
            if args.no_cache:
                search_kwargs["no_cache"] = True
                
            # Run the search interface
            asyncio.run(search_main(**search_kwargs))
    elif args.command == "visualize":
        import asyncio
        
        # Define the visualization function directly
        async def generate_visualization(output_file="vector_visualization.html", perplexity=30):
            from utils.vector_store_visualization import visualize_vector_store
            from app.core.document_rag_loader import create_async_engine, async_sessionmaker
            from app.config import db_config
            import plotly
            import os
            
            print(f"Generating vector store visualization with perplexity={perplexity}...")
            print("This may take a few minutes depending on the size of your vector store.")
            
            # Create engine and session
            engine = create_async_engine(db_config.connection_string)
            async_session = async_sessionmaker(bind=engine, expire_on_commit=False)
            
            try:
                async with async_session() as session:
                    fig = await visualize_vector_store(session, perplexity=perplexity)
                    # Ensure we have an absolute path
                    abs_path = os.path.abspath(output_file)
                    plotly.offline.plot(fig, filename=abs_path, auto_open=True)
                    print(f"✅ Visualization successfully saved to {abs_path}")
                    print(f"The visualization should have opened in your web browser.")
            except Exception as e:
                print(f"❌ Error generating visualization: {str(e)}")
                raise
            finally:
                await engine.dispose()
                
        # Run the visualization function with args
        asyncio.run(generate_visualization(output_file=args.output, perplexity=args.perplexity))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
