#!/usr/bin/env python3
"""
Vector Store Visualization Runner

This script visualizes document embeddings from the vector store in 3D space
using t-SNE dimensionality reduction.

Usage:
    python vector_visualization_runner.py [--output FILENAME]

Options:
    --output FILENAME    Output HTML file name (default: vector_visualization.html)
"""

import asyncio
import argparse
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Ensure src directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import visualization components
from src.services.vector_visualization_adapter import (
    setup_database,
    visualize_vector_store,
    calculate_elapsed_time,
    logger
)

async def main(output_file: str = "vector_visualization.html") -> None:
    """
    Main entry point for vector store visualization
    
    Args:
        output_file: Path to save the HTML visualization
    """
    # Load environment variables
    load_dotenv()
    
    # Validate environment variables
    required_vars = ['POSTGRES_HOST', 'POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DB']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file or set these variables in your environment.")
        return
    
    engine = None
    try:
        logger.info("Starting vector store visualization process")
        start_time = datetime.now()
        
        # Setup database connection
        engine, async_session = setup_database()
        
        async with async_session() as session:
            try:
                # Generate visualization
                fig = await visualize_vector_store(session)
                
                # Save visualization to file
                fig.write_html(output_file)
                logger.info(f"Saved visualization to {output_file}")
                print(f"Visualization saved to {output_file}")
                
                # Show visualization in browser if running interactively
                if sys.stdout.isatty():
                    fig.show()
                    print("Visualization displayed in browser")
                
            except Exception as e:
                logger.error(f"Error during visualization: {str(e)}", exc_info=True)
                print(f"Error: {str(e)}")
                return
            
        total_time = calculate_elapsed_time(start_time)
        logger.info(f"Visualization process completed in {total_time:.2f} seconds")
        print(f"Visualization process completed in {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Critical error in visualization process: {str(e)}", exc_info=True)
        print(f"Critical error: {str(e)}")
    finally:
        if engine:
            await engine.dispose()
            logger.info("Database engine disposed")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Visualize document embeddings in 3D space")
    parser.add_argument(
        "--output", 
        default="vector_visualization.html",
        help="Output HTML file name (default: vector_visualization.html)"
    )
    args = parser.parse_args()
    
    # Run the main function
    asyncio.run(main(args.output)) 