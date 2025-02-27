import logging
from datetime import datetime
from typing import Optional, Tuple
import plotly.graph_objects as go
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.visualization.vector_store import (
    visualize_vector_store,
    setup_visualization_logging,
    calculate_elapsed_time
)

logger = setup_visualization_logging()

class StoreVisualizer:
    """Handles vector store visualization operations"""
    
    def __init__(self, session_maker) -> None:
        self.session_maker = session_maker
        
    async def generate_visualization(self) -> Tuple[Optional[go.Figure], str]:
        """Generate vector store visualization with detailed logging"""
        try:
            start_time = datetime.now()
            logger.info("Starting vector store visualization process")
            
            async with self.session_maker() as session:
                if session is None:
                    error_msg = "Failed to create database session"
                    logger.error(error_msg)
                    return None, error_msg
                
                try:
                    logger.info("Initiating vector store visualization")
                    fig, msg = await visualize_vector_store(session)
                    
                    if fig is None:
                        error_msg = "Visualization generation failed - no figure returned"
                        logger.error(error_msg)
                        return None, error_msg
                    
                    processing_time = calculate_elapsed_time(start_time)
                    success_msg = (
                        f"Visualization generated successfully in {processing_time:.2f} seconds!\n"
                        "Hover over points to see document details."
                    )
                    logger.info(success_msg)
                    return fig, success_msg
                    
                except Exception as e:
                    error_msg = f"Error during visualization: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    return None, error_msg
                    
        except Exception as e:
            error_msg = f"Critical error in visualization process: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg 