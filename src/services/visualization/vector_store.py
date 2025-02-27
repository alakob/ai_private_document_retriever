# Standard library imports
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from packaging import version
import sklearn

# Local imports
from ...config.database import DocumentModel, DocumentChunk, PostgresConfig

# Configuration
def setup_visualization_logging() -> logging.Logger:
    """Configure logging for the visualization module"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('vector_visualization.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_visualization_logging()

def calculate_elapsed_time(start_time: datetime) -> float:
    """Calculate elapsed time in seconds"""
    return (datetime.now() - start_time).total_seconds()

@dataclass
class VectorData:
    """Data class to hold vector information"""
    embeddings: np.ndarray  # Changed to numpy array
    documents: List[str]
    metadata: List[Dict[str, Any]]

class VectorStoreDataAcquisition:
    """Handles data acquisition from PostgreSQL vector store"""
    
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        logger.info("Initialized VectorStoreDataAcquisition")

    async def get_vector_count(self) -> int:
        """Get total count of vectors in store"""
        try:
            result = await self.session.execute(
                select(func.count()).select_from(DocumentChunk)
            )
            count = result.scalar()
            logger.info(f"Found {count} vectors in store")
            return count or 0
        except Exception as e:
            logger.error(f"Error getting vector count: {str(e)}")
            raise

    async def get_vectors_with_metadata(self) -> VectorData:
        """Get all vectors with their documents and metadata"""
        try:
            result = await self.session.execute(
                select(DocumentChunk, DocumentModel)
                .join(DocumentModel)
                .order_by(DocumentChunk.id)
            )
            
            embeddings = []
            documents = []
            metadata = []
            
            for chunk, doc in result:
                # Convert embedding to numpy array if it's not already
                embedding = np.array(chunk.embedding, dtype=np.float32)
                embeddings.append(embedding)
                documents.append(chunk.content)
                
                # Create metadata dictionary with proper None handling
                chunk_metadata = chunk.chunk_metadata if chunk.chunk_metadata is not None else {}
                metadata_dict = {
                    'filename': doc.filename,
                    'page': chunk.page_number,
                    'chunk_index': chunk.chunk_index,
                }
                metadata_dict.update(chunk_metadata)
                metadata.append(metadata_dict)
            
            # Convert list of embeddings to 2D numpy array
            embeddings_array = np.vstack(embeddings)
            
            return VectorData(
                embeddings=embeddings_array,
                documents=documents,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error getting vectors with metadata: {str(e)}")
            raise

# Add constants for visualization
DEFAULT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
]

def get_default_colors(n_colors: int) -> List[str]:
    """Get default colors for visualization"""
    colors = DEFAULT_COLORS.copy()
    while len(colors) < n_colors:
        colors.extend(DEFAULT_COLORS)
    return colors[:n_colors]

async def visualize_vector_store(session: AsyncSession) -> Tuple[Optional[go.Figure], str]:
    """Generate visualization of the vector store"""
    try:
        data_acquisition = VectorStoreDataAcquisition(session)
        vector_data = await data_acquisition.get_vectors_with_metadata()
        
        if len(vector_data.documents) == 0:
            return None, "No vectors found in the database"
        
        # Get unique document types and assign colors
        doc_types = [
            Path(m['filename']).stem.split('_')[0] 
            for m in vector_data.metadata
        ]
        unique_types = sorted(list(set(doc_types)))
        color_map = dict(zip(unique_types, get_default_colors(len(unique_types))))
        
        # Assign colors to each point
        point_colors = [color_map[doc_type] for doc_type in doc_types]
        
        # Perform t-SNE
        tsne = TSNE(n_components=3, random_state=42)
        vectors_3d = tsne.fit_transform(vector_data.embeddings)
        
        # Create main scatter plot
        scatter = go.Scatter3d(
            x=vectors_3d[:, 0],
            y=vectors_3d[:, 1],
            z=vectors_3d[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=point_colors,
                opacity=0.8
            ),
            text=[
                f"Document: {m['filename']}<br>"
                f"Page: {m['page']}<br>"
                f"Content: {d[:100]}..."
                for m, d in zip(vector_data.metadata, vector_data.documents)
            ],
            hoverinfo='text',
            showlegend=False
        )
        
        # Create figure with main scatter plot
        fig = go.Figure(data=[scatter])
        
        # Add legend traces
        for doc_type, color in color_map.items():
            legend_trace = go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=doc_type,
                showlegend=True
            )
            fig.add_trace(legend_trace)
        
        # Update layout with better positioning
        fig.update_layout(
            title=dict(
                text='3D Vector Store Visualization',
                x=0.5,
                xanchor='center',
                y=0.95,
                yanchor='top',
                font=dict(size=20)
            ),
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                domain=dict(
                    x=[0.2, 1.0],  # Shift plot to the right
                    y=[0, 1.0]
                )
            ),
            width=1200,
            height=800,
            margin=dict(r=20, b=10, l=10, t=40),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            showlegend=True
        )
        
        return fig, "Visualization generated successfully"
        
    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")
        return None, f"Error generating visualization: {str(e)}"

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
                    fig = await visualize_vector_store(session)
                    
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