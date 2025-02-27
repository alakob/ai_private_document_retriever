"""
Vector visualization adapter for PostgreSQL vector store.
Provides functionality to visualize document embeddings in 3D space.
"""

# Standard library imports
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Third-party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine
from packaging import version
import sklearn
from dotenv import load_dotenv

# Local imports
from ..models.documents import DocumentModel, DocumentChunk
from ..config.database import PostgresConfig
from ..services.database import create_async_engine, async_sessionmaker
from ..utils.logging_config import setup_logger

# Load environment variables
load_dotenv()

# Get database credentials from environment variables
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', '')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'ragSystem')

# Setup logger
logger = setup_logger("vector_visualization")

def setup_database() -> Tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """Configure database connection using environment variables"""
    try:
        postgres_config = PostgresConfig(
            connection_string=f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
        )
        
        engine = create_async_engine(
            postgres_config.connection_string,
            pool_size=5,
            max_overflow=10,
            echo=False
        )
        
        session_maker = async_sessionmaker(
            engine,
            expire_on_commit=False,
            class_=AsyncSession
        )
        
        return engine, session_maker
    except Exception as e:
        error_msg = f"Failed to setup database connection: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise DatabaseConnectionError(error_msg) from e

# Add constants at the top level
DEFAULT_FIGURE_WIDTH = 900
DEFAULT_FIGURE_HEIGHT = 700
DEFAULT_MARKER_SIZE = 5
DEFAULT_MARKER_OPACITY = 0.8
DEFAULT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
]

# Add constants for vector operations
VECTOR_CHUNK_SIZE = 1000  # Process vectors in chunks if needed
MAX_TEXT_PREVIEW_LENGTH = 100  # Limit text preview length
EMBEDDING_DIMENSION_WARNING = 768  # Standard embedding dimension for many models

# Add performance-related constants
BATCH_SIZE = 1000  # Process vectors in batches
CACHE_TIMEOUT = 3600  # Cache timeout in seconds
MIN_VECTORS_FOR_BATCHING = 5000  # Minimum vectors to trigger batch processing

@dataclass
class VectorData:
    """Data class to hold vector information"""
    embeddings: List[List[float]]
    documents: List[str]
    metadata: List[Dict[str, Any]]
    
    def __post_init__(self):
        logger.info(f"Created VectorData with {len(self.embeddings)} vectors")

# Add utility functions for common operations
def calculate_elapsed_time(start_time: datetime) -> float:
    """Calculate elapsed time in seconds"""
    return (datetime.now() - start_time).total_seconds()

# Add utility function for text truncation
def truncate_text(text: str, max_length: int = MAX_TEXT_PREVIEW_LENGTH) -> str:
    """Truncate text to specified length with ellipsis"""
    return f"{text[:max_length]}..." if len(text) > max_length else text

# Update utility function to use constants
def create_error_figure(
    error_message: str, 
    width: int = DEFAULT_FIGURE_WIDTH, 
    height: int = DEFAULT_FIGURE_HEIGHT
) -> go.Figure:
    """Create a Plotly figure displaying an error message"""
    fig = go.Figure()
    fig.add_annotation(
        text=str(error_message).replace("\n", "<br>"),
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=20, color="red"),
        align='center'
    )
    fig.update_layout(
        title='Vector Store Visualization Error',
        width=width,
        height=height,
        showlegend=False
    )
    return fig

# Add custom exceptions
class VectorStoreError(Exception):
    """Base exception for vector store operations"""
    pass

class DataAcquisitionError(VectorStoreError):
    """Exception raised for errors during data acquisition"""
    pass

class VisualizationError(VectorStoreError):
    """Exception raised for errors during visualization"""
    pass

class InsufficientDataError(VectorStoreError):
    """Exception raised when there is not enough data for visualization"""
    pass

class DatabaseConnectionError(VectorStoreError):
    """Exception raised for database connection issues"""
    pass

class TSNEError(VisualizationError):
    """Exception raised for t-SNE specific errors"""
    pass

# Update the data acquisition class with better error handling
class VectorStoreDataAcquisition:
    """Handles data acquisition from PostgreSQL vector store"""
    
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self._last_count_check = 0
        self._cached_count = 0
        self._cache_timeout = CACHE_TIMEOUT
        logger.info("Initialized VectorStoreDataAcquisition")

    async def get_vector_count(self) -> int:
        """Get total count of vectors in store with manual caching"""
        current_time = datetime.now().timestamp()
        
        # Return cached value if still valid
        if (current_time - self._last_count_check < self._cache_timeout and 
            self._cached_count > 0):
            return self._cached_count

        try:
            async with self.session.begin():
                result = await self.session.execute(
                    select(func.count()).select_from(DocumentChunk)
                )
                count = result.scalar()
                if count is None:
                    raise DataAcquisitionError("Failed to get vector count from database")
                
                # Update cache
                self._cached_count = count
                self._last_count_check = current_time
                logger.info(f"Found {count} vectors in store")
                return count
        except Exception as e:
            error_msg = f"Error getting vector count: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise DataAcquisitionError(error_msg) from e

    def process_chunk_metadata(self, chunk: DocumentChunk, doc: DocumentModel) -> Dict[str, Any]:
        """Process chunk and document metadata into a standardized format"""
        return {
            'doc_type': truncate_text(
                doc.filename.replace('documents/', '').replace('.pdf', ''),
                max_length=20
            ),
            'filename': truncate_text(doc.filename, max_length=100),
            'chunk_index': chunk.chunk_index,
            'page_number': chunk.page_number,
            'embedding_dim': len(chunk.embedding),
            'content_length': len(chunk.content),
            **(chunk.chunk_metadata or {})
        }

    async def _process_batch(
        self,
        result,
        embeddings: List[List[float]],
        documents: List[str],
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Process a batch of database results."""
        try:
            for chunk, doc in result:
                # Add detailed validation logging
                if not isinstance(chunk.embedding, (list, np.ndarray)):
                    logger.warning(
                        f"Skipping chunk {chunk.chunk_index} from {doc.filename}: "
                        f"Invalid embedding type {type(chunk.embedding)}"
                    )
                    continue
                    
                if isinstance(chunk.embedding, np.ndarray) and chunk.embedding.size == 0:
                    logger.warning(
                        f"Skipping chunk {chunk.chunk_index} from {doc.filename}: "
                        "Empty embedding array"
                    )
                    continue
                    
                if not chunk.content:
                    logger.warning(
                        f"Skipping chunk {chunk.chunk_index} from {doc.filename}: "
                        "Missing content"
                    )
                    continue

                # Convert numpy array to list if necessary
                embedding = (
                    chunk.embedding.tolist() 
                    if isinstance(chunk.embedding, np.ndarray) 
                    else chunk.embedding
                )

                # Add validation for embedding dimensions
                if len(embedding) != 1536:  # OpenAI's embedding dimension
                    logger.warning(
                        f"Skipping chunk {chunk.chunk_index} from {doc.filename}: "
                        f"Unexpected embedding dimension {len(embedding)}"
                    )
                    continue

                embeddings.append(embedding)
                documents.append(chunk.content)
                metadata.append(self.process_chunk_metadata(chunk, doc))

        except Exception as e:
            error_msg = f"Error processing batch: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise DataAcquisitionError(error_msg) from e

    async def get_vectors_with_metadata(self) -> VectorData:
        """Get all vectors with their documents and metadata"""
        try:
            start_time = datetime.now()
            logger.info("Starting vector data acquisition")
            
            # Get total count first
            total_count = await self.get_vector_count()
            
            # Initialize containers
            embeddings = []
            documents = []
            metadata = []
            
            async with self.session.begin():
                if total_count >= MIN_VECTORS_FOR_BATCHING:
                    # Process in batches for large datasets
                    for offset in range(0, total_count, BATCH_SIZE):
                        batch_result = await self.session.execute(
                            select(DocumentChunk, DocumentModel)
                            .join(DocumentModel)
                            .order_by(DocumentChunk.id)
                            .offset(offset)
                            .limit(BATCH_SIZE)
                        )
                        await self._process_batch(batch_result, embeddings, documents, metadata)
                else:
                    # Process all at once for small datasets
                    result = await self.session.execute(
                        select(DocumentChunk, DocumentModel)
                        .join(DocumentModel)
                        .order_by(DocumentChunk.id)
                    )
                    await self._process_batch(result, embeddings, documents, metadata)
                
                if not embeddings:
                    raise DataAcquisitionError("No valid vectors found after processing")
                
                vector_data = VectorData(
                    embeddings=embeddings,
                    documents=documents,
                    metadata=metadata
                )
                
                logger.info(
                    f"Data acquisition completed in {calculate_elapsed_time(start_time):.2f} seconds. "
                    f"Processed {len(embeddings)} vectors with metadata"
                )
                return vector_data
                
        except Exception as e:
            error_msg = "Error acquiring vector data"
            logger.error(error_msg, exc_info=True)
            raise DataAcquisitionError(error_msg) from e

def get_tsne_params():
    """Get TSNE parameters based on sklearn version."""
    base_params = {
        'n_components': 3,
        'random_state': 42,
        'perplexity': 30,
        'learning_rate': 'auto',
        'init': 'random'
    }
    
    if version.parse(sklearn.__version__) >= version.parse('1.5.0'):
        base_params['max_iter'] = 1000
    else:
        base_params['n_iter'] = 1000
        
    return base_params

class VectorStoreVisualization:
    """Handles visualization of vector store data"""
    
    def __init__(self, vector_data: VectorData) -> None:
        self.vector_data = vector_data
        self.tsne = TSNE(**get_tsne_params())
        logger.info("Initialized VectorStoreVisualization")
        
    def validate_sample_size(self, n_samples: int) -> None:
        """Validate sample size for visualization"""
        if n_samples == 0:
            raise InsufficientDataError(
                "No vectors found in the database for visualization.\n"
                "Please add some documents to process."
            )
        elif n_samples < 3:
            raise InsufficientDataError(
                f"Found only {n_samples} vectors.\n"
                "At least 3 vectors are required for 3D t-SNE visualization.\n"
                "Please add more documents to the database."
            )
        elif n_samples < 10:
            logger.warning(
                f"Only {n_samples} vectors available. "
                "Visualization may not be meaningful with less than 10 samples."
            )

    def get_default_colors(self, n_colors: int) -> List[str]:
        """Get default colors for visualization"""
        colors = DEFAULT_COLORS.copy()
        while len(colors) < n_colors:
            colors.extend(DEFAULT_COLORS)
        return colors[:n_colors]

    def create_hover_text(self) -> List[str]:
        """Create hover text for data points"""
        return [
            f"Type: {m['doc_type']}<br>"
            f"File: {m['filename']}<br>"
            f"Page: {m['page_number']}<br>"
            f"Embedding Dim: {m.get('embedding_dim', 'unknown')}<br>"
            f"Content Length: {m.get('content_length', 'unknown')}<br>"
            f"Text: {truncate_text(d)}"
            for m, d in zip(self.vector_data.metadata, self.vector_data.documents)
        ]

    def create_scatter_trace(
        self, 
        vectors: np.ndarray, 
        colors: List[str], 
        hover_text: List[str]
    ) -> go.Scatter3d:
        """Create scatter trace for 3D visualization."""
        return go.Scatter3d(
            x=vectors[:, 0],
            y=vectors[:, 1],
            z=vectors[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=colors,
                opacity=0.8
            ),
            text=hover_text,
            hoverinfo='text',
            showlegend=False  # Hide this trace from legend
        )

    def create_legend_traces(self, color_map: Dict[str, str]) -> List[go.Scatter3d]:
        """Create legend traces for document types"""
        return [
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=doc_type,
                showlegend=True
            )
            for doc_type, color in color_map.items()
        ]

    def create_3d_visualization(
        self, 
        color_map: Optional[Dict[str, str]] = None,
        width: int = DEFAULT_FIGURE_WIDTH,
        height: int = DEFAULT_FIGURE_HEIGHT
    ) -> go.Figure:
        """Create 3D visualization of vectors"""
        start_time = datetime.now()
        logger.info("Starting 3D visualization creation")
        
        try:
            # Input validation
            if width <= 0 or height <= 0:
                raise ValueError("Width and height must be positive values")
            
            vectors = np.array(self.vector_data.embeddings)
            if vectors.size == 0:
                raise InsufficientDataError("Empty vector array provided")
                
            n_samples = vectors.shape[0]
            self.validate_sample_size(n_samples)

            # Perform t-SNE reduction with error handling
            try:
                reduced_vectors = self.tsne.fit_transform(vectors)
            except Exception as e:
                logger.error("t-SNE reduction failed", exc_info=True)
                raise VisualizationError(f"Dimensionality reduction failed: {str(e)}")

            # Process colors with validation
            doc_types = [m.get('doc_type', 'unknown') for m in self.vector_data.metadata]
            if not doc_types:
                raise VisualizationError("No document types found in metadata")
                
            unique_types = sorted(list(set(doc_types)))
            if color_map is not None and not all(t in color_map for t in unique_types):
                logger.warning("Some document types missing from provided color map")
            
            color_map = color_map or dict(zip(unique_types, self.get_default_colors(len(unique_types))))
            
            point_colors = self._assign_colors(doc_types, color_map)
            hover_text = self.create_hover_text()
            
            # Create visualization with error handling
            try:
                fig = go.Figure(data=[self.create_scatter_trace(reduced_vectors, point_colors, hover_text)])
                
                for trace in self.create_legend_traces(color_map):
                    fig.add_trace(trace)
                
                self._update_figure_layout(fig, width, height)
            except Exception as e:
                raise VisualizationError(f"Failed to create plot: {str(e)}") from e
            
            processing_time = calculate_elapsed_time(start_time)
            logger.info(f"Visualization created in {processing_time:.2f} seconds")
            
            return fig
            
        except (InsufficientDataError, VisualizationError, ValueError) as e:
            # Known errors are logged and re-raised
            logger.error(str(e), exc_info=True)
            raise
        except Exception as e:
            # Unknown errors are wrapped
            error_msg = f"Unexpected error creating visualization: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise VisualizationError(error_msg) from e

    def _assign_colors(self, doc_types: List[str], color_map: Dict[str, str]) -> List[str]:
        """Assign colors to data points based on document types"""
        return [
            color_map.get(doc_type, '#808080')  # Default to gray for unknown types
            for doc_type in doc_types
        ]

    def _update_figure_layout(self, fig: go.Figure, width: int, height: int) -> None:
        """Update the figure layout with adjusted positioning."""
        fig.update_layout(
            title=dict(
                text='3D PostgreSQL Vector Store Visualization',
                x=0.5,  # Center the title horizontally
                xanchor='center',
                y=0.95,  # Adjust vertical position
                yanchor='top',
                font=dict(
                    size=20  # Optional: make title more prominent
                )
            ),
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z',
                domain=dict(
                    x=[0.2, 1.0],  # Shift plot to the right
                    y=[0, 1.0]
                )
            ),
            width=width,
            height=height,
            margin=dict(r=20, b=10, l=10, t=40),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            autosize=True,
            showlegend=True
        )

# Main visualization function with better error handling
async def visualize_vector_store(session: AsyncSession) -> go.Figure:
    """Main function to create visualization"""
    try:
        start_time = datetime.now()
        logger.info("Starting vector store visualization process")
        
        if session is None:
            raise ValueError("Database session is required")
            
        data_acquisition = VectorStoreDataAcquisition(session)
        
        try:
            vector_count = await data_acquisition.get_vector_count()
        except DataAcquisitionError as e:
            return create_error_figure(str(e))
            
        if vector_count == 0:
            return create_error_figure(
                "No vectors found in the database.\n"
                "Please add some documents first."
            )
            
        try:
            vector_data = await data_acquisition.get_vectors_with_metadata()
        except DataAcquisitionError as e:
            return create_error_figure(str(e))
            
        try:
            visualizer = VectorStoreVisualization(vector_data)
            fig = visualizer.create_3d_visualization()
        except (InsufficientDataError, VisualizationError) as e:
            return create_error_figure(str(e))
        
        logger.info(f"Visualization process completed in {calculate_elapsed_time(start_time):.2f} seconds")
        return fig
        
    except Exception as e:
        logger.error("Unexpected error in visualization process", exc_info=True)
        return create_error_figure(f"An unexpected error occurred: {str(e)}") 