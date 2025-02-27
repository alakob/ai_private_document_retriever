"""
Vector store visualization component.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...models.documents import DocumentModel, DocumentChunk
from ...utils.text import truncate_text

logger = logging.getLogger(__name__)

class VectorStoreVisualizer:
    """Handles visualization of vector store data."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.tsne = TSNE(
            n_components=3,
            random_state=42,
            perplexity=30,
            learning_rate='auto',
            init='random'
        )
    
    async def get_vector_data(self) -> Dict[str, Any]:
        """Get vector data from database."""
        try:
            async with self.session.begin():
                result = await self.session.execute(
                    select(DocumentChunk, DocumentModel)
                    .join(DocumentModel)
                    .order_by(DocumentChunk.id)
                )
                
                embeddings = []
                documents = []
                metadata = []
                
                for chunk, doc in result:
                    if not isinstance(chunk.embedding, (list, np.ndarray)):
                        continue
                        
                    embeddings.append(
                        chunk.embedding.tolist() 
                        if isinstance(chunk.embedding, np.ndarray) 
                        else chunk.embedding
                    )
                    documents.append(chunk.content)
                    metadata.append({
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
                    })
                
                return {
                    'embeddings': embeddings,
                    'documents': documents,
                    'metadata': metadata
                }
                
        except Exception as e:
            logger.error(f"Failed to get vector data: {str(e)}")
            raise

    def create_3d_visualization(
        self,
        vector_data: Dict[str, Any],
        width: int = 900,
        height: int = 700
    ) -> go.Figure:
        """Create 3D visualization of vectors."""
        try:
            vectors = np.array(vector_data['embeddings'])
            if vectors.size == 0:
                raise ValueError("No vectors found for visualization")
                
            # Perform t-SNE reduction
            reduced_vectors = self.tsne.fit_transform(vectors)
            
            # Process document types and colors
            doc_types = [m.get('doc_type', 'unknown') for m in vector_data['metadata']]
            unique_types = sorted(list(set(doc_types)))
            colors = self._get_default_colors(len(unique_types))
            color_map = dict(zip(unique_types, colors))
            
            # Create hover text
            hover_text = [
                f"Type: {m['doc_type']}<br>"
                f"File: {m['filename']}<br>"
                f"Page: {m['page_number']}<br>"
                f"Text: {truncate_text(d)}"
                for m, d in zip(vector_data['metadata'], vector_data['documents'])
            ]
            
            # Create figure
            fig = go.Figure()
            
            # Add scatter trace
            fig.add_trace(go.Scatter3d(
                x=reduced_vectors[:, 0],
                y=reduced_vectors[:, 1],
                z=reduced_vectors[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=[color_map[t] for t in doc_types],
                    opacity=0.8
                ),
                text=hover_text,
                hoverinfo='text',
                showlegend=False
            ))
            
            # Add legend traces
            for doc_type, color in color_map.items():
                fig.add_trace(go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    name=doc_type,
                    showlegend=True
                ))
            
            # Update layout
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
                    xaxis_title='x',
                    yaxis_title='y',
                    zaxis_title='z',
                    domain=dict(x=[0.2, 1.0], y=[0, 1.0])
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
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create visualization: {str(e)}")
            raise

    def _get_default_colors(self, n_colors: int) -> List[str]:
        """Get default colors for visualization."""
        default_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
        ]
        colors = default_colors.copy()
        while len(colors) < n_colors:
            colors.extend(default_colors)
        return colors[:n_colors] 