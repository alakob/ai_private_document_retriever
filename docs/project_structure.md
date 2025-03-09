# AI Private Document Retriever - Project Structure

This document outlines the organized structure of the AI Private Document Retriever project.

## Directory Structure

```
ai_private_document_retriever/
│
├── app/                        # Main application code
│   ├── __init__.py             # Package initialization
│   ├── config.py               # Configuration settings
│   ├── models.py               # Database models
│   │
│   ├── core/                   # Core functionality
│   │   ├── __init__.py         # Package initialization
│   │   └── document_rag_loader.py  # Document processing
│   │
│   ├── services/               # Service layer
│   │   ├── __init__.py         # Package initialization
│   │   └── vector_similarity_search.py  # Vector search
│   │
│   └── ui/                     # User interface components
│       ├── __init__.py         # Package initialization
│       └── chat_interface.py   # Chat interface
│
├── utils/                      # Utility modules
│   ├── __init__.py             # Package initialization
│   └── vector_store_visualization.py  # Visualization
│
├── docs/                       # Documentation
│   ├── project_architecture.md # Architecture documentation
│   ├── project_structure.md    # Structure documentation
│   └── diagram_renderer.html   # Diagram rendering
│
├── documents/                  # Document storage
│
├── .env                        # Environment variables
├── .env.local                  # Local environment variables
├── .gitignore                  # Git ignore rules
├── main.py                     # Application entry point
└── requirements.txt            # Project dependencies
```

## Module Descriptions

### app/config.py
Contains configuration settings for various components of the application, including:
- Database connections
- Embedding settings
- Vector search parameters
- Chat settings
- Rate limiting

### app/models.py
Defines the database models using SQLAlchemy:
- DocumentModel: Represents document metadata
- DocumentChunk: Represents chunks of document content with vector embeddings

### app/core/document_rag_loader.py
Handles document processing:
- Loading documents from various file formats
- Chunking text content
- Generating embeddings
- Storing documents and embeddings in the database

### app/services/vector_similarity_search.py
Provides vector similarity search capabilities:
- Converts queries to vector embeddings
- Performs similarity search against stored document chunks
- Returns relevant documents with similarity scores

### app/ui/chat_interface.py
Implements the chat interface using Gradio:
- Question-answering using the RAG system
- Document visualization
- User interaction

### utils/vector_store_visualization.py
Provides utilities for visualizing vector data:
- T-SNE visualization of document embeddings
- Data acquisition from the vector store
- Interactive plots

### main.py
The application entry point with command-line arguments:
- `process`: Process documents and store in vector database
- `chat`: Start the chat interface

## Running the Application

### Process Documents
```bash
python main.py process --dir documents
```

### Start Chat Interface
```bash
python main.py chat
```

### Reset Database and Process Documents
```bash
python main.py process --reset --dir documents
```
