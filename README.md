# AI Private Document Retriever

A system for private document retrieval using vector similarity search and Large Language Models.

## Project Overview

This project provides a complete RAG (Retrieval-Augmented Generation) system for processing, indexing, and querying private documents. It uses modern vector embedding techniques and integrates with powerful language models to provide accurate, contextual responses based on your document corpus.

## Project Structure

The project has been organized into a structured layout:

```
ai_private_document_retriever/
│
├── app/                        # Main application code
│   ├── config.py               # Configuration settings
│   ├── models.py               # Database models
│   │
│   ├── core/                   # Core functionality
│   │   └── document_rag_loader.py  # Document processing
│   │
│   ├── services/               # Service layer
│   │   └── vector_similarity_search.py  # Vector similarity search
│   │
│   └── ui/                     # User interface components
│       └── chat_interface.py   # Chat interface
│
├── utils/                      # Utility modules
│   └── vector_store_visualization.py  # Vector visualization
│
├── docs/                       # Documentation
│   ├── project_architecture.md # Architecture documentation
│   ├── project_structure.md    # Structure documentation
│   └── diagram_renderer.html   # Diagram rendering
│
├── documents/                  # Document storage
│
├── main.py                     # Application entry point with CLI commands:
│                               # - process: Process documents
│                               # - chat: Start chat interface
│                               # - search: Run vector similarity search
│                               # - visualize: Generate vector visualization
│
└── requirements.txt            # Project dependencies
```

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure your environment:
   - Create a `.env` file with the necessary API keys and configuration
   - Sample `.env` file:
     ```
     OPENAI_API_KEY=your_openai_api_key
     POSTGRES_USER=postgres
     POSTGRES_PASSWORD=yourpassword
     POSTGRES_HOST=localhost
     POSTGRES_PORT=5432
     POSTGRES_DB=ragSystem
     ```

3. Available commands:

   ```
   # Process documents
   python main.py process --dir documents

   # Process documents with Docling (enhanced document conversion)
   python main.py process --dir documents --use-docling

   # Process documents with Mistral OCR API (for better text extraction)
   python main.py process --dir documents --use-mistral

   # Process documents and reset the database (warning: deletes existing data)
   python main.py process --dir documents --reset-db

   # Start chat interface
   python main.py chat

   # Interactive vector search
   python main.py search

   # Direct vector search
   python main.py search --query "your search query"

   # Display cache information
   python main.py search --cache-info

   # Clear embedding cache
   python main.py search --clear-cache

   # Generate visualization
   python main.py visualize
   ```

## Features

- **Document Processing**: Load, chunk, and embed documents from various formats (PDF, DOCX, TXT, etc.)
- **Enhanced Document Conversion**: Optional integration with Docling for improved document structure preservation
- **OCR Capabilities**: Integration with Mistral OCR API for superior text extraction from documents and images
- **Vector Storage**: Store document embeddings in PostgreSQL with pgvector extension
- **Similarity Search**: Find the most relevant document chunks for any query
- **Chat Interface**: Interactive UI for asking questions about your documents
- **Visualization**: Visualize document embeddings in 2D space
- **Embedding Caching**: Dual caching system (file-based and Redis) for embeddings to reduce API calls and improve performance

## Dependencies

- Python 3.8+
- PostgreSQL with pgvector extension
- OpenAI API key
- Mistral API key (optional, for OCR capabilities)
- Redis (optional, for high-performance caching)
- Various Python packages (see requirements.txt)

## Documentation

For more information, see the documentation in the `docs/` directory:
- `project_architecture.md`: Overview of the system architecture
- `project_structure.md`: Details of the project structure and organization
- `docling_integration.md`: Guide to using Docling for enhanced document conversion

## Environment Configuration

The system uses environment variables for configuration, which can be set in a `.env` file:

- `OPENAI_API_KEY`: Required for embedding generation and LLM responses
- `MODEL`: LLM model to use (default: gpt-4o-mini)
- `CHUNK_SIZE`: Size of document chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- Database configuration (`POSTGRES_*` variables)
- Redis caching configuration:
  - `USE_REDIS_CACHE`: Enable Redis caching (default: false, uses file-based caching)
  - `REDIS_HOST`: Redis server hostname (default: localhost)
  - `REDIS_PORT`: Redis server port (default: 6379)
  - `REDIS_DB`: Redis database number (default: 0)
  - `REDIS_PASSWORD`: Redis authentication password (if required)
- Docling configuration (for enhanced document conversion):
  - `USE_DOCLING`: Enable Docling document conversion (default: false)
  - `DOCLING_ARTIFACTS_PATH`: Path to pre-downloaded Docling models (optional)
  - `DOCLING_ENABLE_REMOTE`: Allow Docling to use remote services (default: false)
  - `DOCLING_USE_CACHE`: Use caching for Docling conversions (default: true)
