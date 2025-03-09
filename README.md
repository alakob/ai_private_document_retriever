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
     MISTRAL_API_KEY=your_mistral_api_key
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

   # Start chat interface with a shareable link
   python main.py chat --share

   # Interactive vector search
   python main.py search

   # Direct vector search
   python main.py search --query "your search query"

   # Direct vector search with custom parameters
   python main.py search --query "your search query" --top-k 10 --threshold 0.6

   # Display cache information
   python main.py search --cache-info

   # Clear embedding cache
   python main.py search --clear-cache

   # Generate visualization
   python main.py visualize

   # Generate visualization with custom parameters
   python main.py visualize --output custom_visualization.html --perplexity 50
   ```

## Features

- **Document Processing**: Load, chunk, and embed documents from various formats (PDF, DOCX, TXT, etc.)
- **Enhanced Document Conversion**: Optional integration with Docling for improved document structure preservation
- **OCR Capabilities**: Integration with Mistral OCR API for superior text extraction from documents and images
- **Vector Storage**: Store document embeddings in PostgreSQL with pgvector extension
- **Similarity Search**: Find the most relevant document chunks for any query
- **Chat Interface**: Interactive UI for asking questions about your documents
- **Conceptual Diagrams**: Generate Mermaid diagrams that visually represent the key concepts and relationships in your chat history
- **Visualization**: Visualize document embeddings in 2D space
- **Embedding Caching**: Dual caching system (file-based and Redis) for embeddings to reduce API calls and improve performance
- **Security Features**: Built-in circuit breaker pattern to prevent system overload and rate limiting for diagram generation
- **Asynchronous Processing**: Fully asynchronous document processing pipeline for improved performance
- **Duplicate Document Detection**: Checksums to prevent duplicate document processing
- **Adaptive Batching**: Dynamic batch size adjustments based on system performance
- **Resource Monitoring**: Runtime memory and CPU monitoring to prevent system overload

## Architecture Highlights

- **Modular Structure**: Separation of concerns with distinct components for document processing, vector search, and user interface
- **Resilient Design**: Error handling, retry mechanisms, and circuit breakers to ensure system stability
- **Efficient Document Processing**:
  - Concurrent document processing with thread and process pools
  - Configurable chunking strategies for different document types
  - Automatic checksum calculation to prevent duplicate processing
  - Progress tracking and detailed statistics
- **Advanced Vector Search**:
  - PostgreSQL with pgvector extension for scalable similarity search
  - Multi-level caching system (Redis and file-based) for embeddings
  - Configurable search parameters (top-k, threshold, etc.)
  - Rich result formatting and display

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

## Conceptual Diagrams

The system includes a powerful feature to generate conceptual diagrams from your chat history:

1. **Automatic Analysis**: The system analyzes your chat history to extract key concepts and relationships
2. **Mermaid Diagram Generation**: Visualizes these concepts using Mermaid.js for clear, interactive diagrams
3. **Integration in UI**: Access diagrams directly in the chat interface by clicking the "Generate Conceptual Diagram" button
4. **Multiple Diagram Types**: Supports various diagram formats including flowcharts, mindmaps, and entity-relationship diagrams
5. **Styled Visualization**: Diagrams use color coding and styling to differentiate between different types of concepts
6. **Rate-Limited Generation**: Built-in rate limiting to prevent abuse and ensure system stability

## Environment Configuration

The system uses environment variables for configuration, which can be set in a `.env` file:

### Core Configuration
- `OPENAI_API_KEY`: Required for embedding generation and LLM responses
- `MODEL`: LLM model to use (default: gpt-4o-mini)

### Document Processing
- `CHUNK_SIZE`: Size of document chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `KNOWLEDGE_BASE_DIR`: Directory containing documents (default: documents)

### Database Configuration
- `POSTGRES_HOST`: Database hostname (default: localhost)
- `POSTGRES_PORT`: Database port (default: 5432)
- `POSTGRES_USER`: Database username (default: postgres)
- `POSTGRES_PASSWORD`: Database password
- `POSTGRES_DB`: Database name (default: ragSystem)

### Caching Configuration
- `USE_REDIS_CACHE`: Enable Redis caching (default: false, uses file-based caching)
- `REDIS_HOST`: Redis server hostname (default: localhost)
- `REDIS_PORT`: Redis server port (default: 6379)
- `REDIS_DB`: Redis database number (default: 0)
- `REDIS_PASSWORD`: Redis authentication password (if required)

### Chat Interface Configuration
- `SERVER_NAME`: Server hostname for chat interface (default: 127.0.0.1)
- `TEMPERATURE`: Temperature for LLM responses (default: 0.7)
- `MAX_TOKENS`: Maximum tokens in LLM responses (default: 1000)

### Enhanced Document Processing
- `USE_DOCLING`: Enable Docling document conversion (default: false)
- `DOCLING_ARTIFACTS_PATH`: Path to pre-downloaded Docling models (optional)
- `DOCLING_ENABLE_REMOTE`: Allow Docling to use remote services (default: false)
- `DOCLING_USE_CACHE`: Use caching for Docling conversions (default: true)

### OCR Configuration
- `MISTRAL_API_KEY`: Required for OCR capabilities when using Mistral API
- `USE_MISTRAL`: Enable Mistral OCR API (default: false)
- `MISTRAL_OCR_MODEL`: Mistral OCR model to use (default: mistral-ocr-latest)
- `MISTRAL_INCLUDE_IMAGES`: Include extracted images in OCR results (default: false)

## Advanced Features

### Performance Optimization
- **Rate Limiting**: Configurable rate limiting for API calls to prevent quota exhaustion
- **Exponential Backoff**: Automatic retry with increasing delays for API failures
- **Connection Pooling**: Database connection pooling for efficient resource utilization
- **Adaptive Batching**: Dynamic batch size adjustments based on processing times

### Security Features
- **Circuit Breaker Pattern**: Prevents cascading failures by temporarily disabling failing components
- **Input Validation**: Thorough validation of all user inputs to prevent security issues
- **Rate Limiting**: Protection against abuse with configurable rate limits
- **Error Isolation**: Contained error handling to prevent system-wide failures

### Document Processing
- **Multi-Format Support**: Processing for PDF, DOCX, TXT, MD, CSV, JSON, HTML, XML, and more
- **Structured Extraction**: Preservation of document structure with specialized loaders
- **Duplicate Detection**: SHA-256 checksums to prevent duplicate document processing
- **Progress Tracking**: Detailed statistics and progress monitoring during processing
