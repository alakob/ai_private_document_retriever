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
│   │   └── document_rag_loader.py  # Document processing logic
│   │
│   ├── services/               # Service layer
│   │   ├── __init__.py         # Package initialization
│   │   └── vector_similarity_search.py  # Vector search logic
│   │
│   ├── ui/                     # User interface components
│   │   ├── __init__.py         # Package initialization
│   │   └── chat_interface.py   # Gradio chat interface
│   │
│   └── routers/                # FastAPI API endpoint definitions
│       ├── __init__.py         # Package initialization
│       └── api_router.py       # Main API router aggregating sub-routers
│
├── utils/                      # Utility modules
│   ├── __init__.py             # Package initialization
│   └── vector_store_visualization.py  # Vector visualization functions
│
├── docs/                       # Project documentation (.md files)
│   ├── architecture_overview.md
│   ├── docker_deployment.md
│   ├── chat_flow.md
│   ├── document_processing_flow.md
│   ├── docker_commands.md
│   ├── api_reference.md
│   └── ...                     # Other documentation files
│
├── documents/                  # Default storage location for input documents
│
├── pgadmin/                    # Configuration/scripts for pgAdmin container
│
├── scripts/                    # Utility or development scripts
│   └── PRD.txt                 # Product Requirements Document
│
├── visualizations/             # Default output location for visualizations
│
├── docker-entrypoint-initdb.d/ # Scripts run during PostgreSQL container initialization
│
├── .env.docker                 # Environment variables specifically for Docker Compose
├── .env.example                # Example environment variables
├── .gitignore                  # Git ignore rules
├── Dockerfile                  # Instructions to build the application Docker image
├── docker-bake.hcl             # Docker Buildx bake file definition
├── docker-compose.yml          # Docker Compose configuration for services
├── docker-entrypoint.sh        # Entrypoint script for the application container
├── main.py                     # Main application entry point (CLI)
├── README.md                   # Project overview and quickstart
└── requirements.txt            # Python dependencies
```

## Module Descriptions

### app/config.py
Contains configuration settings loaded from environment variables for various components (database, embedding, chat, processing, etc.) using dataclasses.

### app/models.py
Defines the database models (`DocumentModel`, `DocumentChunk`) using SQLAlchemy, including table structure, relationships, and constraints.

### app/core/document_rag_loader.py
Handles the core document processing pipeline: loading files (using various loaders like Docling, Mistral OCR), chunking, calculating checksums, generating embeddings via OpenAI, and storing data in the PostgreSQL database.

### app/services/vector_similarity_search.py
Provides vector similarity search capabilities against the PostgreSQL/pgvector database, including query embedding and result retrieval logic. Also includes CLI interface for search.

### app/ui/chat_interface.py
Implements the main interactive chat interface using Gradio, orchestrating calls to the retriever, LLM, and visualization components.

### app/routers/
Contains FastAPI routers defining the RESTful API endpoints. `api_router.py` likely aggregates sub-routers for different functionalities (e.g., documents, search, chat).

### utils/vector_store_visualization.py
Provides utility functions for generating interactive 2D visualizations (using t-SNE and Plotly) of the document chunk embeddings stored in the vector store.

### main.py
The main application entry point, providing a Command Line Interface (CLI) using `argparse` to run different parts of the system:
- `process`: Process documents in a specified directory and store them.
- `chat`: Start the interactive Gradio chat interface.
- `api`: Start the FastAPI backend server.
- `search`: Run interactive or direct vector similarity searches.
- `visualize`: Generate the vector store visualization HTML file.

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
