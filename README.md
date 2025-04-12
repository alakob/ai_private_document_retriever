# DocuSeek AI

A system for private document retrieval using vector similarity search and Large Language Models.

## Project Overview

DocuSeek AI provides a complete RAG (Retrieval-Augmented Generation) system for processing, indexing, and querying private documents. It leverages vector embeddings and integrates with LLMs to deliver contextual responses based on your document corpus.

## Quick Start

### Prerequisites

- Python 3.8+
- Docker & Docker Compose (Recommended)
- `docker buildx` plugin enabled (usually included with recent Docker Desktop versions)
- PostgreSQL with pgvector extension (provided by Docker setup)
- OpenAI API Key
- (Optional) Mistral API Key (for OCR)
- (Optional) Redis (for caching, not configured by default)

See [Dependencies](docs/dependencies.md) for full details.

### Using Docker (Recommended)

This setup uses `docker buildx bake` for efficient image building and `docker compose` for service orchestration.

1.  **Configure Environment**: Create a file named `.env.docker` in the project root. Copy the contents from the example in [Environment Configuration](docs/environment_configuration.md) and populate it with your API keys and desired settings (e.g., database password).

2.  **Build Application Image**: Use `docker buildx bake` to build the application image using `docker-bake.hcl`. This step ensures the image is loaded into the local Docker daemon.
    ```bash
    docker buildx bake
    ```
    *(Note: If this is the first time running `bake` or after changes, and compose later fails with image not found, ensure your buildx builder is configured correctly or try `docker buildx bake --load`)*

3.  **Start Services**: Use `docker compose` to start the application (Chat UI by default), PostgreSQL, and pgAdmin. It reads configuration from `.env.docker`.
    ```bash
    docker compose --env-file .env.docker up -d
    ```
    *Troubleshooting Compose Warnings: If `docker compose up` warns about missing `OPENAI_API_KEY` or other variables despite them being in `.env.docker`, check if those variables are set (even if empty) in your shell environment (`echo $VAR_NAME`). Shell variables take precedence; `unset VAR_NAME` in your shell before running `compose` if needed.* 

4.  **Access Services**:
    *   **Chat Interface**: `http://localhost:7861`
    *   **pgAdmin**: `http://localhost:8080` (Login with details from `.env.docker`)

5.  **Process Your Documents**: Once the services are running, place your documents in the `documents/` directory and run the processing command:
    ```bash
    docker compose exec app python main.py process --dir documents
    ```
    *(Note: The first time you run this after resetting the database, it will automatically create the necessary database tables.)*

6.  **Use the Chat**: Interact with your documents via the Chat Interface at `http://localhost:7861`.

See the full [Docker Command Reference](docs/docker_commands.md) for other commands (API server, search, visualization, etc.).

### Manual Setup

*(Note: Docker is strongly recommended for managing dependencies like PostgreSQL)*

1.  **Install Dependencies**: `pip install -r requirements.txt`
2.  **Setup Database**: Manually set up PostgreSQL with the pgvector extension.
3.  **Configure Environment**: Create a `.env` file. See [Environment Configuration](docs/environment_configuration.md). Ensure database connection details match your manual setup.
4.  **Create Database Tables**: Manually run database schema creation logic (e.g., using Alembic if integrated, or a custom script calling `Base.metadata.create_all`).
5.  **Available Commands**:
    ```bash
    # Process documents (replace 'documents' with your directory)
    python main.py process --dir documents

    # Start chat interface (Default: http://127.0.0.1:7860)
    python main.py chat

    # Run vector similarity search (interactive)
    python main.py search

    # Start the API server (Default: http://0.0.0.0:8000)
    python main.py api
    ```
    For more command options, run `python main.py --help`.

## Key Features

- Document Processing (PDF, DOCX, TXT, etc.)
- Vector Storage & Similarity Search (PostgreSQL/pgvector)
- Interactive Chat Interface (Gradio)
- FastAPI Backend & API Endpoints
- Optional Enhanced Conversion (Docling) & OCR (Mistral)
- Caching (File-based/Redis)
- Duplicate Document Detection (Checksums)
- Asynchronous Processing Pipeline
- Vector Visualization
- Conceptual Diagram Generation from Chat

For a full list, see [Features](docs/features.md).

## Documentation

- **Architecture & Flow Diagrams:**
    - [High-Level Architecture Overview](docs/architecture_overview.md)
    - [Docker Deployment View](docs/docker_deployment.md)
    - [Chat Interaction Flow](docs/chat_flow.md)
    - [Document Processing Flow](docs/document_processing_flow.md)
- **Core Documentation:**
    - [Project Structure](docs/project_structure.md)
    - [Docker Command Reference](docs/docker_commands.md)
    - [API Reference](docs/api_reference.md)
    - [Features](docs/features.md)
    - [Conceptual Diagrams Feature](docs/conceptual_diagrams.md)
    - [Environment Configuration](docs/environment_configuration.md)
    - [Dependencies](docs/dependencies.md)
    - [Advanced Features](docs/advanced_features.md)
- **Integrations:**
    - [Docling Integration](docs/docling_integration.md) (If available)
