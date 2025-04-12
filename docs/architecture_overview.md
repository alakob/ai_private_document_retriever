# High-Level Component Architecture

This diagram provides an overview of the major components of the DocuSeek AI system and how they interact.

```mermaid
graph TD
    subgraph User Interaction
        User --\> Browser
    end

    subgraph Application Container [rag-app Container]
        Browser -- HTTPS --\> GradioUI[Gradio Web UI /app/ui/chat_interface.py]
        Browser -- HTTPS --\> FastAPI[FastAPI Backend /main.py, /app/routers]
        
        GradioUI -- Calls methods --> ChatLogic[Chat Logic /app/ui/chat_interface.py]
        FastAPI -- Calls methods --> APILogic[API Logic /app/routers/*]

        ChatLogic -- Uses --> Retriever[Vector Retriever /app/ui/chat_interface.py::PostgresVectorRetriever]
        APILogic -- Uses --> Retriever
        ChatLogic -- Uses --> LLM[LLM Integration /app/ui/chat_interface.py]
        APILogic -- Uses --> DocProcessor[Document Processor /app/core/document_rag_loader.py]
        
        Retriever -- Embeds Query --> OpenAIEmbed[OpenAI Embedding API]
        Retriever -- Similarity Search --> PostgresDB[(Postgres DB w/ pgvector)]
        LLM -- Generates Response --> OpenAI_LLM[OpenAI Chat API]
        
        DocProcessor -- Loads/Splits/Embeds --> OpenAIEmbed
        DocProcessor -- Checks/Stores --> PostgresDB
        DocProcessor -- Uses --> FileLoaders[File Loaders LangChain/Docling/Mistral]

        FileLoaders -- Optional OCR --> MistralOCR[Mistral OCR API]

        subgraph Caching [Optional Caching Layer]
           Retriever -- Optional Cache --> Redis[(Redis Cache)]
        end
    end

    subgraph Database Container [rag-postgres Container]
        PostgresDB -- Stores/Retrieves Data --> Volume[(postgres_data Volume)]
    end

    subgraph External Services
        OpenAIEmbed
        OpenAI_LLM
        MistralOCR
    end

    User -- Admin Tasks --> pgAdminUI[pgAdmin Web UI]
    pgAdminUI -- DB Admin --> PostgresDB
```

**Explanation:**

*   **User Interaction:** The user interacts with the system primarily via a web browser.
*   **Application Container (`rag-app`):** This container hosts the main Python application.
    *   **GradioUI:** Provides the interactive chat interface. It handles user input and displays responses.
    *   **FastAPI Backend:** Provides RESTful API endpoints for programmatic interaction (document upload, search, etc.).
    *   **ChatLogic:** Contains the core logic for handling chat requests, orchestrating retrieval, LLM calls, and history management (currently largely within `chat_interface.py`).
    *   **APILogic:** Contains the logic for handling API requests (within `app/routers`).
    *   **Document Processor (`DocProcessor`):** Handles the ingestion pipeline: loading files, chunking, calculating checksums, embedding, and storing in the database. Triggered via CLI or API.
    *   **Vector Retriever:** Performs similarity searches against the vector database based on user queries. Embeds the query using the Embedding API.
    *   **LLM Integration:** Formats prompts (context + query + history) and interacts with the OpenAI Chat API to generate responses.
    *   **File Loaders:** Uses different strategies (standard LangChain, optional Docling, optional Mistral OCR) to load document content.
    *   **Caching:** An optional Redis layer can cache embeddings.
*   **Database Container (`rag-postgres`):** Runs the PostgreSQL database with the pgvector extension. Stores document metadata, chunks, and vector embeddings.
*   **External Services:** The system relies on external APIs for core functionality: OpenAI for embeddings and chat generation, and optionally Mistral for OCR.
*   **pgAdmin:** A separate container provides a web UI for direct database administration.
*   **Logging/Monitoring:** Not explicitly shown, but Python's `logging` is used. Structured logging and centralized collection are recommended improvements.
*   **Security:** API Keys are managed via `.env.docker`. The app runs as a non-root user. CORS policy needs tightening for production. API endpoint security is basic and needs review/implementation if exposed. 