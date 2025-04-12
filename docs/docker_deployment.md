# Docker Deployment View

This diagram illustrates how the different services of the DocuSeek AI system are containerized and connected using Docker Compose.

```mermaid
graph TD
    subgraph Host Machine
        UserBrowser[User's Browser]
        UserCLI[User's Terminal]
    end

    subgraph Docker Network [rag-network]
        subgraph AppService [rag-app Container]
            direction LR
            AppCode[Python App: main.py / app/*]
            Gradio[Gradio Server (Port 7860)]
            FastAPI[FastAPI Server (Port 8000)]
            AppCode -- Runs --> Gradio
            AppCode -- Runs --> FastAPI
        end

        subgraph DBService [rag-postgres Container]
            direction LR
            Postgres[PostgreSQL Server (Port 5432)] -- Has --> PGVector[pgvector Extension]
            Postgres -- Reads/Writes --> Volume1[(rag-postgres-data Volume)]
        end

        subgraph AdminService [rag-pgadmin Container]
            direction LR
            PGAdmin[pgAdmin UI (Port 80)] -- Reads/Writes --> Volume2[(rag-pgadmin-data Volume)]
        end

        AppService -- TCP --> DBService
        AdminService -- TCP --> DBService
    end

    UserBrowser --> AppService
    UserBrowser --> AppService
    UserBrowser --> AdminService
    UserCLI --> AppService

    AppService --> External_OpenAI[External OpenAI API]
    AppService --> External_Mistral[External Mistral API (Optional)]

    %% Styling
    classDef container fill:#f9f,stroke:#333,stroke-width:2px;
    class AppService,DBService,AdminService container
```

**Explanation:**

1.  **Host Machine:** Represents the user's computer where Docker is running.
2.  **Docker Network (`rag-network`):** A virtual bridge network created by Docker Compose to allow containers to communicate with each other using their service names (e.g., `app` can reach `postgres` at host `postgres`).
3.  **`rag-app` Container:**
    *   Runs the main Python application code.
    *   By default (`CMD ["chat"]`), starts the Gradio server internally on port `7860`.
    *   If started with the `api` command, runs the FastAPI server internally on port `8000`.
    *   Connects to the `rag-postgres` container for database operations.
    *   Makes outbound API calls to external services (OpenAI, Mistral).
4.  **`rag-postgres` Container:**
    *   Runs the PostgreSQL database server with the pgvector extension enabled.
    *   Listens internally on port `5432`.
    *   Persists data to the named Docker volume `rag-postgres-data`.
5.  **`rag-pgadmin` Container:**
    *   Runs the pgAdmin web administration tool.
    *   Listens internally on port `80`.
    *   Connects to the `rag-postgres` container for database management.
    *   Persists its own configuration data to the `rag-pgadmin-data` volume.
6.  **Port Mapping:**
    *   Host port `7861` is mapped to `rag-app` container port `7860` (Gradio UI).
    *   Host port `8000` is mapped to `rag-app` container port `8000` (FastAPI API).
    *   Host port `8080` is mapped to `rag-pgadmin` container port `80`.
7.  **User Access:**
    *   Users access the Chat UI, API, and pgAdmin via `localhost` on the mapped host ports.
    *   Users interact with the application for tasks like processing via `docker compose exec app ...`.

</rewritten_file> 