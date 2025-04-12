# Docker Deployment View

This diagram illustrates how the different services of the DocuSeek AI system are containerized and connected using Docker Compose.

```mermaid
graph TD
    subgraph Host Machine
        UserBrowser["User's Browser"]
        UserCLI["User's Terminal"]
    end

    subgraph Docker Network [rag-network]
        subgraph AppService [rag-app Container]
            direction LR
            AppCode[Python App: main.py / app/*]
            Gradio["Gradio Server (Port 7860)"]
            FastAPI["FastAPI Server (Port 8000)"]
            AppCode -- Runs --> Gradio
            AppCode -- Runs --> FastAPI
        end

        subgraph DBService [rag-postgres Container]
            direction LR
            Postgres["PostgreSQL Server (Port 5432)"] -- Has --> PGVector[pgvector Extension]
            Postgres -- Reads/Writes --> Volume1[(rag-postgres-data Volume)]
        end

        subgraph AdminService [rag-pgadmin Container]
            direction LR
            PGAdmin["pgAdmin UI (Port 80)"] -- Reads/Writes --> Volume2[(rag-pgadmin-data Volume)]
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