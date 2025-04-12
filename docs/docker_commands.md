# Docker Command Reference

With the integration of `docker buildx bake`, the workflow for building and running the application with Docker has changed. The build step is now separate from the run step.

## Building the Application Image

First, build the application image using the `docker-bake.hcl` definition file. This only needs to be done once, or whenever the `Dockerfile` or related source code changes.

```bash
# Build the image defined in docker-bake.hcl (e.g., docuseek-ai-app:latest)
docker buildx bake
```

## Starting Services

Once the image is built, use `docker compose` to start the application and its dependencies (like PostgreSQL and pgAdmin). Note that the `--build` flag is no longer needed as the image is pre-built.

```bash
# Start all services defined in docker-compose.yml in detached mode
docker compose --env-file .env.docker up -d
```

## Interacting with the Running Application

After the services are running (using `docker compose up -d`), you can interact with the application container using `docker compose exec`. All commands follow the pattern:

```bash
docker compose exec app python main.py [command] [options]
```

Here are the Docker equivalents for common application commands:

#### Document Processing
```bash
# Basic document processing
docker compose exec app python main.py process --dir documents

# Process documents with Docling
docker compose exec app python main.py process --dir documents --use-docling

# Process documents with Mistral OCR
docker compose exec app python main.py process --dir documents --use-mistral

# Reset database and process documents
docker compose exec app python main.py process --dir documents --reset-db
```

#### Chat Interface
```bash
# Start the chat interface (accessible at http://localhost:7861)
docker compose exec app python main.py chat
```

#### Vector Search
```bash
# Interactive search
docker compose exec app python main.py search

# Direct search with query
docker compose exec app python main.py search --query "your search query"

# Search with custom parameters
docker compose exec app python main.py search --query "your search query" --top-k 10 --threshold 0.6

# Cache management
docker compose exec app python main.py search --cache-info
docker compose exec app python main.py search --clear-cache
```

#### Visualization
```bash
# Generate default visualization
docker compose exec app python main.py visualize

# Custom visualization
docker compose exec app python main.py visualize --output visualizations/custom_visualization.html --perplexity 50
```

#### API Server
```bash
# Start the API server (accessible at http://localhost:8000)
docker compose exec app python main.py api
``` 