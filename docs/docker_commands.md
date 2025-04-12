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
# Basic document processing (uses default LangChain loaders)
docker compose exec app python main.py process --dir documents

# Process documents using Docling loader
docker compose exec app python main.py process --dir documents --use-docling

# Process documents using Mistral OCR API (requires MISTRAL_API_KEY)
docker compose exec app python main.py process --dir documents --use-mistral

# Reset database tables before processing (deletes existing processed data)
docker compose exec app python main.py process --dir documents --reset-db
```

#### Chat Interface
```bash
# Start the chat interface (accessible at http://localhost:7861)
docker compose exec app python main.py chat
```

#### Vector Search
```bash
# Enter interactive search mode
docker compose exec app python main.py search

# Perform a direct search
docker compose exec app python main.py search --query "your specific search query here"

# Direct search with custom top-k and similarity threshold
docker compose exec app python main.py search --query "details about project X" --top-k 10 --threshold 0.65

# Perform search disabling the embedding cache for this query
docker compose exec app python main.py search --query "latest status update" --no-cache

# Display embedding cache information (size, location, etc.)
docker compose exec app python main.py search --cache-info

# Clear the entire embedding cache
docker compose exec app python main.py search --clear-cache
```

#### Visualization
```