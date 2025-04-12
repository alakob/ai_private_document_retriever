# Docker Setup for AI Private Document Retriever

This guide explains how to run the AI Private Document Retriever application using Docker containers for streamlined deployment and management.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (version 20.10.0 or higher)
- [Docker Compose](https://docs.docker.com/compose/install/) (version 2.0.0 or higher)

## Step-by-Step Deployment Guide

### 1. Prepare Environment Variables

```bash
# Create a .env file for Docker
cp .env.example .env.docker

# Edit the .env.docker file to include your API keys
nano .env.docker
```

Make sure to update these values in your `.env.docker` file:
- `POSTGRES_USER`
- `POSTGRES_PASSWORD` 
- `OPENAI_API_KEY`
- `MISTRAL_API_KEY` (if using Mistral)
- `PGADMIN_EMAIL` (for pgAdmin login)
- `PGADMIN_PASSWORD` (for pgAdmin login)

### 2. Create Required Directories

```bash
# Create directories for mounted volumes
mkdir -p documents
mkdir -p visualizations
mkdir -p docker-entrypoint-initdb.d
```

### 3. Build and Start the Containers

```bash
# Build and start all services in detached mode
docker compose --env-file .env.docker up -d --build
```

This command will:
- Build the application image from the Dockerfile
- Pull the PostgreSQL and pgAdmin images
- Create the necessary networks and volumes
- Start all services in the background

### 4. Monitor Container Startup

```bash
# Check container logs
docker compose logs -f
```

Wait until you see the health checks passing:
- PostgreSQL: "database system is ready to accept connections"
- pgAdmin: Successfully starts up
- Application: "Application startup complete"

### 5. Using the Application

The general format for running commands in Docker is:
```bash
docker compose exec app python main.py [command] [options]
```

#### a. Document Processing Commands

```bash
# Basic document processing
docker compose exec app python main.py process --dir documents

# Process documents with enhanced Docling document conversion
docker compose exec app python main.py process --dir documents --use-docling

# Process documents with Mistral OCR API for better text extraction
docker compose exec app python main.py process --dir documents --use-mistral

# Reset the database and process documents (warning: deletes existing data)
docker compose exec app python main.py process --dir documents --reset-db

# Combine options as needed
docker compose exec app python main.py process --dir documents --use-mistral --use-docling
```

#### b. Chat Interface

```bash
# Start the chat interface
docker compose exec app python main.py chat
```
The chat interface will be accessible at http://localhost:7861 in your browser.

#### c. Vector Search Commands

```bash
# Start interactive vector search CLI
docker compose exec app python main.py search

# Direct search with a specific query
docker compose exec app python main.py search --query "your search query"

# Search with custom parameters
docker compose exec app python main.py search --query "your search query" --top-k 10 --threshold 0.6

# Display embedding cache information
docker compose exec app python main.py search --cache-info

# Clear the embedding cache
docker compose exec app python main.py search --clear-cache
```

#### d. Visualization Commands

```bash
# Generate default visualization
docker compose exec app python main.py visualize

# Generate visualization with custom parameters
docker compose exec app python main.py visualize --output visualizations/custom_visualization.html --perplexity 50
```
The visualization will be saved to your local `./visualizations` directory.

#### e. API Server

```bash
# Start the API server with health check endpoint
docker compose exec app python main.py api
```
The API will be accessible at http://localhost:8000 with a health check endpoint at http://localhost:8000/health.

### 6. Accessing Services

- **Chat Interface**: http://localhost:7861
- **API Health Check**: http://localhost:8000/health
- **pgAdmin Database Admin**: http://localhost:8080 (Login with email and password set in .env)
- **Visualizations**: Open files directly from the `./visualizations` directory

## Database Connection Details

pgAdmin connection settings:
- **Server Type**: PostgreSQL
- **Hostname/address**: postgres
- **Port**: 5432
- **Maintenance database**: ragSystem
- **Username**: postgres
- **Password**: (value from POSTGRES_PASSWORD in .env)

## Container Resource Management

Each container has resource limits configured in the docker-compose.yml file:

- **PostgreSQL**: 8 CPU, 6GB memory
- **pgAdmin**: 0.5 CPU, 500MB memory
- **Application**: 2 CPU, 2GB memory

Adjust these values in docker-compose.yml based on your system resources and application needs.

## Security Features

The Docker setup includes several security best practices:

1. **Environment Variable Management**: Sensitive credentials are managed using environment variables
2. **Network Isolation**: Services communicate over an isolated Docker bridge network
3. **Non-Root Users**: Containers run with non-root users where possible
4. **RBAC**: Database access is controlled using role-based access control
5. **Connection Security**: PostgreSQL uses SCRAM-SHA-256 authentication
6. **Resource Limitations**: Container resources are restricted to prevent resource exhaustion

## Data Persistence

Data persists between container restarts using Docker volumes:

- **postgres_data**: Stores PostgreSQL database files
- **app_data**: Stores application data files
- **pgadmin_data**: Stores pgAdmin configuration and session data

The following directories are mounted from the host:
- `documents`: For easy document management
- `visualizations`: For storing and accessing visualization outputs

## Health Monitoring

Health checks are configured for all containers:

- **PostgreSQL**: Checks database availability every 10 seconds
- **pgAdmin**: Checks web interface every 30 seconds
- **Application**: Checks API endpoint every 30 seconds

## Customization

### PostgreSQL Optimization

Database performance settings are configured in `docker-entrypoint-initdb.d/init-pgvector.sql`:

- **shared_buffers**: Memory for caching data (256MB default)
- **work_mem**: Memory for query operations (16MB default)
- **effective_cache_size**: Estimate of available system memory (512MB default)

Adjust these values based on your database size and system resources.

## Troubleshooting

### Container fails to start

Check the logs for detailed error messages:
```bash
docker compose logs app
```

### 7. Managing Containers

```bash
# Stop all containers
docker compose down

# Stop and remove volumes (CAUTION: this deletes all data)
docker compose down -v

# View container resource usage
docker stats
```

### 8. Troubleshooting

#### Database connection issues

1. Verify the PostgreSQL container is running:
   ```bash
   docker compose ps postgres
   ```

2. Check PostgreSQL logs:
   ```bash
   docker compose logs postgres
   ```

3. Connect directly to the database:
   ```bash
   docker compose exec postgres psql -U postgres -d ragSystem
   ```

#### Chat interface not accessible

1. Check if the ports are correctly mapped:
   ```bash
   docker compose ps
   ```

2. Verify the app logs for any errors:
   ```bash
   docker compose logs app
   ```

3. Restart the chat interface:
   ```bash
   docker compose exec app python main.py chat
   ```

### Performance issues

1. Monitor container resource usage:
   ```bash
   docker stats
   ```

2. Increase container resources in docker-compose.yml if needed.

## Production Deployment Considerations

For production deployments:

1. **Use Strong Passwords**: Replace default passwords with strong, unique credentials
2. **Configure SSL**: Add SSL certificates for secure communications
3. **Regular Backups**: Set up automated PostgreSQL database backups
4. **Monitoring**: Implement a monitoring solution (Prometheus/Grafana)
5. **Load Balancing**: For high-traffic deployments, consider adding a load balancer
6. **CI/CD Integration**: Integrate with CI/CD pipelines for automated deployments
