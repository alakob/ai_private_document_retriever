# Systematic Testing Plan for AI Document Retriever Docker Commands

## Executive Summary

This document provides a comprehensive testing plan for the AI Private Document Retriever when running in Docker containers. Following this guide will help you verify that all system components are functioning correctly. The testing process is organized in a logical progression from basic setup to advanced features.

This guide provides a step-by-step testing plan for all functionalities of the AI Document Retriever when running with Docker.

## Prerequisites

Before beginning, verify your Docker installation:

```bash
# Check Docker version
docker --version

# Check Docker Compose version
docker compose version

# Verify Docker is running
docker info
```

## Initial Setup
```bash
# 1. Create required directories
mkdir -p documents
mkdir -p visualizations
mkdir -p docker-entrypoint-initdb.d

# 2. Copy and configure environment file
cp .env.example .env.docker
nano .env.docker  # Set your API keys and passwords

# 3. Start fresh by removing any existing containers and volumes
docker compose down -v

# 4. Build and start services
docker compose --env-file .env.docker up -d --build

# 5. Monitor initial startup
docker compose logs -f
```

## Test 1: Database and API Health
```bash
# 1. Check if all containers are running
docker compose ps

# 2. Test API health endpoint
curl http://localhost:8000/health

# 3. Verify pgAdmin access
open http://localhost:8080  # Login with credentials from .env.docker
```

### Accessing pgAdmin

pgAdmin should automatically connect to the PostgreSQL database without requiring server details:

1. Open http://localhost:8080 in your browser
2. Login with the following credentials:
   - Email: Value of PGADMIN_EMAIL from .env.docker (default: admin@example.com)
   - Password: Value of PGADMIN_PASSWORD from .env.docker (default: admin)
3. After logging in, you should see "RAG Postgres" server in the left sidebar
4. Click on this server to expand it - it should connect automatically without asking for a password
5. If asked for a password, enter the value of POSTGRES_PASSWORD from .env.docker

If you're still prompted for connection details:
```bash
# Restart pgAdmin container
docker compose restart pgadmin

# Check pgAdmin logs for any issues
docker compose logs pgadmin
```

## Test 2: Document Processing Features
```bash
# 1. Basic document processing
docker compose exec app python main.py process --dir documents

# 2. Test document processing with Docling
docker compose exec app python main.py process --dir documents --use-docling

# 3. Test document processing with Mistral OCR
docker compose exec app python main.py process --dir documents --use-mistral

# 4. Test combined processing
docker compose exec app python main.py process --dir documents --use-docling --use-mistral

# 5. Test database reset and processing
docker compose exec app python main.py process --dir documents --reset-db
```

## Test 3: Vector Search Functionality
```bash
# 1. Test interactive search
docker compose exec app python main.py search

# 2. Test direct search with query
docker compose exec app python main.py search --query "test query"

# 3. Test search with custom parameters
docker compose exec app python main.py search --query "test query" --top-k 10 --threshold 0.6

# 4. Check cache information
docker compose exec app python main.py search --cache-info

# 5. Test cache clearing
docker compose exec app python main.py search --clear-cache
```

## Test 4: Visualization Features
```bash
# 1. Generate default visualization
docker compose exec app python main.py visualize

# 2. Generate visualization with custom parameters
docker compose exec app python main.py visualize --output visualizations/custom_viz.html --perplexity 50

# 3. Verify visualization files
ls -l visualizations/
```

## Test 5: Chat Interface
```bash
# 1. Start chat interface
docker compose exec app python main.py chat

# 2. Verify chat interface access
open http://localhost:7861

# 3. Test chat functionality
# - Open browser and try asking questions
# - Test document retrieval
# - Test different settings in the interface
```

## Test 6: API Server
```bash
# 1. Start API server
docker compose exec app python main.py api

# 2. Test endpoints
curl http://localhost:8000/health
```

## Test 7: Error Handling and Recovery
```bash
# 1. Test container recovery
docker compose stop app
docker compose start app

# 2. Test database recovery
docker compose stop postgres
docker compose start postgres

# 3. Verify service recovery
docker compose logs --tail=100 app
docker compose logs --tail=100 postgres
```

## Test 8: Resource Monitoring
```bash
# 1. Monitor resource usage
docker stats

# 2. Check container logs for any warnings/errors
docker compose logs app | grep -i error
docker compose logs postgres | grep -i error
```

## Test 9: Data Persistence
```bash
# 1. Stop all containers (keeping volumes)
docker compose down

# 2. Start containers again
docker compose --env-file .env.docker up -d

# 3. Verify data persistence
docker compose exec app python main.py search --cache-info
```

## Cleanup After Testing
```bash
# 1. Stop all containers
docker compose down

# 2. Remove all data (if needed)
docker compose down -v

# 3. Clean up test files
rm -rf visualizations/*
```

## Testing Checklist

For each test, record:
- [ ] Command executed
- [ ] Expected output
- [ ] Actual output
- [ ] Any errors encountered
- [ ] Resolution steps (if errors occurred)

## Error Recovery Steps

If you encounter issues:

1. Check logs:
```bash
docker compose logs app
docker compose logs postgres
docker compose logs pgadmin
```

2. Restart specific service:
```bash
docker compose restart [service_name]
```

3. Reset and rebuild (if necessary):
```bash
docker compose down -v
docker compose --env-file .env.docker up -d --build
```

## Important Notes

1. **Order Matters**: Follow the test sequence as document processing needs to be done before testing search or chat features.

2. **Data Requirements**: Ensure you have test documents in the `documents` directory before testing.

3. **API Keys**: Verify your API keys are correctly set in `.env.docker` before testing OCR or embedding features.

4. **Resource Monitoring**: Keep an eye on system resources during testing, especially during document processing.

5. **Network Connectivity**: Ensure you have stable internet connection for API-dependent features.

## Service Access URLs

- Chat Interface: http://localhost:7861
- API Health Check: http://localhost:8000/health
- pgAdmin: http://localhost:8080

## Common Issues and Solutions

### Chat Interface Not Accessible
1. Verify port mapping:
   ```bash
   docker compose ps
   ```
2. Check if the interface is running:
   ```bash
   docker compose logs app | grep -i "Running on"
   ```

### Database Connection Issues
1. Check if PostgreSQL is running:
   ```bash
   docker compose ps postgres
   ```
2. Verify database initialization:
   ```bash
   docker compose logs postgres | grep "database system is ready"
   ```

### pgAdmin Connection Issues
1. If pgAdmin isn't auto-connecting to PostgreSQL:
   ```bash
   # Force recreation of the pgAdmin container
   docker compose down pgadmin
   docker compose up -d pgadmin
   ```
2. Check if the setup script ran successfully:
   ```bash
   docker compose logs pgadmin | grep "PostgreSQL server configuration complete"
   ```
3. Manually connect using the following details if needed:
   - Host: postgres
   - Port: 5432
   - Username: Value of POSTGRES_USER from .env.docker (default: postgres)
   - Password: Value of POSTGRES_PASSWORD from .env.docker

### Document Processing Failures
1. Check API key configuration:
   ```bash
   docker compose exec app env | grep API_KEY
   ```
2. Verify document directory mounting:
   ```bash
   docker compose exec app ls -l /app/documents
   ```

## Support and Feedback

If you encounter any issues not covered in this guide:
1. Check the project documentation in the `docs/` directory
2. Review container logs for specific error messages
3. Open an issue in the project repository with:
   - Steps to reproduce the issue
   - Relevant log outputs
   - Environment details 