#!/bin/bash
# Wrapper script to build the Docker image and start all services.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Starting DocuSeek AI Docker environment..."

# 1. Build the application image using docker-bake.hcl
# This ensures the latest code is built and the image is loaded locally.
echo "🛠️  Building application image with 'docker buildx bake' (if necessary)..."
docker buildx bake
echo "✅ Image build complete."

# 2. Start all services (app, postgres, pgadmin) using docker-compose
# Reads environment variables from .env.docker
echo "🐳 Starting services with 'docker compose up -d'..."
docker compose --env-file .env.docker up -d
echo "✅ Services started successfully."

echo ""
echo "🔗 Access Services:"
echo "   - Chat Interface (Default Service): http://localhost:7861"
echo "   - pgAdmin (Database Admin):         http://localhost:8080"
echo "   - API (If started manually):        http://localhost:8000"

echo ""
echo "💡 Next Steps:"
echo "   - Place documents in the ./documents folder."
echo "   - Process documents using: docker compose exec app python main.py process --dir documents"

echo "✨ Setup complete!" 