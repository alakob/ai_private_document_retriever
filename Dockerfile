FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DOCKER_CONTAINER=true

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    python3-dev \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and set proper permissions for appuser home directory
RUN mkdir -p /home/appuser && \
    chown -R appuser:appuser /home/appuser && \
    chmod 755 /home/appuser

# Set temporary directory to one the user has access to
ENV TMPDIR=/app/tmp

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create temporary directory
RUN mkdir -p /app/tmp && chown -R appuser:appuser /app/tmp

# Set proper permissions
RUN chown -R appuser:appuser /app

# Change to non-root user
USER appuser

# Create necessary directories with proper permissions
RUN mkdir -p /app/documents && \
    mkdir -p /app/data && \
    mkdir -p /app/visualizations

# Expose ports for FastAPI and Gradio
EXPOSE 8000
EXPOSE 7861

# Make entrypoint script executable
RUN chmod +x /app/docker-entrypoint.sh

# Use entrypoint script
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command if none provided (will start chat interface)
CMD ["chat"]
