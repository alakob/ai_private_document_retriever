# Environment Configuration

The system uses environment variables for configuration, which can be set in a `.env` file:

### Core Configuration
- `OPENAI_API_KEY`: Required for embedding generation and LLM responses
- `MODEL`: LLM model to use (default: gpt-4o-mini)

### Document Processing
- `CHUNK_SIZE`: Size of document chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `KNOWLEDGE_BASE_DIR`: Directory containing documents (default: documents)

### Database Configuration
- `POSTGRES_HOST`: Database hostname (default: localhost)
- `POSTGRES_PORT`: Database port (default: 5432)
- `POSTGRES_USER`: Database username (default: postgres)
- `POSTGRES_PASSWORD`: Database password
- `POSTGRES_DB`: Database name (default: ragSystem)

### Caching Configuration
- `USE_REDIS_CACHE`: Enable Redis caching (default: false, uses file-based caching)
- `REDIS_HOST`: Redis server hostname (default: localhost)
- `REDIS_PORT`: Redis server port (default: 6379)
- `REDIS_DB`: Redis database number (default: 0)
- `REDIS_PASSWORD`: Redis authentication password (if required)

### Chat Interface Configuration
- `SERVER_NAME`: Server hostname for chat interface (default: 127.0.0.1)
- `TEMPERATURE`: Temperature for LLM responses (default: 0.7)
- `MAX_TOKENS`: Maximum tokens in LLM responses (default: 1000)

### Enhanced Document Processing
- `USE_DOCLING`: Enable Docling document conversion (default: false)
- `DOCLING_ARTIFACTS_PATH`: Path to pre-downloaded Docling models (optional)
- `DOCLING_ENABLE_REMOTE`: Allow Docling to use remote services (default: false)
- `DOCLING_USE_CACHE`: Use caching for Docling conversions (default: true)

### OCR Configuration
- `MISTRAL_API_KEY`: Required for OCR capabilities when using Mistral API
- `USE_MISTRAL`: Enable Mistral OCR API (default: false)
- `MISTRAL_OCR_MODEL`: Mistral OCR model to use (default: mistral-ocr-latest)
- `MISTRAL_INCLUDE_IMAGES`: Include extracted images in OCR results (default: false) 