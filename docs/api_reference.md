# API Endpoints Reference

When running the API server with `python main.py api` or in Docker, the following endpoints are available at `http://localhost:8000/api/v1/`:

### Document Management

**Upload Document**
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/documents/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path/to/your/document.pdf' \
  -F 'options={"use_docling": true, "use_mistral": false, "reset_db": false}'
```

**List Documents**
```bash
curl -X 'GET' \
  'http://localhost:8000/api/v1/documents/?skip=0&limit=100' \
  -H 'accept: application/json'
```

**Get Document Details**
```bash
curl -X 'GET' \
  'http://localhost:8000/api/v1/documents/1?include_chunks=true' \
  -H 'accept: application/json'
```

**Delete Document**
```bash
curl -X 'DELETE' \
  'http://localhost:8000/api/v1/documents/1?delete_file=false' \
  -H 'accept: application/json'
```

### Search

**Search Documents**
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/search/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "your search query",
    "top_k": 5,
    "threshold": 0.7
  }'
```

**Get Cache Info**
```bash
curl -X 'GET' \
  'http://localhost:8000/api/v1/search/cache-info' \
  -H 'accept: application/json'
```

**Clear Cache**
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/search/clear-cache' \
  -H 'accept: application/json'
```

### Chat

**Chat with Documents**
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/chat/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "your question about documents",
    "chat_history": [],
    "retriever_k": 4,
    "retriever_score_threshold": 0.5,
    "temperature": 0.7
  }'
```

### System

**Get System Info**
```bash
curl -X 'GET' \
  'http://localhost:8000/api/v1/system/info' \
  -H 'accept: application/json'
```

**Reset Database** (requires API key)
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/system/reset?reset_vector_index=true&confirm=true' \
  -H 'accept: */*' \
  -H 'X-API-Key: dev_api_key_not_secure' \
  -d ''
``` 