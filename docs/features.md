# Features

- **Document Processing**: Load, chunk, and embed documents from various formats (PDF, DOCX, TXT, etc.)
- **Enhanced Document Conversion**: Optional integration with Docling for improved document structure preservation
- **OCR Capabilities**: Integration with Mistral OCR API for superior text extraction from documents and images
- **Vector Storage**: Store document embeddings in PostgreSQL with pgvector extension
- **Similarity Search**: Find the most relevant document chunks for any query
- **Chat Interface**: Interactive UI for asking questions about your documents
- **Conceptual Diagrams**: Generate Mermaid diagrams that visually represent the key concepts and relationships in your chat history
- **Visualization**: Visualize document embeddings in 2D space
- **Embedding Caching**: Dual caching system (file-based and Redis) for embeddings to reduce API calls and improve performance
- **Security Features**: Built-in circuit breaker pattern to prevent system overload and rate limiting for diagram generation
- **Asynchronous Processing**: Fully asynchronous document processing pipeline for improved performance
- **Duplicate Document Detection**: Checksums to prevent duplicate document processing
- **Adaptive Batching**: Dynamic batch size adjustments based on system performance
- **Resource Monitoring**: Runtime memory and CPU monitoring to prevent system overload 