# Docling Integration

## Overview

This document explains the integration of [Docling](https://ds4sd.github.io/docling/) as an alternative document loader in the system. Docling provides enhanced document conversion capabilities with better preservation of document structure, tables, and other elements.

## Benefits of Docling

- **Improved Structure Preservation**: Better handling of tables, lists, and other structured content
- **Unified API**: One converter for multiple document formats (PDF, DOCX, PPT, etc.)
- **Advanced PDF Handling**: Better structural recognition for complex documents
- **Offline Capability**: Can use pre-downloaded models for air-gapped environments
- **Export Flexibility**: Can export to multiple formats (Markdown, HTML, JSON)

## Installation

Docling has been added to the `requirements.txt` file. To install it, run:

```bash
pip install -r requirements.txt
```

Or install docling directly with:

```bash
pip install docling
```

## Configuration

You can configure docling through environment variables or command-line arguments:

### Environment Variables

Add these to your `.env` file:

```bash
# Enable docling (default: false)
USE_DOCLING=true

# Path to pre-downloaded models (optional)
DOCLING_ARTIFACTS_PATH=/path/to/models

# Allow remote services (default: false)
DOCLING_ENABLE_REMOTE=false

# Use cache for document conversions (default: true)
DOCLING_USE_CACHE=true
```

### Command-Line Usage

When processing documents, you can enable docling with the `--use-docling` flag:

```bash
python -m app.core.document_rag_loader --directory /path/to/documents --use-docling
```

## Pre-downloading Models for Offline Use

To use docling in air-gapped environments, you can pre-download the required models:

```bash
# Install docling tools
pip install docling-tools

# Download models
docling-tools models download
```

The models will be downloaded to `$HOME/.cache/docling/models`. You can specify this path in the `DOCLING_ARTIFACTS_PATH` environment variable.

## How Docling is Used in the System

The system now includes a `DoclingLoader` class in `app/core/document_rag_loader.py` that provides an alternative to the existing document loaders. When enabled:

1. The system checks if docling is installed
2. If available, it replaces the default loaders for supported file formats (.pdf, .docx, .doc, .ppt, .pptx)
3. Documents are loaded using docling's `DocumentConverter` which preserves structure better
4. The converted content is returned as a LangChain Document object with appropriate metadata

## Comparison with Default Loaders

### Default Loaders
- **PDF**: PyPDFLoader - Basic text extraction
- **DOCX**: Docx2txtLoader - Simple text extraction
- **PPT**: UnstructuredPowerPointLoader - Basic text extraction

### Docling Loader
- **PDF**: Full structure preservation (headers, tables, lists)
- **DOCX**: Better structure preservation
- **PPT**: Better structure preservation

## Troubleshooting

If you encounter issues with docling:

1. Check that docling is properly installed: `pip list | grep docling`
2. Ensure the required dependencies are installed
3. Check your CUDA version if using GPU acceleration
4. Verify that you have sufficient disk space for model downloads

For more information, see the [Docling documentation](https://ds4sd.github.io/docling/).
