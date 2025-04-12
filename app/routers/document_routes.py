"""
Document Management API Router

This module provides endpoints for managing documents in the AI Private Document Retriever system:
- Upload and process documents
- List processed documents
- Get document details
- Delete documents
"""

import os
import shutil
import json
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query, Path, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel, Field

from app.models import DocumentModel, DocumentChunk
from app.core.document_rag_loader import DocumentProcessor, DocumentProcessingError
from app.config import processor_config, db_config
from pathlib import Path as PathLib

# Database session dependency
async def get_db():
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.ext.asyncio import async_sessionmaker
    
    engine = create_async_engine(
        db_config.connection_string,
        pool_size=db_config.pool_size,
        max_overflow=db_config.max_overflow,
        pool_timeout=db_config.pool_timeout,
        pool_recycle=db_config.pool_recycle
    )
    
    async_session = async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        class_=AsyncSession
    )
    
    async with async_session() as session:
        yield session

# Pydantic models for request/response schemas
class DocumentMetadata(BaseModel):
    """Document metadata schema"""
    filename: str
    file_size: int
    file_type: str
    checksum: str
    created_at: str
    num_chunks: Optional[int] = None
    additional_metadata: Optional[dict] = None

class DocumentResponse(BaseModel):
    """Document response schema"""
    id: int
    filename: str
    metadata: DocumentMetadata
    
class DocumentListResponse(BaseModel):
    """Response schema for document list"""
    documents: List[DocumentResponse]
    total: int
    
class DocumentDetail(BaseModel):
    """Detailed document information including chunks"""
    id: int
    filename: str
    metadata: DocumentMetadata
    chunks: Optional[List[dict]] = None
    
class ProcessingOptions(BaseModel):
    """Document processing options"""
    use_docling: bool = Field(False, description="Use Docling for enhanced document conversion")
    use_mistral: bool = Field(False, description="Use Mistral OCR API for document processing")
    reset_db: bool = Field(False, description="Reset database before processing")

# Create router
router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    responses={404: {"description": "Not found"}},
)

# Document upload directory
UPLOAD_DIR = PathLib("documents")

# Ensure upload directory exists
if not UPLOAD_DIR.exists():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Background task to process documents
async def process_document_task(file_path: str, options: ProcessingOptions):
    """Process a document in the background"""
    # Configure processor with options
    config_dict = {
        "directory": str(UPLOAD_DIR),
        "reset_db": options.reset_db,
        "use_docling": options.use_docling,
        "use_mistral": options.use_mistral
    }
    
    # Import async_main from the document_rag_loader
    from app.core.document_rag_loader import async_main
    import asyncio
    
    # Process the file
    try:
        await async_main(**config_dict)
    except Exception as e:
        # Log error but don't raise since this is a background task
        import logging
        logging.error(f"Error processing document {file_path}: {str(e)}")

@router.post("/", response_model=DocumentResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    options: Optional[str] = Form(
        None,
        description=(
            "JSON string with processing options. Example: "
            '{"use_docling": true, "use_mistral": false, "reset_db": false}'
        )
    ),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload and process a new document
    
    This endpoint accepts document uploads, saves them to the storage directory,
    and initiates background processing to extract text and create vector embeddings.
    """
    # Parse options from JSON string if provided
    processing_options = ProcessingOptions()
    if options and options.strip().startswith('{'):
        try:
            options_dict = json.loads(options)
            processing_options = ProcessingOptions(**options_dict)
        except json.JSONDecodeError:
            # Fall back to default options if JSON is invalid
            import logging
            logging.warning(f"Invalid JSON in options field, using defaults: {options}")
        except Exception as e:
            # Fall back to default options if there's any other error
            import logging
            logging.warning(f"Error parsing options, using defaults: {str(e)}")
    
    # Validate file type
    file_ext = os.path.splitext(file.filename)[1].lower()
    supported_extensions = [".pdf", ".txt", ".md", ".docx", ".csv", ".json", ".html", ".htm", ".xml"]
    
    if file_ext not in supported_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported types: {', '.join(supported_extensions)}"
        )
    
    # Create file path for saving
    file_path = UPLOAD_DIR / file.filename
    
    # Check if file already exists
    if file_path.exists():
        # Check if document with same filename already exists in DB
        result = await db.execute(select(DocumentModel).where(DocumentModel.filename == file.filename))
        existing_doc = result.scalars().first()
        
        if existing_doc:
            # Return existing document
            return JSONResponse(
                status_code=200,
                content={
                    "id": existing_doc.id,
                    "filename": existing_doc.filename,
                    "metadata": existing_doc.doc_metadata,
                    "message": "Document already exists"
                }
            )
    
    # Save uploaded file
    try:
        # Create file with content from uploaded file
        with open(file_path, "wb") as f:
            # Read file chunk by chunk to handle large files
            chunk_size = 1024 * 1024  # 1MB chunks
            while content := await file.read(chunk_size):
                f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    # Queue background processing task
    background_tasks.add_task(process_document_task, str(file_path), processing_options)
    
    # Calculate basic file metadata
    file_size = file_path.stat().st_size
    file_type = file_ext.replace(".", "")
    
    # Calculate checksum
    from app.core.document_rag_loader import calculate_file_checksum
    checksum = calculate_file_checksum(str(file_path))
    
    # Create response with basic info (processing will happen in background)
    response = {
        "id": 0,  # Will be assigned during processing
        "filename": file.filename,
        "metadata": {
            "filename": file.filename,
            "file_size": file_size,
            "file_type": file_type,
            "checksum": checksum,
            "created_at": str(PathLib(file_path).stat().st_mtime),
            "status": "processing"
        }
    }
    
    return JSONResponse(
        status_code=202, 
        content=response
    )

@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """
    List all processed documents
    
    Returns a paginated list of all documents that have been processed by the system.
    """
    import logging
    from sqlalchemy.exc import SQLAlchemyError
    
    # Create an empty response structure to use in case of errors
    empty_response = {
        "documents": [],
        "total": 0
    }
    
    try:
        # First attempt to get total count - this will fail if table doesn't exist
        try:
            result = await db.execute(select(func.count()).select_from(DocumentModel))
            total_count = result.scalar() or 0
        except SQLAlchemyError as e:
            logging.warning(f"Could not count documents, table may not exist: {str(e)}")
            return empty_response
            
        # If we get here, the table exists, so proceed with pagination query
        try:
            result = await db.execute(
                select(DocumentModel)
                .order_by(DocumentModel.id)
                .offset(skip)
                .limit(limit)
            )
            documents = result.scalars().all()
        except SQLAlchemyError as e:
            logging.error(f"Error fetching documents: {str(e)}")
            return empty_response
        
        # Process documents if we have them
        docs_list = []
        for doc in documents:
            try:
                # Count chunks for each document
                chunk_count_result = await db.execute(
                    select(func.count())
                    .select_from(DocumentChunk)
                    .where(DocumentChunk.document_id == doc.id)
                )
                chunk_count = chunk_count_result.scalar() or 0
                
                # Extract metadata
                metadata = doc.doc_metadata or {}
                
                # Get information from the actual file if possible
                file_path = os.path.join(UPLOAD_DIR, os.path.basename(doc.filename))
                file_size = 0
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                elif metadata.get('file_size'):
                    # Try to get from existing metadata
                    file_size = metadata.get('file_size')
                
                # Determine file type from extension
                file_ext = os.path.splitext(doc.filename)[1].lower()
                file_type = metadata.get('file_type', "application/octet-stream")
                
                if not file_type or file_type == "application/octet-stream":
                    # Only determine file type if we don't already have it
                    if not file_ext:
                        file_type = "text/plain"  # Default
                    elif file_ext in [".pdf"]:
                        file_type = "application/pdf"
                    elif file_ext in [".txt"]:
                        file_type = "text/plain"
                    elif file_ext in [".docx", ".doc"]:
                        file_type = "application/msword"
                    elif file_ext in [".xlsx", ".xls"]:
                        file_type = "application/excel"
                    elif file_ext in [".pptx", ".ppt"]:
                        file_type = "application/powerpoint"
                    elif file_ext in [".html", ".htm"]:
                        file_type = "text/html"
                    elif file_ext in [".md"]:
                        file_type = "text/markdown"
                    else:
                        file_type = "application/octet-stream"
                        
                # Add required fields to metadata
                metadata.update({
                    "filename": doc.filename,
                    "checksum": doc.checksum,
                    "created_at": str(doc.created_at),
                    "num_chunks": chunk_count,
                    "file_size": file_size,
                    "file_type": file_type
                })
                
                docs_list.append({
                    "id": doc.id,
                    "filename": doc.filename,
                    "metadata": metadata
                })
            except SQLAlchemyError as e:
                # Log error but continue processing other documents
                logging.error(f"Error processing document {doc.id}: {str(e)}")
                continue
        
        return {
            "documents": docs_list,
            "total": total_count
        }
    except Exception as e:
        # Catch all other exceptions and return empty result
        logging.error(f"Unexpected error listing documents: {str(e)}")
        return empty_response

@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: int = Path(..., description="The ID of the document to retrieve"),
    include_chunks: bool = Query(False, description="Include document chunks in response"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get document details
    
    Returns detailed information about a specific document, optionally including its chunks.
    """
    # Get document
    result = await db.execute(
        select(DocumentModel).where(DocumentModel.id == document_id)
    )
    document = result.scalars().first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Count chunks
    chunk_count_result = await db.execute(
        select(func.count())
        .select_from(DocumentChunk)
        .where(DocumentChunk.document_id == document.id)
    )
    chunk_count = chunk_count_result.scalar()
    
    # Extract metadata
    metadata = document.doc_metadata or {}
    
    # Get information from the actual file if possible
    file_path = os.path.join(UPLOAD_DIR, os.path.basename(document.filename))
    file_size = 0
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
    elif metadata.get('file_size'):
        # Try to get from existing metadata
        file_size = metadata.get('file_size')
    
    # Determine file type from extension
    file_ext = os.path.splitext(document.filename)[1].lower()
    file_type = metadata.get('file_type', "application/octet-stream")
    
    if not file_type or file_type == "application/octet-stream":
        # Only determine file type if we don't already have it
        if not file_ext:
            file_type = "text/plain"  # Default
        elif file_ext in [".pdf"]:
            file_type = "application/pdf"
        elif file_ext in [".txt"]:
            file_type = "text/plain"
        elif file_ext in [".docx", ".doc"]:
            file_type = "application/msword"
        elif file_ext in [".xlsx", ".xls"]:
            file_type = "application/excel"
        elif file_ext in [".pptx", ".ppt"]:
            file_type = "application/powerpoint"
        elif file_ext in [".html", ".htm"]:
            file_type = "text/html"
        elif file_ext in [".md"]:
            file_type = "text/markdown"
        else:
            file_type = "application/octet-stream"
    
    metadata.update({
        "filename": document.filename,
        "checksum": document.checksum,
        "created_at": str(document.created_at),
        "num_chunks": chunk_count,
        "file_size": file_size,
        "file_type": file_type
    })
    
    response = {
        "id": document.id,
        "filename": document.filename,
        "metadata": metadata
    }
    
    # Include chunks if requested
    if include_chunks:
        chunks_result = await db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document.id)
            .order_by(DocumentChunk.chunk_index)
        )
        chunks = chunks_result.scalars().all()
        
        response["chunks"] = [
            {
                "id": chunk.id,
                "content": chunk.content,
                "metadata": chunk.chunk_metadata,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "chunk_index": chunk.chunk_index
            }
            for chunk in chunks
        ]
    
    return response

@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: int = Path(..., description="The ID of the document to delete"),
    delete_file: bool = Query(False, description="Also delete the source file"),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a document
    
    Removes a document and its chunks from the database. Optionally also deletes the source file.
    """
    # Check if document exists
    result = await db.execute(
        select(DocumentModel).where(DocumentModel.id == document_id)
    )
    document = result.scalars().first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get filename for possible file deletion
    filename = document.filename
    
    # Delete document (cascades to chunks due to relationship configuration)
    await db.delete(document)
    await db.commit()
    
    # Delete the file if requested
    if delete_file and filename:
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            try:
                os.remove(file_path)
            except Exception as e:
                # Log error but don't fail the request
                import logging
                logging.error(f"Error deleting file {file_path}: {str(e)}")
    
    return None  # 204 No Content
