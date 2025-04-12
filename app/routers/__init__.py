"""
API Routers Package

This package contains all API router modules for the AI Private Document Retriever:
- document_routes: Document management (upload, list, get details, delete)
- search_routes: Vector similarity search operations
- chat_routes: Chat interface and question answering
- system_routes: System information and management
"""

from fastapi import APIRouter

from app.routers.document_routes import router as document_router
from app.routers.search_routes import router as search_router
from app.routers.chat_routes import router as chat_router
from app.routers.system_routes import router as system_router

# Create a combined router
api_router = APIRouter()

# Include all routers
api_router.include_router(document_router)
api_router.include_router(search_router)
api_router.include_router(chat_router)
api_router.include_router(system_router)
