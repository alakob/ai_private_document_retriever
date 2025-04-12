"""
Chat API Router

This module provides endpoints for chat operations:
- Submit questions to the chat interface
- Retrieve chat history
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Body, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
import uuid
import time
import asyncio
from datetime import datetime

from app.routers.document_routes import get_db
from app.config import chat_config

# Pydantic models for request/response schemas
class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    """Chat request model"""
    question: str = Field(..., description="The question to ask")
    chat_history: Optional[List[ChatMessage]] = Field(default_factory=list, description="Previous chat history")
    retriever_k: Optional[int] = Field(None, description="Number of documents to retrieve")
    retriever_score_threshold: Optional[float] = Field(None, description="Similarity threshold for retrieved documents")
    temperature: Optional[float] = Field(None, description="LLM temperature parameter")
    stream: Optional[bool] = Field(True, description="Whether to stream the response")

class ChatResponse(BaseModel):
    """Chat response model"""
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents used for the answer")
    chat_id: str
    elapsed_time_ms: float

class ChatHistoryResponse(BaseModel):
    """Chat history response model"""
    chat_id: str
    messages: List[ChatMessage]
    created_at: datetime
    updated_at: datetime

# Create router
router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)

# In-memory chat history store - in a real application, this would be persisted to a database
chat_history_store = {}

async def get_chat_processor():
    """Initialize chat components dynamically to avoid circular imports"""
    from app.services.vector_similarity_search import VectorSimilaritySearch
    from app.config import db_config
    from langchain_openai import ChatOpenAI
    import os
    
    # Initialize search service
    search_service = VectorSimilaritySearch(
        connection_string=db_config.connection_string
    )
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=chat_config.model,
        temperature=chat_config.temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    return {
        "search_service": search_service,
        "llm": llm
    }

@router.post("/", response_model=ChatResponse)
async def chat_question(
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Submit a question to the chat interface
    
    This endpoint processes a question using the RAG system and returns an AI-generated answer.
    """
    # Generate a unique chat ID if not provided in history
    chat_id = str(uuid.uuid4())
    
    # Process the question
    start_time = time.time()
    
    try:
        # Get chat components
        chat_components = await get_chat_processor()
        search_service = chat_components["search_service"]
        llm = chat_components["llm"]
        
        # Set up parameters
        retriever_k = chat_request.retriever_k or chat_config.retriever_k
        retriever_score_threshold = chat_request.retriever_score_threshold or chat_config.retriever_score_threshold
        temperature = chat_request.temperature or chat_config.temperature  # Fixed parameter name
        
        # Set parameters on the search service instance
        search_service.top_k = retriever_k
        search_service.score_threshold = retriever_score_threshold
        
        # Step 1: Retrieve relevant documents
        search_results = await search_service.search(
            query=chat_request.question
        )
        
        # Format the context from retrieved documents
        context = "\n\n".join([
            f"Document: {result.get('document_name', 'Unknown')}\n"
            f"Content: {result.get('content', '')}"
            for result in search_results
        ])
        
        # Build chat history in the format LLM expects
        formatted_history = []
        for msg in chat_request.chat_history:
            formatted_history.append({"role": msg.role, "content": msg.content})
        
        # Step 2: Generate the response using the LLM
        from app.config import prompt_config
        
        # Format the prompt
        messages = [
            {"role": "system", "content": prompt_config.qa_template.format(
                context=context,
                chat_history=formatted_history if formatted_history else "No previous conversation.",
                question=chat_request.question
            )}
        ]
        
        # Get response from LLM
        response = llm.invoke(messages)
        answer = response.content
        
        # Calculate elapsed time
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Store in chat history
        current_time = datetime.utcnow()
        if chat_id not in chat_history_store:
            chat_history_store[chat_id] = {
                "messages": [],
                "created_at": current_time,
                "updated_at": current_time
            }
            
        # Add user message
        chat_history_store[chat_id]["messages"].append({
            "role": "user",
            "content": chat_request.question,
            "timestamp": current_time
        })
        
        # Add assistant message
        chat_history_store[chat_id]["messages"].append({
            "role": "assistant",
            "content": answer,
            "timestamp": current_time
        })
        
        # Update the timestamp
        chat_history_store[chat_id]["updated_at"] = current_time
        
        # Format sources for response
        sources = [
            {
                "document_id": result.get("document_id"),
                "document_name": result.get("document_name", "Unknown"),
                "chunk_id": result.get("chunk_id"),
                "page_number": result.get("metadata", {}).get("page_number"),
                "section_title": result.get("metadata", {}).get("section_title"),
                "score": result.get("score", 0.0)
            }
            for result in search_results
        ]
        
        # Clean up resources in background
        background_tasks.add_task(search_service.cleanup)
        
        return {
            "answer": answer,
            "sources": sources,
            "chat_id": chat_id,
            "elapsed_time_ms": elapsed_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")

@router.get("/history", response_model=List[ChatHistoryResponse])
async def get_chat_history():
    """
    Retrieve chat history
    
    Returns a list of all chat sessions with their messages.
    """
    # Format chat history for response
    history_list = []
    for chat_id, chat_data in chat_history_store.items():
        history_list.append({
            "chat_id": chat_id,
            "messages": chat_data["messages"],
            "created_at": chat_data["created_at"],
            "updated_at": chat_data["updated_at"]
        })
    
    return history_list

@router.get("/history/{chat_id}", response_model=ChatHistoryResponse)
async def get_chat_session(chat_id: str):
    """
    Retrieve a specific chat session
    
    Returns the messages from a specific chat session.
    """
    if chat_id not in chat_history_store:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    chat_data = chat_history_store[chat_id]
    return {
        "chat_id": chat_id,
        "messages": chat_data["messages"],
        "created_at": chat_data["created_at"],
        "updated_at": chat_data["updated_at"]
    }
