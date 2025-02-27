"""
Chat interface component handling the core chat functionality.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
import tiktoken
import asyncio
from datetime import datetime
from langchain_core.messages import trim_messages
from pathlib import Path
import shutil
import tempfile

from ...services.retriever import PostgresVectorRetriever
from ...services.embeddings import get_embeddings
from ...services.database import create_async_db_engine, create_async_session_maker
from ...config.database import PostgresConfig, DocumentModel, DocumentChunk
from ...config.processor import ProcessorConfig
from ..visualization.store_visualizer import StoreVisualizer
import plotly.graph_objects as go
from ..document_processing.processor import DocumentProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatInterface:
    """Handles chat interface and conversation"""
    
    def __init__(self):
        """Initialize chat interface with document processor."""
        # Initialize configs
        self.postgres_config = PostgresConfig(
            connection_string="postgresql+asyncpg://postgres:1%40SSongou2@192.168.1.185:5432/ragSystem",
            pre_delete_collection=True,
            drop_existing=True,
            collection_name="documents",
            embedding_dimension=1536
        )
        
        self.processor_config = ProcessorConfig(
            chunk_size=1000,
            chunk_overlap=200,
            batch_size=100,
            max_workers=4,
            db_pool_size=20,
            postgres_config=self.postgres_config,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Initialize document processor
        self.processor = DocumentProcessor(self.processor_config)
        
        # Initialize database connection
        self.engine = create_async_db_engine(self.postgres_config.connection_string)
        self.async_session = create_async_session_maker(self.engine)
        
        # Initialize OpenAI components
        self.embeddings = get_embeddings()
        
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-4o-mini",
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Initialize retriever
        self.retriever = PostgresVectorRetriever(
            session_maker=self.async_session,
            embeddings=self.embeddings,
            k=4,
            score_threshold=0.5
        )
        
        # Initialize system message
        self.system_message = SystemMessage(
            content="You are a helpful assistant that answers questions based on the provided documents."
        )
        
        # Initialize conversation chain
        self._initialize_conversation_chain()
        
        # Initialize conversation history
        self.messages = [self.system_message]
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.max_token_limit = 4000
        
        # Add debug logging
        self.debug = True
        
        # Add visualizer
        self.visualizer = StoreVisualizer(self.async_session)

    async def initialize(self):
        """Initialize necessary components."""
        try:
            await self.processor.initialize_database()
        except Exception as e:
            logger.error(f"Failed to initialize chat interface: {str(e)}")
            raise

    async def process_uploaded_files(self, files: List[tempfile._TemporaryFileWrapper]) -> Optional[str]:
        """Process uploaded files using document processor."""
        try:
            if not files:
                return "No files uploaded"

            logger.info(f"Received {len(files)} files")
            
            # Process each file
            for file in files:
                try:
                    file_path = Path(file.name)
                    if not file_path.exists():
                        logger.warning(f"File not found: {file_path}")
                        continue
                        
                    # Process the file
                    await self.processor.process_file(file_path)
                    
                except Exception as e:
                    logger.error(f"Error processing {file.name}: {str(e)}")
                    continue

            return "Files processed successfully"
            
        except Exception as e:
            error_msg = f"Error processing uploaded files: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def cleanup(self):
        """Cleanup resources properly"""
        try:
            if hasattr(self, 'engine'):
                await self.engine.dispose()
                logger.info("Database engine disposed")
            if hasattr(self, 'processor'):
                await self.processor.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def _initialize_conversation_chain(self):
        """Initialize the conversation chain with prompts"""
        document_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}"
        )

        qa_prompt = ChatPromptTemplate.from_template("""
        You are a helpful AI assistant that answers questions based on the provided documents.
        
        Instructions:
        1. Use ONLY the following context to answer the question
        2. If the context doesn't contain enough information, explain what's missing
        3. Always cite your sources
        4. Be detailed but concise
        
        Context: {context}
        
        Previous conversation:
        {chat_history}
        
        Question: {question}
        
        Answer: """)

        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            verbose=True,
            return_source_documents=True,
            chain_type="stuff",
            combine_docs_chain_kwargs={
                "document_prompt": document_prompt,
                "document_variable_name": "context",
                "prompt": qa_prompt
            }
        )

    async def chat(self, message: str, history: List[Dict[str, str]]) -> Dict[str, str]:
        """Process chat message and return response."""
        try:
            if self.debug:
                print(f"\nQuestion: {message}")
                print(f"History: {history}")
            
            # Convert history to messages format
            if not history:
                self.messages = [self.system_message]
            
            # Add user message
            user_message = HumanMessage(content=message)
            self.messages.append(user_message)
            
            # Trim history if needed
            self._trim_history()
            
            # Format chat history as tuples
            chat_history = []
            messages = self.messages[1:]  # Skip system message
            for i in range(0, len(messages)-1, 2):
                if i+1 < len(messages):
                    human_msg = messages[i]
                    ai_msg = messages[i+1]
                    if isinstance(human_msg, HumanMessage) and isinstance(ai_msg, AIMessage):
                        chat_history.append((human_msg.content, ai_msg.content))
            
            # Invoke chain
            result = await self.conversation_chain.ainvoke({
                "question": message,
                "chat_history": chat_history
            })
            
            # Create assistant message
            assistant_message = AIMessage(content=result["answer"])
            self.messages.append(assistant_message)
            
            # Format response with sources
            response = result["answer"]
            if "source_documents" in result and result["source_documents"]:
                unique_sources = {
                    doc.metadata['source']: doc.metadata.get('page', 'N/A') 
                    for doc in result["source_documents"]
                }
                sources = [
                    f"Source: {source} (Page: {page})" 
                    for source, page in unique_sources.items()
                ]
                response = f"{response}\n\nSources:\n" + "\n".join(sources)
            
            # Add debug logging for retrieved documents
            if self.debug and "source_documents" in result:
                print("\nRetrieved Documents:")
                for doc in result["source_documents"]:
                    print(f"Source: {doc.metadata['source']}")
                    print(f"Content: {doc.page_content[:200]}...")
            
            # Return in OpenAI message format
            return {"role": "assistant", "content": response}
            
        except Exception as e:
            import traceback
            print(f"Error: {str(e)}")
            print(traceback.format_exc())
            return {"role": "assistant", "content": f"Error processing question: {str(e)}"}

    def _count_tokens(self, messages: list) -> int:
        """Count tokens in concatenated message content"""
        try:
            text_content = " ".join([
                str(msg.content) 
                for msg in messages 
                if hasattr(msg, 'content') and msg.content is not None
            ])
            return len(self.tokenizer.encode(text_content))
        except Exception as e:
            print(f"Token counting error: {str(e)}")
            return 0
    
    def _trim_history(self):
        """Trim conversation history using LangChain's utilities"""
        try:
            # Validate messages
            validated_messages = [
                msg for msg in self.messages 
                if isinstance(msg, BaseMessage) and hasattr(msg, 'content')
            ]
            
            # Trim messages
            self.messages = trim_messages(
                messages=validated_messages,
                max_tokens=self.max_token_limit,
                token_counter=self._count_tokens,
                strategy="last"  # Keep most recent messages
            )
        except Exception as e:
            print(f"History trimming error: {str(e)}")
            # Keep only system message and last few messages if trimming fails
            self.messages = [self.system_message] + self.messages[-4:]

    async def handle_file_upload(self, files: List[str]) -> str:
        """Handle initial file upload to temporary storage."""
        try:
            logger.info(f"\n📤 Received {len(files)} files")
            
            # Create documents directory if it doesn't exist
            documents_dir = Path("documents")
            documents_dir.mkdir(exist_ok=True)
            
            successful_uploads = []
            for file_path in files:
                try:
                    # Convert to Path object
                    source_path = Path(file_path)
                    
                    # Create destination path in documents directory
                    dest_path = documents_dir / source_path.name
                    
                    # Handle duplicate filenames
                    counter = 1
                    while dest_path.exists():
                        stem = source_path.stem
                        suffix = source_path.suffix
                        new_name = f"{stem}_{counter}{suffix}"
                        dest_path = documents_dir / new_name
                        counter += 1
                    
                    logger.info(f"Moving {source_path} to {dest_path}")
                    
                    # Move file to documents directory
                    shutil.move(str(source_path), str(dest_path))
                    
                    successful_uploads.append(dest_path.name)
                    logger.info(f"✅ Successfully moved {dest_path.name}")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing {file_path}: {str(e)}")
                    continue

            if successful_uploads:
                return (f"Successfully uploaded {len(successful_uploads)} files:\n" + 
                       "\n".join(f"- {f}" for f in successful_uploads) + 
                       "\n\nClick 'Process Documents' to proceed with document processing.")
            else:
                return "No files were successfully uploaded"
                
        except Exception as e:
            logger.error(f"❌ Upload error: {str(e)}", exc_info=True)
            return f"Error uploading files: {str(e)}"

    async def process_documents(self) -> str:
        """Process the uploaded documents."""
        try:
            # Process documents directory
            docs = await self.processor.process_directory(Path("documents"))
            
            if not docs:
                return "No documents were processed. Please check the files and try again."
            
            return f"Successfully processed {len(docs)} documents into chunks. You can now ask questions about them."
            
        except Exception as e:
            logger.error(f"❌ Processing error: {str(e)}", exc_info=True)
            return f"Error processing documents: {str(e)}"

    async def visualize_vector_store(self) -> tuple[go.Figure, str]:
        """Generate visualization of the vector store."""
        try:
            async with self.async_session() as session:
                visualizer = VectorStoreVisualizer(session)
                
                # Get vector data
                vector_data = await visualizer.get_vector_data()
                
                if not vector_data['embeddings']:
                    return (
                        go.Figure(),
                        "No vectors found in the database. Please process some documents first."
                    )
                
                # Create visualization
                fig = visualizer.create_3d_visualization(vector_data)
                
                return (
                    fig,
                    "Visualization generated successfully! Hover over points to see document details."
                )
                
        except Exception as e:
            logger.error(f"Failed to generate visualization: {str(e)}", exc_info=True)
            return (
                go.Figure(),
                f"Error generating visualization: {str(e)}"
            )

    async def generate_visualization(self):
        """Generate vector store visualization"""
        return await self.visualizer.generate_visualization()

    # ... rest of ChatInterface methods ... 