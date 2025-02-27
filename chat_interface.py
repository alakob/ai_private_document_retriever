import os
from typing import List, Dict, Any, Callable, Optional, Union, AsyncGenerator
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.memory import BaseMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from sqlalchemy import select, func
from sqlalchemy.sql import text
from dotenv import load_dotenv
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import trim_messages
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
import tiktoken
import asyncio
from pgvector.sqlalchemy import Vector
import numpy as np
import shutil
from pathlib import Path
import logging
import plotly.graph_objects as go
from vector_store_visualization import (
    VectorStoreDataAcquisition,
    VectorStoreVisualization,
    visualize_vector_store
)
from sqlalchemy.ext.asyncio import AsyncSession
import hashlib
from datetime import datetime

from config import chat_config, db_config, processor_config, PostgresConfig, ProcessorConfig

from document_rag_loader import (
    DocumentModel,
    DocumentChunk,
    PostgresConfig,
    async_sessionmaker,
    create_async_engine,
    OpenAIEmbeddings,
    DocumentProcessor,
    ProcessorConfig,
    calculate_file_checksum,
    check_duplicate_document,
    sanitize_text
)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostgresVectorRetriever(BaseRetriever, BaseModel):
    """Custom retriever for PostgreSQL vector store"""
    
    session_maker: Callable = Field(..., description="Async session maker for PostgreSQL")
    embeddings: Any = Field(..., description="Embeddings model")
    k: int = Field(default=chat_config.retriever_k, description="Number of documents to retrieve")
    score_threshold: float = Field(default=chat_config.retriever_score_threshold, description="Minimum similarity score threshold")
    
    class Config:
        arbitrary_types_allowed = True
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for a query using vector similarity search."""
        try:
            # Generate embedding for query
            query_embedding = await self.embeddings.aembed_query(query)
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.astype(np.float32).tolist()
            
            # Convert embedding to PostgreSQL vector string format
            query_embedding_str = f"[{','.join(map(str, query_embedding))}]"
            
            print(f"✓ Generated embedding for query: {query[:50]}...")
            
            async with self.session_maker() as session:
                # Ensure pgvector extension is installed
                await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                
                # SQL query for similarity search using cosine distance
                sql = text("""
                    WITH similarity_scores AS (
                        SELECT 
                            dc.content,
                            dc.chunk_index,
                            dc.chunk_size,
                            dc.page_number,
                            dm.filename,
                            dc.chunk_metadata,
                            1 - (dc.embedding <=> CAST(:embedding AS vector)) as similarity_score
                        FROM document_chunks dc
                        JOIN documents dm ON dc.document_id = dm.id
                        WHERE 1 - (dc.embedding <=> CAST(:embedding AS vector)) > :threshold
                        ORDER BY similarity_score DESC
                        LIMIT :limit
                    )
                    SELECT * FROM similarity_scores;
                """)
                
                # Execute search with parameters
                result = await session.execute(
                    sql,
                    {
                        "embedding": query_embedding_str,
                        "threshold": self.score_threshold,
                        "limit": self.k
                    }
                )
                
                # Process results into Documents
                documents = []
                for row in result:
                    doc = Document(
                        page_content=row.content,
                        metadata={
                            'source': row.filename,
                            'page': row.page_number,
                            'chunk_index': row.chunk_index,
                            'chunk_size': row.chunk_size,
                            'similarity_score': row.similarity_score,
                            **(row.chunk_metadata or {})
                        }
                    )
                    documents.append(doc)
                
                print(f"✓ Found {len(documents)} matching documents")
                return documents
                
        except Exception as e:
            print("\n=== Document Retrieval Error ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Traceback:")
            import traceback
            print(traceback.format_exc())
            return []
    
    def _get_relevant_documents(
        self, 
        query: str,
        *,
        runnable_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Synchronous method to get relevant documents"""
        raise NotImplementedError("This retriever only supports async operations")

class ChatInterface:
    """Handles chat interface and conversation"""
    
    def __init__(self):
        self.documents_dir = Path(chat_config.documents_dir)
        self.documents_dir.mkdir(exist_ok=True)
        
        # Create PostgreSQL connection string
        self.postgres_config = PostgresConfig(
            connection_string=db_config.connection_string,
            pre_delete_collection=False,
            drop_existing=False
        )
        
        # Create async engine and session
        self.engine = create_async_engine(
            self.postgres_config.connection_string,
            echo=False,
            pool_size=5,
            max_overflow=10
        )
        self.async_session = async_sessionmaker(self.engine, expire_on_commit=False)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-ada-002"
        )
        
        # Initialize retriever
        self.retriever = PostgresVectorRetriever(
            session_maker=self.async_session,
            embeddings=self.embeddings,
            k=chat_config.retriever_k,
            score_threshold=chat_config.retriever_score_threshold
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=chat_config.model,
            temperature=chat_config.temperature,
            max_tokens=chat_config.max_tokens,
            streaming=chat_config.streaming
        )
        
        # Initialize chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True
        )
        
        # Initialize system message
        self.system_message = SystemMessage(
            content="You are a helpful assistant that answers questions based on the provided documents."
        )
        
        # Define prompt templates
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

        # Initialize conversation chain with correct parameters and prompts
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
        
        # Initialize conversation history
        self.messages = [self.system_message]
        
        # Initialize tokenizer and set max tokens
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.max_token_limit = 4000  # Adjust based on your needs
        
        # Add debug logging
        self.debug = True
        
        # Test database connection at initialization
        if self.debug:
            asyncio.create_task(self.test_db_connection())
    
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
    
    async def chat(self, message: str, history: List[Dict[str, str]]) -> Dict[str, str]:
        """Process chat message and return response"""
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
    
    async def cleanup(self):
        """Cleanup resources properly"""
        try:
            # Close any active sessions
            if hasattr(self, 'async_session'):
                async with self.async_session() as session:
                    await session.close()
            
            # Dispose of the engine
            if hasattr(self, 'engine'):
                await self.engine.dispose()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def test_db_connection(self):
        """Test database connection and document retrieval"""
        try:
            async with self.async_session() as session:
                # Try to get one document
                result = await session.execute(
                    select(DocumentModel).limit(1)
                )
                doc = result.scalar_one_or_none()
                if doc:
                    print("\n=== 🔍 Database Connection Test ===")
                    print("✅ Successfully connected to database")
                    print(f"📄 Found document: {doc.filename}")
                    
                    # Test vector retrieval
                    test_query = "test query"
                    docs = await self.retriever._aget_relevant_documents(
                        test_query
                    )
                    print(f"✅ Vector retrieval test: found {len(docs)} documents")
                    return True
                else:
                    print("⚠️ No documents found in database")
                    return False
        except Exception as e:
            print("\n=== ❌ Database Connection Error ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            print("Traceback:")
            print(traceback.format_exc())
            return False

    async def handle_file_upload(self, files: List[str]) -> str:
        """Handle initial file upload to temporary storage"""
        try:
            logger.info(f"\n📤 Received {len(files)} files")
            
            html_output = ["<div>"]
            html_output.append(f"<b>📤 Received {len(files)} files</b><br>")
            
            successful_uploads = []
            for file_path in files:
                try:
                    # Convert to Path object
                    source_path = Path(file_path)
                    
                    # Create destination path in documents directory
                    dest_path = self.documents_dir / source_path.name
                    
                    # Handle duplicate filenames
                    counter = 1
                    while dest_path.exists():
                        stem = source_path.stem
                        suffix = source_path.suffix
                        new_name = f"{stem}_{counter}{suffix}"
                        dest_path = self.documents_dir / new_name
                        counter += 1
                    
                    logger.info(f"Moving {source_path} to {dest_path}")
                    
                    # Move file to documents directory
                    shutil.move(str(source_path), str(dest_path))
                    
                    successful_uploads.append(dest_path.name)
                    html_output.append(f"✅ Successfully moved <span style='color: green;'>{dest_path.name}</span><br>")
                    logger.info(f"✅ Successfully moved {dest_path.name}")
                    
                except Exception as e:
                    html_output.append(f"❌ Error processing <span style='color: red;'>{file_path}</span>: {str(e)}<br>")
                    logger.error(f"❌ Error processing {file_path}: {str(e)}")
                    continue

            if successful_uploads:
                html_output.append("<br><b>Successfully uploaded files:</b><br>")
                for f in successful_uploads:
                    html_output.append(f"- <span style='color: green;'>{f}</span><br>")
                html_output.append("<br><b>Click 'Process Documents' to proceed with document processing.</b>")
            else:
                html_output.append("<br><span style='color: red;'>No files were successfully uploaded</span>")
            
            html_output.append("</div>")
            return "".join(html_output)
            
        except Exception as e:
            logger.error(f"❌ Upload error: {str(e)}", exc_info=True)
            return f"<div style='color: red;'>Error uploading files: {str(e)}</div>"

    def calculate_file_checksum(self, file_path: str) -> str:
        """
        Calculate SHA-256 checksum for a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hexadecimal digest of the SHA-256 hash
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                # Read the file in chunks to handle large files efficiently
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"Error calculating checksum for {file_path}: {e}")
            return None

    async def process_single_file(self, file_path: Path, processor_config: ProcessorConfig) -> Optional[List[Document]]:
        """Process a single file with document processor."""
        try:
            print(f"\n=== Processing File: {file_path.name} ===")
            print(f"Checksum: {self.calculate_file_checksum(str(file_path))}")
            
            # Initialize processor for embeddings
            processor = DocumentProcessor(processor_config)
            
            # Load and process the document to get chunks and embeddings
            documents = await processor._load_document(str(file_path))
            if not documents:
                print(f"⚠️ No content extracted from {file_path.name}")
                return None
            
            # Process in batches like document_rag_loader
            all_chunks = []
            
            # Calculate checksum once for the file
            checksum = self.calculate_file_checksum(str(file_path))
            
            # Check for duplicate file first
            async with self.async_session() as session:
                result = await session.execute(
                    select(DocumentModel).where(DocumentModel.checksum == checksum)
                )
                if result.scalar_one_or_none():
                    print(f"⏩ Document {file_path.name} already exists with same checksum")
                    return None
            
            # Process document in batches
            for i in range(0, len(documents), processor_config.batch_size):
                batch = documents[i:i + processor_config.batch_size]
                
                # Get text content
                texts = [doc.page_content for doc in batch]
                
                # Get embeddings
                embeddings = await processor._get_embeddings_batch(texts)
                
                # Validate chunks before database operations
                valid_chunks = []
                valid_embeddings = []
                
                for idx, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                    content = sanitize_text(chunk.page_content)
                    if not content:
                        print(f"Skipping chunk {idx}: Empty or invalid content")
                        continue
                    
                    chunk.page_content = content
                    valid_chunks.append(chunk)
                    valid_embeddings.append(embedding)
                
                if not valid_chunks:
                    print(f"No valid chunks found in batch")
                    continue
                
                # Process database operations in a single transaction
                async with self.async_session() as session:
                    try:
                        async with session.begin():
                            # Final duplicate check before insert
                            result = await session.execute(
                                select(DocumentModel).where(DocumentModel.checksum == checksum)
                            )
                            if result.scalar_one_or_none():
                                print(f"⚠️ Race condition detected! Document was inserted by another process")
                                return None
                            
                            # Create document record if this is the first batch
                            if not all_chunks:  # Only create document on first batch
                                doc = DocumentModel(
                                    filename=file_path.name,
                                    checksum=checksum,
                                    doc_metadata={
                                        'source': file_path.name,
                                        'file_path': str(file_path),
                                        'processed_at': datetime.now().isoformat()
                                    }
                                )
                                session.add(doc)
                                await session.flush()
                                doc_id = doc.id
                            
                            # Create chunk records
                            for idx, (chunk, embedding) in enumerate(zip(valid_chunks, valid_embeddings)):
                                chunk_record = DocumentChunk(
                                    document_id=doc_id,
                                    content=chunk.page_content,
                                    embedding=embedding,
                                    chunk_index=len(all_chunks) + idx,  # Global index across all batches
                                    chunk_size=len(chunk.page_content),
                                    page_number=chunk.metadata.get('page', None),
                                    section_title=chunk.metadata.get('section_title', None),
                                    chunk_metadata={
                                        **chunk.metadata,
                                        'processed_at': datetime.now().isoformat(),
                                        'embedding_model': 'text-embedding-ada-002',
                                        'batch_index': i // processor_config.batch_size
                                    }
                                )
                                session.add(chunk_record)
                        
                        print(f"✅ Stored batch with {len(valid_chunks)} chunks")
                        all_chunks.extend(valid_chunks)
                        
                    except Exception as e:
                        print(f"❌ Transaction error: {str(e)}")
                        raise
            
            if all_chunks:
                print(f"✅ Successfully processed {file_path.name} with {len(all_chunks)} total chunks")
            return all_chunks

        except Exception as e:
            print(f"❌ Error processing file {file_path}: {str(e)}")
            logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
            raise

    async def process_documents(self) -> str:
        """Process the uploaded documents with progress tracking"""
        try:
            logger.info("Starting document processing...")
            print("\n=== Document Processing Started ===")
            
            # Initialize database tables if needed
            async with self.async_session() as session:
                # Create extensions if they don't exist
                await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                await session.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto;"))
                await session.commit()
                
                # Create index if it doesn't exist
                try:
                    await session.execute(
                        text("CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_checksum ON documents (checksum);")
                    )
                    await session.commit()
                    logger.info("Tables and indices created successfully")
                except Exception as e:
                    logger.warning(f"Index creation warning (can be ignored if already exists): {str(e)}")
            
            # Get all files in documents directory
            all_files = list(self.documents_dir.glob('*.*'))
            total_files = len(all_files)
            print(f"Found {total_files} files to process")
            
            # Process files with checksum verification
            total_chunks = 0
            processed_files = []
            skipped_files = []
            error_files = []
            
            # First, get all existing checksums in a separate session
            existing_checksums = {}
            existing_filenames = {}
            
            async with self.async_session() as session:
                result = await session.execute(select(DocumentModel))
                existing_docs = result.scalars().all()
                
                for doc in existing_docs:
                    existing_checksums[doc.checksum] = doc.id
                    existing_filenames[doc.filename] = doc.checksum
            
            print(f"\n=== Existing Documents in Database ===")
            print(f"Found {len(existing_checksums)} documents in database")
            print("All checksums in database:")
            for checksum, doc_id in existing_checksums.items():
                print(f"ID: {doc_id} | Checksum: {checksum[:8]}...")
            
            # Create processor config
            postgres_config = PostgresConfig(
                connection_string=self.postgres_config.connection_string,
                pre_delete_collection=False,
                drop_existing=False
            )
            
            processor_config = ProcessorConfig(
                chunk_size=500,
                chunk_overlap=50,
                vector_store_type="postgres",
                postgres_config=postgres_config,
                batch_size=50,
                max_workers=1,
                db_pool_size=5,
                openai_api_key=OPENAI_API_KEY
            )
            
            # Process each file
            for file_path in all_files:
                try:
                    # Calculate checksum
                    file_checksum = calculate_file_checksum(str(file_path))
                    print(f"\n===== Checking file: {file_path.name} =====")
                    print(f"Calculated checksum: {file_checksum}")
                    
                    # Check if file exists in database by checksum
                    checksum_exists = file_checksum in existing_checksums
                    print(f"Checksum exists in database? {checksum_exists}")
                    
                    if checksum_exists:
                        print(f"⏩ Skipping {file_path.name} - found matching checksum in database")
                        logger.info(f"Skipping {file_path.name} - already exists with checksum {file_checksum}")
                        skipped_files.append(file_path.name)
                        continue
                    
                    # Process file
                    print(f"🔄 Processing {file_path.name}...")
                    logger.info(f"Processing {file_path.name}")
                    chunks = await self.process_single_file(file_path, processor_config)
                    
                    if chunks:
                        total_chunks += len(chunks)
                        processed_files.append(file_path.name)
                        print(f"✅ Processed {file_path.name} - created {len(chunks)} chunks")
                        
                        # Update our local cache
                        existing_checksums[file_checksum] = True
                        existing_filenames[file_path.name] = file_checksum
                    
                except Exception as e:
                    print(f"❌ Error processing {file_path.name}: {str(e)}")
                    logger.error(f"Error processing {file_path.name}: {str(e)}")
                    error_files.append(file_path.name)
                    continue
            
            # Create status message with HTML formatting
            status_message = []
            status_message.append("<div style='padding: 10px; background-color: #1a1a1a; border-radius: 5px;'>")
            status_message.append("<h3 style='text-align: center; color: #ffffff;'>=== Processing Summary ===</h3>")
            status_message.append(f"<p><b>Total files:</b> {len(all_files)}</p>")
            
            if processed_files:
                status_message.append(f"<p><b>Successfully processed files:</b> {len(processed_files)}</p>")
                status_message.append("<ul>")
                for file in processed_files:
                    status_message.append(f"<li style='color: #00cc00;'>{file}</li>")
                status_message.append("</ul>")
                status_message.append(f"<p><b>Total chunks created:</b> {total_chunks}</p>")
            
            if skipped_files:
                status_message.append(f"<p><b>Skipped files:</b> {len(skipped_files)}</p>")
                status_message.append("<ul>")
                for file in skipped_files:
                    status_message.append(f"<li style='color: orange;'>{file}</li>")
                status_message.append("</ul>")
            
            if error_files:
                status_message.append(f"<p><b>Files with errors:</b> {len(error_files)}</p>")
                status_message.append("<ul>")
                for file in error_files:
                    status_message.append(f"<li style='color: #ff3333;'>{file}</li>")
                status_message.append("</ul>")
                status_message.append("<p>Check logs for detailed error information.</p>")
            
            status_message.append("</div>")
            return "".join(status_message)
            
        except Exception as e:
            error_message = f"<div style='color: red; padding: 10px;'>Error processing documents: {str(e)}</div>"
            logger.error(error_message, exc_info=True)
            return error_message

    async def stream_chat(self, message: str, history: List[Dict[str, str]]) -> AsyncGenerator[Dict[str, str], None]:
        """Process chat message and stream response"""
        try:
            # Add debug flag to track streaming
            streaming_debug = True
            
            if streaming_debug:
                print(f"\n🔄 Starting streaming response for: {message}")
            
            if self.debug:
                print(f"\n📝 Question: {message}")
                print(f"📚 History length: {len(history) if history else 0}")
            
            # Convert history to messages format if needed
            if not self.messages or len(self.messages) <= 1:  # Only system message or empty
                self.messages = [self.system_message]
                
                # Convert history to messages if available
                if history:
                    for h_msg, a_msg in history:
                        self.messages.append(HumanMessage(content=h_msg))
                        self.messages.append(AIMessage(content=a_msg))
            
            # Add user message
            user_message = HumanMessage(content=message)
            self.messages.append(user_message)
            
            # Trim history if needed
            self._trim_history()
            
            # Format chat history as tuples for the prompt
            chat_history = []
            messages = self.messages[1:]  # Skip system message
            for i in range(0, len(messages)-1, 2):
                if i+1 < len(messages):
                    human_msg = messages[i]
                    ai_msg = messages[i+1]
                    if isinstance(human_msg, HumanMessage) and isinstance(ai_msg, AIMessage):
                        chat_history.append((human_msg.content, ai_msg.content))
            
            # Prepare for streaming - get relevant documents
            docs = await self.retriever._aget_relevant_documents(message)
            
            if self.debug:
                print("\n📄 Retrieved Documents:")
                for i, doc in enumerate(docs):
                    print(f"  📑 Doc {i+1}: {doc.metadata['source']} (Page: {doc.metadata.get('page', 'N/A')})")
                    print(f"  📝 Content: {doc.page_content[:100]}...")
            
            # Format documents for context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create a streaming LLM
            streaming_llm = ChatOpenAI(
                model=chat_config.model,
                temperature=chat_config.temperature,
                max_tokens=chat_config.max_tokens,
                streaming=True
            )
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_template("""
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
            
            # Format chat history for prompt
            formatted_chat_history = ""
            for human, ai in chat_history:
                formatted_chat_history += f"Human: {human}\nAI: {ai}\n\n"
            
            # Create the chain
            chain = prompt | streaming_llm
            
            # Start streaming response
            full_response = ""
            
            # Invoke the chain with streaming
            async for chunk in chain.astream({
                "context": context,
                "chat_history": formatted_chat_history,
                "question": message
            }):
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    full_response += content
                    
                    if streaming_debug and len(content) > 0:
                        print(f"🔤 Streaming chunk: {content[:20]}...")
                    
                    # Yield the current state of the response
                    yield {"role": "assistant", "content": full_response}
            
            if streaming_debug:
                print(f"✅ Streaming complete, final length: {len(full_response)}")
            
            # After streaming is complete, add sources information
            if docs:
                unique_sources = {
                    doc.metadata['source']: doc.metadata.get('page', 'N/A') 
                    for doc in docs
                }
                sources = [
                    f"Source: {source} (Page: {page})" 
                    for source, page in unique_sources.items()
                ]
                final_response = f"{full_response}\n\nSources:\n" + "\n".join(sources)
                
                # Yield the final response with sources
                yield {"role": "assistant", "content": final_response}
                
                # Create assistant message for history
                assistant_message = AIMessage(content=final_response)
                self.messages.append(assistant_message)
            else:
                # Create assistant message for history
                assistant_message = AIMessage(content=full_response)
                self.messages.append(assistant_message)
                
                # Yield the final response
                yield {"role": "assistant", "content": full_response}
            
        except Exception as e:
            import traceback
            print(f"❌ Error: {str(e)}")
            print(traceback.format_exc())
            yield {"role": "assistant", "content": f"Error processing question: {str(e)}"}

    async def check_database_integrity(self):
        """Check database tables for data integrity"""
        try:
            async with self.async_session() as session:
                # Check documents table
                print("\n=== Checking Documents Table ===")
                result = await session.execute(select(DocumentModel))
                documents = result.scalars().all()
                print(f"Total documents: {len(documents)}")
                
                for doc in documents:
                    print(f"\nDocument ID: {doc.id}")
                    print(f"Filename: {doc.filename}")
                    print(f"Checksum: {doc.checksum}")
                    print(f"Metadata: {doc.doc_metadata}")
                    
                    # Check associated chunks
                    chunk_result = await session.execute(
                        select(DocumentChunk)
                        .where(DocumentChunk.document_id == doc.id)
                    )
                    chunks = chunk_result.scalars().all()
                    print(f"Number of chunks: {len(chunks)}")
                    
                    if chunks:
                        # Check first and last chunk
                        print("\nFirst chunk details:")
                        print(f"Content length: {len(chunks[0].content)}")
                        print(f"Embedding dimension: {len(chunks[0].embedding)}")
                        print(f"Metadata: {chunks[0].chunk_metadata}")
                        
                        print("\nLast chunk details:")
                        print(f"Content length: {len(chunks[-1].content)}")
                        print(f"Embedding dimension: {len(chunks[-1].embedding)}")
                        print(f"Metadata: {chunks[-1].chunk_metadata}")
                    
                # Check for orphaned chunks
                print("\n=== Checking for Orphaned Chunks ===")
                orphaned_result = await session.execute(
                    select(DocumentChunk)
                    .outerjoin(DocumentModel)
                    .where(DocumentModel.id == None)
                )
                orphaned_chunks = orphaned_result.scalars().all()
                if orphaned_chunks:
                    print(f"Warning: Found {len(orphaned_chunks)} orphaned chunks!")
                else:
                    print("No orphaned chunks found.")
                
                return True
                
        except Exception as e:
            print(f"Error checking database: {str(e)}")
            logger.error(f"Database check error: {str(e)}", exc_info=True)
            return False

async def main():
    """Main function to run the chat interface"""
    chat_interface = None
    try:
        chat_interface = ChatInterface()
        
        with gr.Blocks() as demo:
            gr.Markdown("""
                <div style="text-align: center; margin-bottom: 20px;">
                    <h1>Document Q&A System</h1>
                    <p>Ask questions about your documents. The system will search through the document collection and provide relevant answers.</p>
                </div>
            """)
            
            # Document Upload Accordion
            with gr.Accordion("Document Upload/Processing", open=False):
                upload_button = gr.UploadButton(
                    "Click to Upload Files",
                    variant="huggingface",
                    size="sm",
                    file_types=[".pdf", ".txt", ".doc", ".docx"],
                    file_count="multiple"
                )
                
                with gr.Row():
                    process_btn = gr.Button("Process Documents", variant="primary", size="sm")
                    clear_btn = gr.Button("Clear", variant="stop", size="sm")
                
                progress_box = gr.HTML(
                    label="Status",
                    value="<div style='padding: 10px;'>Upload files and click Process to begin...</div>",
                    elem_id="progress-box"
                )
                
            # Add custom CSS for the progress box
            gr.Markdown("""
                <style>
                    /* Style for the progress box */
                    #progress-box {
                        min-height: 250px;
                        max-height: 500px;
                        overflow-y: auto;
                        background-color: #1a1a1a;
                        border-radius: 4px;
                        margin-top: 10px;
                        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    }
                    
                    /* Style for lists in the progress box */
                    #progress-box ul {
                        margin-top: 5px;
                        margin-bottom: 15px;
                        padding-left: 25px;
                    }
                    
                    #progress-box li {
                        margin-bottom: 3px;
                    }
                    
                    /* Make sure the progress bar is visible */
                    .progress-container {
                        width: 100% !important;
                        height: 20px !important;
                        background-color: #333 !important;
                        border-radius: 10px !important;
                        margin: 10px 0 !important;
                        overflow: hidden !important;
                        display: block !important;
                    }
                    
                    .progress-bar {
                        height: 100% !important;
                        background-color: #ff5500 !important;
                        border-radius: 10px !important;
                        transition: width 0.3s ease !important;
                        display: block !important;
                    }
                    
                    /* Ensure Gradio's progress bar is visible */
                    .gradio-container .progress {
                        display: block !important;
                        visibility: visible !important;
                        opacity: 1 !important;
                    }
                </style>
            """)

            # Create a container for the chat interface
            with gr.Column():
                chat = gr.Chatbot()
                msg = gr.Textbox(label="Message")
                with gr.Row():
                    submit = gr.Button("Submit", variant="huggingface", size="sm")
                    clear = gr.Button("Clear", variant="stop", size="sm")

                # Add example questions
                gr.Examples(
                    examples=chat_config.example_questions,
                    inputs=msg
                )

            # Vector Visualization Accordion
            with gr.Accordion("Vector Store Visualization", open=False):
                with gr.Row():
                    visualize_btn = gr.Button("Generate Visualization", variant="huggingface", size="sm")
                    plot_status = gr.Textbox(
                        label="Visualization Status",
                        placeholder="Click Generate Visualization to begin...",
                        interactive=False
                    )
                
                # Center the plot using a container with CSS
                with gr.Column(elem_classes="center-plot"):
                    # Plotly output component
                    plot_output = gr.Plot(
                        label="Vector Store Visualization",
                        show_label=True,
                        container=True,
                        elem_classes="plot-container"
                    )

            # Add custom CSS for centering
            gr.Markdown("""
                <style>
                    /* Center the plot container */
                    .center-plot {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        width: 100%;
                        min-height: 700px;  /* Add minimum height */
                    }
                    
                    /* Style the plot container */
                    .plot-container {
                        width: 100%;
                        max-width: 1200px;
                        margin: 0 auto;
                    }
                    
                    /* Center the Plotly plot itself */
                    .plot-container > div {
                        display: flex;
                        justify-content: center;
                        width: 100%;
                    }
                    
                    /* Ensure the SVG is centered */
                    .js-plotly-plot {
                        margin: 0 auto !important;
                    }
                    
                    /* Add responsive behavior */
                    @media (max-width: 768px) {
                        .plot-container {
                            max-width: 100%;
                            padding: 10px;
                        }
                    }
                </style>
            """)

            # Event handlers
            async def chat_response(message, history):
                history = history or []
                
                # Add user message to history immediately
                history.append((message, ""))
                
                # Create a generator for streaming responses
                try:
                    # First yield the history with empty response to show user message immediately
                    yield history
                    
                    # Get the full response (non-streaming first to debug)
                    full_response = ""
                    
                    # Stream the response
                    async for partial_response in chat_interface.stream_chat(message, history[:-1]):
                        # Update the last assistant message with the current content
                        full_response = partial_response['content']
                        history[-1] = (message, full_response)
                        
                        # Yield the updated history
                        yield history
                    
                except Exception as e:
                    logger.error(f"Error in chat response: {str(e)}", exc_info=True)
                    history[-1] = (message, f"Error: {str(e)}")
                    yield history

            async def process_documents():
                try:
                    logger.info("Starting document processing...")
                    print("\n=== Document Processing Started ===")
                    
                    # Fix: Create config objects first
                    postgres_config = PostgresConfig(
                        connection_string=chat_interface.postgres_config.connection_string,
                        pre_delete_collection=False,
                        drop_existing=False
                    )
                    
                    processor_config = ProcessorConfig(
                        chunk_size=500,
                        chunk_overlap=50,
                        vector_store_type="postgres",
                        postgres_config=postgres_config,
                        batch_size=50,
                        max_workers=8,
                        db_pool_size=20,
                        openai_api_key=OPENAI_API_KEY
                    )
                    
                    # Initialize document processor with config object
                    processor = DocumentProcessor(config=processor_config)
                    
                    # Initialize database
                    await processor.initialize_database()
                    
                    # Get all files in documents directory
                    all_files = list(chat_interface.documents_dir.glob('*.*'))
                    print(f"Found {len(all_files)} files to process")
                    
                    # Process files with checksum verification
                    total_chunks = 0
                    processed_files = []
                    skipped_files = []
                    
                    # First, get all existing checksums in a separate session
                    existing_checksums = {}
                    existing_filenames = {}
                    
                    async with chat_interface.async_session() as session:
                        # Get all existing documents
                        result = await session.execute(select(DocumentModel))
                        existing_docs = result.scalars().all()
                        
                        for doc in existing_docs:
                            existing_checksums[doc.checksum] = doc.id
                            existing_filenames[doc.filename] = doc.checksum
                        
                    print(f"\n=== Existing Documents in Database ===")
                    print(f"Found {len(existing_checksums)} documents in database")
                    
                    # Now process each file individually without an outer session
                    for file_path in all_files:
                        try:
                            # Calculate checksum
                            file_checksum = chat_interface.calculate_file_checksum(str(file_path))
                            print(f"\n===== Checking file: {file_path.name} =====")
                            print(f"Calculated checksum: {file_checksum}")
                            
                            # Check if file exists in database by checksum
                            checksum_exists = file_checksum in existing_checksums
                            print(f"Checksum exists in database? {checksum_exists}")
                            
                            if checksum_exists:
                                print(f"⏩ Skipping {file_path.name} - found matching checksum in database")
                                logger.info(f"Skipping {file_path.name} - already exists with checksum {file_checksum}")
                                skipped_files.append(file_path.name)
                                continue
                            
                            # Also check by filename as a fallback
                            if file_path.name in existing_filenames:
                                db_checksum = existing_filenames[file_path.name]
                                print(f"⚠️ File {file_path.name} exists in DB but with different checksum!")
                                print(f"  DB checksum: {db_checksum[:8]}...")
                                print(f"  File checksum: {file_checksum[:8]}...")
                                # Skip if filename matches but checksum doesn't
                                skipped_files.append(file_path.name)
                                continue
                            
                            # Process file - let the processor handle its own session
                            print(f"🔄 Processing {file_path.name}...")
                            logger.info(f"Processing {file_path.name}")
                            chunks = await chat_interface.process_single_file(file_path, processor_config)
                            
                            if chunks:
                                total_chunks += len(chunks)
                                processed_files.append(file_path.name)
                                print(f"✅ Processed {file_path.name} - created {len(chunks)} chunks")
                                
                                # Update our local cache of checksums and filenames
                                existing_checksums[file_checksum] = True
                                existing_filenames[file_path.name] = file_checksum
                            
                        except Exception as e:
                            print(f"❌ Error processing {file_path.name}: {str(e)}")
                            logger.error(f"Error processing {file_path.name}: {str(e)}")
                            continue
                    
                    # Create status message
                    status_message = []
                    status_message.append(f"=== Processing Summary ===")
                    status_message.append(f"Total files: {len(all_files)}")
                    status_message.append(f"Skipped files: {len(skipped_files)}")
                    status_message.append(f"Processed files: {len(processed_files)}")
                    status_message.append(f"Total chunks created: {total_chunks}")
                    
                    print("\n".join(status_message))
                    return "\n".join(status_message)
                    
                except Exception as e:
                    error_message = f"Error processing documents: {str(e)}"
                    logger.error(error_message, exc_info=True)
                    return error_message

            async def generate_visualization():
                try:
                    plot_status.value = "Generating visualization... This may take a while."
                    
                    # Create async session
                    async with chat_interface.async_session() as session:
                        # Generate visualization
                        fig = await visualize_vector_store(session)
                        
                        return (
                            fig,
                            "Visualization generated successfully! Hover over points to see document details."
                        )
                except Exception as e:
                    error_msg = f"Error generating visualization: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    return (
                        None,
                        error_msg
                    )

            def clear_upload():
                return "<div style='padding: 10px;'>Upload files and click Process to begin...</div>"

            # Connect the chat components with proper Gradio streaming configuration
            submit.click(
                chat_response,
                inputs=[msg, chat],
                outputs=chat,
                api_name="submit",
            ).then(
                lambda: "",  # Clear the message box after submission
                None,
                msg
            )

            # Also enable streaming for the Enter key
            msg.submit(
                chat_response,
                inputs=[msg, chat],
                outputs=chat,
                api_name="submit_msg",
            ).then(
                lambda: "",  # Clear the message box after submission
                None,
                msg
            )

            clear.click(lambda: None, None, chat)

            # Connect file upload components
            upload_button.upload(
                fn=chat_interface.handle_file_upload,
                inputs=upload_button,
                outputs=progress_box,
            )
            
            process_btn.click(
                fn=chat_interface.process_documents,
                inputs=None,
                outputs=progress_box,
                show_progress=True,
            )
            
            clear_btn.click(
                fn=clear_upload,
                inputs=None,
                outputs=progress_box,
            )
            
            # Connect visualization button
            visualize_btn.click(
                fn=generate_visualization,
                inputs=None,
                outputs=[plot_output, plot_status],
            )

        await demo.launch(
            server_name=chat_config.server_name,
            share=chat_config.share,
            inbrowser=chat_config.inbrowser
        )
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise
    finally:
        if chat_interface:
            await chat_interface.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application failed: {str(e)}") 