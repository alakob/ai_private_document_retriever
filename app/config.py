from dataclasses import dataclass
from typing import Optional, Literal, List
import os
from dotenv import load_dotenv
import psutil

load_dotenv()

@dataclass
class RateLimitConfig:
    """Enhanced configuration for rate limiting."""
    max_requests_per_minute: int = 150
    max_concurrent_requests: int = 20  # New: limit concurrent requests
    batch_size: int = 50  # Reduced from 100
    min_delay: float = 0.1  # Minimum delay between requests
    max_delay: float = 30.0  # Maximum delay for backoff
    initial_delay: float = 1.0  # Starting delay for backoff
    backoff_factor: float = 2.0  # Exponential backoff multiplier
    jitter: float = 0.1  # Add randomness to prevent thundering herd
    max_retries: int = 3  # Add max_retries attribute
    retry_delay: float = 5.0  # Add retry_delay attribute

@dataclass
class ResourceConfig:
    """Configuration for resource monitoring."""
    memory_threshold_mb: int = 1000
    cpu_threshold_percent: float = 80.0
    min_free_memory_mb: int = 500

@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL connection."""
    connection_string: str
    collection_name: str = "documents"
    pre_delete_collection: bool = False
    embedding_dimension: int = 1536
    drop_existing: bool = True

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str = "text-embedding-ada-002"
    embedding_dimension: int = 1536
    batch_size: int = 100
    max_retries: int = 3
    timeout: int = 60
    max_tokens_per_batch: int = 8000

@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    chunk_size: int = int(os.getenv('CHUNK_SIZE', 1000))
    chunk_overlap: int = int(os.getenv('CHUNK_OVERLAP', 200))
    length_function: str = "token_count"  # or "character_count"

@dataclass
class FileProcessingConfig:
    """Configuration for file processing."""
    supported_extensions: List[str] = None
    max_file_size_mb: int = 50
    min_chunk_size: int = 50
    max_chunk_size: int = 2000
    
    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = [
                ".pdf", ".txt", ".md", ".docx", ".csv", 
                ".json", ".html", ".htm", ".xml"
            ]

@dataclass
class IndexConfig:
    """Configuration for vector indexes."""
    faiss_index_type: str = "Flat"  # or "HNSW", "IVF", etc.
    chroma_distance_function: str = "cosine"  # or "l2", "ip"
    postgres_index_type: str = "ivfflat"  # or "hnsw"
    postgres_lists: int = 100
    postgres_probes: int = 10

@dataclass
class ProcessorConfig:
    """Configuration for document processing."""
    # Basic configuration
    chunk_size: int = int(os.getenv('CHUNK_SIZE', 1000))
    chunk_overlap: int = int(os.getenv('CHUNK_OVERLAP', 200))
    vector_store_type: Literal["postgres", "faiss", "chroma"] = "postgres"
    postgres_config: Optional[PostgresConfig] = None
    persist_directory: Optional[str] = None
    batch_size: int = 10
    max_workers: int = 4
    db_pool_size: int = 5
    openai_api_key: str = os.getenv('OPENAI_API_KEY')
    
    # Sub-configurations
    rate_limit_config: RateLimitConfig = RateLimitConfig()
    resource_config: ResourceConfig = ResourceConfig()
    embedding_config: EmbeddingConfig = EmbeddingConfig()
    chunking_config: ChunkingConfig = None
    file_processing_config: FileProcessingConfig = FileProcessingConfig()
    index_config: IndexConfig = IndexConfig()
    
    # Docling configuration
    use_docling: bool = os.getenv('USE_DOCLING', '').lower() in ('true', '1', 'yes')
    docling_artifacts_path: Optional[str] = os.getenv('DOCLING_ARTIFACTS_PATH')
    docling_enable_remote: bool = os.getenv('DOCLING_ENABLE_REMOTE', '').lower() in ('true', '1', 'yes')
    docling_use_cache: bool = os.getenv('DOCLING_USE_CACHE', 'true').lower() in ('true', '1', 'yes')
    
    # Mistral OCR configuration
    use_mistral: bool = os.getenv('USE_MISTRAL', '').lower() in ('true', '1', 'yes')
    mistral_api_key: Optional[str] = os.getenv('MISTRAL_API_KEY')
    mistral_ocr_model: str = os.getenv('MISTRAL_OCR_MODEL', 'mistral-ocr-latest')
    mistral_include_images: bool = os.getenv('MISTRAL_INCLUDE_IMAGES', '').lower() in ('true', '1', 'yes')

    def __post_init__(self):
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        if self.chunking_config is None:
            self.chunking_config = ChunkingConfig(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

@dataclass
class ChatConfig:
    """Configuration for chat interface."""
    model: str = os.getenv('MODEL', 'gpt-4o-mini')
    server_name: str = os.getenv('SERVER_NAME', '0.0.0.0')
    server_port: int = int(os.getenv('GRADIO_SERVER_PORT', 7860))
    share: bool = os.getenv('GRADIO_SHARE', 'false').lower() in ('true', '1', 'yes')
    inbrowser: bool = True
    documents_dir: str = os.getenv('KNOWLEDGE_BASE_DIR', 'documents')
    retriever_k: int = 4
    retriever_score_threshold: float = 0.5
    temperature: float = 0.7
    max_tokens: int = 1000
    streaming: bool = True
    example_questions: List[str] = None
    
    def __post_init__(self):
        if self.example_questions is None:
            self.example_questions = [
                "What are the main topics covered in the documents?",
                "Can you summarize the key points in the documents?",
                "What are the best practices mentioned in the documents?"
            ]

@dataclass
class DatabaseConfig:
    """Configuration for database connection."""
    host: str = os.getenv('POSTGRES_HOST', 'localhost')
    port: int = int(os.getenv('POSTGRES_PORT', 5432))
    user: str = os.getenv('POSTGRES_USER', 'postgres')
    password: str = os.getenv('POSTGRES_PASSWORD', '')
    database: str = os.getenv('POSTGRES_DB', 'ragSystem')
    pool_size: int = int(os.getenv('POSTGRES_POOL_SIZE', 5))
    max_overflow: int = int(os.getenv('POSTGRES_MAX_OVERFLOW', 10))
    pool_timeout: int = int(os.getenv('POSTGRES_POOL_TIMEOUT', 30))
    pool_recycle: int = int(os.getenv('POSTGRES_POOL_RECYCLE', 1800))  # 30 minutes
    connect_retries: int = int(os.getenv('POSTGRES_CONNECT_RETRIES', 5))
    connect_retry_delay: int = int(os.getenv('POSTGRES_CONNECT_RETRY_DELAY', 5))
    
    @property
    def connection_string(self) -> str:
        # Docker-friendly connection string with expanded parameters
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class RedisConfig:
    """Configuration for Redis connection."""
    host: str = os.getenv('REDIS_HOST', 'localhost')
    port: int = int(os.getenv('REDIS_PORT', 6379))
    db: int = int(os.getenv('REDIS_DB', 0))
    password: str = os.getenv('REDIS_PASSWORD', '')
    timeout: int = 5
    enabled: bool = os.getenv('USE_REDIS_CACHE', '').lower() in ('true', '1', 'yes')
    use_redis_cache: bool = os.getenv('USE_REDIS_CACHE', '').lower() in ('true', '1', 'yes')  # Added for compatibility with embedding service
    cache_expiry: int = 24 * 60 * 60  # 24 hours in seconds
    
    @property
    def connection_url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

@dataclass
class VectorSearchConfig:
    """Configuration for vector similarity search."""
    top_k: int = 5
    score_threshold: float = 0.7
    pool_size: int = 5
    max_overflow: int = 10
    max_retries: int = 3
    base_delay: float = 1.0
    embedding_model: str = "text-embedding-ada-002"
    cache_expiry: int = 24 * 60 * 60  # 24 hours in seconds
    use_redis_cache: bool = False     # Whether to use Redis for caching (if available)
    
    # Add a method to create a connection string from database config
    @classmethod
    def get_connection_string(cls, db_config: DatabaseConfig) -> str:
        return db_config.connection_string

def get_optimal_workers() -> int:
    """Determine optimal number of workers based on system resources."""
    try:
        cpu_count = os.cpu_count() or 1
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Allocate 1 worker per 2GB of RAM, up to CPU count
        workers = min(
            cpu_count,
            max(1, int(memory_gb / 2)),
            10  # Maximum workers limit
        )
        return workers
    except Exception:
        return 4  # Default fallback


@dataclass
class PromptConfig:

    """Configuration for prompts used in the application."""
    # Document prompt template for formatting document content
    document_template: str = "{page_content}"

    # QA prompt template for answering questions based on context
    qa_template: str = """
    You are a knowledgeable AI assistant tasked with providing accurate, evidence-based answers.

    INSTRUCTIONS:
    1. Base your answer EXCLUSIVELY on the provided context
    2. If the context lacks sufficient information, clearly state what specific information is missing
    3. Always cite your sources
    4. Prioritize the most relevant information first
    5. Structure longer answers with clear headings or bullet points for readability
    6. Maintain a professional, objective tone
    7. Be detailed but concise


    CONTEXT:
    {context}

    CONVERSATION HISTORY:
    {chat_history}

    QUESTION:
    {question}

    ANSWER:
    """

    # Diagram generation prompt for creating mermaid diagrams from chat history
    diagram_template: str = """
    Create a visual representation of the conversation by analyzing the chat history and extracting key concepts and relationships.

    ANALYSIS INSTRUCTIONS:
    1. Identify 3-7 main topics or themes from the chat history
    2. Determine logical relationships between topics (causation, hierarchy, sequence, etc.)
    3. Extract critical insights, conclusions, or decision points
    4. Note any significant shifts in discussion direction
    5. Identify recurring themes or patterns

    DIAGRAM REQUIREMENTS:
    1. Choose the MOST APPROPRIATE diagram type:
    - Flowchart: For processes, sequences, or decision paths
    - Mindmap: For hierarchical concept exploration from central ideas
    - Entity-Relationship: For showing how different elements interact
    - Gantt: For timeline-based discussions
    - Class Diagram: For structural relationships

    2. Design principles:
    - Use distinctive node shapes for different concept types
    - Apply color coding consistently (max 4 colors)
    - Keep text labels concise (3-5 words max)
    - Ensure the diagram has clear entry and exit points if applicable
    - Limit diagram complexity to maximum 15-20 nodes

    3. Technical specifications:
    - Use proper mermaid syntax and indentation
    - Include appropriate direction (TD, LR, etc.) based on diagram complexity
    - Incorporate subgraphs to group related concepts
    - Use appropriate connector types (-->, ---|, -.->)
    - Add brief node descriptions when necessary

    OUTPUT INSTRUCTIONS:
    1. Provide ONLY the valid mermaid code block
    2. Ensure all syntax is correct and renderable

    Chat History:
    {chat_history}
    """


    # """Configuration for prompts used in the application."""
    
    # # Document prompt template for formatting document content
    # document_template: str = "{page_content}"
    
    # # QA prompt template for answering questions based on context
    # qa_template: str = """
    # You are a helpful AI assistant that answers questions based on the provided documents.
    
    # Instructions:
    # 1. Use ONLY the following context to answer the question
    # 2. If the context doesn't contain enough information, explain what's missing
    # 3. Always cite your sources
    # 4. Be detailed but concise
    
    # Context: {context}
    
    # Previous conversation:
    # {chat_history}
    
    # Question: {question}
    
    # Answer: """
    
    # # Diagram generation prompt for creating mermaid diagrams from chat history
    # diagram_template: str = """
    # Analyze the provided chat history and extract the key concepts, themes, and relationships between them. 
    # Use this information to generate a conceptual diagram that visually represents the structure and flow of ideas. 
    # The diagram should include:

    # Main Topics: Identify the primary subjects discussed.
    # Subtopics & Relationships: Show how different ideas are connected.
    # Key Insights: Highlight important takeaways from the conversation.
    # Flow & Interaction: Represent how ideas evolve over time, if applicable.
    
    # Ensure the diagram is clear, organized, and easy to understand. Use labeled nodes and arrows to indicate relationships. 
    # If possible, group related concepts for better visualization.

    # Format your response as a Mermaid diagram. Use one of the following diagram types that best represents the conversation:
    # 1. Flowchart (for sequential processes)
    # 2. Mindmap (for hierarchical relationships)
    # 3. Graph (for complex relationships)
    
    # Example Mermaid syntax for a flowchart:
    # ```mermaid
    # flowchart TD
    #   A[Main Topic] --> B[Subtopic 1]
    #   A --> C[Subtopic 2]
    #   B --> D[Detail 1]
    #   C --> E[Detail 2]
    # ```
    
    # Example Mermaid syntax for a mindmap:
    # ```mermaid
    # mindmap
    #   root((Main Topic))
    #     Subtopic 1
    #       Detail 1
    #       Detail 2
    #     Subtopic 2
    #       Detail 3
    # ```
    
    # Example Mermaid syntax for a graph:
    # ```mermaid
    # graph TD
    #   A[Topic 1] --- B[Topic 2]
    #   A --- C[Topic 3]
    #   B --- D[Topic 4]
    #   C --- D
    # ```
    
    # Use appropriate styling to differentiate between different types of nodes (main topics, subtopics, insights, etc.).
    # Only include the Mermaid diagram code in your response, nothing else.
    
    # Chat History:
    # {chat_history}
    # """

# Create instances of configs for easy import
embedding_config = EmbeddingConfig()
chunking_config = ChunkingConfig(
    chunk_size=int(os.getenv('CHUNK_SIZE', 1000)),
    chunk_overlap=int(os.getenv('CHUNK_OVERLAP', 200))
)
file_processing_config = FileProcessingConfig()
index_config = IndexConfig()

chat_config = ChatConfig()
db_config = DatabaseConfig()
redis_config = RedisConfig()
processor_config = ProcessorConfig(
    chunk_size=int(os.getenv('CHUNK_SIZE', 1000)),
    chunk_overlap=int(os.getenv('CHUNK_OVERLAP', 200)),
    max_workers=get_optimal_workers(),
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    embedding_config=embedding_config,
    chunking_config=chunking_config,
    file_processing_config=file_processing_config,
    index_config=index_config
)

# Configure vector search with Redis if enabled
use_redis_cache = redis_config.enabled
vector_search_config = VectorSearchConfig(use_redis_cache=use_redis_cache)

# Create prompt config
prompt_config = PromptConfig()