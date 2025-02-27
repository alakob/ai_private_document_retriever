"""
Embeddings service for generating vector embeddings.
"""

import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def get_embeddings():
    """Get OpenAI embeddings instance."""
    return OpenAIEmbeddings(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model="text-embedding-ada-002"
    ) 