"""
config.py - Configuration settings for the application
"""

import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings.
    """
    # API settings
    PROJECT_NAME: str = "Streaming RAG API"
    PROJECT_DESCRIPTION: str = "A lightweight Streaming Retrieval-Augmented Generation (RAG) API service"
    VERSION: str = "0.1.0"
    API_PREFIX: str = ""
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # LLM settings
    LLM_MODEL_NAME: str = "gpt-4o-mini"
    MODEL_PROVIDER: str = "openai"
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    
    # Vector store settings
    CHROMA_DB_PATH: str = "./chroma_langchain_db"
    
    # Embedding model settings
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()