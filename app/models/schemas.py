"""
schemas.py - Pydantic models for request and response validation
"""

from pydantic import BaseModel, Field, validator


class QueryRequest(BaseModel):
    """
    Request schema for the query endpoint.
    """
    query: str = Field(..., description="The user query")
    top_k: int = Field(3, description="Number of contexts to retrieve (optional, default: 3)")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v) > 1000:
            raise ValueError("Query too long, maximum 1000 characters")
        return v
    
    @validator('top_k')
    def validate_top_k(cls, v):
        if v < 1:
            raise ValueError("top_k must be at least 1")
        if v > 10:
            raise ValueError("top_k must be at most 10")
        return v


class Document(BaseModel):
    """
    Represents a document in the vector store.
    """
    content: str = Field(..., description="Content of the document")
    metadata: dict = Field(default_factory=dict, description="Metadata for the document")


class StreamChunk(BaseModel):
    """
    Represents a single chunk of the streamed response.
    """
    text: str = Field(..., description="Text chunk of the response")