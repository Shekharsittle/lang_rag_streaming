"""
endpoints.py - API endpoints for the Streaming RAG service
"""

import logging
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Depends, Header, Query
from fastapi.responses import StreamingResponse

from app.models.schemas import QueryRequest
from app.services.lang_rag import get_rag_system, LangChainRAG

# Set up logger
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query")
async def query_endpoint(
    request: QueryRequest,
    rag_system: LangChainRAG = Depends(get_rag_system),
    x_thread_id: str = Header(None)
) -> StreamingResponse:
    """
    Process a user query, retrieve relevant contexts, and stream the generated response.
    
    Args:
        request: The query request containing the question and top_k parameter
        rag_system: The RAG system instance (injected via dependency)
        x_thread_id: Optional thread ID header for maintaining conversation state
        
    Returns:
        A streaming response with the generated answer
    """
    logger.info(f"Received query: {request.query} with top_k={request.top_k}")
    
    try:
        # Set up streaming response using the RAG system
        return StreamingResponse(
            rag_system.stream_query(
                request.query, 
                top_k=request.top_k,
                thread_id=x_thread_id
            ),
            media_type="text/plain",
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing your request: {str(e)}")


# @router.post("/query/complete")
# async def complete_query_endpoint(
#     request: QueryRequest,
#     rag_system: LangChainRAG = Depends(get_rag_system),
#     x_thread_id: str = Header(None)
# ):
#     """
#     Process a user query and return the complete result (non-streaming).
    
#     This endpoint is useful for clients that don't support streaming responses.
    
#     Args:
#         request: The query request containing the question and top_k parameter
#         rag_system: The RAG system instance (injected via dependency)
#         x_thread_id: Optional thread ID header for maintaining conversation state
        
#     Returns:
#         The complete result including question, retrieved context, and answer
#     """
#     logger.info(f"Received complete query: {request.query} with top_k={request.top_k}")
    
#     try:
#         # Process the query through the RAG system
#         result = await rag_system.query(
#             request.query, 
#             top_k=request.top_k,
#             thread_id=x_thread_id
#         )
        
#         # Return the complete result
#         return {
#             "question": result["question"],
#             "context_count": len(result.get("context", [])),
#             "answer": result.get("answer", "No answer generated")
#         }
    
#     except Exception as e:
#         logger.error(f"Error processing complete query: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error processing your request: {str(e)}")