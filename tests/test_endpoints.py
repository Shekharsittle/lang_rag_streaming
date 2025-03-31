import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import asyncio

from app.main import app
from app.services.lang_rag import LangChainRAG


@pytest.fixture
def test_client():
    return TestClient(app)


class MockRAGSystem:
    """Mock RAG system with controlled streaming behavior"""
    
    async def stream_query(self, query, top_k=3, thread_id=None):
        """Mock implementation of stream_query method"""
        # Record the call parameters for later assertion
        self.called_with = {"query": query, "top_k": top_k, "thread_id": thread_id}
        
        # Yield chunks with a small delay to simulate streaming
        yield "This "
        await asyncio.sleep(0.01)
        yield "is "
        await asyncio.sleep(0.01)
        yield "a "
        await asyncio.sleep(0.01)
        yield "streaming "
        await asyncio.sleep(0.01)
        yield "response."


@pytest.mark.asyncio
async def test_query_endpoint_streaming(test_client):
    """Test that the /query endpoint returns a properly streaming response"""
    # Create our mock RAG system
    mock_system = MockRAGSystem()
    
    # Patch the get_rag_system function to return our mock
    with patch("app.api.endpoints.get_rag_system", return_value=mock_system):
        # Make the request
        response = test_client.post(
            "/query",
            json={"query": "Tell me about Shekhar", "top_k": 3}
        )
        
        # Check response status
        assert response.status_code == 200
        
        # Check that we get the expected content
        content = response.content.decode()
        
        
