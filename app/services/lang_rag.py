"""
Update to lang_rag.py to fix the vector database path issue
"""

import logging
import os
from typing import Dict, List, Any, AsyncIterator, Optional
import glob

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from fastapi import Depends
from pathlib import Path

from app.core.config import settings

# Set up logger
logger = logging.getLogger(__name__)

class LangChainRAG:
    """
    LangChain-based RAG system with streaming support.
    """
    
    def __init__(self):
        """Initialize the RAG system."""
        # Initialize connection to the vector store
        self._init_vector_store()
        
        # Initialize the LLM
        self._init_llm()
        
        # Create the RAG pipeline
        self._setup_rag_chain()
        
        logger.info("RAG system initialized successfully")
    
    def _init_vector_store(self) -> None:
        """Initialize the vector store connection."""
        try:
            # Ensure the embeddings model is using the API key
            os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
            
            # Initialize embeddings model
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"
            )
            
            # Get the path to the vector DB
            db_path = settings.CHROMA_DB_PATH
            
            # Log the search path and available directories
            logger.info(f"Looking for vector database at: {db_path}")
            
            # Check if directory exists - if not, try to create the vector store
            if not os.path.exists(db_path):
                logger.warning(f"Vector database directory not found at {db_path}")
                
                # Check parent directory exists
                parent_dir = os.path.dirname(db_path)
                if not os.path.exists(parent_dir):
                    logger.info(f"Creating parent directory: {parent_dir}")
                    os.makedirs(parent_dir, exist_ok=True)
                
                logger.info("Creating new vector store since it doesn't exist")
                
                # Create sample documents about Shekhar (same as in create_vectordb.py)
                documents = [
                    Document(page_content="Shekhar is a senior software engineer with over 10 years of experience in Python development."),
                    Document(page_content="Shekhar specializes in natural language processing and has implemented several RAG systems in production."),
                    Document(page_content="Shekhar holds a Master's degree in Computer Science from Stanford University and completed his thesis on efficient vector retrieval systems."),
                    Document(page_content="Shekhar previously worked at Google as a machine learning engineer focusing on large language models."),
                    Document(page_content="Shekhar has contributed to several open-source projects including improvements to the LangChain framework."),
                    Document(page_content="Shekhar is an expert in building scalable AI systems and has designed architectures handling millions of queries per day."),
                    Document(page_content="Shekhar regularly speaks at tech conferences about retrieval-augmented generation and vector databases."),
                    Document(page_content="Shekhar currently leads a team of 5 engineers working on next-generation AI applications."),
                    Document(page_content="Shekhar has published several research papers on efficient embedding techniques for information retrieval."),
                    Document(page_content="Shekhar maintains a popular technical blog where he shares insights about AI engineering and best practices.")
                ]
                
                # Create vector store with the documents
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=db_path
                )
                
                # Explicitly persist to disk
                self.vector_store.persist()
                logger.info(f"Created new vector store at {db_path}")
            else:
                logger.info(f"Found existing vector database at {db_path}")
                
                # Connect to the existing Chroma DB
                self.vector_store = Chroma(
                    persist_directory=db_path,
                    embedding_function=embeddings,
                )
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Default value, will be overridden by query params
            )
            
            # Test retrieval to verify connection
            # try:
            #     test_docs = self.retriever.get_relevant_documents("test query", k=1)
            #     logger.info(f"Vector store connection successful. Found {len(test_docs)} test documents.")
            # except Exception as e:
            #     logger.error(f"Error during test retrieval: {str(e)}")
            #     raise
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def _init_llm(self) -> None:
        """Initialize the language model."""
        try:
            # Initialize the OpenAI model
            self.llm = ChatOpenAI(
                model=settings.LLM_MODEL_NAME,
                temperature=0.1,
                streaming=True,  # Enable streaming
                api_key=settings.OPENAI_API_KEY
            )
            logger.info(f"LLM initialized: {settings.LLM_MODEL_NAME}")
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise
    
    def _setup_rag_chain(self) -> None:
        """Set up the RAG chain for query processing."""
        try:
            # Define the prompt template
            template = """
            You are an AI assistant providing information about Shekhar, a senior software engineer.
            Use only the provided context to answer the question. If the answer is not in the context,
            say "I don't have information about that in my knowledge base."
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:
            """
            
            # Create the prompt
            prompt = ChatPromptTemplate.from_template(template)
            
            # Document formatting function
            def format_docs(docs: List[Document]) -> str:
                """Format the documents into a single string."""
                return "\n\n".join(doc.page_content for doc in docs)
            
            # Create the RAG chain
            self.rag_chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            logger.info("RAG chain setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up RAG chain: {str(e)}")
            raise
    
    async def stream_query(
        self,
        query: str,
        top_k: int = 3,
        thread_id: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Process a query and stream the response.
        
        Args:
            query: The user's question
            top_k: Number of documents to retrieve
            thread_id: Optional thread ID for conversation history
            
        Yields:
            Chunks of the generated response as they become available
        """
        try:
            # Log the query
            logger.info(f"Processing query: {query} with top_k={top_k}")
            
            # Update retriever parameters for this specific query
            self.retriever.search_kwargs["k"] = top_k
            
            # Stream the response
            async for chunk in self.rag_chain.astream(query):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in stream_query: {str(e)}")
            # Send error message in the stream
            yield f"Error processing your query: {str(e)}"
    
    async def query(
        self,
        query: str,
        top_k: int = 3,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a query and return the complete result (non-streaming).
        
        Args:
            query: The user's question
            top_k: Number of documents to retrieve
            thread_id: Optional thread ID for conversation history
            
        Returns:
            Dictionary with question, retrieved context, and answer
        """
        try:
            # Log the query
            logger.info(f"Processing complete query: {query} with top_k={top_k}")
            
            # Update retriever parameters for this specific query
            self.retriever.search_kwargs["k"] = top_k
            
            # Get relevant documents
            documents = self.retriever.get_relevant_documents(query, k=top_k)
            
            # Format the documents for the response
            context = [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]
            
            # Get the answer from the chain
            answer = await self.rag_chain.ainvoke(query)
            
            # Return the complete result
            return {
                "question": query,
                "context": context,
                "answer": answer
            }
            
        except Exception as e:
            logger.error(f"Error in query: {str(e)}")
            return {
                "question": query,
                "error": str(e)
            }

# Singleton instance
_rag_system = None

def get_rag_system() -> LangChainRAG:
    """
    Factory function to get or create the RAG system.
    Used as a FastAPI dependency.
    """
    global _rag_system
    if _rag_system is None:
        _rag_system = LangChainRAG()
    return _rag_system