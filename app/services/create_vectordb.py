"""
create_vectordb.py - Creates a new ChromaDB vector database with sample documents about Shekhar
"""

import os
import logging
from pathlib import Path
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def resolve_relative_path(relative_path):
    """Resolve a relative path to an absolute path"""
    if relative_path.startswith('./'):
        relative_path = relative_path[2:]
        
    base_dir = Path.cwd()
    absolute_path = os.path.join(base_dir, relative_path)
    return os.path.normpath(absolute_path)

def create_vector_database(db_path="./chroma_langchain_db", collection_name="data_resume"):
    """Create a vector database with sample documents about Shekhar"""
    
    # Resolve the path
    db_path = resolve_relative_path(db_path)
    logger.info(f"Creating vector database at: {db_path}")
    
    # Delete previous content if it exists
    import shutil
    if os.path.exists(db_path):
        logger.info(f"Deleting existing vector database at: {db_path}")
        shutil.rmtree(db_path)
        logger.info("Previous database deleted successfully")
    
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    
    # Create sample documents about Shekhar
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
    
    # Create and initialize the vector store
    try:
        # Check if directory exists, create if it doesn't
        os.makedirs(db_path, exist_ok=True)
        
        # Create vector store
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=db_path
        )
        
        # Add documents
        vector_store.add_documents(documents)
        
        # Explicitly persist to disk
        vector_store.persist()
        
        # Verify documents were added
        count = vector_store._collection.count()
        logger.info(f"Created vector database with {count} documents")
        
        # Test retrieval
        test_query = "Who is Shekhar?"
        results = vector_store.similarity_search(test_query, k=1)
        if results:
            logger.info(f"Test retrieval successful. Sample result: {results[0].page_content[:50]}...")
        else:
            logger.warning("Test retrieval returned no results.")
        
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating vector database: {str(e)}")
        raise

# Run the function if executed directly
if __name__ == "__main__":
    create_vector_database()
    logger.info("Vector database creation completed")