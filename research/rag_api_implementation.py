"""
RAG API Implementation and Testing Script

This script implements and tests additional RAG endpoints for the FastAPI application:
1. Implements a RAG router with endpoints for question answering and document retrieval
2. Tests the performance and quality of the RAG endpoints
3. Evaluates different configurations for the RAG pipeline
4. Provides recommendations for production deployment
"""

import os
import sys
import time
import logging
import json
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, Depends, HTTPException, Query, Body
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.session import get_db
from src.vector.processor import generate_embedding
from src.utils.groq_client import get_groq_client, DEFAULT_MODEL, LARGER_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('research/rag_api_implementation.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Test questions for RAG evaluation
TEST_QUESTIONS = [
    "What are the latest developments in renewable energy technology?",
    "How is artificial intelligence being used in healthcare?",
    "What are the economic impacts of climate change?",
    "What are the current trends in cryptocurrency markets?",
    "How are autonomous vehicles evolving and what challenges do they face?"
]

# Define RAG request and response models
class RAGQuestionRequest(BaseModel):
    """Request model for RAG question answering."""
    question: str = Field(..., description="The question to answer")
    model: str = Field(DEFAULT_MODEL, description="The model to use for generation")
    max_tokens: int = Field(1024, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Temperature for generation")
    use_hybrid_search: bool = Field(True, description="Whether to use hybrid search")
    num_documents: int = Field(5, description="Number of documents to retrieve")

class RAGQuestionResponse(BaseModel):
    """Response model for RAG question answering."""
    answer: str = Field(..., description="The generated answer")
    model: str = Field(..., description="The model used for generation")
    documents: List[Dict[str, Any]] = Field(..., description="The retrieved documents")
    retrieval_time: float = Field(..., description="Time taken for retrieval (seconds)")
    generation_time: float = Field(..., description="Time taken for generation (seconds)")
    total_time: float = Field(..., description="Total time taken (seconds)")

class DocumentRetrievalRequest(BaseModel):
    """Request model for document retrieval."""
    query: str = Field(..., description="The query to search for")
    use_hybrid_search: bool = Field(True, description="Whether to use hybrid search")
    num_documents: int = Field(5, description="Number of documents to retrieve")

class DocumentRetrievalResponse(BaseModel):
    """Response model for document retrieval."""
    documents: List[Dict[str, Any]] = Field(..., description="The retrieved documents")
    retrieval_time: float = Field(..., description="Time taken for retrieval (seconds)")

# Create RAG router
rag_router = APIRouter(prefix="/rag", tags=["rag"])

@rag_router.post("/question", response_model=RAGQuestionResponse)
async def rag_question_answering(request: RAGQuestionRequest, db: Session = Depends(get_db)):
    """
    Answer a question using RAG (Retrieval-Augmented Generation).
    
    This endpoint retrieves relevant documents from the database and uses them to generate
    an answer to the question using the Groq API.
    """
    try:
        start_time = time.time()
        
        # Retrieve relevant documents
        retrieval_start_time = time.time()
        documents = retrieve_documents(
            db=db,
            query=request.question,
            use_hybrid_search=request.use_hybrid_search,
            num_documents=request.num_documents
        )
        retrieval_time = time.time() - retrieval_start_time
        
        # Generate answer
        generation_start_time = time.time()
        answer = generate_answer(
            question=request.question,
            documents=documents,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        generation_time = time.time() - generation_start_time
        
        total_time = time.time() - start_time
        
        return {
            "answer": answer,
            "model": request.model,
            "documents": documents,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time
        }
    
    except Exception as e:
        logger.error(f"Error in RAG question answering: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@rag_router.post("/retrieve", response_model=DocumentRetrievalResponse)
async def retrieve_documents_endpoint(request: DocumentRetrievalRequest, db: Session = Depends(get_db)):
    """
    Retrieve relevant documents for a query.
    
    This endpoint retrieves documents from the database that are relevant to the query
    using either vector search or hybrid search.
    """
    try:
        start_time = time.time()
        
        documents = retrieve_documents(
            db=db,
            query=request.query,
            use_hybrid_search=request.use_hybrid_search,
            num_documents=request.num_documents
        )
        
        retrieval_time = time.time() - start_time
        
        return {
            "documents": documents,
            "retrieval_time": retrieval_time
        }
    
    except Exception as e:
        logger.error(f"Error in document retrieval: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

def retrieve_documents(db: Session, query: str, use_hybrid_search: bool = True, num_documents: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents for a query.
    
    Args:
        db: Database session
        query: The query to search for
        use_hybrid_search: Whether to use hybrid search
        num_documents: Number of documents to retrieve
        
    Returns:
        List of relevant documents
    """
    if use_hybrid_search:
        # Hybrid search (vector + keyword)
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        query_vector = str(query_embedding).replace('[', '{').replace(']', '}')
        
        # Convert query to tsquery format
        tsquery = ' & '.join(query.split())
        
        # Execute hybrid search query
        result = db.execute(f"""
            WITH vector_search AS (
                SELECT id, url, title, content, source_domain, published_date,
                       1 - (vector_embedding <=> '{query_vector}'::vector) AS vector_similarity
                FROM articles
                WHERE vector_embedding IS NOT NULL
                ORDER BY vector_embedding <=> '{query_vector}'
                LIMIT 100
            ),
            text_search AS (
                SELECT id, url, title, content, source_domain, published_date,
                       ts_rank(to_tsvector('english', title || ' ' || content), to_tsquery('english', '{tsquery}')) AS text_rank
                FROM articles
                WHERE to_tsvector('english', title || ' ' || content) @@ to_tsquery('english', '{tsquery}')
                LIMIT 100
            )
            SELECT v.id, v.url, v.title, v.content, v.source_domain, v.published_date,
                   (0.7 * v.vector_similarity + 0.3 * COALESCE(t.text_rank, 0)) AS combined_score
            FROM vector_search v
            LEFT JOIN text_search t ON v.id = t.id
            ORDER BY combined_score DESC
            LIMIT {num_documents}
        """)
    else:
        # Vector search only
        query_embedding = generate_embedding(query)
        query_vector = str(query_embedding).replace('[', '{').replace(']', '}')
        
        # Execute vector search query
        result = db.execute(f"""
            SELECT id, url, title, content, source_domain, published_date,
                   1 - (vector_embedding <=> '{query_vector}'::vector) AS similarity
            FROM articles
            WHERE vector_embedding IS NOT NULL
            ORDER BY vector_embedding <=> '{query_vector}'
            LIMIT {num_documents}
        """)
    
    # Process results
    documents = []
    for row in result:
        documents.append({
            "id": str(row[0]),
            "url": row[1],
            "title": row[2],
            "content": row[3][:1000] + "..." if len(row[3]) > 1000 else row[3],  # Truncate content for readability
            "source_domain": row[4],
            "published_date": row[5].isoformat() if row[5] else None,
            "score": float(row[6])
        })
    
    return documents

def generate_answer(question: str, documents: List[Dict[str, Any]], model: str = DEFAULT_MODEL, 
                   max_tokens: int = 1024, temperature: float = 0.7) -> str:
    """
    Generate an answer to a question using the retrieved documents.
    
    Args:
        question: The question to answer
        documents: The retrieved documents
        model: The model to use for generation
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for generation
        
    Returns:
        The generated answer
    """
    # Create context from documents
    context = ""
    for i, doc in enumerate(documents):
        context += f"\nDocument {i+1}:\nTitle: {doc['title']}\nSource: {doc['source_domain']}\nContent: {doc['content']}\n"
    
    # Create prompt
    system_message = """
    You are a helpful assistant that provides accurate, comprehensive answers based on the provided context.
    Follow these guidelines:
    1. Base your answer ONLY on the information in the context
    2. If the context doesn't contain the answer, acknowledge that you don't have enough information
    3. Do not make up or infer information that isn't explicitly stated in the context
    4. Cite specific parts of the context to support your answer
    5. Organize your answer in a clear, structured format
    6. Be concise but thorough
    """
    
    prompt = f"""
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    # Generate answer using Groq API
    client = get_groq_client()
    answer = client.generate_completion(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system_message=system_message
    )
    
    return answer

def create_test_app():
    """Create a test FastAPI app with the RAG router."""
    app = FastAPI(title="RAG API Test")
    app.include_router(rag_router)
    return app

def test_rag_endpoints():
    """Test the RAG endpoints."""
    logger.info("Testing RAG endpoints...")
    
    # Create test app and client
    app = create_test_app()
    client = TestClient(app)
    
    # Test question answering endpoint
    question_results = []
    for question in TEST_QUESTIONS:
        logger.info(f"Testing question: {question}")
        
        # Test with different configurations
        configs = [
            {"use_hybrid_search": True, "model": DEFAULT_MODEL},
            {"use_hybrid_search": False, "model": DEFAULT_MODEL},
            {"use_hybrid_search": True, "model": LARGER_MODEL}
        ]
        
        for config in configs:
            try:
                request_data = {
                    "question": question,
                    "model": config["model"],
                    "use_hybrid_search": config["use_hybrid_search"],
                    "num_documents": 5,
                    "max_tokens": 512,
                    "temperature": 0.7
                }
                
                response = client.post("/rag/question", json=request_data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    question_results.append({
                        "question": question,
                        "model": config["model"],
                        "use_hybrid_search": config["use_hybrid_search"],
                        "retrieval_time": result["retrieval_time"],
                        "generation_time": result["generation_time"],
                        "total_time": result["total_time"],
                        "num_documents": len(result["documents"]),
                        "answer_length": len(result["answer"]),
                        "success": True
                    })
                    
                    logger.info(f"Success! Total time: {result['total_time']:.2f}s")
                else:
                    logger.error(f"Error: {response.status_code} - {response.text}")
                    question_results.append({
                        "question": question,
                        "model": config["model"],
                        "use_hybrid_search": config["use_hybrid_search"],
                        "retrieval_time": 0,
                        "generation_time": 0,
                        "total_time": 0,
                        "num_documents": 0,
                        "answer_length": 0,
                        "success": False,
                        "error": response.text
                    })
            
            except Exception as e:
                logger.error(f"Error testing question: {e}")
                question_results.append({
                    "question": question,
                    "model": config["model"],
                    "use_hybrid_search": config["use_hybrid_search"],
                    "retrieval_time": 0,
                    "generation_time": 0,
                    "total_time": 0,
                    "num_documents": 0,
                    "answer_length": 0,
                    "success": False,
                    "error": str(e)
                })
    
    # Test document retrieval endpoint
    retrieval_results = []
    for query in TEST_QUESTIONS:
        logger.info(f"Testing document retrieval for query: {query}")
        
        # Test with different configurations
        configs = [
            {"use_hybrid_search": True, "num_documents": 5},
            {"use_hybrid_search": False, "num_documents": 5},
            {"use_hybrid_search": True, "num_documents": 10}
        ]
        
        for config in configs:
            try:
                request_data = {
                    "query": query,
                    "use_hybrid_search": config["use_hybrid_search"],
                    "num_documents": config["num_documents"]
                }
                
                response = client.post("/rag/retrieve", json=request_data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    retrieval_results.append({
                        "query": query,
                        "use_hybrid_search": config["use_hybrid_search"],
                        "num_documents": config["num_documents"],
                        "retrieval_time": result["retrieval_time"],
                        "actual_num_documents": len(result["documents"]),
                        "success": True
                    })
                    
                    logger.info(f"Success! Retrieval time: {result['retrieval_time']:.2f}s")
                else:
                    logger.error(f"Error: {response.status_code} - {response.text}")
                    retrieval_results.append({
                        "query": query,
                        "use_hybrid_search": config["use_hybrid_search"],
                        "num_documents": config["num_documents"],
                        "retrieval_time": 0,
                        "actual_num_documents": 0,
                        "success": False,
                        "error": response.text
                    })
            
            except Exception as e:
                logger.error(f"Error testing document retrieval: {e}")
                retrieval_results.append({
                    "query": query,
                    "use_hybrid_search": config["use_hybrid_search"],
                    "num_documents": config["num_documents"],
                    "retrieval_time": 0,
                    "actual_num_documents": 0,
                    "success": False,
                    "error": str(e)
                })
    
    # Save results to CSV
    question_df = pd.DataFrame(question_results)
    question_df.to_csv('research/results/rag_question_results.csv', index=False)
    
    retrieval_df = pd.DataFrame(retrieval_results)
    retrieval_df.to_csv('research/results/rag_retrieval_results.csv', index=False)
    
    # Create summary by configuration
    question_summary = question_df[question_df["success"]].groupby(["model", "use_hybrid_search"]).agg({
        "retrieval_time": "mean",
        "generation_time": "mean",
        "total_time": "mean",
        "answer_length": "mean",
        "success": "mean"
    }).reset_index()
    
    question_summary.to_csv('research/results/rag_question_summary.csv', index=False)
    
    retrieval_summary = retrieval_df[retrieval_df["success"]].groupby(["use_hybrid_search", "num_documents"]).agg({
        "retrieval_time": "mean",
        "actual_num_documents": "mean",
        "success": "mean"
    }).reset_index()
    
    retrieval_summary.to_csv('research/results/rag_retrieval_summary.csv', index=False)
    
    logger.info("RAG endpoint testing completed")
    return question_results, retrieval_results

def implement_rag_router():
    """Implement the RAG router in the FastAPI application."""
    logger.info("Implementing RAG router...")
    
    # Create router implementation file
    router_code = """
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
import time

from src.database.session import get_db
from src.vector.processor import generate_embedding
from src.utils.groq_client import get_groq_client, DEFAULT_MODEL, LARGER_MODEL

router = APIRouter(prefix="/rag", tags=["rag"])

class RAGQuestionRequest(BaseModel):
    \"\"\"Request model for RAG question answering.\"\"\"
    question: str = Field(..., description="The question to answer")
    model: str = Field(DEFAULT_MODEL, description="The model to use for generation")
    max_tokens: int = Field(1024, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Temperature for generation")
    use_hybrid_search: bool = Field(True, description="Whether to use hybrid search")
    num_documents: int = Field(5, description="Number of documents to retrieve")

class RAGQuestionResponse(BaseModel):
    \"\"\"Response model for RAG question answering.\"\"\"
    answer: str = Field(..., description="The generated answer")
    model: str = Field(..., description="The model used for generation")
    documents: List[Dict[str, Any]] = Field(..., description="The retrieved documents")
    retrieval_time: float = Field(..., description="Time taken for retrieval (seconds)")
    generation_time: float = Field(..., description="Time taken for generation (seconds)")
    total_time: float = Field(..., description="Total time taken (seconds)")

class DocumentRetrievalRequest(BaseModel):
    \"\"\"Request model for document retrieval.\"\"\"
    query: str = Field(..., description="The query to search for")
    use_hybrid_search: bool = Field(True, description="Whether to use hybrid search")
    num_documents: int = Field(5, description="Number of documents to retrieve")

class DocumentRetrievalResponse(BaseModel):
    \"\"\"Response model for document retrieval.\"\"\"
    documents: List[Dict[str, Any]] = Field(..., description="The retrieved documents")
    retrieval_time: float = Field(..., description="Time taken for retrieval (seconds)")

@router.post("/question", response_model=RAGQuestionResponse)
async def rag_question_answering(request: RAGQuestionRequest, db: Session = Depends(get_db)):
    \"\"\"
    Answer a question using RAG (Retrieval-Augmented Generation).
    
    This endpoint retrieves relevant documents from the database and uses them to generate
    an answer to the question using the Groq API.
    \"\"\"
    try:
        start_time = time.time()
        
        # Retrieve relevant documents
        retrieval_start_time = time.time()
        documents = retrieve_documents(
            db=db,
            query=request.question,
            use_hybrid_search=request.use_hybrid_search,
            num_documents=request.num_documents
        )
        retrieval_time = time.time() - retrieval_start_time
        
        # Generate answer
        generation_start_time = time.time()
        answer = generate_answer(
            question=request.question,
            documents=documents,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        generation_time = time.time() - generation_start_time
        
        total_time = time.time() - start_time
        
        return {
            "answer": answer,
            "model": request.model,
            "documents": documents,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@router.post("/retrieve", response_model=DocumentRetrievalResponse)
async def retrieve_documents_endpoint(request: DocumentRetrievalRequest, db: Session = Depends(get_db)):
    \"\"\"
    Retrieve relevant documents for a query.
    
    This endpoint retrieves documents from the database that are relevant to the query
    using either vector search or hybrid search.
    \"\"\"
    try:
        start_time = time.time()
        
        documents = retrieve_documents(
            db=db,
            query=request.query,
            use_hybrid_search=request.use_hybrid_search,
            num_documents=request.num_documents
        )
        
        retrieval_time = time.time() - start_time
        
        return {
            "documents": documents,
            "retrieval_time": retrieval_time
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

def retrieve_documents(db: Session, query: str, use_hybrid_search: bool = True, num_documents: int = 5) -> List[Dict[str, Any]]:
    \"\"\"
    Retrieve relevant documents for a query.
    
    Args:
        db: Database session
        query: The query to search for
        use_hybrid_search: Whether to use hybrid search
        num_documents: Number of documents to retrieve
        
    Returns:
        List of relevant documents
    \"\"\"
    if use_hybrid_search:
        # Hybrid search (vector + keyword)
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        query_vector = str(query_embedding).replace('[', '{').replace(']', '}')
        
        # Convert query to tsquery format
        tsquery = ' & '.join(query.split())
        
        # Execute hybrid search query
        result = db.execute(f\"\"\"
            WITH vector_search AS (
                SELECT id, url, title, content, source_domain, published_date,
                       1 - (vector_embedding <=> '{query_vector}'::vector) AS vector_similarity
                FROM articles
                WHERE vector_embedding IS NOT NULL
                ORDER BY vector_embedding <=> '{query_vector}'
                LIMIT 100
            ),
            text_search AS (
                SELECT id, url, title, content, source_domain, published_date,
                       ts_rank(to_tsvector('english', title || ' ' || content), to_tsquery('english', '{tsquery}')) AS text_rank
                FROM articles
                WHERE to_tsvector('english', title || ' ' || content) @@ to_tsquery('english', '{tsquery}')
                LIMIT 100
            )
            SELECT v.id, v.url, v.title, v.content, v.source_domain, v.published_date,
                   (0.7 * v.vector_similarity + 0.3 * COALESCE(t.text_rank, 0)) AS combined_score
            FROM vector_search v
            LEFT JOIN text_search t ON v.id = t.id
            ORDER BY combined_score DESC
            LIMIT {num_documents}
        \"\"\")
    else:
        # Vector search only
        query_embedding = generate_embedding(query)
        query_vector = str(query_embedding).replace('[', '{').replace(']', '}')
        
        # Execute vector search query
        result = db.execute(f\"\"\"
            SELECT id, url, title, content, source_domain, published_date,
                   1 - (vector_embedding <=> '{query_vector}'::vector) AS similarity
            FROM articles
            WHERE vector_embedding IS NOT NULL
            ORDER BY vector_embedding <=> '{query_vector}'
            LIMIT {num_documents}
        \"\"\")
    
    # Process results
    documents = []
    for row in result:
        documents.append({
            "id": str(row[0]),
            "url": row[1],
            "title": row[2],
            "content": row[3][:1000] + "..." if len(row[3]) > 1000 else row[3],  # Truncate content for readability
            "source_domain": row[4],
            "published_date": row[5].isoformat() if row[5] else None,
            "score": float(row[6])
        })
    
    return documents

def generate_answer(question: str, documents: List[Dict[str, Any]], model: str = DEFAULT_MODEL, 
                   max_tokens: int = 1024, temperature: float = 0.7) -> str:
    \"\"\"
    Generate an answer to a question using the retrieved documents.
    
    Args:
        question: The question to answer
        documents: The retrieved documents
        model: The model to use for generation
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for generation
        
    Returns:
        The generated answer
    \"\"\"
    # Create context from documents
    context = ""
    for i, doc in enumerate(documents):
        context += f"\\nDocument {i+1}:\\nTitle: {doc['title']}\\nSource: {doc['source_domain']}\\nContent: {doc['content']}\\n"
    
    # Create prompt
    system_message = \"\"\"
    You are a helpful assistant that provides accurate, comprehensive answers based on the provided context.
    Follow these guidelines:
    1. Base your answer ONLY on the information in the context
    2. If the context doesn't contain the answer, acknowledge that you don't have enough information
    3. Do not make up or infer information that isn't explicitly stated in the context
    4. Cite specific parts of the context to support your answer
    5. Organize your answer in a clear, structured format
    6. Be concise but thorough
    \"\"\"
    
    prompt = f"\"\"\"
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    \"\"\"
    
    # Generate answer using Groq API
    client = get_groq_client()
    answer = client.generate_completion(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system_message=system_message
    )
    
    return answer
"""
    
    # Save router implementation to file
    with open('research/results/rag_router.py', 'w') as f:
        f.write(router_code)
    
    # Create main.py update instructions
    main_update = """
# Add this import at the top with other router imports
from src.api.routers import rag

# Add this line with other router includes
app.include_router(rag.router)
"""
    
    # Save main.py update instructions to file
    with open('research/results/main_py_update.txt', 'w') as f:
        f.write(main_update)
    
    logger.info("RAG router implementation completed")

def main():
    """Main function to run all implementations and tests."""
    logger.info("Starting RAG API implementation and testing...")
    
    # Create results directory if it doesn't exist
    os.makedirs('research/results', exist_ok=True)
    
    # Implement RAG router
    implement_rag_router()
    
    # Test RAG endpoints
    test_rag_endpoints()
    
    logger.info("RAG API implementation and testing completed")

if __name__ == "__main__":
    main() 