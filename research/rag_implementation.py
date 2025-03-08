"""
Retrieval-Augmented Generation (RAG) Implementation and Testing Script

This script implements and tests a complete RAG pipeline using:
1. Vector search for relevant article retrieval
2. Hybrid search combining vector and keyword search
3. GroqAI for text generation
4. Local LLM fallback options
5. Performance and quality evaluation
"""

import os
import sys
import time
import logging
import json
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.groq_client import get_groq_client, DEFAULT_MODEL, LARGER_MODEL, MIXTRAL_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('research/rag_implementation.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://newscrawler:newscrawler_password@localhost:5432/newscrawler")
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# Define test questions
TEST_QUESTIONS = [
    "What are the latest developments in renewable energy technology?",
    "How is artificial intelligence being used in healthcare?",
    "What are the economic impacts of climate change?",
    "What are the current trends in cryptocurrency markets?",
    "How are autonomous vehicles evolving and what challenges do they face?"
]

# Define embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# RAG prompt templates
STANDARD_RAG_TEMPLATE = """
You are a helpful assistant that provides accurate information based on the context provided.
If the information is not in the context, acknowledge that you don't know rather than making up an answer.

Context:
{context}

Question: {question}

Answer:
"""

ENHANCED_RAG_TEMPLATE = """
You are a helpful assistant that provides accurate, comprehensive answers based on the provided context.
Follow these guidelines:
1. Base your answer ONLY on the information in the context
2. If the context doesn't contain the answer, acknowledge that you don't have enough information
3. Do not make up or infer information that isn't explicitly stated in the context
4. Cite specific parts of the context to support your answer
5. Organize your answer in a clear, structured format
6. Be concise but thorough

Context:
{context}

Question: {question}

Answer:
"""

def get_embedding_model():
    """Get the embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def setup_pgvector_connection():
    """Set up connection to PostgreSQL with pgvector extension."""
    connection_string = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://")
    return connection_string

def create_vectorstore():
    """Create a vector store using PGVector."""
    logger.info("Creating vector store...")
    
    connection_string = setup_pgvector_connection()
    embedding_model = get_embedding_model()
    
    vectorstore = PGVector(
        connection_string=connection_string,
        embedding_function=embedding_model,
        collection_name="articles",
        pre_delete_collection=False
    )
    
    return vectorstore

def setup_groq_llm(model_name=DEFAULT_MODEL, temperature=0.7):
    """Set up GroqAI LLM."""
    logger.info(f"Setting up GroqAI LLM with model {model_name}...")
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")
    
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature
    )
    
    return llm

def setup_local_llm():
    """Set up local LLM using HuggingFace."""
    logger.info("Setting up local LLM...")
    
    try:
        # Try to load a smaller model suitable for local execution
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True  # Use 8-bit quantization for memory efficiency
        )
        
        # Create a text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        # Create LangChain wrapper
        llm = HuggingFacePipeline(pipeline=pipe)
        
        return llm
    
    except Exception as e:
        logger.error(f"Error setting up local LLM: {e}")
        logger.warning("Falling back to GroqAI LLM")
        return setup_groq_llm()

def setup_rag_pipeline(vectorstore, use_local_llm=False, use_enhanced_prompt=False, use_hybrid_search=False):
    """Set up the RAG pipeline using LangChain."""
    logger.info("Setting up RAG pipeline...")
    
    # Create retriever from vector store
    search_kwargs = {"k": 5}  # Retrieve top 5 documents
    
    if use_hybrid_search:
        # Use hybrid search with custom retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 10,
                "score_threshold": 0.5,
                "filter": None
            }
        )
    else:
        # Use standard vector search
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
    
    # Create prompt template
    if use_enhanced_prompt:
        prompt = PromptTemplate(
            template=ENHANCED_RAG_TEMPLATE,
            input_variables=["context", "question"]
        )
    else:
        prompt = PromptTemplate(
            template=STANDARD_RAG_TEMPLATE,
            input_variables=["context", "question"]
        )
    
    # Set up LLM - either local or GroqAI
    if use_local_llm:
        try:
            llm = setup_local_llm()
        except Exception as e:
            logger.error(f"Error setting up local LLM: {e}")
            logger.warning("Falling back to GroqAI LLM")
            llm = setup_groq_llm()
    else:
        llm = setup_groq_llm()
    
    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Simple approach that stuffs all retrieved documents into the prompt
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

def test_direct_retrieval(vectorstore, query: str) -> Tuple[List[Document], float]:
    """Test direct retrieval from vector store."""
    logger.info(f"Testing direct retrieval for query: '{query}'")
    
    # Measure retrieval time
    start_time = time.time()
    docs = vectorstore.similarity_search(query, k=5)
    retrieval_time = time.time() - start_time
    
    return docs, retrieval_time

def test_hybrid_retrieval(db_session, query: str, limit: int = 5) -> Tuple[List[Dict[str, Any]], float]:
    """Test hybrid retrieval combining vector and keyword search."""
    logger.info(f"Testing hybrid retrieval for query: '{query}'")
    
    # Generate embedding for the query
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.embed_query(query)
    query_vector = str(query_embedding).replace('[', '{').replace(']', '}')
    
    # Convert query to tsquery format
    tsquery = ' & '.join(query.split())
    
    # Construct SQL query for hybrid search
    sql_query = text(f"""
        WITH vector_search AS (
            SELECT id, url, title, content, source_domain,
                   1 - (vector_embedding <=> :query_vector) AS vector_similarity
            FROM articles
            WHERE vector_embedding IS NOT NULL
            ORDER BY vector_embedding <=> :query_vector
            LIMIT 100
        ),
        text_search AS (
            SELECT id, url, title, content, source_domain,
                   ts_rank(to_tsvector('english', title || ' ' || content), to_tsquery('english', :tsquery)) AS text_rank
            FROM articles
            WHERE to_tsvector('english', title || ' ' || content) @@ to_tsquery('english', :tsquery)
            LIMIT 100
        )
        SELECT v.id, v.url, v.title, v.content, v.source_domain,
               (0.7 * v.vector_similarity + 0.3 * COALESCE(t.text_rank, 0)) AS combined_score
        FROM vector_search v
        LEFT JOIN text_search t ON v.id = t.id
        ORDER BY combined_score DESC
        LIMIT :limit
    """)
    
    # Execute search query and measure time
    start_time = time.time()
    result = db_session.execute(sql_query, {
        "query_vector": query_vector, 
        "tsquery": tsquery,
        "limit": limit
    })
    retrieval_time = time.time() - start_time
    
    # Process results
    results = []
    for row in result:
        results.append({
            "id": str(row[0]),
            "url": row[1],
            "title": row[2],
            "content": row[3],
            "source_domain": row[4],
            "combined_score": float(row[5])
        })
    
    return results, retrieval_time

def test_rag_pipeline(qa_chain, question: str) -> Tuple[str, float, float]:
    """Test the RAG pipeline with a question."""
    logger.info(f"Testing RAG pipeline with question: '{question}'")
    
    # Measure total time
    start_time = time.time()
    
    # Get answer
    result = qa_chain({"query": question})
    
    total_time = time.time() - start_time
    
    # Extract retrieval time if available
    retrieval_time = result.get("retrieval_time", 0)
    
    return result["result"], total_time, retrieval_time

def evaluate_rag_configurations():
    """Evaluate different RAG configurations."""
    logger.info("Evaluating different RAG configurations...")
    
    # Create vector store
    vectorstore = create_vectorstore()
    
    # Define configurations to test
    configurations = [
        {
            "name": "Standard RAG with GroqAI",
            "use_local_llm": False,
            "use_enhanced_prompt": False,
            "use_hybrid_search": False,
            "model": DEFAULT_MODEL
        },
        {
            "name": "Enhanced RAG with GroqAI",
            "use_local_llm": False,
            "use_enhanced_prompt": True,
            "use_hybrid_search": False,
            "model": DEFAULT_MODEL
        },
        {
            "name": "Hybrid Search RAG with GroqAI",
            "use_local_llm": False,
            "use_enhanced_prompt": True,
            "use_hybrid_search": True,
            "model": DEFAULT_MODEL
        },
        {
            "name": "Enhanced RAG with Larger GroqAI Model",
            "use_local_llm": False,
            "use_enhanced_prompt": True,
            "use_hybrid_search": False,
            "model": LARGER_MODEL
        }
    ]
    
    # Try to add local LLM configuration if possible
    try:
        # Test if local LLM can be initialized
        setup_local_llm()
        
        # Add local LLM configuration
        configurations.append({
            "name": "Enhanced RAG with Local LLM",
            "use_local_llm": True,
            "use_enhanced_prompt": True,
            "use_hybrid_search": False,
            "model": "local"
        })
    except Exception as e:
        logger.warning(f"Local LLM not available: {e}")
    
    results = []
    
    for config in configurations:
        logger.info(f"Testing configuration: {config['name']}")
        
        # Set up RAG pipeline with this configuration
        if not config["use_local_llm"]:
            # Override the LLM model for GroqAI
            llm = setup_groq_llm(model_name=config["model"])
        else:
            llm = setup_local_llm()
        
        qa_chain = setup_rag_pipeline(
            vectorstore=vectorstore,
            use_local_llm=config["use_local_llm"],
            use_enhanced_prompt=config["use_enhanced_prompt"],
            use_hybrid_search=config["use_hybrid_search"]
        )
        
        # Test with each question
        config_results = []
        for question in TEST_QUESTIONS:
            try:
                answer, total_time, retrieval_time = test_rag_pipeline(qa_chain, question)
                
                config_results.append({
                    "question": question,
                    "answer": answer,
                    "total_time": total_time,
                    "retrieval_time": retrieval_time,
                    "generation_time": total_time - retrieval_time
                })
                
                logger.info(f"Question: '{question}' - Total time: {total_time:.2f}s")
            
            except Exception as e:
                logger.error(f"Error testing configuration {config['name']} with question '{question}': {e}")
        
        # Add to overall results
        results.append({
            "configuration": config["name"],
            "results": config_results,
            "avg_total_time": np.mean([r["total_time"] for r in config_results]) if config_results else 0,
            "avg_retrieval_time": np.mean([r["retrieval_time"] for r in config_results]) if config_results else 0,
            "avg_generation_time": np.mean([r["generation_time"] for r in config_results]) if config_results else 0
        })
    
    # Save detailed results to JSON
    with open('research/results/rag_configurations_detailed.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary DataFrame
    summary = []
    for result in results:
        summary.append({
            "configuration": result["configuration"],
            "avg_total_time": result["avg_total_time"],
            "avg_retrieval_time": result["avg_retrieval_time"],
            "avg_generation_time": result["avg_generation_time"],
            "avg_answer_length": np.mean([len(r["answer"]) for r in result["results"]]) if result["results"] else 0
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('research/results/rag_configurations_summary.csv', index=False)
    
    logger.info("RAG configuration evaluation completed")
    
    return results

def evaluate_retrieval_methods():
    """Evaluate different retrieval methods for RAG."""
    logger.info("Evaluating different retrieval methods...")
    
    # Create vector store
    vectorstore = create_vectorstore()
    
    # Create database session for hybrid search
    db = Session()
    
    try:
        results = []
        
        # Test each question with different retrieval methods
        for question in TEST_QUESTIONS:
            logger.info(f"Testing retrieval methods for question: '{question}'")
            
            # Vector search
            vector_docs, vector_time = test_direct_retrieval(vectorstore, question)
            
            # Hybrid search
            hybrid_results, hybrid_time = test_hybrid_retrieval(db, question, limit=5)
            
            # Convert hybrid results to documents
            hybrid_docs = [
                Document(
                    page_content=result["content"],
                    metadata={
                        "id": result["id"],
                        "url": result["url"],
                        "title": result["title"],
                        "source_domain": result["source_domain"],
                        "score": result["combined_score"]
                    }
                )
                for result in hybrid_results
            ]
            
            # Save results for this question
            results.append({
                "question": question,
                "vector_retrieval_time": vector_time,
                "hybrid_retrieval_time": hybrid_time,
                "vector_docs": [
                    {
                        "page_content": doc.page_content[:200] + "...",  # Truncate for readability
                        "metadata": doc.metadata
                    }
                    for doc in vector_docs
                ],
                "hybrid_docs": [
                    {
                        "page_content": doc.page_content[:200] + "...",  # Truncate for readability
                        "metadata": doc.metadata
                    }
                    for doc in hybrid_docs
                ]
            })
        
        # Save detailed results to JSON
        with open('research/results/retrieval_methods_detailed.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary DataFrame
        summary = []
        for result in results:
            summary.append({
                "question": result["question"],
                "vector_retrieval_time": result["vector_retrieval_time"],
                "hybrid_retrieval_time": result["hybrid_retrieval_time"],
                "vector_docs_count": len(result["vector_docs"]),
                "hybrid_docs_count": len(result["hybrid_docs"])
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv('research/results/retrieval_methods_summary.csv', index=False)
        
        logger.info("Retrieval methods evaluation completed")
        
        return results
    
    finally:
        db.close()

def main():
    """Main function to run all evaluations."""
    logger.info("Starting RAG implementation and testing...")
    
    # Create results directory if it doesn't exist
    os.makedirs('research/results', exist_ok=True)
    
    # Run evaluations
    evaluate_retrieval_methods()
    evaluate_rag_configurations()
    
    logger.info("RAG implementation and testing completed")

if __name__ == "__main__":
    main() 