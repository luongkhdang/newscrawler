"""
Retrieval-Augmented Generation (RAG) Pipeline Test Script

This script designs and tests a complete RAG pipeline using:
1. Vector search for relevant article retrieval
2. LangChain for context augmentation
3. GroqAI and local LLMs for text generation
4. Evaluation metrics for quality assessment
"""

import os
import sys
import time
import logging
import json
from typing import List, Dict, Any, Tuple
import pandas as pd
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import PGVector
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import Groq
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.session import SessionLocal
from src.database.models import Article
from src.utils.groq_client import get_groq_client, DEFAULT_MODEL, LARGER_MODEL, MIXTRAL_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('research/rag_pipeline.log')
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

# RAG prompt template
RAG_PROMPT_TEMPLATE = """
You are a helpful assistant that provides accurate information based on the context provided.
If the information is not in the context, acknowledge that you don't know rather than making up an answer.

Context:
{context}

Question: {question}

Answer:
"""

def setup_pgvector_connection():
    """Set up connection to PostgreSQL with pgvector extension."""
    # Get database connection string from environment
    connection_string = os.getenv("DATABASE_URL", "postgresql://newscrawler:newscrawler_password@localhost:5432/newscrawler")
    
    # Create connection string for PGVector
    connection_string = connection_string.replace("postgresql://", "postgresql+psycopg2://")
    
    return connection_string

def create_vectorstore():
    """Create a vector store using PGVector."""
    logger.info("Creating vector store...")
    
    connection_string = setup_pgvector_connection()
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # Create PGVector instance
    # Note: We're using the 'articles' table that already has vector_embedding column
    vectorstore = PGVector(
        connection_string=connection_string,
        embedding_function=embedding_model,
        collection_name="articles",
        pre_delete_collection=False  # Don't delete existing collection
    )
    
    return vectorstore

def setup_groq_llm():
    """Set up GroqAI LLM."""
    logger.info("Setting up GroqAI LLM...")
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")
    
    llm = Groq(
        api_key=groq_api_key,
        model_name=DEFAULT_MODEL
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

def setup_rag_pipeline(vectorstore, use_local_llm=False):
    """Set up the RAG pipeline using LangChain."""
    logger.info("Setting up RAG pipeline...")
    
    # Create retriever from vector store
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve top 5 documents
    )
    
    # Create prompt template
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
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

def test_direct_retrieval(vectorstore, question: str) -> Tuple[List[Dict[str, Any]], float]:
    """
    Test direct retrieval from the vector store without LLM generation.
    
    Args:
        vectorstore: The vector store to retrieve from
        question: The question to retrieve documents for
        
    Returns:
        Tuple of (retrieved_docs, retrieval_time)
    """
    logger.info(f"Testing direct retrieval for: {question}")
    
    # Measure retrieval time
    start_time = time.time()
    docs = vectorstore.similarity_search(question, k=5)
    retrieval_time = time.time() - start_time
    
    # Process retrieved documents
    retrieved_docs = []
    for i, doc in enumerate(docs):
        retrieved_docs.append({
            "index": i,
            "content": doc.page_content[:200] + "...",  # First 200 chars
            "metadata": doc.metadata
        })
    
    logger.info(f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.4f}s")
    
    return retrieved_docs, retrieval_time

def test_rag_pipeline(qa_chain, question: str) -> Tuple[str, float, float]:
    """
    Test the RAG pipeline with a question.
    
    Args:
        qa_chain: The QA chain to use
        question: The question to answer
        
    Returns:
        Tuple of (answer, total_time, retrieval_time)
    """
    logger.info(f"Testing RAG pipeline for: {question}")
    
    # Measure total time
    start_time = time.time()
    
    # Get answer from the chain
    result = qa_chain({"query": question})
    
    total_time = time.time() - start_time
    
    # Extract answer and retrieval time (if available)
    answer = result.get("result", "No answer generated")
    retrieval_time = result.get("retrieval_time", 0)
    
    logger.info(f"Generated answer in {total_time:.4f}s")
    logger.info(f"Answer: {answer[:100]}...")  # Log first 100 chars
    
    return answer, total_time, retrieval_time

def evaluate_rag_pipeline():
    """Evaluate the RAG pipeline with test questions."""
    logger.info("Starting RAG pipeline evaluation...")
    
    try:
        # Create vector store
        vectorstore = create_vectorstore()
        
        # Test direct retrieval
        retrieval_results = []
        for question in TEST_QUESTIONS:
            try:
                retrieved_docs, retrieval_time = test_direct_retrieval(vectorstore, question)
                
                retrieval_results.append({
                    "question": question,
                    "retrieval_time": retrieval_time,
                    "num_docs": len(retrieved_docs),
                    "retrieved_docs": retrieved_docs
                })
            
            except Exception as e:
                logger.error(f"Error in direct retrieval for '{question}': {e}")
        
        # Test RAG pipeline with GroqAI
        logger.info("Testing RAG pipeline with GroqAI...")
        groq_qa_chain = setup_rag_pipeline(vectorstore, use_local_llm=False)
        groq_results = []
        
        for question in TEST_QUESTIONS:
            try:
                answer, total_time, retrieval_time = test_rag_pipeline(groq_qa_chain, question)
                
                groq_results.append({
                    "question": question,
                    "answer": answer,
                    "total_time": total_time,
                    "retrieval_time": retrieval_time,
                    "llm_type": "GroqAI"
                })
            
            except Exception as e:
                logger.error(f"Error in GroqAI RAG pipeline for '{question}': {e}")
        
        # Test RAG pipeline with local LLM if available
        local_results = []
        try:
            logger.info("Testing RAG pipeline with local LLM...")
            local_qa_chain = setup_rag_pipeline(vectorstore, use_local_llm=True)
            
            for question in TEST_QUESTIONS:
                try:
                    answer, total_time, retrieval_time = test_rag_pipeline(local_qa_chain, question)
                    
                    local_results.append({
                        "question": question,
                        "answer": answer,
                        "total_time": total_time,
                        "retrieval_time": retrieval_time,
                        "llm_type": "Local"
                    })
                
                except Exception as e:
                    logger.error(f"Error in local LLM RAG pipeline for '{question}': {e}")
        
        except Exception as e:
            logger.error(f"Error setting up local LLM RAG pipeline: {e}")
            logger.warning("Skipping local LLM tests")
        
        # Combine results
        rag_results = groq_results + local_results
        
        # Save results
        results = {
            "retrieval_results": retrieval_results,
            "rag_results": rag_results
        }
        
        with open('research/rag_pipeline_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary DataFrame
        summary = []
        for result in rag_results:
            summary.append({
                "question": result["question"],
                "llm_type": result["llm_type"],
                "answer_length": len(result["answer"]),
                "total_time": result["total_time"],
                "retrieval_time": result.get("retrieval_time", 0),
                "generation_time": result["total_time"] - result.get("retrieval_time", 0)
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv('research/rag_pipeline_summary.csv', index=False)
        
        logger.info("RAG pipeline evaluation completed. Results saved to rag_pipeline_results.json")
    
    except Exception as e:
        logger.error(f"Error in RAG pipeline evaluation: {e}")

def main():
    """Main function to run the RAG pipeline test."""
    try:
        evaluate_rag_pipeline()
    except Exception as e:
        logger.error(f"Error in RAG pipeline test: {e}")

if __name__ == "__main__":
    main() 