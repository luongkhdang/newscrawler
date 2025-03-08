"""
Embedding Analysis Script

This script analyzes the current embedding implementation in the NewsCrawler project.
It tests different embedding models, chunking strategies, and performance metrics.
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vector.processor import get_embedding_model, generate_embedding, create_text_chunks
from src.database.session import SessionLocal
from src.database.models import Article

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('research/embedding_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def analyze_current_implementation():
    """Analyze the current embedding implementation."""
    logger.info("Analyzing current embedding implementation...")
    
    # Get the current embedding model
    model = get_embedding_model()
    logger.info(f"Current embedding model: {model.__class__.__name__}")
    
    # Test embedding generation
    test_text = "This is a test article about artificial intelligence and machine learning."
    
    start_time = time.time()
    embedding = generate_embedding(test_text)
    end_time = time.time()
    
    logger.info(f"Embedding dimension: {len(embedding)}")
    logger.info(f"Embedding generation time: {end_time - start_time:.4f} seconds")
    
    # Test text chunking
    metadata = {"source": "test", "title": "Test Article"}
    chunks = create_text_chunks(test_text, metadata)
    
    logger.info(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk {i+1}: {len(chunk.page_content)} characters")
        logger.info(f"Chunk {i+1} metadata: {chunk.metadata}")
    
    return {
        "model_name": model.__class__.__name__,
        "embedding_dimension": len(embedding),
        "embedding_time": end_time - start_time,
        "num_chunks": len(chunks)
    }

def analyze_database_embeddings():
    """Analyze embeddings stored in the database."""
    logger.info("Analyzing database embeddings...")
    
    db = SessionLocal()
    try:
        # Count articles with and without embeddings
        total_articles = db.query(Article).count()
        articles_with_embeddings = db.query(Article).filter(Article.vector_embedding.is_not(None)).count()
        articles_without_embeddings = db.query(Article).filter(Article.vector_embedding.is_(None)).count()
        
        logger.info(f"Total articles: {total_articles}")
        logger.info(f"Articles with embeddings: {articles_with_embeddings}")
        logger.info(f"Articles without embeddings: {articles_without_embeddings}")
        
        # Sample a few embeddings to analyze their properties
        if articles_with_embeddings > 0:
            sample_articles = db.query(Article).filter(Article.vector_embedding.is_not(None)).limit(5).all()
            
            for i, article in enumerate(sample_articles):
                embedding = article.vector_embedding
                logger.info(f"Sample {i+1} embedding dimension: {len(embedding)}")
                
                # Calculate basic statistics
                embedding_array = np.array(embedding)
                logger.info(f"Sample {i+1} embedding mean: {np.mean(embedding_array):.6f}")
                logger.info(f"Sample {i+1} embedding std: {np.std(embedding_array):.6f}")
                logger.info(f"Sample {i+1} embedding min: {np.min(embedding_array):.6f}")
                logger.info(f"Sample {i+1} embedding max: {np.max(embedding_array):.6f}")
        
        return {
            "total_articles": total_articles,
            "articles_with_embeddings": articles_with_embeddings,
            "articles_without_embeddings": articles_without_embeddings,
            "embedding_coverage": articles_with_embeddings / total_articles if total_articles > 0 else 0
        }
    
    finally:
        db.close()

def main():
    """Main function to run the embedding analysis."""
    logger.info("Starting embedding analysis...")
    
    # Analyze current implementation
    implementation_results = analyze_current_implementation()
    
    # Analyze database embeddings
    database_results = analyze_database_embeddings()
    
    # Combine results
    results = {**implementation_results, **database_results}
    
    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_df.to_csv('research/embedding_analysis_results.csv', index=False)
    
    logger.info("Embedding analysis completed. Results saved to embedding_analysis_results.csv")

if __name__ == "__main__":
    main() 