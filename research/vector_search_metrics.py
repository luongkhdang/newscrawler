"""
Vector Search Metrics Analysis Script

This script tests similarity search using pgvector and evaluates different distance metrics:
1. Cosine similarity
2. Euclidean distance
3. Dot product

It also tests different index types and search configurations.
"""

import os
import sys
import time
import logging
import json
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from langchain.embeddings import HuggingFaceEmbeddings

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.session import SessionLocal
from src.database.models import Article
from src.vector.processor import generate_embedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('research/vector_search_metrics.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Test queries for search evaluation
TEST_QUERIES = [
    "Latest developments in artificial intelligence",
    "Climate change impact on agriculture",
    "Cryptocurrency market trends",
    "Advances in renewable energy technology",
    "Global economic outlook post-pandemic"
]

# Distance metrics to test
DISTANCE_METRICS = [
    {"name": "cosine", "function": "cosine_similarity", "order": "DESC"},
    {"name": "euclidean", "function": "euclidean_distance", "order": "ASC"},
    {"name": "dot", "function": "dot_product", "order": "DESC"}
]

def setup_test_data(db, num_articles=10):
    """
    Set up test data if needed.
    This function checks if there are enough articles with embeddings for testing.
    If not, it processes more articles to generate embeddings.
    """
    # Check if we have enough articles with embeddings
    articles_with_embeddings = db.query(Article).filter(Article.vector_embedding.is_not(None)).count()
    logger.info(f"Found {articles_with_embeddings} articles with embeddings")
    
    if articles_with_embeddings >= num_articles:
        logger.info(f"Sufficient articles with embeddings for testing")
        return True
    
    # Process more articles to generate embeddings
    articles_to_process = db.query(Article).filter(
        Article.vector_embedding.is_(None)
    ).limit(num_articles - articles_with_embeddings).all()
    
    if not articles_to_process:
        logger.warning(f"Not enough articles available for processing")
        return False
    
    logger.info(f"Processing {len(articles_to_process)} articles to generate embeddings")
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    for article in articles_to_process:
        try:
            # Combine title and content for embedding
            text_to_embed = f"{article.title}\n\n{article.content}"
            
            # Generate embedding
            embedding = embedding_model.embed_query(text_to_embed)
            
            # Update article with embedding
            article.vector_embedding = embedding
            
            logger.info(f"Generated embedding for article: {article.title[:50]}...")
        
        except Exception as e:
            logger.error(f"Error generating embedding for article {article.id}: {e}")
    
    # Commit changes
    db.commit()
    
    # Verify we now have enough articles with embeddings
    articles_with_embeddings = db.query(Article).filter(Article.vector_embedding.is_not(None)).count()
    logger.info(f"Now have {articles_with_embeddings} articles with embeddings")
    
    return articles_with_embeddings >= num_articles

def test_vector_search(db, query: str, metric: Dict[str, str], limit: int = 5) -> Tuple[List[Dict[str, Any]], float]:
    """
    Test vector search using the specified distance metric.
    
    Args:
        db: Database session
        query: Search query
        metric: Distance metric configuration
        limit: Maximum number of results to return
        
    Returns:
        Tuple of (results, search_time)
    """
    # Generate embedding for the query
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    query_embedding = embedding_model.embed_query(query)
    
    # Construct SQL query based on the distance metric
    if metric["name"] == "cosine":
        sql_query = text(f"""
            SELECT id, title, source_domain, 
                   cosine_similarity(vector_embedding, :embedding) as similarity
            FROM articles
            WHERE vector_embedding IS NOT NULL
            ORDER BY similarity DESC
            LIMIT :limit
        """)
    elif metric["name"] == "euclidean":
        sql_query = text(f"""
            SELECT id, title, source_domain, 
                   euclidean_distance(vector_embedding, :embedding) as distance
            FROM articles
            WHERE vector_embedding IS NOT NULL
            ORDER BY distance ASC
            LIMIT :limit
        """)
    elif metric["name"] == "dot":
        sql_query = text(f"""
            SELECT id, title, source_domain, 
                   vector_embedding <#> :embedding as dot_product
            FROM articles
            WHERE vector_embedding IS NOT NULL
            ORDER BY dot_product DESC
            LIMIT :limit
        """)
    else:
        raise ValueError(f"Unsupported distance metric: {metric['name']}")
    
    # Execute search query and measure time
    start_time = time.time()
    result = db.execute(sql_query, {"embedding": query_embedding, "limit": limit})
    search_time = time.time() - start_time
    
    # Process results
    results = []
    for row in result:
        results.append({
            "id": str(row[0]),
            "title": row[1],
            "source_domain": row[2],
            "score": float(row[3])
        })
    
    return results, search_time

def test_hybrid_search(db, query: str, limit: int = 5) -> Tuple[List[Dict[str, Any]], float]:
    """
    Test hybrid search combining vector similarity and full-text search.
    
    Args:
        db: Database session
        query: Search query
        limit: Maximum number of results to return
        
    Returns:
        Tuple of (results, search_time)
    """
    # Generate embedding for the query
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    query_embedding = embedding_model.embed_query(query)
    
    # Construct SQL query for hybrid search
    # This combines vector similarity with text similarity using a weighted approach
    sql_query = text(f"""
        SELECT id, title, source_domain,
               (0.7 * cosine_similarity(vector_embedding, :embedding) + 
                0.3 * ts_rank(to_tsvector('english', title || ' ' || content), 
                             plainto_tsquery('english', :text_query))) as hybrid_score
        FROM articles
        WHERE vector_embedding IS NOT NULL
        ORDER BY hybrid_score DESC
        LIMIT :limit
    """)
    
    # Execute search query and measure time
    start_time = time.time()
    result = db.execute(sql_query, {
        "embedding": query_embedding, 
        "text_query": query,
        "limit": limit
    })
    search_time = time.time() - start_time
    
    # Process results
    results = []
    for row in result:
        results.append({
            "id": str(row[0]),
            "title": row[1],
            "source_domain": row[2],
            "score": float(row[3])
        })
    
    return results, search_time

def evaluate_search_metrics():
    """Evaluate different search metrics and approaches."""
    logger.info("Starting vector search metrics evaluation...")
    
    db = SessionLocal()
    try:
        # Ensure we have test data
        if not setup_test_data(db, num_articles=20):
            logger.error("Failed to set up test data")
            return
        
        results = []
        
        # Test each distance metric
        for metric in DISTANCE_METRICS:
            logger.info(f"Testing {metric['name']} distance metric...")
            
            metric_results = {
                "metric_name": metric["name"],
                "queries": []
            }
            
            total_search_time = 0
            for query in TEST_QUERIES:
                try:
                    search_results, search_time = test_vector_search(db, query, metric)
                    total_search_time += search_time
                    
                    metric_results["queries"].append({
                        "query": query,
                        "search_time": search_time,
                        "results": search_results
                    })
                    
                    logger.info(f"  Query: '{query}' - Search time: {search_time:.4f}s - Results: {len(search_results)}")
                
                except Exception as e:
                    logger.error(f"Error testing {metric['name']} with query '{query}': {e}")
            
            metric_results["avg_search_time"] = total_search_time / len(TEST_QUERIES) if TEST_QUERIES else 0
            results.append(metric_results)
        
        # Test hybrid search
        logger.info("Testing hybrid search...")
        hybrid_results = {
            "metric_name": "hybrid",
            "queries": []
        }
        
        total_search_time = 0
        for query in TEST_QUERIES:
            try:
                search_results, search_time = test_hybrid_search(db, query)
                total_search_time += search_time
                
                hybrid_results["queries"].append({
                    "query": query,
                    "search_time": search_time,
                    "results": search_results
                })
                
                logger.info(f"  Query: '{query}' - Search time: {search_time:.4f}s - Results: {len(search_results)}")
            
            except Exception as e:
                logger.error(f"Error testing hybrid search with query '{query}': {e}")
        
        hybrid_results["avg_search_time"] = total_search_time / len(TEST_QUERIES) if TEST_QUERIES else 0
        results.append(hybrid_results)
        
        # Save detailed results to JSON
        with open('research/vector_search_metrics_detailed.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary DataFrame
        summary = []
        for result in results:
            summary.append({
                "metric_name": result["metric_name"],
                "avg_search_time": result["avg_search_time"]
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv('research/vector_search_metrics_summary.csv', index=False)
        
        logger.info("Vector search metrics evaluation completed. Results saved to vector_search_metrics_summary.csv")
    
    finally:
        db.close()

def main():
    """Main function to run the vector search metrics analysis."""
    try:
        evaluate_search_metrics()
    except Exception as e:
        logger.error(f"Error in vector search metrics analysis: {e}")

if __name__ == "__main__":
    main() 