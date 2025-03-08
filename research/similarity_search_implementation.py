"""
Similarity Search Implementation and Testing Script

This script implements and tests similarity search using pgvector with:
1. Different distance metrics (cosine, euclidean, dot product)
2. Different index types (IVFFlat, HNSW)
3. Performance benchmarking with varying dataset sizes
4. Hybrid search combining vector and keyword search
"""

import os
import sys
import time
import logging
import json
import random
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, Column, String, Text, DateTime, func
from sqlalchemy.dialects.postgresql import UUID, VECTOR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('research/similarity_search_implementation.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://newscrawler:newscrawler_password@localhost:5432/newscrawler")
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# Define test queries
TEST_QUERIES = [
    "Latest developments in artificial intelligence",
    "Climate change impact on agriculture",
    "Cryptocurrency market trends",
    "Advances in renewable energy technology",
    "Global economic outlook post-pandemic"
]

# Define embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Define distance metrics to test
DISTANCE_METRICS = [
    {"name": "cosine", "operator": "<=>", "order": "ASC", "similarity_func": "1 - (vector_embedding <=> :query_vector)"},
    {"name": "euclidean", "operator": "<->", "order": "ASC", "similarity_func": "1 / (1 + (vector_embedding <-> :query_vector))"},
    {"name": "inner_product", "operator": "<#>", "order": "DESC", "similarity_func": "vector_embedding <#> :query_vector"}
]

# Define index configurations to test
INDEX_CONFIGS = [
    {
        "name": "No Index",
        "create_sql": None,
        "drop_sql": None
    },
    {
        "name": "IVFFlat-100",
        "create_sql": "CREATE INDEX ON articles USING ivfflat (vector_embedding vector_cosine_ops) WITH (lists = 100)",
        "drop_sql": "DROP INDEX IF EXISTS articles_vector_embedding_idx"
    },
    {
        "name": "IVFFlat-200",
        "create_sql": "CREATE INDEX ON articles USING ivfflat (vector_embedding vector_cosine_ops) WITH (lists = 200)",
        "drop_sql": "DROP INDEX IF EXISTS articles_vector_embedding_idx"
    },
    {
        "name": "HNSW-16-64",
        "create_sql": "CREATE INDEX ON articles USING hnsw (vector_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
        "drop_sql": "DROP INDEX IF EXISTS articles_vector_embedding_idx"
    },
    {
        "name": "HNSW-32-100",
        "create_sql": "CREATE INDEX ON articles USING hnsw (vector_embedding vector_cosine_ops) WITH (m = 32, ef_construction = 100)",
        "drop_sql": "DROP INDEX IF EXISTS articles_vector_embedding_idx"
    }
]

def get_embedding_model():
    """Get the embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a text."""
    embedding_model = get_embedding_model()
    return embedding_model.embed_query(text)

def vector_to_pg_format(vector: List[float]) -> str:
    """Convert vector to PostgreSQL format."""
    return str(vector).replace('[', '{').replace(']', '}')

def test_vector_search(db_session, query: str, metric: Dict[str, str], limit: int = 10) -> Tuple[List[Dict[str, Any]], float]:
    """
    Test vector search using the specified distance metric.
    
    Args:
        db_session: Database session
        query: Search query
        metric: Distance metric configuration
        limit: Maximum number of results to return
        
    Returns:
        Tuple of (results, search_time)
    """
    # Generate embedding for the query
    query_embedding = generate_embedding(query)
    query_vector = vector_to_pg_format(query_embedding)
    
    # Construct SQL query based on the distance metric
    sql_query = text(f"""
        SELECT id, url, title, source_domain, 
               {metric['similarity_func']} as similarity_score
        FROM articles
        WHERE vector_embedding IS NOT NULL
        ORDER BY vector_embedding {metric['operator']} :query_vector {metric['order']}
        LIMIT :limit
    """)
    
    # Execute search query and measure time
    start_time = time.time()
    result = db_session.execute(sql_query, {"query_vector": query_vector, "limit": limit})
    search_time = time.time() - start_time
    
    # Process results
    results = []
    for row in result:
        results.append({
            "id": str(row[0]),
            "url": row[1],
            "title": row[2],
            "source_domain": row[3],
            "similarity_score": float(row[4])
        })
    
    return results, search_time

def test_hybrid_search(db_session, query: str, limit: int = 10) -> Tuple[List[Dict[str, Any]], float]:
    """
    Test hybrid search combining vector similarity and full-text search.
    
    Args:
        db_session: Database session
        query: Search query
        limit: Maximum number of results to return
        
    Returns:
        Tuple of (results, search_time)
    """
    # Generate embedding for the query
    query_embedding = generate_embedding(query)
    query_vector = vector_to_pg_format(query_embedding)
    
    # Convert query to tsquery format
    tsquery = ' & '.join(query.split())
    
    # Construct SQL query for hybrid search
    sql_query = text(f"""
        WITH vector_search AS (
            SELECT id, url, title, source_domain,
                   1 - (vector_embedding <=> :query_vector) AS vector_similarity
            FROM articles
            WHERE vector_embedding IS NOT NULL
            ORDER BY vector_embedding <=> :query_vector
            LIMIT 100
        ),
        text_search AS (
            SELECT id, url, title, source_domain,
                   ts_rank(to_tsvector('english', title || ' ' || content), to_tsquery('english', :tsquery)) AS text_rank
            FROM articles
            WHERE to_tsvector('english', title || ' ' || content) @@ to_tsquery('english', :tsquery)
            LIMIT 100
        )
        SELECT v.id, v.url, v.title, v.source_domain,
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
    search_time = time.time() - start_time
    
    # Process results
    results = []
    for row in result:
        results.append({
            "id": str(row[0]),
            "url": row[1],
            "title": row[2],
            "source_domain": row[3],
            "combined_score": float(row[4])
        })
    
    return results, search_time

def benchmark_index_performance():
    """Benchmark performance of different index configurations."""
    logger.info("Benchmarking index performance...")
    
    db = Session()
    try:
        results = []
        
        # Get count of articles with embeddings
        count_query = text("SELECT COUNT(*) FROM articles WHERE vector_embedding IS NOT NULL")
        article_count = db.execute(count_query).scalar()
        logger.info(f"Found {article_count} articles with embeddings")
        
        # Generate query embeddings
        query_embeddings = []
        for query in TEST_QUERIES:
            embedding = generate_embedding(query)
            query_embeddings.append(embedding)
        
        # Test each index configuration
        for index_config in INDEX_CONFIGS:
            logger.info(f"Testing index configuration: {index_config['name']}")
            
            # Create index if needed
            if index_config["create_sql"]:
                logger.info(f"Creating index: {index_config['create_sql']}")
                db.execute(text(index_config["create_sql"]))
                db.commit()
            
            # Test each distance metric
            for metric in DISTANCE_METRICS:
                logger.info(f"Testing distance metric: {metric['name']}")
                
                metric_results = []
                for i, query_embedding in enumerate(query_embeddings):
                    query = TEST_QUERIES[i]
                    query_vector = vector_to_pg_format(query_embedding)
                    
                    # Construct SQL query
                    sql_query = text(f"""
                        SELECT id, title, 
                               {metric['similarity_func']} as similarity_score
                        FROM articles
                        WHERE vector_embedding IS NOT NULL
                        ORDER BY vector_embedding {metric['operator']} :query_vector {metric['order']}
                        LIMIT 10
                    """)
                    
                    # Execute search query and measure time
                    start_time = time.time()
                    db.execute(sql_query, {"query_vector": query_vector})
                    search_time = time.time() - start_time
                    
                    metric_results.append({
                        "query": query,
                        "search_time": search_time
                    })
                
                # Calculate average search time
                avg_search_time = sum(r["search_time"] for r in metric_results) / len(metric_results)
                
                results.append({
                    "index_name": index_config["name"],
                    "metric_name": metric["name"],
                    "avg_search_time": avg_search_time,
                    "queries_per_second": 1.0 / avg_search_time if avg_search_time > 0 else 0
                })
                
                logger.info(f"Average search time: {avg_search_time:.4f}s ({1.0/avg_search_time if avg_search_time > 0 else 0:.2f} queries/sec)")
            
            # Drop index for next test
            if index_config["drop_sql"]:
                logger.info(f"Dropping index: {index_config['drop_sql']}")
                db.execute(text(index_config["drop_sql"]))
                db.commit()
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv('research/results/index_performance_benchmark.csv', index=False)
        
        logger.info("Index performance benchmark completed")
        
        return df
    
    finally:
        db.close()

def benchmark_hybrid_search():
    """Benchmark hybrid search performance."""
    logger.info("Benchmarking hybrid search performance...")
    
    db = Session()
    try:
        # Create HNSW index for optimal performance
        logger.info("Creating HNSW index for hybrid search benchmark")
        db.execute(text("DROP INDEX IF EXISTS articles_vector_embedding_idx"))
        db.execute(text("CREATE INDEX ON articles USING hnsw (vector_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"))
        db.commit()
        
        # Create tsvector index if it doesn't exist
        logger.info("Creating tsvector index for text search")
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS articles_content_tsvector_idx 
            ON articles USING GIN (to_tsvector('english', title || ' ' || content))
        """))
        db.commit()
        
        # Test vector search, text search, and hybrid search
        search_types = ["vector", "text", "hybrid"]
        results = []
        
        for query in TEST_QUERIES:
            logger.info(f"Testing search performance for query: '{query}'")
            
            # Vector search
            query_embedding = generate_embedding(query)
            query_vector = vector_to_pg_format(query_embedding)
            
            # Vector search
            vector_query = text("""
                SELECT id, title
                FROM articles
                WHERE vector_embedding IS NOT NULL
                ORDER BY vector_embedding <=> :query_vector
                LIMIT 10
            """)
            
            start_time = time.time()
            db.execute(vector_query, {"query_vector": query_vector})
            vector_time = time.time() - start_time
            
            # Text search
            tsquery = ' & '.join(query.split())
            text_query = text("""
                SELECT id, title
                FROM articles
                WHERE to_tsvector('english', title || ' ' || content) @@ to_tsquery('english', :tsquery)
                ORDER BY ts_rank(to_tsvector('english', title || ' ' || content), to_tsquery('english', :tsquery)) DESC
                LIMIT 10
            """)
            
            start_time = time.time()
            db.execute(text_query, {"tsquery": tsquery})
            text_time = time.time() - start_time
            
            # Hybrid search
            hybrid_query = text("""
                WITH vector_search AS (
                    SELECT id, url, title,
                           1 - (vector_embedding <=> :query_vector) AS vector_similarity
                    FROM articles
                    WHERE vector_embedding IS NOT NULL
                    ORDER BY vector_embedding <=> :query_vector
                    LIMIT 100
                ),
                text_search AS (
                    SELECT id, url, title,
                           ts_rank(to_tsvector('english', title || ' ' || content), to_tsquery('english', :tsquery)) AS text_rank
                    FROM articles
                    WHERE to_tsvector('english', title || ' ' || content) @@ to_tsquery('english', :tsquery)
                    LIMIT 100
                )
                SELECT v.id, v.title,
                       (0.7 * v.vector_similarity + 0.3 * COALESCE(t.text_rank, 0)) AS combined_score
                FROM vector_search v
                LEFT JOIN text_search t ON v.id = t.id
                ORDER BY combined_score DESC
                LIMIT 10
            """)
            
            start_time = time.time()
            db.execute(hybrid_query, {"query_vector": query_vector, "tsquery": tsquery})
            hybrid_time = time.time() - start_time
            
            results.append({
                "query": query,
                "vector_search_time": vector_time,
                "text_search_time": text_time,
                "hybrid_search_time": hybrid_time,
                "vector_queries_per_second": 1.0 / vector_time if vector_time > 0 else 0,
                "text_queries_per_second": 1.0 / text_time if text_time > 0 else 0,
                "hybrid_queries_per_second": 1.0 / hybrid_time if hybrid_time > 0 else 0
            })
            
            logger.info(f"Vector search: {vector_time:.4f}s, Text search: {text_time:.4f}s, Hybrid search: {hybrid_time:.4f}s")
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv('research/results/hybrid_search_benchmark.csv', index=False)
        
        # Calculate averages
        avg_results = {
            "avg_vector_search_time": df["vector_search_time"].mean(),
            "avg_text_search_time": df["text_search_time"].mean(),
            "avg_hybrid_search_time": df["hybrid_search_time"].mean(),
            "avg_vector_qps": df["vector_queries_per_second"].mean(),
            "avg_text_qps": df["text_queries_per_second"].mean(),
            "avg_hybrid_qps": df["hybrid_queries_per_second"].mean()
        }
        
        logger.info(f"Average search times - Vector: {avg_results['avg_vector_search_time']:.4f}s, Text: {avg_results['avg_text_search_time']:.4f}s, Hybrid: {avg_results['avg_hybrid_search_time']:.4f}s")
        
        return df
    
    finally:
        db.close()

def evaluate_search_quality():
    """Evaluate search quality for different search methods."""
    logger.info("Evaluating search quality...")
    
    db = Session()
    try:
        # Create HNSW index for optimal performance
        logger.info("Creating HNSW index for search quality evaluation")
        db.execute(text("DROP INDEX IF EXISTS articles_vector_embedding_idx"))
        db.execute(text("CREATE INDEX ON articles USING hnsw (vector_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"))
        db.commit()
        
        # Create tsvector index if it doesn't exist
        logger.info("Creating tsvector index for text search")
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS articles_content_tsvector_idx 
            ON articles USING GIN (to_tsvector('english', title || ' ' || content))
        """))
        db.commit()
        
        results = []
        
        for query in TEST_QUERIES:
            logger.info(f"Evaluating search quality for query: '{query}'")
            
            # Vector search
            vector_results, vector_time = test_vector_search(db, query, DISTANCE_METRICS[0], limit=10)
            
            # Hybrid search
            hybrid_results, hybrid_time = test_hybrid_search(db, query, limit=10)
            
            # Convert query to tsquery format
            tsquery = ' & '.join(query.split())
            
            # Text search
            text_query = text("""
                SELECT id, url, title, source_domain,
                       ts_rank(to_tsvector('english', title || ' ' || content), to_tsquery('english', :tsquery)) AS text_rank
                FROM articles
                WHERE to_tsvector('english', title || ' ' || content) @@ to_tsquery('english', :tsquery)
                ORDER BY text_rank DESC
                LIMIT 10
            """)
            
            start_time = time.time()
            text_result = db.execute(text_query, {"tsquery": tsquery})
            text_time = time.time() - start_time
            
            text_results = []
            for row in text_result:
                text_results.append({
                    "id": str(row[0]),
                    "url": row[1],
                    "title": row[2],
                    "source_domain": row[3],
                    "text_rank": float(row[4])
                })
            
            # Save results for this query
            results.append({
                "query": query,
                "vector_results": vector_results,
                "text_results": text_results,
                "hybrid_results": hybrid_results,
                "vector_time": vector_time,
                "text_time": text_time,
                "hybrid_time": hybrid_time
            })
        
        # Save detailed results to JSON
        with open('research/results/search_quality_evaluation.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Search quality evaluation completed")
        
        return results
    
    finally:
        db.close()

def main():
    """Main function to run all benchmarks and evaluations."""
    logger.info("Starting similarity search implementation and testing...")
    
    # Create results directory if it doesn't exist
    os.makedirs('research/results', exist_ok=True)
    
    # Run benchmarks
    benchmark_index_performance()
    benchmark_hybrid_search()
    evaluate_search_quality()
    
    logger.info("Similarity search implementation and testing completed")

if __name__ == "__main__":
    main() 