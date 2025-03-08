#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pgvector Benchmark Script

This script benchmarks the performance of pgvector with different configurations
for the NewsCrawler project. It tests various index types, parameters, and query
patterns to determine the optimal configuration.

Usage:
    python pgvector_benchmark.py --connection-string "postgresql://user:pass@localhost:5432/dbname" --dataset-size 10000
"""

import argparse
import csv
import json
import logging
import os
import random
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psycopg2
import sqlalchemy
from sqlalchemy import Column, String, Text, create_engine
from sqlalchemy.dialects.postgresql import UUID, VECTOR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pgvector_benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create SQLAlchemy base
Base = declarative_base()

class Article(Base):
    """Article model for benchmarking."""
    __tablename__ = "benchmark_articles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    vector_embedding = Column(VECTOR(1536), nullable=True)

    def __repr__(self):
        return f"<Article(id={self.id}, title={self.title})>"


class PgVectorBenchmark:
    """Benchmark pgvector performance with different configurations."""

    def __init__(self, connection_string: str, dataset_size: int = 10000, dimension: int = 1536):
        """
        Initialize the benchmark.

        Args:
            connection_string: PostgreSQL connection string
            dataset_size: Number of articles to generate for benchmarking
            dimension: Dimension of the vector embeddings
        """
        self.connection_string = connection_string
        self.dataset_size = dataset_size
        self.dimension = dimension
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "dataset_size": dataset_size,
            "dimension": dimension,
            "index_benchmarks": [],
            "query_benchmarks": [],
            "scaling_benchmarks": [],
        }

    def setup_database(self):
        """Set up the database for benchmarking."""
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Clear existing data
        with self.Session() as session:
            session.execute(f"TRUNCATE TABLE {Article.__tablename__} CASCADE")
            session.commit()
            
        logger.info(f"Database setup complete. Tables created and cleared.")

    def generate_test_data(self):
        """Generate test data for benchmarking."""
        logger.info(f"Generating {self.dataset_size} test articles with {self.dimension}-dimensional embeddings")
        
        # Generate random articles with embeddings
        batch_size = 1000
        num_batches = (self.dataset_size + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Generating test data"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, self.dataset_size)
            batch_size_actual = end_idx - start_idx
            
            with self.Session() as session:
                articles = []
                for i in range(batch_size_actual):
                    # Generate random embedding
                    embedding = np.random.randn(self.dimension).astype(np.float32)
                    # Normalize to unit length for cosine similarity
                    embedding = embedding / np.linalg.norm(embedding)
                    
                    article = Article(
                        id=uuid.uuid4(),
                        title=f"Test Article {start_idx + i}",
                        content=f"This is test article {start_idx + i} for pgvector benchmarking.",
                        vector_embedding=embedding.tolist()
                    )
                    articles.append(article)
                
                session.add_all(articles)
                session.commit()
        
        logger.info(f"Test data generation complete. {self.dataset_size} articles created.")

    def benchmark_index_creation(self):
        """Benchmark the creation time for different index types."""
        logger.info("Benchmarking index creation time")
        
        index_configs = [
            {
                "name": "No Index",
                "sql": None,
                "drop_sql": None
            },
            {
                "name": "IVFFlat (lists=100)",
                "sql": f"CREATE INDEX ON {Article.__tablename__} USING ivfflat (vector_embedding vector_cosine_ops) WITH (lists = 100)",
                "drop_sql": f"DROP INDEX IF EXISTS {Article.__tablename__}_vector_embedding_idx"
            },
            {
                "name": "IVFFlat (lists={})".format(int(np.sqrt(self.dataset_size))),
                "sql": f"CREATE INDEX ON {Article.__tablename__} USING ivfflat (vector_embedding vector_cosine_ops) WITH (lists = {int(np.sqrt(self.dataset_size))})",
                "drop_sql": f"DROP INDEX IF EXISTS {Article.__tablename__}_vector_embedding_idx"
            },
            {
                "name": "HNSW (m=16, ef_construction=64)",
                "sql": f"CREATE INDEX ON {Article.__tablename__} USING hnsw (vector_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
                "drop_sql": f"DROP INDEX IF EXISTS {Article.__tablename__}_vector_embedding_idx"
            }
        ]
        
        for config in index_configs:
            # Skip if no index
            if config["sql"] is None:
                self.results["index_benchmarks"].append({
                    "index_type": config["name"],
                    "creation_time": 0,
                    "index_size": 0
                })
                continue
                
            # Drop existing index if any
            if config["drop_sql"]:
                with self.engine.connect() as conn:
                    conn.execute(config["drop_sql"])
                    
            # Create index and measure time
            start_time = time.time()
            with self.engine.connect() as conn:
                conn.execute(config["sql"])
            end_time = time.time()
            creation_time = end_time - start_time
            
            # Get index size
            with self.engine.connect() as conn:
                result = conn.execute(f"""
                    SELECT pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
                           pg_relation_size(indexrelid) as index_size_bytes
                    FROM pg_stat_user_indexes
                    WHERE relname = '{Article.__tablename__}'
                    ORDER BY indexrelid DESC
                    LIMIT 1
                """).fetchone()
                
                index_size = result[0] if result else "N/A"
                index_size_bytes = result[1] if result else 0
            
            self.results["index_benchmarks"].append({
                "index_type": config["name"],
                "creation_time": creation_time,
                "index_size": index_size,
                "index_size_bytes": index_size_bytes
            })
            
            logger.info(f"Index {config['name']} created in {creation_time:.2f} seconds. Size: {index_size}")
            
            # Drop index for next test
            if config["drop_sql"]:
                with self.engine.connect() as conn:
                    conn.execute(config["drop_sql"])

    def benchmark_query_performance(self):
        """Benchmark query performance with different index types."""
        logger.info("Benchmarking query performance")
        
        # Generate random query vectors
        num_queries = 100
        query_vectors = []
        for _ in range(num_queries):
            embedding = np.random.randn(self.dimension).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            query_vectors.append(embedding.tolist())
        
        index_configs = [
            {
                "name": "No Index",
                "sql": None,
                "drop_sql": None
            },
            {
                "name": "IVFFlat (lists=100)",
                "sql": f"CREATE INDEX ON {Article.__tablename__} USING ivfflat (vector_embedding vector_cosine_ops) WITH (lists = 100)",
                "drop_sql": f"DROP INDEX IF EXISTS {Article.__tablename__}_vector_embedding_idx"
            },
            {
                "name": "HNSW (m=16, ef_construction=64)",
                "sql": f"CREATE INDEX ON {Article.__tablename__} USING hnsw (vector_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
                "drop_sql": f"DROP INDEX IF EXISTS {Article.__tablename__}_vector_embedding_idx"
            }
        ]
        
        for config in index_configs:
            # Create index if needed
            if config["sql"]:
                with self.engine.connect() as conn:
                    conn.execute(config["sql"])
            
            # Run queries and measure time
            total_time = 0
            for query_vector in tqdm(query_vectors, desc=f"Testing {config['name']}"):
                query_vector_str = str(query_vector).replace('[', '{').replace(']', '}')
                
                start_time = time.time()
                with self.engine.connect() as conn:
                    conn.execute(f"""
                        SELECT id, title, 1 - (vector_embedding <=> '{query_vector_str}'::vector) AS similarity
                        FROM {Article.__tablename__}
                        ORDER BY vector_embedding <=> '{query_vector_str}'::vector
                        LIMIT 10
                    """)
                end_time = time.time()
                total_time += (end_time - start_time)
            
            avg_query_time = total_time / num_queries
            
            self.results["query_benchmarks"].append({
                "index_type": config["name"],
                "avg_query_time": avg_query_time,
                "queries_per_second": 1.0 / avg_query_time if avg_query_time > 0 else 0
            })
            
            logger.info(f"Index {config['name']} average query time: {avg_query_time*1000:.2f}ms ({1.0/avg_query_time if avg_query_time > 0 else 0:.2f} queries/sec)")
            
            # Drop index for next test
            if config["drop_sql"]:
                with self.engine.connect() as conn:
                    conn.execute(config["drop_sql"])

    def benchmark_scaling(self):
        """Benchmark how performance scales with dataset size."""
        logger.info("Benchmarking scaling behavior")
        
        # Define dataset sizes to test
        if self.dataset_size >= 100000:
            sizes = [10000, 50000, 100000, self.dataset_size]
        elif self.dataset_size >= 50000:
            sizes = [10000, 25000, 50000, self.dataset_size]
        elif self.dataset_size >= 10000:
            sizes = [1000, 5000, 10000, self.dataset_size]
        else:
            sizes = [100, 500, 1000, self.dataset_size]
        
        # Generate a random query vector
        query_embedding = np.random.randn(self.dimension).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_vector_str = str(query_embedding.tolist()).replace('[', '{').replace(']', '}')
        
        index_configs = [
            {
                "name": "IVFFlat",
                "sql": lambda size: f"CREATE INDEX ON {Article.__tablename__} USING ivfflat (vector_embedding vector_cosine_ops) WITH (lists = {int(np.sqrt(size))})",
                "drop_sql": f"DROP INDEX IF EXISTS {Article.__tablename__}_vector_embedding_idx"
            },
            {
                "name": "HNSW",
                "sql": lambda size: f"CREATE INDEX ON {Article.__tablename__} USING hnsw (vector_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
                "drop_sql": f"DROP INDEX IF EXISTS {Article.__tablename__}_vector_embedding_idx"
            }
        ]
        
        for size in sizes:
            for config in index_configs:
                # Create index
                with self.engine.connect() as conn:
                    conn.execute(config["drop_sql"])
                    conn.execute(config["sql"](size))
                
                # Run queries with LIMIT size
                num_queries = 20
                total_time = 0
                
                for _ in range(num_queries):
                    start_time = time.time()
                    with self.engine.connect() as conn:
                        conn.execute(f"""
                            SELECT id, title, 1 - (vector_embedding <=> '{query_vector_str}'::vector) AS similarity
                            FROM {Article.__tablename__}
                            ORDER BY vector_embedding <=> '{query_vector_str}'::vector
                            LIMIT 10
                        """)
                    end_time = time.time()
                    total_time += (end_time - start_time)
                
                avg_query_time = total_time / num_queries
                
                self.results["scaling_benchmarks"].append({
                    "index_type": config["name"],
                    "dataset_size": size,
                    "avg_query_time": avg_query_time,
                    "queries_per_second": 1.0 / avg_query_time if avg_query_time > 0 else 0
                })
                
                logger.info(f"Index {config['name']} with {size} articles: {avg_query_time*1000:.2f}ms ({1.0/avg_query_time if avg_query_time > 0 else 0:.2f} queries/sec)")
                
                # Drop index
                with self.engine.connect() as conn:
                    conn.execute(config["drop_sql"])

    def benchmark_distance_metrics(self):
        """Benchmark different distance metrics."""
        logger.info("Benchmarking distance metrics")
        
        # Generate a random query vector
        query_embedding = np.random.randn(self.dimension).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_vector_str = str(query_embedding.tolist()).replace('[', '{').replace(']', '}')
        
        # Create IVFFlat index
        with self.engine.connect() as conn:
            conn.execute(f"DROP INDEX IF EXISTS {Article.__tablename__}_vector_embedding_idx")
            conn.execute(f"CREATE INDEX ON {Article.__tablename__} USING ivfflat (vector_embedding vector_cosine_ops) WITH (lists = 100)")
        
        distance_metrics = [
            {
                "name": "Cosine Distance",
                "operator": "<=>",
                "order": "ASC"
            },
            {
                "name": "Euclidean Distance",
                "operator": "<->",
                "order": "ASC"
            },
            {
                "name": "Inner Product",
                "operator": "<#>",
                "order": "DESC"  # Higher inner product is better
            }
        ]
        
        for metric in distance_metrics:
            # Run queries
            num_queries = 50
            total_time = 0
            
            for _ in range(num_queries):
                start_time = time.time()
                with self.engine.connect() as conn:
                    conn.execute(f"""
                        SELECT id, title
                        FROM {Article.__tablename__}
                        ORDER BY vector_embedding {metric['operator']} '{query_vector_str}'::vector {metric['order']}
                        LIMIT 10
                    """)
                end_time = time.time()
                total_time += (end_time - start_time)
            
            avg_query_time = total_time / num_queries
            
            self.results["distance_metrics"] = self.results.get("distance_metrics", [])
            self.results["distance_metrics"].append({
                "metric": metric["name"],
                "avg_query_time": avg_query_time,
                "queries_per_second": 1.0 / avg_query_time if avg_query_time > 0 else 0
            })
            
            logger.info(f"Distance metric {metric['name']}: {avg_query_time*1000:.2f}ms ({1.0/avg_query_time if avg_query_time > 0 else 0:.2f} queries/sec)")
        
        # Drop index
        with self.engine.connect() as conn:
            conn.execute(f"DROP INDEX IF EXISTS {Article.__tablename__}_vector_embedding_idx")

    def run_benchmarks(self):
        """Run all benchmarks."""
        logger.info("Starting pgvector benchmarks")
        
        # Setup database
        self.setup_database()
        
        # Generate test data
        self.generate_test_data()
        
        # Run benchmarks
        self.benchmark_index_creation()
        self.benchmark_query_performance()
        self.benchmark_scaling()
        self.benchmark_distance_metrics()
        
        # Save results
        self.save_results()
        
        logger.info("Benchmarks complete")

    def save_results(self, output_file: str = "pgvector_benchmark_results.json"):
        """
        Save benchmark results to a file.
        
        Args:
            output_file: Path to the output file
        """
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        # Also save CSV summaries for easy analysis
        self._save_csv_summaries()

    def _save_csv_summaries(self):
        """Save CSV summaries of benchmark results."""
        # Index creation benchmarks
        if self.results["index_benchmarks"]:
            df = pd.DataFrame(self.results["index_benchmarks"])
            df.to_csv("pgvector_index_benchmarks.csv", index=False)
        
        # Query performance benchmarks
        if self.results["query_benchmarks"]:
            df = pd.DataFrame(self.results["query_benchmarks"])
            df.to_csv("pgvector_query_benchmarks.csv", index=False)
        
        # Scaling benchmarks
        if self.results["scaling_benchmarks"]:
            df = pd.DataFrame(self.results["scaling_benchmarks"])
            df.to_csv("pgvector_scaling_benchmarks.csv", index=False)
        
        # Distance metrics benchmarks
        if "distance_metrics" in self.results and self.results["distance_metrics"]:
            df = pd.DataFrame(self.results["distance_metrics"])
            df.to_csv("pgvector_distance_metrics.csv", index=False)


def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark pgvector performance")
    parser.add_argument(
        "--connection-string", 
        type=str, 
        default="postgresql://newscrawler:newscrawler_password@localhost:5432/newscrawler",
        help="PostgreSQL connection string"
    )
    parser.add_argument(
        "--dataset-size", 
        type=int, 
        default=10000,
        help="Number of articles to generate for benchmarking"
    )
    parser.add_argument(
        "--dimension", 
        type=int, 
        default=1536,
        help="Dimension of the vector embeddings"
    )
    args = parser.parse_args()

    benchmark = PgVectorBenchmark(
        connection_string=args.connection_string,
        dataset_size=args.dataset_size,
        dimension=args.dimension
    )
    benchmark.run_benchmarks()

    # Print summary
    print("\nBenchmark Summary:")
    
    print("\nIndex Creation:")
    for result in benchmark.results["index_benchmarks"]:
        print(f"  {result['index_type']}: {result['creation_time']:.2f}s, Size: {result['index_size']}")
    
    print("\nQuery Performance:")
    for result in benchmark.results["query_benchmarks"]:
        print(f"  {result['index_type']}: {result['avg_query_time']*1000:.2f}ms ({result['queries_per_second']:.2f} queries/sec)")
    
    print("\nDistance Metrics:")
    if "distance_metrics" in benchmark.results:
        for result in benchmark.results["distance_metrics"]:
            print(f"  {result['metric']}: {result['avg_query_time']*1000:.2f}ms ({result['queries_per_second']:.2f} queries/sec)")


if __name__ == "__main__":
    main() 