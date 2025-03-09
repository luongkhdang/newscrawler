"""
Database optimization utilities for the NewsCrawler system.
This module provides functions for optimizing database performance.
"""

import logging
import time
from sqlalchemy import create_engine, event, text, Index
from sqlalchemy.orm import Session
from sqlalchemy.pool import _ConnectionRecord

from src.database.models import Article, Source, CrawlLog
from src.database.session import engine, Base

logger = logging.getLogger(__name__)

def configure_connection_pooling(database_url, pool_size=20, max_overflow=10, 
                               pool_timeout=30, pool_recycle=1800):
    """
    Configure an optimized SQLAlchemy engine with connection pooling.
    
    Args:
        database_url: Database connection URL
        pool_size: The number of connections to keep open
        max_overflow: The maximum overflow size of the pool
        pool_timeout: The number of seconds to wait before giving up on getting a connection
        pool_recycle: Number of seconds after which a connection is recycled
        
    Returns:
        Configured SQLAlchemy engine
    """
    engine = create_engine(
        database_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        pool_recycle=pool_recycle,
        pool_pre_ping=True
    )
    
    # Add event listeners for monitoring query performance
    @event.listens_for(engine, "before_cursor_execute")
    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        conn.info.setdefault('query_start_time', []).append(time.time())
        logger.debug("Start Query: %s", statement)

    @event.listens_for(engine, "after_cursor_execute")
    def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        total = time.time() - conn.info['query_start_time'].pop()
        logger.debug("Query Complete: %s", statement)
        logger.debug("Total Time: %f", total)
        if total > 0.5:  # Log slow queries
            logger.warning("Slow Query (%.2fs): %s", total, statement)
    
    return engine

def log_pool_status():
    """Log the current status of the connection pool."""
    status = {
        'checkedin': 0,
        'checkedout': 0,
    }
    for conn in engine.pool._pool:
        if isinstance(conn, _ConnectionRecord):
            if conn.in_use:
                status['checkedout'] += 1
            else:
                status['checkedin'] += 1
    logger.info(f"Connection pool status: {status}")

def create_indexes(db: Session):
    """
    Create database indexes for frequently queried fields.
    
    Args:
        db: SQLAlchemy database session
    """
    logger.info("Creating database indexes for performance optimization")
    
    # Create indexes for Article table
    db.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_articles_published_date ON articles (published_date);
        CREATE INDEX IF NOT EXISTS ix_articles_source_domain ON articles (source_domain);
        CREATE INDEX IF NOT EXISTS ix_articles_category ON articles (category);
        CREATE INDEX IF NOT EXISTS ix_articles_created_at ON articles (created_at);
        CREATE INDEX IF NOT EXISTS ix_articles_updated_at ON articles (updated_at);
    """))
    
    # Create partial index for relevant articles
    db.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_articles_relevant ON articles (published_date, relevance_score)
        WHERE is_relevant = true;
    """))
    
    # Create indexes for Source table
    db.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_sources_active ON sources (active);
        CREATE INDEX IF NOT EXISTS ix_sources_last_crawled ON sources (last_crawled);
        CREATE INDEX IF NOT EXISTS ix_sources_crawl_frequency ON sources (crawl_frequency);
    """))
    
    # Create indexes for CrawlLog table
    db.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_crawl_logs_start_time ON crawl_logs (start_time);
        CREATE INDEX IF NOT EXISTS ix_crawl_logs_status ON crawl_logs (status);
    """))
    
    # Create composite indexes for common query patterns
    db.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_articles_domain_date ON articles (source_domain, published_date);
        CREATE INDEX IF NOT EXISTS ix_articles_domain_relevance ON articles (source_domain, relevance_score);
    """))
    
    db.commit()
    logger.info("Database indexes created successfully")

def optimize_vector_search():
    """
    Configure pgvector for optimal performance.
    """
    with Session(engine) as db:
        # Enable pgvector extension if not already enabled
        db.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        
        # Create HNSW index for vector_embedding column
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_articles_vector_hnsw ON articles 
            USING hnsw (vector_embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """))
        
        db.commit()
        logger.info("Vector search optimization completed")

def initialize_optimizations():
    """
    Initialize all database optimizations.
    """
    with Session(engine) as db:
        create_indexes(db)
        optimize_vector_search()
    logger.info("Database optimizations initialized") 