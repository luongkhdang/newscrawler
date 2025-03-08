# System Scalability and Performance Optimization Plan

## Overview
This plan outlines the implementation of scalability and performance enhancements for the NewsCrawler project, focusing on distributed processing, database optimization, and caching strategies.

## 1. Distributed Crawling Architecture (Week 1)

### Tasks:
- [ ] Implement a task queue system using Celery and Redis
- [ ] Create worker nodes for parallel crawling operations
- [ ] Develop a scheduler for distributing crawling tasks
- [ ] Implement monitoring and failure recovery mechanisms

### Implementation Details:
```python
# src/crawler/distributed/tasks.py
from celery import Celery
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from src.database.session import SessionLocal
from src.database.models import Source, CrawlLog
from src.scrapers.scraper_factory import ScraperFactory

# Configure Celery
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery('crawler_tasks', broker=redis_url, backend=redis_url)

# Configure logging
logger = logging.getLogger(__name__)

@app.task(bind=True, max_retries=3)
def crawl_source(self, source_id: str) -> Dict[str, Any]:
    """Crawl a specific source and store articles."""
    db = SessionLocal()
    try:
        # Get source information
        source = db.query(Source).filter(Source.id == source_id).first()
        if not source:
            logger.error(f"Source with ID {source_id} not found")
            return {"status": "error", "message": f"Source not found: {source_id}"}
        
        # Create crawl log
        crawl_log = CrawlLog(
            source_id=source.id,
            start_time=datetime.utcnow(),
            status="in_progress"
        )
        db.add(crawl_log)
        db.commit()
        db.refresh(crawl_log)
        
        # Get appropriate scraper
        scraper = ScraperFactory.get_scraper(source.scraper_type)
        
        # Crawl the source
        results = scraper.crawl(source.base_url)
        
        # Process results
        articles_found = len(results)
        articles_added = 0
        articles_updated = 0
        
        for article_data in results:
            # Process and store article
            # This is simplified - actual implementation would handle duplicates, etc.
            articles_added += 1
        
        # Update crawl log
        crawl_log.end_time = datetime.utcnow()
        crawl_log.articles_found = articles_found
        crawl_log.articles_added = articles_added
        crawl_log.articles_updated = articles_updated
        crawl_log.status = "completed"
        
        # Update source last_crawled time
        source.last_crawled = datetime.utcnow()
        
        db.commit()
        
        return {
            "status": "success",
            "source_id": str(source.id),
            "articles_found": articles_found,
            "articles_added": articles_added,
            "articles_updated": articles_updated,
            "crawl_log_id": str(crawl_log.id)
        }
        
    except Exception as e:
        # Handle failure
        logger.exception(f"Error crawling source {source_id}: {str(e)}")
        
        # Update crawl log with error
        if 'crawl_log' in locals():
            crawl_log.end_time = datetime.utcnow()
            crawl_log.status = "failed"
            crawl_log.error_message = str(e)
            db.commit()
        
        # Retry with exponential backoff
        retry_in = 60 * (2 ** self.request.retries)  # 60s, 120s, 240s
        raise self.retry(exc=e, countdown=retry_in)
    
    finally:
        db.close()

@app.task
def schedule_crawls() -> List[Dict[str, Any]]:
    """Schedule crawling tasks for sources that need to be crawled."""
    db = SessionLocal()
    try:
        # Find sources that need to be crawled
        now = datetime.utcnow()
        sources_to_crawl = db.query(Source).filter(
            Source.active == True,
            (Source.last_crawled == None) | 
            (Source.last_crawled <= now - timedelta(hours=Source.crawl_frequency))
        ).all()
        
        results = []
        for source in sources_to_crawl:
            # Schedule crawl task
            task = crawl_source.delay(str(source.id))
            results.append({
                "source_id": str(source.id),
                "task_id": task.id,
                "domain": source.domain
            })
            
        return results
    
    finally:
        db.close()
```

```python
# src/crawler/distributed/scheduler.py
import time
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.crawler.distributed.tasks import schedule_crawls

# Configure logging
logger = logging.getLogger(__name__)

class CrawlScheduler:
    """Scheduler for distributed crawling tasks."""
    
    def __init__(self, check_interval: int = 300):
        """
        Initialize the scheduler.
        
        Args:
            check_interval: Time between scheduling checks in seconds (default: 5 minutes)
        """
        self.check_interval = check_interval
    
    def run(self):
        """Run the scheduler continuously."""
        logger.info("Starting crawl scheduler")
        
        while True:
            try:
                logger.info("Checking for sources to crawl")
                result = schedule_crawls.delay()
                scheduled_tasks = result.get()
                
                if scheduled_tasks:
                    logger.info(f"Scheduled {len(scheduled_tasks)} crawl tasks")
                else:
                    logger.info("No sources need crawling at this time")
                
                # Sleep until next check
                logger.info(f"Sleeping for {self.check_interval} seconds")
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.exception(f"Error in scheduler: {str(e)}")
                # Sleep a bit before retrying
                time.sleep(60)
```

## 2. Database Optimization (Week 2)

### Tasks:
- [ ] Implement database sharding for article storage
- [ ] Optimize indexes for common query patterns
- [ ] Set up connection pooling for efficient resource usage
- [ ] Implement query caching for frequently accessed data

### Implementation Details:
```python
# src/database/sharding.py
import hashlib
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

class ShardManager:
    """Manages database shards for article storage."""
    
    def __init__(self, shard_count: int = 4):
        """
        Initialize the shard manager.
        
        Args:
            shard_count: Number of database shards to use
        """
        self.shard_count = shard_count
        self.engines = {}
        self.session_factories = {}
        
        # Initialize connections to all shards
        for i in range(shard_count):
            # In production, these would be different physical databases
            # For development, we're using the same database with different schemas
            db_url = os.getenv(f"DATABASE_URL_SHARD_{i}", os.getenv("DATABASE_URL"))
            
            # Create engine with connection pooling
            engine = create_engine(
                db_url,
                pool_size=10,
                max_overflow=20,
                pool_recycle=3600,
                pool_pre_ping=True
            )
            
            self.engines[i] = engine
            self.session_factories[i] = sessionmaker(bind=engine)
    
    def get_shard_for_article(self, article_url: str) -> int:
        """
        Determine which shard an article should be stored in.
        
        Args:
            article_url: URL of the article
            
        Returns:
            Shard index (0 to shard_count-1)
        """
        # Use consistent hashing to determine shard
        hash_value = int(hashlib.md5(article_url.encode()).hexdigest(), 16)
        return hash_value % self.shard_count
    
    def get_session_for_article(self, article_url: str):
        """
        Get a database session for the appropriate shard for an article.
        
        Args:
            article_url: URL of the article
            
        Returns:
            SQLAlchemy session for the appropriate shard
        """
        shard_index = self.get_shard_for_article(article_url)
        return self.session_factories[shard_index]()
    
    def get_all_sessions(self):
        """
        Get sessions for all shards.
        
        Returns:
            List of SQLAlchemy sessions, one for each shard
        """
        return [factory() for factory in self.session_factories.values()]
```

```python
# src/database/query_optimization.py
from sqlalchemy import text, Index
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional

from src.database.models import Article, Source, CrawlLog
from src.database.session import engine

def create_optimized_indexes():
    """Create optimized indexes for common query patterns."""
    with engine.connect() as conn:
        # Create GIN index for full-text search on article content
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_articles_content_gin 
            ON articles USING gin(to_tsvector('english', content));
        """))
        
        # Create index for source domain lookups
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_articles_source_domain 
            ON articles(source_domain);
        """))
        
        # Create index for published date (for time-based queries)
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_articles_published_date 
            ON articles(published_date);
        """))
        
        # Create composite index for category and published date
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_articles_category_date 
            ON articles(category, published_date);
        """))
        
        # Create HNSW index for vector similarity search
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_articles_vector_hnsw 
            ON articles USING hnsw (vector_embedding vector_cosine_ops) 
            WITH (m=16, ef_construction=64);
        """))
        
        conn.commit()

class OptimizedQueries:
    """Optimized query methods for common operations."""
    
    @staticmethod
    def search_articles_by_text(db: Session, query_text: str, limit: int = 10) -> List[Article]:
        """
        Perform optimized full-text search on articles.
        
        Args:
            db: Database session
            query_text: Text to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching articles
        """
        # Use to_tsquery for full-text search with ranking
        sql = text("""
            SELECT id, ts_rank(to_tsvector('english', content), query) AS rank
            FROM articles, to_tsquery('english', :query) query
            WHERE to_tsvector('english', content) @@ query
            ORDER BY rank DESC
            LIMIT :limit
        """)
        
        # Format query for ts_query (replace spaces with &)
        formatted_query = " & ".join(query_text.split())
        
        result = db.execute(sql, {"query": formatted_query, "limit": limit})
        article_ids = [row[0] for row in result]
        
        # Fetch full articles
        return db.query(Article).filter(Article.id.in_(article_ids)).all()
    
    @staticmethod
    def find_similar_articles(db: Session, article_id: str, limit: int = 5) -> List[Article]:
        """
        Find articles similar to a given article using vector similarity.
        
        Args:
            db: Database session
            article_id: ID of the reference article
            limit: Maximum number of results to return
            
        Returns:
            List of similar articles
        """
        # First get the reference article's embedding
        reference = db.query(Article).filter(Article.id == article_id).first()
        if not reference or not reference.vector_embedding:
            return []
        
        # Use the <-> operator for cosine distance
        sql = text("""
            SELECT id, 1 - (vector_embedding <-> :embedding) AS similarity
            FROM articles
            WHERE id != :article_id
            ORDER BY vector_embedding <-> :embedding
            LIMIT :limit
        """)
        
        result = db.execute(sql, {
            "embedding": reference.vector_embedding, 
            "article_id": article_id,
            "limit": limit
        })
        
        article_ids = [row[0] for row in result]
        
        # Fetch full articles
        return db.query(Article).filter(Article.id.in_(article_ids)).all()
```

## 3. Caching and Performance Optimization (Week 3)

### Tasks:
- [ ] Implement Redis-based caching for API responses
- [ ] Create a vector cache for embedding operations
- [ ] Develop batch processing for embedding generation
- [ ] Implement request throttling and rate limiting

### Implementation Details:
```python
# src/utils/caching.py
import json
import hashlib
import pickle
from typing import Any, Optional, Callable, Dict, List, Tuple
import redis
import os
import functools
import time
import logging

# Configure Redis connection
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL)

# Configure logging
logger = logging.getLogger(__name__)

class ResponseCache:
    """Caches API responses to improve performance."""
    
    def __init__(self, expiration: int = 3600):
        """
        Initialize the cache.
        
        Args:
            expiration: Cache expiration time in seconds (default: 1 hour)
        """
        self.redis = redis_client
        self.expiration = expiration
    
    def _generate_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate a cache key from endpoint and parameters."""
        # Sort params to ensure consistent keys
        param_str = json.dumps(params, sort_keys=True)
        key = f"api:{endpoint}:{hashlib.md5(param_str.encode()).hexdigest()}"
        return key
    
    def get(self, endpoint: str, params: Dict[str, Any]) -> Optional[Any]:
        """
        Get a cached response.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Cached response or None if not found
        """
        key = self._generate_key(endpoint, params)
        cached = self.redis.get(key)
        
        if cached:
            logger.debug(f"Cache hit for {key}")
            return json.loads(cached)
        
        logger.debug(f"Cache miss for {key}")
        return None
    
    def set(self, endpoint: str, params: Dict[str, Any], response: Any) -> None:
        """
        Cache a response.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            response: Response to cache
        """
        key = self._generate_key(endpoint, params)
        self.redis.setex(key, self.expiration, json.dumps(response))
        logger.debug(f"Cached response for {key}")
    
    def invalidate(self, endpoint: str = None) -> None:
        """
        Invalidate cache for an endpoint or all endpoints.
        
        Args:
            endpoint: Endpoint to invalidate, or None for all endpoints
        """
        if endpoint:
            pattern = f"api:{endpoint}:*"
        else:
            pattern = "api:*"
            
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
            logger.info(f"Invalidated {len(keys)} cache entries for pattern {pattern}")


class VectorCache:
    """Caches vector embeddings to avoid redundant computation."""
    
    def __init__(self, expiration: int = 86400 * 7):  # 1 week default
        """
        Initialize the vector cache.
        
        Args:
            expiration: Cache expiration time in seconds
        """
        self.redis = redis_client
        self.expiration = expiration
    
    def _generate_key(self, text: str) -> str:
        """Generate a cache key from text."""
        return f"vector:{hashlib.md5(text.encode()).hexdigest()}"
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        Get a cached vector embedding.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached vector embedding or None if not found
        """
        key = self._generate_key(text)
        cached = self.redis.get(key)
        
        if cached:
            return pickle.loads(cached)
        
        return None
    
    def set(self, text: str, vector: List[float]) -> None:
        """
        Cache a vector embedding.
        
        Args:
            text: Text the embedding is for
            vector: Vector embedding to cache
        """
        key = self._generate_key(text)
        self.redis.setex(key, self.expiration, pickle.dumps(vector))


def cache_response(expiration: int = 3600):
    """
    Decorator to cache API endpoint responses.
    
    Args:
        expiration: Cache expiration time in seconds
        
    Returns:
        Decorated function
    """
    cache = ResponseCache(expiration=expiration)
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from kwargs or args
            request = kwargs.get('request', args[0] if args else None)
            
            # Get endpoint name from function
            endpoint = func.__name__
            
            # Extract query parameters
            params = {}
            if hasattr(request, 'query_params'):
                params = dict(request.query_params)
            
            # Check cache
            cached_response = cache.get(endpoint, params)
            if cached_response:
                return cached_response
            
            # Call original function
            response = await func(*args, **kwargs)
            
            # Cache response
            cache.set(endpoint, params, response)
            
            return response
        
        return wrapper
    
    return decorator


class RateLimiter:
    """Implements rate limiting for API endpoints."""
    
    def __init__(self, limit: int = 100, window: int = 60):
        """
        Initialize the rate limiter.
        
        Args:
            limit: Maximum number of requests allowed in the window
            window: Time window in seconds
        """
        self.redis = redis_client
        self.limit = limit
        self.window = window
    
    def is_allowed(self, client_id: str) -> Tuple[bool, int]:
        """
        Check if a request is allowed under the rate limit.
        
        Args:
            client_id: Identifier for the client (e.g., IP address)
            
        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        key = f"ratelimit:{client_id}"
        current = self.redis.get(key)
        
        if current is None:
            # First request in the window
            self.redis.setex(key, self.window, 1)
            return True, self.limit - 1
        
        current_count = int(current)
        if current_count < self.limit:
            # Increment the counter
            self.redis.incr(key)
            return True, self.limit - current_count - 1
        
        # Rate limit exceeded
        ttl = self.redis.ttl(key)
        return False, 0
```

```python
# src/vector/batch_processor.py
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm
import logging
from sqlalchemy.orm import Session
import time

from src.database.session import SessionLocal
from src.database.models import Article
from src.vector.processor import generate_embedding
from src.utils.caching import VectorCache

# Configure logging
logger = logging.getLogger(__name__)

class BatchEmbeddingProcessor:
    """Processes article embeddings in batches for efficiency."""
    
    def __init__(self, batch_size: int = 32, db: Optional[Session] = None):
        """
        Initialize the batch processor.
        
        Args:
            batch_size: Number of articles to process in each batch
            db: Database session
        """
        self.batch_size = batch_size
        self.db = db or SessionLocal()
        self.vector_cache = VectorCache()
    
    def process_pending_articles(self, limit: int = 1000) -> int:
        """
        Process articles that don't have embeddings yet.
        
        Args:
            limit: Maximum number of articles to process
            
        Returns:
            Number of articles processed
        """
        # Find articles without embeddings
        articles = self.db.query(Article).filter(
            Article.vector_embedding.is_(None),
            Article.content.isnot(None)
        ).limit(limit).all()
        
        if not articles:
            logger.info("No pending articles to process")
            return 0
        
        logger.info(f"Processing embeddings for {len(articles)} articles")
        
        # Process in batches
        processed_count = 0
        for i in range(0, len(articles), self.batch_size):
            batch = articles[i:i+self.batch_size]
            self._process_batch(batch)
            processed_count += len(batch)
            
            # Commit after each batch
            self.db.commit()
            
            # Small delay to avoid overloading the embedding service
            time.sleep(0.5)
        
        logger.info(f"Completed processing {processed_count} articles")
        return processed_count
    
    def _process_batch(self, articles: List[Article]) -> None:
        """
        Process a batch of articles.
        
        Args:
            articles: List of articles to process
        """
        for article in tqdm(articles, desc="Generating embeddings"):
            try:
                # Check cache first
                cached_vector = self.vector_cache.get(article.content)
                
                if cached_vector:
                    article.vector_embedding = cached_vector
                else:
                    # Generate embedding
                    embedding = generate_embedding(article.content)
                    article.vector_embedding = embedding
                    
                    # Cache the embedding
                    self.vector_cache.set(article.content, embedding)
                
            except Exception as e:
                logger.error(f"Error generating embedding for article {article.id}: {str(e)}")
```

## Integration Plan

1. Add Redis to the Docker Compose configuration
2. Implement the distributed crawling architecture
3. Apply database optimizations and create indexes
4. Integrate caching mechanisms with the API

## Testing Strategy

1. Benchmark crawling performance before and after distributed implementation
2. Test database query performance with different index configurations
3. Measure API response times with and without caching
4. Conduct load testing to verify scalability improvements 