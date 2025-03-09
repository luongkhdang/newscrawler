from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, or_, and_, text
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging
from datetime import datetime, timedelta

from src.database.session import get_db
from src.database.models import Article
from src.vector.processor import generate_embedding
from src.utils.cache import search_cache, cached

router = APIRouter()
logger = logging.getLogger(__name__)

class ArticleResponse(BaseModel):
    """Article response model."""
    id: str
    url: str
    title: str
    summary: Optional[str] = None
    published_date: Optional[str] = None
    author: Optional[str] = None
    source_domain: str
    category: Optional[str] = None
    score: Optional[float] = None

    class Config:
        from_attributes = True

class PaginatedSearchResponse(BaseModel):
    """Paginated search response model."""
    items: List[ArticleResponse]
    total: int
    page: int
    size: int
    pages: int
    query: str

@router.get("/text", response_model=PaginatedSearchResponse)
@cached(search_cache)
async def text_search(
    q: str = Query(..., description="Search query"),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(10, ge=1, le=100, description="Page size"),
    source_domain: Optional[str] = Query(None, description="Filter by source domain"),
    category: Optional[str] = Query(None, description="Filter by category"),
    author: Optional[str] = Query(None, description="Filter by author"),
    published_after: Optional[datetime] = Query(None, description="Filter by publication date (after)"),
    published_before: Optional[datetime] = Query(None, description="Filter by publication date (before)"),
    db: Session = Depends(get_db)
):
    """
    Perform a text-based search on articles with pagination and filtering.
    
    Args:
        q: Search query
        page: Page number (1-indexed)
        size: Page size
        source_domain: Filter by source domain
        category: Filter by category
        author: Filter by author
        published_after: Filter by publication date (after)
        published_before: Filter by publication date (before)
        db: Database session
        
    Returns:
        PaginatedSearchResponse: Paginated list of matching articles
    """
    try:
        # Calculate offset
        offset = (page - 1) * size
        
        # Simple text search using LIKE
        search_term = f"%{q}%"
        
        # Build query
        query = db.query(Article).filter(
            or_(
                Article.title.ilike(search_term),
                Article.content.ilike(search_term)
            )
        )
        
        # Apply filters
        if source_domain:
            query = query.filter(Article.source_domain == source_domain)
        
        if category:
            query = query.filter(Article.category == category)
        
        if author:
            query = query.filter(Article.author == author)
        
        if published_after:
            query = query.filter(Article.published_date >= published_after)
        
        if published_before:
            query = query.filter(Article.published_date <= published_before)
        
        # Get total count
        total = query.count()
        
        # Apply sorting and pagination
        articles = query.order_by(desc(Article.published_date)).offset(offset).limit(size).all()
        
        # Calculate total pages
        total_pages = (total + size - 1) // size
        
        # Format response
        result = {
            "items": [
                ArticleResponse(
                    id=str(article.id),
                    url=article.url,
                    title=article.title,
                    summary=article.summary,
                    published_date=article.published_date.isoformat() if article.published_date else None,
                    author=article.author,
                    source_domain=article.source_domain,
                    category=article.category
                ) for article in articles
            ],
            "total": total,
            "page": page,
            "size": size,
            "pages": total_pages,
            "query": q
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error in text search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing search: {str(e)}"
        )

@router.get("/semantic", response_model=List[SearchResult])
async def semantic_search(
    query: str = Query(..., min_length=3, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results to return"),
    threshold: float = Query(0.7, ge=0, le=1, description="Similarity threshold"),
    topic: Optional[str] = Query(None, description="Filter by topic"),
    relevant_only: bool = Query(False, description="Only return relevant articles"),
    min_relevance: Optional[float] = Query(None, ge=0, le=1, description="Minimum relevance score"),
    db: Session = Depends(get_db)
):
    """
    Perform semantic search on articles using vector similarity.
    """
    try:
        # Generate embedding for query
        query_embedding = generate_embedding(query)
        
        # Build base query
        base_query = """
            SELECT 
                id, 
                title, 
                content, 
                summary, 
                published_date, 
                author, 
                source_domain, 
                url,
                1 - (vector_embedding <=> :query_embedding) AS similarity
            FROM articles
            WHERE vector_embedding IS NOT NULL
        """
        
        # Add filters
        filters = []
        params = {"query_embedding": query_embedding, "threshold": threshold, "limit": limit}
        
        if topic:
            filters.append("topics @> ARRAY[:topic]::text[]")
            params["topic"] = topic
        
        if relevant_only:
            filters.append("is_relevant = true")
        
        if min_relevance is not None:
            filters.append("relevance_score >= :min_relevance")
            params["min_relevance"] = min_relevance
        
        # Add filters to query
        if filters:
            base_query += " AND " + " AND ".join(filters)
        
        # Add similarity threshold and ordering
        base_query += """
            AND 1 - (vector_embedding <=> :query_embedding) >= :threshold
            ORDER BY similarity DESC
            LIMIT :limit
        """
        
        # Execute query
        results = db.execute(text(base_query), params).fetchall()
        
        # Format results
        search_results = []
        for row in results:
            search_results.append({
                "id": row.id,
                "title": row.title,
                "content": row.content,
                "summary": row.summary,
                "published_date": row.published_date,
                "author": row.author,
                "source_domain": row.source_domain,
                "url": row.url,
                "similarity": row.similarity
            })
        
        return search_results
    
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in semantic search: {str(e)}"
        )

@router.get("/hybrid", response_model=List[SearchResult])
async def hybrid_search(
    query: str = Query(..., min_length=3, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results to return"),
    semantic_weight: float = Query(0.7, ge=0, le=1, description="Weight for semantic search (0-1)"),
    topic: Optional[str] = Query(None, description="Filter by topic"),
    relevant_only: bool = Query(False, description="Only return relevant articles"),
    min_relevance: Optional[float] = Query(None, ge=0, le=1, description="Minimum relevance score"),
    db: Session = Depends(get_db)
):
    """
    Perform hybrid search combining semantic and keyword search.
    """
    try:
        # Generate embedding for query
        query_embedding = generate_embedding(query)
        
        # Prepare text search query
        text_search_query = " | ".join(query.split())
        
        # Build hybrid search query
        hybrid_query = """
            SELECT 
                id, 
                title, 
                content, 
                summary, 
                published_date, 
                author, 
                source_domain, 
                url,
                (
                    :semantic_weight * (1 - (vector_embedding <=> :query_embedding)) + 
                    (1 - :semantic_weight) * (ts_rank(to_tsvector('english', title || ' ' || content), to_tsquery('english', :text_query)) / 10)
                ) AS score
            FROM articles
            WHERE vector_embedding IS NOT NULL
        """
        
        # Add filters
        filters = []
        params = {
            "query_embedding": query_embedding, 
            "text_query": text_search_query,
            "semantic_weight": semantic_weight,
            "limit": limit
        }
        
        if topic:
            filters.append("topics @> ARRAY[:topic]::text[]")
            params["topic"] = topic
        
        if relevant_only:
            filters.append("is_relevant = true")
        
        if min_relevance is not None:
            filters.append("relevance_score >= :min_relevance")
            params["min_relevance"] = min_relevance
        
        # Add filters to query
        if filters:
            hybrid_query += " AND " + " AND ".join(filters)
        
        # Add text search condition and ordering
        hybrid_query += """
            AND to_tsvector('english', title || ' ' || content) @@ to_tsquery('english', :text_query)
            ORDER BY score DESC
            LIMIT :limit
        """
        
        # Execute query
        results = db.execute(text(hybrid_query), params).fetchall()
        
        # Format results
        search_results = []
        for row in results:
            search_results.append({
                "id": row.id,
                "title": row.title,
                "content": row.content,
                "summary": row.summary,
                "published_date": row.published_date,
                "author": row.author,
                "source_domain": row.source_domain,
                "url": row.url,
                "similarity": row.score
            })
        
        return search_results
    
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in hybrid search: {str(e)}"
        ) 