from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Optional
from pydantic import BaseModel
import logging

from src.database.session import get_db
from src.database.models import Article
from src.vector.processor import generate_embedding

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

@router.get("/text", response_model=List[ArticleResponse])
async def text_search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum number of results to return"),
    offset: int = Query(0, description="Number of results to skip"),
    db: Session = Depends(get_db)
):
    """
    Perform a text-based search on articles.
    
    Args:
        q: Search query
        limit: Maximum number of results to return
        offset: Number of results to skip
        db: Database session
        
    Returns:
        List[ArticleResponse]: List of matching articles
    """
    try:
        # Simple text search using LIKE
        search_term = f"%{q}%"
        articles = db.query(Article).filter(
            (Article.title.ilike(search_term)) | 
            (Article.content.ilike(search_term))
        ).order_by(desc(Article.published_date)).offset(offset).limit(limit).all()
        
        return [
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
        ]
    
    except Exception as e:
        logger.error(f"Error in text search: {e}")
        raise HTTPException(status_code=500, detail="Error performing search")

@router.get("/semantic", response_model=List[ArticleResponse])
async def semantic_search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum number of results to return"),
    offset: int = Query(0, description="Number of results to skip"),
    db: Session = Depends(get_db)
):
    """
    Perform a semantic search on articles using vector embeddings.
    
    Args:
        q: Search query
        limit: Maximum number of results to return
        offset: Number of results to skip
        db: Database session
        
    Returns:
        List[ArticleResponse]: List of matching articles with similarity scores
    """
    try:
        # Generate embedding for the query
        query_embedding = generate_embedding(q)
        
        # Perform vector search using cosine similarity
        # Note: This requires the pgvector extension to be installed
        articles = db.query(
            Article,
            func.cosine_similarity(Article.vector_embedding, query_embedding).label("similarity")
        ).filter(
            Article.vector_embedding.is_not(None)
        ).order_by(
            desc("similarity")
        ).offset(offset).limit(limit).all()
        
        return [
            ArticleResponse(
                id=str(article.id),
                url=article.url,
                title=article.title,
                summary=article.summary,
                published_date=article.published_date.isoformat() if article.published_date else None,
                author=article.author,
                source_domain=article.source_domain,
                category=article.category,
                score=float(similarity)
            ) for article, similarity in articles
        ]
    
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise HTTPException(status_code=500, detail="Error performing semantic search") 