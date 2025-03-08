from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
import logging
from datetime import datetime

from src.database.session import get_db
from src.database.models import Article

router = APIRouter()
logger = logging.getLogger(__name__)

class ArticleBase(BaseModel):
    """Base article model."""
    url: str
    title: str
    content: str
    summary: Optional[str] = None
    published_date: Optional[datetime] = None
    author: Optional[str] = None
    source_domain: str
    category: Optional[str] = None
    keywords: Optional[List[str]] = None

class ArticleCreate(ArticleBase):
    """Article creation model."""
    pass

class ArticleUpdate(BaseModel):
    """Article update model."""
    title: Optional[str] = None
    content: Optional[str] = None
    summary: Optional[str] = None
    published_date: Optional[datetime] = None
    author: Optional[str] = None
    category: Optional[str] = None
    keywords: Optional[List[str]] = None

class ArticleResponse(ArticleBase):
    """Article response model."""
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

@router.get("", response_model=List[ArticleResponse])
async def get_articles(
    limit: int = Query(10, description="Maximum number of results to return"),
    offset: int = Query(0, description="Number of results to skip"),
    source_domain: Optional[str] = Query(None, description="Filter by source domain"),
    db: Session = Depends(get_db)
):
    """
    Get a list of articles.
    
    Args:
        limit: Maximum number of results to return
        offset: Number of results to skip
        source_domain: Filter by source domain
        db: Database session
        
    Returns:
        List[ArticleResponse]: List of articles
    """
    query = db.query(Article)
    
    if source_domain:
        query = query.filter(Article.source_domain == source_domain)
    
    articles = query.order_by(Article.published_date.desc()).offset(offset).limit(limit).all()
    
    return articles

@router.get("/{article_id}", response_model=ArticleResponse)
async def get_article(
    article_id: str = Path(..., description="Article ID"),
    db: Session = Depends(get_db)
):
    """
    Get an article by ID.
    
    Args:
        article_id: Article ID
        db: Database session
        
    Returns:
        ArticleResponse: Article details
    """
    article = db.query(Article).filter(Article.id == article_id).first()
    
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    return article

@router.post("", response_model=ArticleResponse, status_code=201)
async def create_article(
    article: ArticleCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new article.
    
    Args:
        article: Article data
        db: Database session
        
    Returns:
        ArticleResponse: Created article
    """
    # Check if article with the same URL already exists
    existing_article = db.query(Article).filter(Article.url == article.url).first()
    if existing_article:
        raise HTTPException(status_code=400, detail="Article with this URL already exists")
    
    # Create new article
    db_article = Article(**article.dict())
    
    db.add(db_article)
    db.commit()
    db.refresh(db_article)
    
    return db_article

@router.put("/{article_id}", response_model=ArticleResponse)
async def update_article(
    article_id: str = Path(..., description="Article ID"),
    article_update: ArticleUpdate = None,
    db: Session = Depends(get_db)
):
    """
    Update an article.
    
    Args:
        article_id: Article ID
        article_update: Article update data
        db: Database session
        
    Returns:
        ArticleResponse: Updated article
    """
    db_article = db.query(Article).filter(Article.id == article_id).first()
    
    if not db_article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    # Update article fields
    update_data = article_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_article, key, value)
    
    # Update the updated_at timestamp
    db_article.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(db_article)
    
    return db_article

@router.delete("/{article_id}", status_code=204)
async def delete_article(
    article_id: str = Path(..., description="Article ID"),
    db: Session = Depends(get_db)
):
    """
    Delete an article.
    
    Args:
        article_id: Article ID
        db: Database session
    """
    db_article = db.query(Article).filter(Article.id == article_id).first()
    
    if not db_article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    db.delete(db_article)
    db.commit()
    
    return None 