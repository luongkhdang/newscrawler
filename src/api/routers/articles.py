from fastapi import APIRouter, Depends, HTTPException, Query, Path, status, Body
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, or_
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging
from datetime import datetime, timedelta
from uuid import UUID

from src.database.session import get_db
from src.database.models import Article
from src.vector.processor import process_article
from src.topics.processor import process_article_topics, update_article_relevance

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

class PaginatedResponse(BaseModel):
    """Paginated response model."""
    items: List[ArticleResponse]
    total: int
    page: int
    size: int
    pages: int

@router.get("/", response_model=List[ArticleResponse])
async def get_articles(
    skip: int = Query(0, ge=0, description="Number of articles to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of articles to return"),
    source_domain: Optional[str] = Query(None, description="Filter by source domain"),
    category: Optional[str] = Query(None, description="Filter by category"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    topic: Optional[str] = Query(None, description="Filter by topic"),
    relevant_only: bool = Query(False, description="Only return relevant articles"),
    min_relevance: Optional[float] = Query(None, ge=0, le=1, description="Minimum relevance score"),
    db: Session = Depends(get_db)
):
    """
    Get a list of articles with pagination and filtering options.
    """
    query = db.query(Article)
    
    # Apply filters
    if source_domain:
        query = query.filter(Article.source_domain == source_domain)
    
    if category:
        query = query.filter(Article.category == category)
    
    if start_date:
        query = query.filter(Article.published_date >= start_date)
    
    if end_date:
        query = query.filter(Article.published_date <= end_date)
    
    if topic:
        query = query.filter(Article.topics.contains([topic]))
    
    if relevant_only:
        query = query.filter(Article.is_relevant == True)
    
    if min_relevance is not None:
        query = query.filter(Article.relevance_score >= min_relevance)
    
    # Apply pagination
    total_count = query.count()
    articles = query.order_by(Article.published_date.desc()).offset(skip).limit(limit).all()
    
    # Add pagination metadata
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
    try:
        article = db.query(Article).filter(Article.id == article_id).first()
        
        if not article:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Article with ID {article_id} not found"
            )
        
        return article
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error getting article {article_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting article: {str(e)}"
        )

@router.post("", response_model=ArticleResponse, status_code=status.HTTP_201_CREATED)
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
    try:
        # Check if article with the same URL already exists
        existing_article = db.query(Article).filter(Article.url == article.url).first()
        if existing_article:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Article with URL '{article.url}' already exists"
            )
        
        # Create new article
        db_article = Article(**article.dict())
        
        db.add(db_article)
        db.commit()
        db.refresh(db_article)
        
        return db_article
    
    except HTTPException:
        raise
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating article: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating article: {str(e)}"
        )

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
    try:
        db_article = db.query(Article).filter(Article.id == article_id).first()
        
        if not db_article:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Article with ID {article_id} not found"
            )
        
        # Update article fields
        update_data = article_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_article, key, value)
        
        # Update the updated_at timestamp
        db_article.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(db_article)
        
        return db_article
    
    except HTTPException:
        raise
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating article {article_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating article: {str(e)}"
        )

@router.delete("/{article_id}", status_code=status.HTTP_204_NO_CONTENT)
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
    try:
        db_article = db.query(Article).filter(Article.id == article_id).first()
        
        if not db_article:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Article with ID {article_id} not found"
            )
        
        db.delete(db_article)
        db.commit()
        
        return None
    
    except HTTPException:
        raise
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting article {article_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting article: {str(e)}"
        )

@router.post("/{article_id}/process", response_model=ArticleResponse)
async def process_article_endpoint(
    article_id: UUID = Path(..., description="The ID of the article to process"),
    generate_embedding: bool = Query(True, description="Generate vector embedding"),
    classify_topics: bool = Query(True, description="Classify topics and calculate relevance"),
    db: Session = Depends(get_db)
):
    """
    Process an article to generate embedding and/or classify topics.
    """
    # Check if article exists
    article = db.query(Article).filter(Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    # Process article
    if generate_embedding:
        if not process_article(str(article_id), db, process_topics=False):
            raise HTTPException(status_code=500, detail="Failed to generate embedding")
    
    if classify_topics:
        if not process_article_topics(str(article_id), db):
            raise HTTPException(status_code=500, detail="Failed to classify topics")
    
    # Refresh article from database
    db.refresh(article)
    
    return article

@router.get("/topics", response_model=List[str])
async def get_all_topics(db: Session = Depends(get_db)):
    """
    Get a list of all topics found in articles.
    """
    # Query distinct topics from articles
    result = db.execute("SELECT DISTINCT unnest(topics) as topic FROM articles WHERE topics IS NOT NULL")
    topics = [row[0] for row in result]
    
    return sorted(topics)

@router.get("/topics/{topic}", response_model=List[ArticleResponse])
async def get_articles_by_topic(
    topic: str = Path(..., description="The topic to filter by"),
    skip: int = Query(0, ge=0, description="Number of articles to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of articles to return"),
    min_relevance: Optional[float] = Query(None, ge=0, le=1, description="Minimum relevance score"),
    db: Session = Depends(get_db)
):
    """
    Get articles by topic.
    """
    query = db.query(Article).filter(Article.topics.contains([topic]))
    
    if min_relevance is not None:
        query = query.filter(Article.relevance_score >= min_relevance)
    
    articles = query.order_by(Article.published_date.desc()).offset(skip).limit(limit).all()
    
    return articles

@router.get("/entities/{entity_type}", response_model=List[str])
async def get_entities_by_type(
    entity_type: str = Path(..., description="The entity type to retrieve"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of entities to return"),
    db: Session = Depends(get_db)
):
    """
    Get a list of entities of a specific type found in articles.
    """
    # Query entities of the specified type
    query = f"""
    SELECT DISTINCT jsonb_array_elements_text(entities->'{entity_type}') as entity
    FROM articles
    WHERE entities IS NOT NULL AND entities->'{entity_type}' IS NOT NULL
    LIMIT {limit}
    """
    
    result = db.execute(query)
    entities = [row[0] for row in result]
    
    return sorted(entities)

@router.get("/entities/{entity_type}/{entity}", response_model=List[ArticleResponse])
async def get_articles_by_entity(
    entity_type: str = Path(..., description="The entity type"),
    entity: str = Path(..., description="The entity value"),
    skip: int = Query(0, ge=0, description="Number of articles to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of articles to return"),
    db: Session = Depends(get_db)
):
    """
    Get articles that contain a specific entity.
    """
    # Query articles containing the specified entity
    query = f"""
    SELECT id FROM articles
    WHERE entities IS NOT NULL 
    AND entities->'{entity_type}' IS NOT NULL
    AND entities->'{entity_type}' @> '"{entity}"'
    """
    
    result = db.execute(query)
    article_ids = [row[0] for row in result]
    
    articles = db.query(Article).filter(Article.id.in_(article_ids)).order_by(
        Article.published_date.desc()
    ).offset(skip).limit(limit).all()
    
    return articles 