"""
Article processing module for topic classification, entity extraction, and relevance scoring.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from src.database.session import SessionLocal
from src.database.models import Article
from src.topics.classifier import get_topic_classifier
from src.topics.entities import get_entity_extractor
from src.topics.relevance import get_relevance_scorer

logger = logging.getLogger(__name__)

def process_article_topics(article_id: str, db: Session) -> bool:
    """
    Process an article to classify topics, extract entities, and calculate relevance score.
    
    Args:
        article_id: ID of the article to process
        db: Database session
        
    Returns:
        True if processing was successful, False otherwise
    """
    try:
        # Get article from database
        article = db.query(Article).filter(Article.id == article_id).first()
        if not article:
            logger.error(f"Article with ID {article_id} not found")
            return False
        
        # Get processors
        topic_classifier = get_topic_classifier()
        entity_extractor = get_entity_extractor()
        relevance_scorer = get_relevance_scorer()
        
        # Combine title and content for processing
        text_to_process = f"{article.title}\n\n{article.content}"
        
        # Classify topics
        logger.info(f"Classifying topics for article: {article.title}")
        topics = topic_classifier.classify(text_to_process)
        
        # Extract entities
        logger.info(f"Extracting entities from article: {article.title}")
        entities = entity_extractor.extract_entities(text_to_process)
        
        # Calculate relevance score
        logger.info(f"Calculating relevance score for article: {article.title}")
        relevance_score = relevance_scorer.calculate_score(
            text=text_to_process,
            topics=topics,
            entities=entities
        )
        
        # Determine if article is relevant based on threshold
        is_relevant = relevance_score >= relevance_scorer.relevance_threshold
        
        # Update article with topics, entities, and relevance information
        article.topics = topics
        article.entities = entities
        article.relevance_score = relevance_score
        article.is_relevant = is_relevant
        
        db.commit()
        
        logger.info(f"Successfully processed topics for article: {article.title}")
        logger.info(f"Topics: {topics}")
        logger.info(f"Relevance score: {relevance_score}")
        logger.info(f"Is relevant: {is_relevant}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing topics for article {article_id}: {e}")
        db.rollback()
        return False

def process_unclassified_articles(batch_size: int = 10) -> int:
    """
    Process articles that don't have topic classification yet.
    
    Args:
        batch_size: Number of articles to process in one batch
        
    Returns:
        Number of successfully processed articles
    """
    db = SessionLocal()
    try:
        # Get articles without topic classification
        articles = db.query(Article).filter(Article.topics.is_(None)).limit(batch_size).all()
        
        processed_count = 0
        for article in articles:
            if process_article_topics(article.id, db):
                processed_count += 1
            
            # Sleep briefly to avoid overwhelming the system
            time.sleep(0.1)
        
        return processed_count
    
    finally:
        db.close()

def update_article_relevance(article_id: str, db: Session) -> bool:
    """
    Update the relevance score for an article that already has topics and entities.
    
    Args:
        article_id: ID of the article to update
        db: Database session
        
    Returns:
        True if update was successful, False otherwise
    """
    try:
        # Get article from database
        article = db.query(Article).filter(Article.id == article_id).first()
        if not article:
            logger.error(f"Article with ID {article_id} not found")
            return False
        
        # Check if article has topics and entities
        if not article.topics or not article.entities:
            logger.error(f"Article {article_id} does not have topics or entities")
            return False
        
        # Get relevance scorer
        relevance_scorer = get_relevance_scorer()
        
        # Combine title and content for processing
        text_to_process = f"{article.title}\n\n{article.content}"
        
        # Calculate relevance score
        logger.info(f"Recalculating relevance score for article: {article.title}")
        relevance_score = relevance_scorer.calculate_score(
            text=text_to_process,
            topics=article.topics,
            entities=article.entities
        )
        
        # Determine if article is relevant based on threshold
        is_relevant = relevance_score >= relevance_scorer.relevance_threshold
        
        # Update article with new relevance information
        article.relevance_score = relevance_score
        article.is_relevant = is_relevant
        
        db.commit()
        
        logger.info(f"Successfully updated relevance for article: {article.title}")
        logger.info(f"New relevance score: {relevance_score}")
        logger.info(f"Is relevant: {is_relevant}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error updating relevance for article {article_id}: {e}")
        db.rollback()
        return False

def update_all_relevance_scores(batch_size: int = 50) -> int:
    """
    Update relevance scores for all articles with topics and entities.
    Useful when relevance scoring algorithm changes.
    
    Args:
        batch_size: Number of articles to process in one batch
        
    Returns:
        Number of successfully updated articles
    """
    db = SessionLocal()
    try:
        # Get articles with topics and entities
        articles = db.query(Article).filter(
            Article.topics.isnot(None),
            Article.entities.isnot(None)
        ).limit(batch_size).all()
        
        updated_count = 0
        for article in articles:
            if update_article_relevance(article.id, db):
                updated_count += 1
            
            # Sleep briefly to avoid overwhelming the system
            time.sleep(0.05)
        
        return updated_count
    
    finally:
        db.close()

def main():
    """Main function for processing articles."""
    logger.info("Starting topic classification processing")
    
    # Process unclassified articles
    processed_count = process_unclassified_articles(batch_size=20)
    logger.info(f"Processed {processed_count} articles")
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run main function
    main() 