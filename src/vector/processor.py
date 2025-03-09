import logging
import os
import time
import hashlib
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.database.session import SessionLocal
from src.database.models import Article
from src.utils.cache import vector_cache, cached
from src.topics.processor import process_article_topics

logger = logging.getLogger(__name__)

# Configure embedding model
EMBEDDING_MODEL_TYPE = os.getenv("EMBEDDING_MODEL_TYPE", "huggingface")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_MODEL_NAME = os.getenv("HUGGINGFACE_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")

# Cache for embedding models
_embedding_model = None

def get_embedding_model():
    """Get the embedding model based on configuration."""
    global _embedding_model
    
    # Return cached model if available
    if _embedding_model is not None:
        return _embedding_model
    
    # Create new model
    if EMBEDDING_MODEL_TYPE.lower() == "openai":
        if not OPENAI_API_KEY:
            logger.warning("OpenAI API key not set, falling back to HuggingFace embeddings")
            _embedding_model = HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL_NAME)
        else:
            _embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    else:
        _embedding_model = HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL_NAME)
    
    return _embedding_model

def create_text_chunks(text: str, metadata: Dict[str, Any]) -> List[Document]:
    """Split text into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.create_documents([text], [metadata])
    return chunks

@cached(vector_cache)
def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text with caching."""
    try:
        # Get embedding model
        model = get_embedding_model()
        
        # Generate embedding
        embedding = model.embed_query(text)
        
        return embedding
    
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        # Return empty embedding in case of error
        return []

def process_article(article_id: str, db: Session, process_topics: bool = True) -> bool:
    """
    Process an article to generate and store its embedding and optionally classify topics.
    
    Args:
        article_id: ID of the article to process
        db: Database session
        process_topics: Whether to also process topics, entities, and relevance
        
    Returns:
        True if processing was successful, False otherwise
    """
    try:
        # Get article from database
        article = db.query(Article).filter(Article.id == article_id).first()
        if not article:
            logger.error(f"Article with ID {article_id} not found")
            return False
        
        # Generate embedding for the article
        logger.info(f"Generating embedding for article: {article.title}")
        
        # Combine title and content for better embedding
        text_to_embed = f"{article.title}\n\n{article.content}"
        
        # Generate embedding
        embedding = generate_embedding(text_to_embed)
        
        # Update article with embedding
        article.vector_embedding = embedding
        db.commit()
        
        logger.info(f"Successfully generated embedding for article: {article.title}")
        
        # Process topics if requested
        if process_topics:
            logger.info(f"Processing topics for article: {article.title}")
            if not process_article_topics(article_id, db):
                logger.warning(f"Failed to process topics for article: {article.title}")
                # Continue processing even if topic classification fails
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing article {article_id}: {e}")
        db.rollback()
        return False

def process_unembedded_articles(batch_size: int = 10, process_topics: bool = True) -> int:
    """
    Process articles that don't have embeddings yet.
    
    Args:
        batch_size: Number of articles to process in one batch
        process_topics: Whether to also process topics, entities, and relevance
        
    Returns:
        Number of successfully processed articles
    """
    db = SessionLocal()
    try:
        # Get articles without embeddings
        articles = db.query(Article).filter(Article.vector_embedding.is_(None)).limit(batch_size).all()
        
        processed_count = 0
        for article in articles:
            if process_article(article.id, db, process_topics=process_topics):
                processed_count += 1
            
            # Sleep briefly to avoid overwhelming the embedding API
            time.sleep(0.1)
        
        return processed_count
    
    finally:
        db.close()

def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the vector cache."""
    return {
        "size": len(vector_cache),
        "hits": vector_cache.hits,
        "misses": vector_cache.misses,
    }

def clear_cache() -> None:
    """Clear the vector cache."""
    vector_cache.clear()
    logger.info("Vector cache cleared")

def main():
    """Main function for processing articles."""
    logger.info("Starting vector processing")
    
    # Process unembedded articles
    processed_count = process_unembedded_articles(batch_size=10, process_topics=True)
    logger.info(f"Processed {processed_count} articles")
    
    # Log cache statistics
    logger.info(f"Cache stats: {get_cache_stats()}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run main function
    main() 