import logging
import os
import time
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.database.session import SessionLocal
from src.database.models import Article

logger = logging.getLogger(__name__)

# Configure embedding model
EMBEDDING_MODEL_TYPE = os.getenv("EMBEDDING_MODEL_TYPE", "huggingface")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_MODEL_NAME = os.getenv("HUGGINGFACE_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")

def get_embedding_model():
    """Get the embedding model based on configuration."""
    if EMBEDDING_MODEL_TYPE.lower() == "openai":
        if not OPENAI_API_KEY:
            logger.warning("OpenAI API key not set, falling back to HuggingFace embeddings")
            return HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL_NAME)
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    else:
        return HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL_NAME)

def create_text_chunks(text: str, metadata: Dict[str, Any]) -> List[Document]:
    """Split text into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.create_documents(
        texts=[text],
        metadatas=[metadata]
    )
    
    return chunks

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a text."""
    embedding_model = get_embedding_model()
    return embedding_model.embed_query(text)

def process_article(article_id: str, db: Session) -> bool:
    """Process an article to generate and store its embedding."""
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
        return True
    
    except Exception as e:
        logger.error(f"Error generating embedding for article {article_id}: {e}")
        db.rollback()
        return False

def process_unembedded_articles(batch_size: int = 10) -> int:
    """Process articles that don't have embeddings yet."""
    db = SessionLocal()
    try:
        # Get articles without embeddings
        articles = db.query(Article).filter(Article.vector_embedding.is_(None)).limit(batch_size).all()
        
        processed_count = 0
        for article in articles:
            if process_article(article.id, db):
                processed_count += 1
            
            # Sleep briefly to avoid overwhelming the embedding API
            time.sleep(0.1)
        
        return processed_count
    
    finally:
        db.close()

def main():
    """Main function to run the vector processor."""
    logger.info("Starting vector processor")
    
    while True:
        try:
            processed_count = process_unembedded_articles()
            
            if processed_count == 0:
                # If no articles were processed, wait longer before checking again
                logger.info("No articles to process, waiting...")
                time.sleep(60)
            else:
                logger.info(f"Processed {processed_count} articles")
                time.sleep(5)
        
        except Exception as e:
            logger.error(f"Error in vector processor: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main() 