"""
Command-line interface for the NewsCrawler system.
This module provides a simple CLI for testing the NewsCrawler functionality.
"""

import argparse
import logging
import sys
import os
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

from src.scrapers.scraper_factory import ScraperFactory
from src.database.session import SessionLocal
from src.database.models import Article, Source
from src.vector.processor import process_article
from src.llm.rag import RAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def crawl_url(url: str) -> Dict[str, Any]:
    """
    Crawl a single URL and return the extracted article.
    
    Args:
        url: The URL to crawl
        
    Returns:
        The extracted article data
    """
    factory = ScraperFactory()
    scraper = factory.get_scraper(url)
    
    logger.info(f"Crawling URL: {url}")
    article_data = scraper.extract_article(url)
    
    if article_data:
        logger.info(f"Successfully extracted article: {article_data.get('title', 'No title')}")
        return article_data
    else:
        logger.error(f"Failed to extract article from URL: {url}")
        return {}

def save_article(article_data: Dict[str, Any]) -> Optional[str]:
    """
    Save an article to the database.
    
    Args:
        article_data: The article data to save
        
    Returns:
        The ID of the saved article, or None if saving failed
    """
    if not article_data:
        logger.error("No article data to save")
        return None
    
    db = SessionLocal()
    try:
        # Check if article already exists
        existing_article = db.query(Article).filter(Article.url == article_data["url"]).first()
        if existing_article:
            logger.info(f"Article already exists: {article_data.get('title', 'No title')}")
            return str(existing_article.id)
        
        # Create new article
        article = Article(
            url=article_data["url"],
            title=article_data["title"],
            content=article_data["content"],
            published_date=article_data.get("published_date"),
            author=article_data.get("author"),
            source_domain=article_data["source_domain"],
            summary=article_data.get("summary")
        )
        
        db.add(article)
        db.commit()
        db.refresh(article)
        
        logger.info(f"Saved article to database: {article.title}")
        return str(article.id)
    
    except Exception as e:
        logger.error(f"Error saving article to database: {e}")
        db.rollback()
        return None
    
    finally:
        db.close()

def generate_embedding(article_id: str) -> bool:
    """
    Generate embedding for an article.
    
    Args:
        article_id: The ID of the article
        
    Returns:
        True if successful, False otherwise
    """
    db = SessionLocal()
    try:
        success = process_article(article_id, db)
        if success:
            logger.info(f"Generated embedding for article: {article_id}")
        else:
            logger.error(f"Failed to generate embedding for article: {article_id}")
        return success
    
    finally:
        db.close()

def ask_question(question: str) -> Dict[str, Any]:
    """
    Ask a question using RAG.
    
    Args:
        question: The question to ask
        
    Returns:
        The answer and sources
    """
    rag_system = RAGSystem()
    try:
        logger.info(f"Asking question: {question}")
        result = rag_system.answer_question(query=question)
        logger.info("Question answered successfully")
        return result
    
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return {"answer": f"Error: {str(e)}", "sources": []}
    
    finally:
        rag_system.close()

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="NewsCrawler CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Crawl command
    crawl_parser = subparsers.add_parser("crawl", help="Crawl a URL")
    crawl_parser.add_argument("url", help="URL to crawl")
    crawl_parser.add_argument("--save", action="store_true", help="Save the article to the database")
    crawl_parser.add_argument("--embed", action="store_true", help="Generate embedding for the article")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question using RAG")
    ask_parser.add_argument("question", help="Question to ask")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process articles")
    process_parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    
    args = parser.parse_args()
    
    if args.command == "crawl":
        article_data = crawl_url(args.url)
        print(json.dumps(article_data, indent=2, default=str))
        
        if args.save and article_data:
            article_id = save_article(article_data)
            if article_id and args.embed:
                generate_embedding(article_id)
    
    elif args.command == "ask":
        result = ask_question(args.question)
        print("\nAnswer:")
        print(result["answer"])
        print("\nSources:")
        for i, source in enumerate(result["sources"], 1):
            print(f"{i}. {source['title']} ({source['source_domain']})")
            print(f"   URL: {source['url']}")
            if source.get('published_date'):
                print(f"   Date: {source['published_date']}")
            print(f"   Similarity: {source['similarity']:.2f}")
            print()
    
    elif args.command == "process":
        from src.vector.processor import process_unembedded_articles
        count = process_unembedded_articles(batch_size=args.batch_size)
        print(f"Processed {count} articles")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 