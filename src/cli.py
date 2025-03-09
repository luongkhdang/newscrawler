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
import time
import click

from src.scrapers.scraper_factory import ScraperFactory
from src.database.session import SessionLocal
from src.database.models import Article, Source
from src.vector.processor import process_article
from src.topics.processor import process_article_topics, update_article_relevance
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

@click.group("crawler")
def crawler_cli():
    """Commands for managing the crawler."""
    pass


@crawler_cli.command("start")
def start_crawler():
    """Start the crawler scheduler."""
    from src.scrapers.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    scheduler.start()
    
    print("Crawler scheduler started")
    print("Press Ctrl+C to stop")
    
    try:
        # Keep the process running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping crawler scheduler...")
        scheduler.stop()
        print("Crawler scheduler stopped")


@crawler_cli.command("stop")
def stop_crawler():
    """Stop the crawler scheduler."""
    from src.scrapers.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    scheduler.stop()
    
    print("Crawler scheduler stopped")


@crawler_cli.command("status")
def crawler_status():
    """Show crawler status."""
    from src.scrapers.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    jobs = scheduler.get_all_jobs()
    
    print("Crawler Status:")
    print(f"  Pending jobs: {len(jobs['pending'])}")
    print(f"  Running jobs: {len(jobs['running'])}")
    print(f"  Completed jobs: {len(jobs['completed'])}")
    print(f"  Failed jobs: {len(jobs['failed'])}")
    
    if jobs['running']:
        print("\nRunning Jobs:")
        for job in jobs['running']:
            print(f"  - {job['job_id']}: {job['source_name']} ({job['scraper_type']})")
    
    if jobs['pending']:
        print("\nPending Jobs:")
        for job in jobs['pending']:
            print(f"  - {job['job_id']}: {job['source_name']} ({job['scraper_type']})")


@crawler_cli.command("schedule")
@click.argument("source_id")
@click.option("--priority", type=click.Choice(["high", "medium", "low"]), default="medium", help="Job priority")
@click.option("--max-urls", type=int, default=100, help="Maximum URLs to crawl")
@click.option("--respect-robots", type=bool, default=True, help="Respect robots.txt")
@click.option("--crawl-delay", type=int, default=1, help="Delay between requests in seconds")
def schedule_crawl(source_id, priority, max_urls, respect_robots, crawl_delay):
    """Schedule a crawl job for a source."""
    from src.scrapers.scheduler import get_scheduler, JobPriority
    from src.database.session import SessionLocal
    from src.database.models import Source
    
    # Get source from database
    db = SessionLocal()
    source = db.query(Source).filter(Source.id == source_id).first()
    
    if not source:
        print(f"Error: Source with ID {source_id} not found")
        db.close()
        return
    
    # Map priority string to enum
    priority_map = {
        "high": JobPriority.HIGH,
        "medium": JobPriority.MEDIUM,
        "low": JobPriority.LOW
    }
    
    # Schedule the job
    scheduler = get_scheduler()
    job_id = scheduler.schedule_job(
        source_id=str(source.id),
        source_name=source.name,
        source_url=source.base_url,
        scraper_type=source.scraper_type,
        priority=priority_map[priority],
        max_urls=max_urls,
        respect_robots_txt=respect_robots,
        crawl_delay=crawl_delay
    )
    
    print(f"Scheduled crawl job {job_id} for source {source.name}")
    db.close()


@crawler_cli.command("cancel")
@click.argument("job_id")
def cancel_job(job_id):
    """Cancel a crawl job."""
    from src.scrapers.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    success = scheduler.cancel_job(job_id)
    
    if success:
        print(f"Canceled job {job_id}")
    else:
        print(f"Error: Job {job_id} not found or could not be canceled")


@crawler_cli.command("job-status")
@click.argument("job_id")
def job_status(job_id):
    """Show status of a specific job."""
    from src.scrapers.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    status = scheduler.get_job_status(job_id)
    
    if not status:
        print(f"Error: Job {job_id} not found")
        return
    
    print(f"Job ID: {status['job_id']}")
    print(f"Source: {status['source_name']}")
    print(f"Status: {status['status']}")
    print(f"Created: {datetime.fromtimestamp(status['created_at']).strftime('%Y-%m-%d %H:%M:%S')}")
    
    if status['start_time']:
        print(f"Started: {datetime.fromtimestamp(status['start_time']).strftime('%Y-%m-%d %H:%M:%S')}")
    
    if status['end_time']:
        print(f"Ended: {datetime.fromtimestamp(status['end_time']).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calculate duration
        duration = status['end_time'] - status['start_time']
        print(f"Duration: {duration:.2f} seconds")
    
    if status['error_message']:
        print(f"Error: {status['error_message']}")
    
    if status['status'] == "completed":
        print(f"Articles found: {status['articles_found']}")
        print(f"Articles added: {status['articles_added']}")
        print(f"Articles updated: {status['articles_updated']}")


@crawler_cli.command("list-sources")
def list_sources():
    """List available sources for crawling."""
    from src.database.session import SessionLocal
    from src.database.models import Source
    
    db = SessionLocal()
    sources = db.query(Source).all()
    
    print(f"Found {len(sources)} sources:")
    for source in sources:
        last_crawled = source.last_crawled.strftime("%Y-%m-%d %H:%M:%S") if source.last_crawled else "Never"
        status = "Active" if source.active else "Inactive"
        print(f"  - {source.id}: {source.name} ({source.scraper_type}) - {status}, Last crawled: {last_crawled}")
    
    db.close()

@click.group()
def cli():
    """NewsCrawler CLI for testing and management."""
    pass

@cli.command("crawl")
@click.argument("url")
@click.option("--save", is_flag=True, help="Save the article to the database")
@click.option("--embed", is_flag=True, help="Generate embedding for the article")
@click.option("--classify", is_flag=True, help="Classify topics and calculate relevance")
def crawl_command(url: str, save: bool, embed: bool, classify: bool):
    """Crawl a URL and extract article data."""
    article_data = crawl_url(url)
    
    if not article_data:
        logger.error("Failed to extract article")
        sys.exit(1)
    
    # Print article data
    print(f"Title: {article_data.get('title', 'No title')}")
    print(f"Published: {article_data.get('published_date', 'Unknown')}")
    print(f"Author: {article_data.get('author', 'Unknown')}")
    print(f"Content length: {len(article_data.get('content', ''))}")
    
    article_id = None
    
    # Save to database if requested
    if save:
        article_id = save_article(article_data)
        if article_id:
            print(f"Article saved with ID: {article_id}")
        else:
            logger.error("Failed to save article")
            sys.exit(1)
    
    # Generate embedding if requested
    if embed and article_id:
        db = SessionLocal()
        try:
            if process_article(article_id, db, process_topics=False):
                print("Embedding generated successfully")
            else:
                logger.error("Failed to generate embedding")
        finally:
            db.close()
    
    # Classify topics if requested
    if classify and article_id:
        db = SessionLocal()
        try:
            if process_article_topics(article_id, db):
                # Fetch the updated article to display topics and relevance
                article = db.query(Article).filter(Article.id == article_id).first()
                print("\nTopic Classification Results:")
                print(f"Topics: {article.topics}")
                print(f"Relevance Score: {article.relevance_score}")
                print(f"Is Relevant: {article.is_relevant}")
                
                # Print entities if available
                if article.entities:
                    print("\nExtracted Entities:")
                    for entity_type, entities in article.entities.items():
                        if entities:
                            print(f"{entity_type}: {', '.join(entities)}")
            else:
                logger.error("Failed to classify topics")
        finally:
            db.close()

@cli.command("ask")
@click.argument("question")
@click.option("--model", default="groq/llama3-70b-8192", help="LLM model to use")
def ask_command(question: str, model: str):
    """Ask a question using the RAG system."""
    result = ask_question(question, model)
    
    if result:
        print("\nAnswer:")
        print(result.get("answer", "No answer generated"))
        
        print("\nSources:")
        for i, source in enumerate(result.get("sources", []), 1):
            print(f"{i}. {source.get('title')} ({source.get('url')})")
    else:
        logger.error("Failed to generate answer")
        sys.exit(1)

@cli.command("process")
@click.option("--batch-size", default=10, help="Number of articles to process in one batch")
@click.option("--embeddings-only", is_flag=True, help="Only generate embeddings, skip topic classification")
@click.option("--topics-only", is_flag=True, help="Only classify topics, skip embedding generation")
def process_command(batch_size: int, embeddings_only: bool, topics_only: bool):
    """Process articles in the database."""
    from src.vector.processor import process_unembedded_articles
    from src.topics.processor import process_unclassified_articles
    
    if embeddings_only and topics_only:
        logger.error("Cannot specify both --embeddings-only and --topics-only")
        sys.exit(1)
    
    if embeddings_only:
        # Process only embeddings
        processed_count = process_unembedded_articles(batch_size=batch_size, process_topics=False)
        print(f"Processed embeddings for {processed_count} articles")
    elif topics_only:
        # Process only topics
        processed_count = process_unclassified_articles(batch_size=batch_size)
        print(f"Processed topics for {processed_count} articles")
    else:
        # Process both embeddings and topics
        embedding_count = process_unembedded_articles(batch_size=batch_size, process_topics=True)
        print(f"Processed {embedding_count} articles with embeddings and topics")
        
        # Also process any articles that have embeddings but no topics
        topic_count = process_unclassified_articles(batch_size=batch_size)
        print(f"Processed topics for additional {topic_count} articles")

@cli.command("update-relevance")
@click.option("--batch-size", default=50, help="Number of articles to update in one batch")
@click.option("--article-id", help="Specific article ID to update")
def update_relevance_command(batch_size: int, article_id: Optional[str]):
    """Update relevance scores for articles."""
    from src.topics.processor import update_article_relevance, update_all_relevance_scores
    
    if article_id:
        # Update specific article
        db = SessionLocal()
        try:
            if update_article_relevance(article_id, db):
                print(f"Updated relevance score for article {article_id}")
            else:
                logger.error(f"Failed to update relevance score for article {article_id}")
                sys.exit(1)
        finally:
            db.close()
    else:
        # Update batch of articles
        updated_count = update_all_relevance_scores(batch_size=batch_size)
        print(f"Updated relevance scores for {updated_count} articles")

if __name__ == "__main__":
    # Register command groups
    cli.add_command(crawler_cli)
    
    # Run CLI
    cli() 