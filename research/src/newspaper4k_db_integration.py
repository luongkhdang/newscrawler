#!/usr/bin/env python
"""
Newspaper4k Database Integration Test - NewsCrawler Project

This script demonstrates how to integrate Newspaper4k with a database,
using SQLite for simplicity in the research phase.
"""

import os
import time
import argparse
import logging
from datetime import datetime
import newspaper
from newspaper import Article, Config
import sqlite3
import json
from urllib.parse import urlparse

# Directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("newspaper4k_db.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_database(db_path):
    """
    Create SQLite database with schema for storing articles.
    
    Args:
        db_path: Path to SQLite database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE,
        domain TEXT,
        name TEXT,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id INTEGER,
        url TEXT UNIQUE,
        title TEXT,
        text TEXT,
        summary TEXT,
        authors TEXT,
        publish_date TIMESTAMP,
        keywords TEXT,
        meta_data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (source_id) REFERENCES sources(id)
    )
    ''')
    
    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_publish_date ON articles(publish_date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_sources_domain ON sources(domain)')
    
    conn.commit()
    conn.close()
    
    logger.info(f"Database created at {db_path}")

def get_or_create_source(conn, url):
    """
    Get or create a source record in the database.
    
    Args:
        conn: Database connection
        url: Source URL
        
    Returns:
        int: Source ID
    """
    cursor = conn.cursor()
    
    # Extract domain from URL
    domain = urlparse(url).netloc
    
    # Check if source exists
    cursor.execute('SELECT id FROM sources WHERE url = ?', (url,))
    result = cursor.fetchone()
    
    if result:
        return result[0]
    
    # Create new source
    cursor.execute(
        'INSERT INTO sources (url, domain, name, description) VALUES (?, ?, ?, ?)',
        (url, domain, domain, f"News source from {domain}")
    )
    conn.commit()
    
    return cursor.lastrowid

def store_article(conn, source_id, article):
    """
    Store an article in the database.
    
    Args:
        conn: Database connection
        source_id: Source ID
        article: Newspaper Article object
        
    Returns:
        bool: Success status
    """
    cursor = conn.cursor()
    
    try:
        # Convert authors list to JSON string
        authors_json = json.dumps(article.authors)
        
        # Convert keywords list to JSON string
        keywords_json = json.dumps(article.keywords if hasattr(article, 'keywords') else [])
        
        # Convert metadata to JSON string
        meta_data = {
            'language': article.meta_lang,
            'description': article.meta_description,
            'tags': article.tags
        }
        meta_data_json = json.dumps(meta_data)
        
        # Convert publish date to string
        publish_date = article.publish_date.isoformat() if article.publish_date else None
        
        # Insert article
        cursor.execute('''
        INSERT OR REPLACE INTO articles 
        (source_id, url, title, text, summary, authors, publish_date, keywords, meta_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            source_id,
            article.url,
            article.title,
            article.text,
            article.summary if hasattr(article, 'summary') else None,
            authors_json,
            publish_date,
            keywords_json,
            meta_data_json
        ))
        
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error storing article {article.url}: {e}")
        return False

def process_article(url, db_path):
    """
    Process an article and store it in the database.
    
    Args:
        url: Article URL
        db_path: Path to SQLite database file
        
    Returns:
        bool: Success status
    """
    # Configure newspaper
    config = Config()
    config.fetch_images = False
    config.fetch_videos = False
    config.request_timeout = 10
    
    try:
        # Download and parse article
        article = Article(url, config=config)
        article.download()
        article.parse()
        
        # Optional NLP operations
        try:
            article.nlp()
        except Exception as e:
            logger.warning(f"NLP processing failed for {url}: {e}")
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        
        # Get or create source
        source_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        source_id = get_or_create_source(conn, source_url)
        
        # Store article
        success = store_article(conn, source_id, article)
        
        conn.close()
        
        if success:
            logger.info(f"Successfully processed and stored article: {url}")
        else:
            logger.error(f"Failed to store article: {url}")
        
        return success
    except Exception as e:
        logger.error(f"Error processing article {url}: {e}")
        return False

def query_articles(db_path, query=None, limit=10):
    """
    Query articles from the database.
    
    Args:
        db_path: Path to SQLite database file
        query: Search query (optional)
        limit: Maximum number of results
        
    Returns:
        list: List of article dictionaries
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if query:
        # Simple full-text search
        cursor.execute('''
        SELECT a.*, s.domain 
        FROM articles a
        JOIN sources s ON a.source_id = s.id
        WHERE a.title LIKE ? OR a.text LIKE ?
        ORDER BY a.publish_date DESC
        LIMIT ?
        ''', (f'%{query}%', f'%{query}%', limit))
    else:
        # Get most recent articles
        cursor.execute('''
        SELECT a.*, s.domain 
        FROM articles a
        JOIN sources s ON a.source_id = s.id
        ORDER BY a.publish_date DESC
        LIMIT ?
        ''', (limit,))
    
    results = []
    for row in cursor.fetchall():
        article = dict(row)
        
        # Parse JSON fields
        article['authors'] = json.loads(article['authors']) if article['authors'] else []
        article['keywords'] = json.loads(article['keywords']) if article['keywords'] else []
        article['meta_data'] = json.loads(article['meta_data']) if article['meta_data'] else {}
        
        results.append(article)
    
    conn.close()
    
    return results

def process_urls_from_file(file_path, db_path):
    """
    Process articles from a file and store them in the database.
    
    Args:
        file_path: Path to file containing URLs
        db_path: Path to SQLite database file
        
    Returns:
        tuple: (success_count, failure_count)
    """
    with open(file_path, 'r') as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    success_count = 0
    failure_count = 0
    
    for url in urls:
        logger.info(f"Processing {url}")
        if process_article(url, db_path):
            success_count += 1
        else:
            failure_count += 1
    
    return success_count, failure_count

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description='Newspaper4k Database Integration Test')
    parser.add_argument('--urls', help='File containing URLs to process, one per line')
    parser.add_argument('--url', help='Single URL to process')
    parser.add_argument('--db', default='newspaper4k.db', help='SQLite database file')
    parser.add_argument('--query', help='Search query for articles')
    args = parser.parse_args()
    
    # Create database if it doesn't exist
    if not os.path.exists(args.db):
        create_database(args.db)
    
    if args.query:
        # Query mode
        logger.info(f"Querying articles with search term: {args.query}")
        articles = query_articles(args.db, args.query)
        
        print(f"\nFound {len(articles)} articles matching '{args.query}':\n")
        for article in articles:
            print(f"Title: {article['title']}")
            print(f"Source: {article['domain']}")
            print(f"Date: {article['publish_date']}")
            print(f"URL: {article['url']}")
            print(f"Keywords: {', '.join(article['keywords'][:5])}")
            print(f"Summary: {article['summary'][:200]}...\n")
    
    elif args.urls:
        # Process URLs from file
        logger.info(f"Processing URLs from {args.urls}")
        success, failure = process_urls_from_file(args.urls, args.db)
        logger.info(f"Processed {success + failure} URLs: {success} succeeded, {failure} failed")
    
    elif args.url:
        # Process single URL
        logger.info(f"Processing URL: {args.url}")
        if process_article(args.url, args.db):
            logger.info("Processing succeeded")
        else:
            logger.error("Processing failed")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 