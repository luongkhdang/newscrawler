#!/usr/bin/env python
"""
NewsCrawler Research - Main Entry Point

This script serves as the main entry point for the NewsCrawler research code,
providing a unified interface to the different research scripts.
"""

import os
import argparse
import logging
from datetime import datetime

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"newspaper4k_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_paths():
    """Set up paths for data, logs, and results directories."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    logs_dir = os.path.join(base_dir, "logs")
    results_dir = os.path.join(base_dir, "results")
    
    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    return {
        "base_dir": base_dir,
        "data_dir": data_dir,
        "logs_dir": logs_dir,
        "results_dir": results_dir
    }

def run_article_extraction(args, paths):
    """Run article extraction research."""
    from newspaper4k_poc import run_benchmarks, load_urls_from_file
    
    logger.info("Running article extraction research")
    
    # Determine URLs to process
    urls = []
    if args.urls:
        urls_file = os.path.join(paths["data_dir"], args.urls)
        urls = load_urls_from_file(urls_file)
    elif args.url:
        urls = [args.url]
    else:
        # Default test URLs
        urls_file = os.path.join(paths["data_dir"], "current_urls.txt")
        if os.path.exists(urls_file):
            urls = load_urls_from_file(urls_file)
        else:
            logger.error(f"Default URLs file not found: {urls_file}")
            return
    
    # Run benchmarks
    output_file = os.path.join(paths["results_dir"], args.output or "benchmark_results.csv")
    run_benchmarks(urls, output_file)
    
    logger.info(f"Article extraction research completed. Results saved to {output_file}")

def run_threading_test(args, paths):
    """Run threading performance research."""
    from newspaper4k_threading import test_threading, load_urls_from_file
    
    logger.info("Running threading performance research")
    
    # Determine URLs to process
    urls = []
    if args.urls:
        urls_file = os.path.join(paths["data_dir"], args.urls)
        urls = load_urls_from_file(urls_file)
    elif args.url:
        urls = [args.url]
    else:
        # Default test URLs
        urls_file = os.path.join(paths["data_dir"], "current_urls.txt")
        if os.path.exists(urls_file):
            urls = load_urls_from_file(urls_file)
        else:
            logger.error(f"Default URLs file not found: {urls_file}")
            return
    
    # Parse thread counts
    thread_counts = [int(tc) for tc in args.threads.split(',')]
    
    # Run threading test
    output_file = os.path.join(paths["results_dir"], args.output or "threading_results.csv")
    test_threading(urls, thread_counts, output_file)
    
    logger.info(f"Threading performance research completed. Results saved to {output_file}")

def run_db_integration(args, paths):
    """Run database integration research."""
    from newspaper4k_db_integration import process_article, process_urls_from_file, query_articles, create_database
    
    logger.info("Running database integration research")
    
    # Set up database
    db_path = os.path.join(paths["results_dir"], args.db or "newspaper4k.db")
    if not os.path.exists(db_path):
        create_database(db_path)
    
    if args.query:
        # Query mode
        logger.info(f"Querying articles with search term: {args.query}")
        articles = query_articles(db_path, args.query)
        
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
        urls_file = os.path.join(paths["data_dir"], args.urls)
        logger.info(f"Processing URLs from {urls_file}")
        success, failure = process_urls_from_file(urls_file, db_path)
        logger.info(f"Processed {success + failure} URLs: {success} succeeded, {failure} failed")
    
    elif args.url:
        # Process single URL
        logger.info(f"Processing URL: {args.url}")
        if process_article(args.url, db_path):
            logger.info("Processing succeeded")
        else:
            logger.error("Processing failed")
    
    else:
        logger.error("No action specified for database integration research")
    
    logger.info("Database integration research completed")

def main():
    """Main function to run the research code."""
    parser = argparse.ArgumentParser(description='NewsCrawler Research')
    subparsers = parser.add_subparsers(dest='command', help='Research command to run')
    
    # Article extraction parser
    extract_parser = subparsers.add_parser('extract', help='Run article extraction research')
    extract_parser.add_argument('--urls', help='File containing URLs to process (relative to data directory)')
    extract_parser.add_argument('--url', help='Single URL to process')
    extract_parser.add_argument('--output', help='Output CSV file for benchmark results (relative to results directory)')
    
    # Threading test parser
    threading_parser = subparsers.add_parser('threading', help='Run threading performance research')
    threading_parser.add_argument('--urls', help='File containing URLs to process (relative to data directory)')
    threading_parser.add_argument('--url', help='Single URL to process')
    threading_parser.add_argument('--threads', default='1,2,4,8,16', help='Comma-separated list of thread counts to test')
    threading_parser.add_argument('--output', help='Output CSV file for test results (relative to results directory)')
    
    # Database integration parser
    db_parser = subparsers.add_parser('db', help='Run database integration research')
    db_parser.add_argument('--urls', help='File containing URLs to process (relative to data directory)')
    db_parser.add_argument('--url', help='Single URL to process')
    db_parser.add_argument('--db', help='SQLite database file (relative to results directory)')
    db_parser.add_argument('--query', help='Search query for articles')
    
    args = parser.parse_args()
    
    # Set up paths
    paths = setup_paths()
    
    # Run the specified command
    if args.command == 'extract':
        run_article_extraction(args, paths)
    elif args.command == 'threading':
        run_threading_test(args, paths)
    elif args.command == 'db':
        run_db_integration(args, paths)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 