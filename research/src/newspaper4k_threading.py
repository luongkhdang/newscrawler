#!/usr/bin/env python
"""
Newspaper4k Threading Test - NewsCrawler Project

This script tests the multi-threading capabilities of Newspaper4k,
measuring performance with different thread counts.
"""

import os
import time
import csv
import argparse
import logging
from datetime import datetime
import newspaper
from newspaper import Source, Config
import concurrent.futures
import psutil
import matplotlib.pyplot as plt
import numpy as np

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
        logging.FileHandler("newspaper4k_threading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def process_source(url, thread_count):
    """
    Process a news source with specified thread count.
    
    Args:
        url: URL of the news source
        thread_count: Number of threads to use
        
    Returns:
        dict: Processing results
    """
    start_time = time.time()
    start_memory = get_memory_usage()
    
    # Configure newspaper
    config = Config()
    config.fetch_images = False
    config.fetch_videos = False
    config.thread_pool_size = thread_count
    config.request_timeout = 10
    
    try:
        # Build source
        source = Source(url, config=config, memoize_articles=False)
        source.build()
        
        # Get article count
        article_count = len(source.articles)
        
        # Process a sample of articles (up to 10)
        processed_articles = 0
        successful_articles = 0
        
        sample_size = min(10, article_count)
        for i in range(sample_size):
            try:
                article = source.articles[i]
                article.download()
                article.parse()
                successful_articles += 1
            except Exception as e:
                logger.error(f"Error processing article {i} from {url}: {e}")
            finally:
                processed_articles += 1
        
        success = True
        error = None
    except Exception as e:
        success = False
        error = str(e)
        article_count = 0
        processed_articles = 0
        successful_articles = 0
        logger.error(f"Error processing source {url}: {e}")
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    # Calculate metrics
    processing_time = end_time - start_time
    memory_increase = end_memory - start_memory
    
    # Prepare results
    results = {
        'url': url,
        'thread_count': thread_count,
        'success': success,
        'error': error,
        'processing_time': processing_time,
        'memory_increase': memory_increase,
        'article_count': article_count,
        'processed_articles': processed_articles,
        'successful_articles': successful_articles,
        'timestamp': datetime.now().isoformat()
    }
    
    return results

def test_threading(urls, thread_counts, output_file='threading_results.csv'):
    """
    Test different thread counts for processing news sources.
    
    Args:
        urls: List of news source URLs to test
        thread_counts: List of thread counts to test
        output_file: CSV file to write results to
    """
    results = []
    
    for url in urls:
        for thread_count in thread_counts:
            logger.info(f"Processing {url} with {thread_count} threads")
            result = process_source(url, thread_count)
            results.append(result)
    
    # Write results to CSV
    if results:
        fieldnames = results[0].keys()
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        logger.info(f"Threading test results written to {output_file}")
        
        # Generate performance chart
        generate_performance_chart(results, 'threading_performance.png')
    else:
        logger.warning("No test results to write")

def generate_performance_chart(results, output_file):
    """
    Generate a chart showing performance across different thread counts.
    
    Args:
        results: List of test results
        output_file: Output image file
    """
    # Group results by URL and thread count
    data = {}
    thread_counts = set()
    
    for result in results:
        if result['success']:
            url = result['url']
            thread_count = result['thread_count']
            thread_counts.add(thread_count)
            
            if url not in data:
                data[url] = {}
            
            data[url][thread_count] = result['processing_time']
    
    if not data:
        logger.warning("No successful results to chart")
        return
    
    # Prepare chart data
    thread_counts = sorted(thread_counts)
    urls = list(data.keys())
    
    plt.figure(figsize=(12, 8))
    
    # Create a bar chart
    bar_width = 0.8 / len(urls)
    index = np.arange(len(thread_counts))
    
    for i, url in enumerate(urls):
        times = [data[url].get(tc, 0) for tc in thread_counts]
        plt.bar(index + i * bar_width, times, bar_width, label=url.split('//')[1].split('/')[0])
    
    plt.xlabel('Thread Count')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Newspaper4k Processing Time by Thread Count')
    plt.xticks(index + bar_width * (len(urls) - 1) / 2, thread_counts)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_file)
    logger.info(f"Performance chart saved to {output_file}")

def load_urls_from_file(file_path):
    """Load URLs from a file, one URL per line."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description='Newspaper4k Threading Test')
    parser.add_argument('--urls', help='File containing URLs to process, one per line')
    parser.add_argument('--url', help='Single URL to process')
    parser.add_argument('--threads', default='1,2,4,8,16', help='Comma-separated list of thread counts to test')
    parser.add_argument('--output', default='threading_results.csv', help='Output CSV file for test results')
    args = parser.parse_args()
    
    # Parse thread counts
    thread_counts = [int(tc) for tc in args.threads.split(',')]
    
    # Collect URLs to process
    urls = []
    if args.urls:
        urls = load_urls_from_file(args.urls)
    elif args.url:
        urls = [args.url]
    else:
        # Default test URLs if none provided
        urls = [
            'https://www.bbc.com',
            'https://www.cnn.com',
            'https://www.nytimes.com',
            'https://www.washingtonpost.com',
            'https://www.theguardian.com'
        ]
    
    if not urls:
        logger.error("No URLs to process. Provide URLs with --url or --urls.")
        return
    
    logger.info(f"Starting threading test with {len(urls)} URLs and thread counts: {thread_counts}")
    test_threading(urls, thread_counts, args.output)
    logger.info("Threading test completed")

if __name__ == "__main__":
    main() 