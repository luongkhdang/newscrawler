#!/usr/bin/env python
"""
Newspaper4k Proof of Concept - NewsCrawler Project

This script demonstrates how to use Newspaper4k with image extraction disabled,
and includes performance benchmarking to measure the impact of different configurations.
"""

import os
import time
import csv
import argparse
import logging
from datetime import datetime
import newspaper
from newspaper import Article, Config
import psutil
import requests
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
        logging.FileHandler("newspaper4k_poc.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NetworkMonitor:
    """Simple class to monitor network requests."""
    
    def __init__(self):
        self.requests = []
        self.original_session_send = requests.Session.send
        
    def start_monitoring(self):
        """Start monitoring network requests."""
        def patched_send(*args, **kwargs):
            request = args[1]
            self.requests.append({
                'url': request.url,
                'method': request.method,
                'timestamp': datetime.now(),
                'content_type': request.headers.get('Content-Type', 'unknown')
            })
            return self.original_session_send(*args, **kwargs)
        
        requests.Session.send = patched_send
        
    def stop_monitoring(self):
        """Stop monitoring and restore original functionality."""
        requests.Session.send = self.original_session_send
        
    def get_image_requests(self):
        """Return all requests that appear to be for images."""
        image_types = ['image/', '.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.ico']
        return [r for r in self.requests if any(img_type in r['url'].lower() or 
                                               (r['content_type'] and img_type in r['content_type'].lower()) 
                                               for img_type in image_types)]

def get_memory_usage():
    """Get current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def benchmark_article_extraction(url, disable_images=True, disable_videos=True, timeout=10):
    """
    Benchmark article extraction with specified configuration.
    
    Args:
        url: URL of the article to extract
        disable_images: Whether to disable image extraction
        disable_videos: Whether to disable video extraction
        timeout: Request timeout in seconds
        
    Returns:
        dict: Benchmark results
    """
    # Record starting memory
    start_memory = get_memory_usage()
    
    # Start network monitoring
    monitor = NetworkMonitor()
    monitor.start_monitoring()
    
    # Configure newspaper
    config = Config()
    config.fetch_images = not disable_images
    config.fetch_videos = not disable_videos
    config.request_timeout = timeout
    
    # Start timing
    start_time = time.time()
    
    try:
        # Download and parse article
        article = Article(url, config=config)
        article.download()
        article.parse()
        
        # Optional NLP operations
        article.nlp()
        
        success = True
        error = None
    except Exception as e:
        success = False
        error = str(e)
        logger.error(f"Error processing {url}: {e}")
    
    # End timing
    end_time = time.time()
    
    # Stop network monitoring
    monitor.stop_monitoring()
    
    # Record ending memory
    end_memory = get_memory_usage()
    
    # Calculate metrics
    processing_time = end_time - start_time
    memory_increase = end_memory - start_memory
    image_requests = monitor.get_image_requests()
    
    # Prepare results
    results = {
        'url': url,
        'success': success,
        'error': error,
        'processing_time': processing_time,
        'memory_increase': memory_increase,
        'image_requests_count': len(image_requests),
        'has_images': len(article.images) > 0 if success else False,
        'images_disabled': disable_images,
        'videos_disabled': disable_videos,
        'domain': urlparse(url).netloc,
        'timestamp': datetime.now().isoformat()
    }
    
    if success:
        results.update({
            'title': article.title,
            'authors': article.authors,
            'publish_date': article.publish_date,
            'text_length': len(article.text),
            'keywords': article.keywords,
        })
    
    return results

def run_benchmarks(urls, output_file='benchmark_results.csv'):
    """
    Run benchmarks on a list of URLs with different configurations.
    
    Args:
        urls: List of URLs to benchmark
        output_file: CSV file to write results to
    """
    results = []
    
    # Test with images disabled (our primary configuration)
    logger.info("Running benchmarks with images disabled...")
    for url in urls:
        logger.info(f"Processing {url} with images disabled")
        result = benchmark_article_extraction(url, disable_images=True)
        results.append(result)
    
    # For comparison, test with images enabled
    logger.info("Running benchmarks with images enabled for comparison...")
    for url in urls:
        logger.info(f"Processing {url} with images enabled")
        result = benchmark_article_extraction(url, disable_images=False)
        results.append(result)
    
    # Write results to CSV
    if results:
        fieldnames = results[0].keys()
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        logger.info(f"Benchmark results written to {output_file}")
    else:
        logger.warning("No benchmark results to write")

def load_urls_from_file(file_path):
    """Load URLs from a file, one URL per line."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description='Newspaper4k Proof of Concept')
    parser.add_argument('--urls', help='File containing URLs to process, one per line')
    parser.add_argument('--url', help='Single URL to process')
    parser.add_argument('--output', default='benchmark_results.csv', help='Output CSV file for benchmark results')
    args = parser.parse_args()
    
    # Collect URLs to process
    urls = []
    if args.urls:
        urls = load_urls_from_file(args.urls)
    elif args.url:
        urls = [args.url]
    else:
        # Default test URLs if none provided
        urls = [
            'https://www.bbc.com/news/world-us-canada-68465050',
            'https://www.cnn.com/2023/03/08/tech/meta-facebook-layoffs/index.html',
            'https://www.nytimes.com/2023/03/08/business/economy/federal-reserve-powell-interest-rates.html',
            'https://www.washingtonpost.com/technology/2023/03/08/ai-voice-deep-fakes/',
            'https://www.theguardian.com/environment/2023/mar/08/revealed-1000-super-emitting-methane-leaks-risk-triggering-climate-tipping-points'
        ]
    
    if not urls:
        logger.error("No URLs to process. Provide URLs with --url or --urls.")
        return
    
    logger.info(f"Starting benchmark with {len(urls)} URLs")
    run_benchmarks(urls, args.output)
    logger.info("Benchmark completed")

if __name__ == "__main__":
    main() 