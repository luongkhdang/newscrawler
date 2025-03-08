"""
Batch processing infrastructure for handling multiple URLs efficiently.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Callable
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass, field
import csv
import os
import json
from datetime import datetime

from src.models.article import Article
from src.utils.exceptions import ConfigurationError


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_workers: int = 5
    batch_size: int = 100
    timeout: int = 30
    retry_count: int = 3
    output_dir: str = "output"
    scraper_config: Dict[str, Any] = field(default_factory=dict)


class BatchProcessor:
    """
    Process URLs in batches with appropriate parallelization.
    """
    
    def __init__(self, scraper_factory, config: BatchConfig = None):
        """
        Initialize the batch processor.
        
        Args:
            scraper_factory: Factory for creating scrapers
            config: Configuration for batch processing
        """
        self.config = config or BatchConfig()
        self.scraper_factory = scraper_factory
        self.queue = Queue()
        self.results = []
        self.errors = []
        self.stats = {
            'total': 0,
            'success': 0,
            'error': 0,
            'start_time': None,
            'end_time': None,
            'domains': {}
        }
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)
    
    def load_urls_from_csv(self, csv_path: str, url_column: str = 'url'):
        """
        Load URLs from a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            url_column: Name of the column containing URLs
        """
        self.logger.info(f"Loading URLs from {csv_path}")
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                urls = [row[url_column] for row in reader if url_column in row and row[url_column]]
                self.add_urls(urls)
                self.logger.info(f"Loaded {len(urls)} URLs from {csv_path}")
        except Exception as e:
            self.logger.error(f"Error loading URLs from {csv_path}: {str(e)}")
            raise
    
    def add_urls(self, urls: List[str]):
        """
        Add URLs to the processing queue.
        
        Args:
            urls: List of URLs to process
        """
        for url in urls:
            self.queue.put(url)
            self.stats['total'] += 1
    
    def process(self, callback: Callable[[Article], None] = None) -> List[Article]:
        """
        Process all URLs in the queue.
        
        Args:
            callback: Optional callback function to call for each processed article
            
        Returns:
            List of processed articles
        """
        if self.queue.empty():
            self.logger.warning("Queue is empty, nothing to process")
            return []
        
        self.stats['start_time'] = datetime.now()
        worker_count = min(self.config.max_workers, self.queue.qsize())
        self.logger.info(f"Starting batch processing with {worker_count} workers")
        
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = []
            while not self.queue.empty():
                url = self.queue.get()
                futures.append(executor.submit(self._process_url, url))
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        with self.lock:
                            self.results.append(result)
                            self.stats['success'] += 1
                            
                            # Update domain stats
                            domain = result.metadata.source_domain
                            if domain not in self.stats['domains']:
                                self.stats['domains'][domain] = {'success': 0, 'error': 0}
                            self.stats['domains'][domain]['success'] += 1
                        
                        if callback:
                            callback(result)
                except Exception as e:
                    self.logger.error(f"Error in worker thread: {str(e)}")
                    with self.lock:
                        self.stats['error'] += 1
        
        self.stats['end_time'] = datetime.now()
        self.logger.info(f"Batch processing complete. Processed {len(self.results)} articles successfully, {self.stats['error']} errors")
        
        # Save stats
        self._save_stats()
        
        return self.results
    
    def _process_url(self, url: str) -> Optional[Article]:
        """
        Process a single URL.
        
        Args:
            url: The URL to process
            
        Returns:
            Processed article or None if processing failed
        """
        try:
            self.logger.debug(f"Processing URL: {url}")
            scraper = self.scraper_factory.get_scraper(url)
            article = scraper.scrape(url)
            
            if article:
                # Save article to file
                self._save_article(article)
                return article
            
            return None
        except Exception as e:
            self.logger.error(f"Failed to process URL {url}: {str(e)}")
            with self.lock:
                self.errors.append({
                    'url': url,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                
                # Update domain stats
                domain = url.split('/')[2] if '://' in url else url.split('/')[0]
                if domain not in self.stats['domains']:
                    self.stats['domains'][domain] = {'success': 0, 'error': 0}
                self.stats['domains'][domain]['error'] += 1
            
            return None
    
    def _save_article(self, article: Article):
        """
        Save an article to a file.
        
        Args:
            article: The article to save
        """
        try:
            # Create a filename based on the domain and a timestamp
            domain = article.metadata.source_domain
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{domain}_{timestamp}.json"
            filepath = os.path.join(self.config.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(article.to_dict(), f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Saved article to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving article: {str(e)}")
    
    def _save_stats(self):
        """Save processing statistics to a file."""
        try:
            stats_dict = {
                'total': self.stats['total'],
                'success': self.stats['success'],
                'error': self.stats['error'],
                'start_time': self.stats['start_time'].isoformat() if self.stats['start_time'] else None,
                'end_time': self.stats['end_time'].isoformat() if self.stats['end_time'] else None,
                'duration_seconds': (self.stats['end_time'] - self.stats['start_time']).total_seconds() if self.stats['end_time'] and self.stats['start_time'] else None,
                'domains': self.stats['domains']
            }
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.config.output_dir, f"stats_{timestamp}.json")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(stats_dict, f, indent=2)
            
            self.logger.info(f"Saved processing stats to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving stats: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary of processing statistics
        """
        return self.stats
    
    def get_errors(self) -> List[Dict[str, Any]]:
        """
        Get processing errors.
        
        Returns:
            List of processing errors
        """
        return self.errors 