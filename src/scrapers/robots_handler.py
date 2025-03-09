"""
Robots.txt handler for the crawler.
This module provides functionality for parsing and respecting robots.txt rules.
"""

import logging
import time
from typing import Dict, Optional, Set, Tuple
from urllib.parse import urlparse
import requests
from reppy.robots import Robots
from reppy.cache import RobotsCache

logger = logging.getLogger(__name__)


class RobotsHandler:
    """
    Handler for robots.txt compliance.
    Provides functionality to check if a URL can be crawled according to robots.txt rules.
    """
    
    def __init__(self, user_agent: str = "NewsCrawler", cache_capacity: int = 100, cache_timeout: int = 3600):
        """
        Initialize the robots handler.
        
        Args:
            user_agent: User agent to use for robots.txt requests
            cache_capacity: Maximum number of robots.txt files to cache
            cache_timeout: Time in seconds to cache robots.txt files
        """
        self.user_agent = user_agent
        self.cache_capacity = cache_capacity
        self.cache_timeout = cache_timeout
        
        # Initialize robots cache
        self.robots_cache = RobotsCache(capacity=cache_capacity)
        
        # Track domains with no robots.txt
        self.no_robots_domains: Set[str] = set()
        
        # Track crawl delays
        self.crawl_delays: Dict[str, float] = {}
        
        # Track last access times
        self.last_access: Dict[str, float] = {}
        
        logger.info(f"Initialized robots handler with user agent: {user_agent}")
    
    def can_fetch(self, url: str) -> bool:
        """
        Check if a URL can be fetched according to robots.txt rules.
        
        Args:
            url: URL to check
            
        Returns:
            True if the URL can be fetched, False otherwise
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # If domain has no robots.txt, allow fetching
            if domain in self.no_robots_domains:
                return True
            
            # Get robots.txt
            robots = self._get_robots(url)
            
            # If no robots.txt, allow fetching
            if robots is None:
                self.no_robots_domains.add(domain)
                return True
            
            # Check if URL can be fetched
            return robots.allowed(url, self.user_agent)
        
        except Exception as e:
            logger.error(f"Error checking robots.txt for {url}: {e}")
            # In case of error, be conservative and allow fetching
            return True
    
    def get_crawl_delay(self, url: str) -> float:
        """
        Get the crawl delay for a domain.
        
        Args:
            url: URL to get crawl delay for
            
        Returns:
            Crawl delay in seconds (0 if no delay specified)
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Check if we already have the crawl delay
            if domain in self.crawl_delays:
                return self.crawl_delays[domain]
            
            # Get robots.txt
            robots = self._get_robots(url)
            
            # If no robots.txt, no delay
            if robots is None:
                self.crawl_delays[domain] = 0
                return 0
            
            # Get crawl delay
            agent = robots.agent(self.user_agent)
            delay = agent.delay or 0
            
            # Cache crawl delay
            self.crawl_delays[domain] = delay
            
            return delay
        
        except Exception as e:
            logger.error(f"Error getting crawl delay for {url}: {e}")
            # In case of error, be conservative and use a default delay
            return 1
    
    def wait_if_needed(self, url: str) -> None:
        """
        Wait if needed according to crawl delay.
        
        Args:
            url: URL to check
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Get crawl delay
            delay = self.get_crawl_delay(url)
            
            # If no delay, no need to wait
            if delay <= 0:
                return
            
            # Check if we need to wait
            current_time = time.time()
            if domain in self.last_access:
                last_time = self.last_access[domain]
                elapsed = current_time - last_time
                
                # If not enough time has elapsed, wait
                if elapsed < delay:
                    wait_time = delay - elapsed
                    logger.debug(f"Waiting {wait_time:.2f}s for {domain} (crawl delay: {delay}s)")
                    time.sleep(wait_time)
            
            # Update last access time
            self.last_access[domain] = time.time()
        
        except Exception as e:
            logger.error(f"Error waiting for crawl delay for {url}: {e}")
    
    def _get_robots(self, url: str) -> Optional[Robots]:
        """
        Get robots.txt for a URL.
        
        Args:
            url: URL to get robots.txt for
            
        Returns:
            Robots object or None if no robots.txt
        """
        try:
            # Use reppy's cached fetching
            return self.robots_cache.fetch(url)
        
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error fetching robots.txt for {url}: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error fetching robots.txt for {url}: {e}")
            return None


# Singleton instance
_robots_handler = None

def get_robots_handler(user_agent: str = "NewsCrawler") -> RobotsHandler:
    """
    Get or create a singleton instance of the RobotsHandler.
    
    Args:
        user_agent: User agent to use for robots.txt requests
        
    Returns:
        RobotsHandler instance
    """
    global _robots_handler
    if _robots_handler is None:
        _robots_handler = RobotsHandler(user_agent=user_agent)
    return _robots_handler 