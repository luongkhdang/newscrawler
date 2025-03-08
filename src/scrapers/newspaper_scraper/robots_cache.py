"""
Robots.txt cache for the Newspaper4k scraper.
"""

import logging
from typing import Dict, Optional
from urllib.parse import urlparse
import time

from reppy.robots import Robots
from reppy.exceptions import ReppyException


class RobotsCache:
    """
    Cache for robots.txt files to avoid repeated downloads.
    Implements robots.txt parsing and checking for URL access permission.
    """

    def __init__(self, cache_ttl: int = 3600, user_agent: str = "NewsCrawler/1.0"):
        """
        Initialize the robots.txt cache.

        Args:
            cache_ttl: Time-to-live for cached robots.txt files in seconds (default: 1 hour)
            user_agent: User agent to use for robots.txt checking
        """
        self.cache: Dict[str, tuple[Robots, float]] = {}
        self.cache_ttl = cache_ttl
        self.user_agent = user_agent
        self.logger = logging.getLogger(__name__)

    def can_fetch(self, url: str) -> bool:
        """
        Check if the URL can be fetched according to robots.txt.

        Args:
            url: The URL to check

        Returns:
            True if the URL can be fetched, False otherwise
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Get robots.txt URL
            robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"
            
            # Check if we need to refresh the cache
            robots = self._get_robots(robots_url)
            
            if robots:
                return robots.allowed(url, self.user_agent)
            
            # If we can't get robots.txt, assume we can fetch
            return True
        
        except Exception as e:
            self.logger.warning(f"Error checking robots.txt for {url}: {str(e)}")
            # If there's an error, we'll be conservative and allow the fetch
            return True

    def _get_robots(self, robots_url: str) -> Optional[Robots]:
        """
        Get the robots.txt parser for a domain, using cache if available.

        Args:
            robots_url: URL to the robots.txt file

        Returns:
            Robots parser object or None if not available
        """
        current_time = time.time()
        
        # Check if we have a cached version that's still valid
        if robots_url in self.cache:
            robots, timestamp = self.cache[robots_url]
            if current_time - timestamp < self.cache_ttl:
                return robots
        
        # Fetch and cache the robots.txt
        try:
            self.logger.debug(f"Fetching robots.txt from {robots_url}")
            robots = Robots.fetch(robots_url)
            self.cache[robots_url] = (robots, current_time)
            return robots
        
        except ReppyException as e:
            self.logger.warning(f"Error fetching robots.txt from {robots_url}: {str(e)}")
            return None 