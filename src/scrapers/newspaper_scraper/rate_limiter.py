"""
Rate limiter for the Newspaper4k scraper.
"""

import time
from typing import Dict
import logging
from urllib.parse import urlparse


class RateLimiter:
    """
    Rate limiter for web scraping to avoid overloading servers.
    Implements per-domain rate limiting with configurable delays.
    """

    def __init__(self, rate_limits: Dict[str, float]):
        """
        Initialize the rate limiter.

        Args:
            rate_limits: Dictionary mapping domains to minimum delay in seconds between requests.
                         The key 'default' is used for domains not explicitly specified.
        """
        self.rate_limits = rate_limits
        self.last_request_time: Dict[str, float] = {}
        self.logger = logging.getLogger(__name__)

    def wait(self, domain: str) -> None:
        """
        Wait until it's appropriate to make a request to the specified domain.

        Args:
            domain: The domain to check rate limits for.
        """
        current_time = time.time()
        
        # Get the rate limit for this domain, or use default
        rate_limit = self.rate_limits.get(domain, self.rate_limits.get('default', 1.0))
        
        # Check if we need to wait
        if domain in self.last_request_time:
            elapsed = current_time - self.last_request_time[domain]
            if elapsed < rate_limit:
                wait_time = rate_limit - elapsed
                self.logger.debug(f"Rate limiting: waiting {wait_time:.2f}s for {domain}")
                time.sleep(wait_time)
        
        # Update the last request time
        self.last_request_time[domain] = time.time()

    def get_domain_from_url(self, url: str) -> str:
        """
        Extract the domain from a URL.

        Args:
            url: The URL to extract the domain from.

        Returns:
            The domain part of the URL.
        """
        return urlparse(url).netloc 