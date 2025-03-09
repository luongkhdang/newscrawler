"""
Rate limiter for the crawler.
This module provides functionality for limiting the rate of requests to websites.
"""

import logging
import time
from typing import Dict, Optional, Set, Tuple
from urllib.parse import urlparse
import threading

logger = logging.getLogger(__name__)


class DomainRateLimiter:
    """
    Rate limiter for domains.
    Implements a token bucket algorithm to limit the rate of requests to a domain.
    """
    
    def __init__(self, 
                 default_rate: float = 1.0, 
                 default_per: float = 1.0,
                 default_burst: float = 3.0):
        """
        Initialize the domain rate limiter.
        
        Args:
            default_rate: Default rate limit (requests per time period)
            default_per: Default time period in seconds
            default_burst: Default burst capacity
        """
        self.default_rate = default_rate
        self.default_per = default_per
        self.default_burst = default_burst
        
        # Domain-specific rate limits
        self.domain_limits: Dict[str, Tuple[float, float, float]] = {}
        
        # Token buckets for each domain
        self.tokens: Dict[str, float] = {}
        
        # Last update time for each domain
        self.last_update: Dict[str, float] = {}
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Initialized domain rate limiter with default rate: {default_rate}/{default_per}s, burst: {default_burst}")
    
    def set_domain_limit(self, domain: str, rate: float, per: float, burst: Optional[float] = None) -> None:
        """
        Set a rate limit for a specific domain.
        
        Args:
            domain: Domain to set limit for
            rate: Rate limit (requests per time period)
            per: Time period in seconds
            burst: Burst capacity (defaults to rate)
        """
        with self.lock:
            if burst is None:
                burst = rate
            
            self.domain_limits[domain] = (rate, per, burst)
            
            # Initialize token bucket if not already present
            if domain not in self.tokens:
                self.tokens[domain] = burst
                self.last_update[domain] = time.time()
            
            logger.info(f"Set rate limit for {domain}: {rate}/{per}s, burst: {burst}")
    
    def get_domain_limit(self, domain: str) -> Tuple[float, float, float]:
        """
        Get the rate limit for a domain.
        
        Args:
            domain: Domain to get limit for
            
        Returns:
            Tuple of (rate, per, burst)
        """
        with self.lock:
            return self.domain_limits.get(domain, (self.default_rate, self.default_per, self.default_burst))
    
    def _update_tokens(self, domain: str) -> None:
        """
        Update the token count for a domain based on elapsed time.
        
        Args:
            domain: Domain to update tokens for
        """
        current_time = time.time()
        
        # Get domain limit
        rate, per, burst = self.get_domain_limit(domain)
        
        # If domain not in tokens, initialize
        if domain not in self.tokens:
            self.tokens[domain] = burst
            self.last_update[domain] = current_time
            return
        
        # Calculate elapsed time
        elapsed = current_time - self.last_update[domain]
        
        # Calculate tokens to add
        tokens_to_add = elapsed * (rate / per)
        
        # Update tokens
        self.tokens[domain] = min(burst, self.tokens[domain] + tokens_to_add)
        self.last_update[domain] = current_time
    
    def can_request(self, url: str, cost: float = 1.0) -> bool:
        """
        Check if a request can be made to a URL.
        
        Args:
            url: URL to check
            cost: Cost of the request in tokens
            
        Returns:
            True if the request can be made, False otherwise
        """
        with self.lock:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Update tokens
            self._update_tokens(domain)
            
            # Check if enough tokens
            return self.tokens[domain] >= cost
    
    def wait_if_needed(self, url: str, cost: float = 1.0) -> None:
        """
        Wait if needed to respect rate limits.
        
        Args:
            url: URL to check
            cost: Cost of the request in tokens
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Wait until we can make the request
        while True:
            with self.lock:
                # Update tokens
                self._update_tokens(domain)
                
                # Check if enough tokens
                if self.tokens[domain] >= cost:
                    # Consume tokens
                    self.tokens[domain] -= cost
                    break
            
            # Wait a bit before checking again
            rate, per, _ = self.get_domain_limit(domain)
            wait_time = per / rate
            logger.debug(f"Waiting {wait_time:.2f}s for {domain} (rate limit)")
            time.sleep(wait_time)
    
    def request(self, url: str, cost: float = 1.0) -> bool:
        """
        Make a request to a URL, consuming tokens.
        
        Args:
            url: URL to request
            cost: Cost of the request in tokens
            
        Returns:
            True if the request was allowed, False otherwise
        """
        with self.lock:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Update tokens
            self._update_tokens(domain)
            
            # Check if enough tokens
            if self.tokens[domain] >= cost:
                # Consume tokens
                self.tokens[domain] -= cost
                return True
            
            return False
    
    def get_wait_time(self, url: str, cost: float = 1.0) -> float:
        """
        Get the time to wait before making a request.
        
        Args:
            url: URL to check
            cost: Cost of the request in tokens
            
        Returns:
            Time to wait in seconds (0 if no wait needed)
        """
        with self.lock:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Update tokens
            self._update_tokens(domain)
            
            # Check if enough tokens
            if self.tokens[domain] >= cost:
                return 0
            
            # Calculate time to wait
            rate, per, _ = self.get_domain_limit(domain)
            tokens_needed = cost - self.tokens[domain]
            return tokens_needed * (per / rate)


class CrawlerRateLimiter:
    """
    Rate limiter for the crawler.
    Combines domain-specific rate limiting with robots.txt compliance.
    """
    
    def __init__(self, 
                 default_rate: float = 1.0, 
                 default_per: float = 1.0,
                 default_burst: float = 3.0,
                 user_agent: str = "NewsCrawler"):
        """
        Initialize the crawler rate limiter.
        
        Args:
            default_rate: Default rate limit (requests per time period)
            default_per: Default time period in seconds
            default_burst: Default burst capacity
            user_agent: User agent to use for robots.txt requests
        """
        self.domain_limiter = DomainRateLimiter(
            default_rate=default_rate,
            default_per=default_per,
            default_burst=default_burst
        )
        
        # Import here to avoid circular imports
        from src.scrapers.robots_handler import get_robots_handler
        self.robots_handler = get_robots_handler(user_agent=user_agent)
        
        logger.info(f"Initialized crawler rate limiter with user agent: {user_agent}")
    
    def set_domain_limit(self, domain: str, rate: float, per: float, burst: Optional[float] = None) -> None:
        """
        Set a rate limit for a specific domain.
        
        Args:
            domain: Domain to set limit for
            rate: Rate limit (requests per time period)
            per: Time period in seconds
            burst: Burst capacity (defaults to rate)
        """
        self.domain_limiter.set_domain_limit(domain, rate, per, burst)
    
    def can_fetch(self, url: str) -> bool:
        """
        Check if a URL can be fetched according to robots.txt rules.
        
        Args:
            url: URL to check
            
        Returns:
            True if the URL can be fetched, False otherwise
        """
        return self.robots_handler.can_fetch(url)
    
    def can_request(self, url: str, cost: float = 1.0) -> bool:
        """
        Check if a request can be made to a URL.
        
        Args:
            url: URL to check
            cost: Cost of the request in tokens
            
        Returns:
            True if the request can be made, False otherwise
        """
        # Check robots.txt first
        if not self.can_fetch(url):
            return False
        
        # Then check rate limits
        return self.domain_limiter.can_request(url, cost)
    
    def wait_if_needed(self, url: str, cost: float = 1.0) -> None:
        """
        Wait if needed to respect rate limits and robots.txt.
        
        Args:
            url: URL to check
            cost: Cost of the request in tokens
        """
        # Wait for robots.txt crawl delay
        self.robots_handler.wait_if_needed(url)
        
        # Wait for rate limits
        self.domain_limiter.wait_if_needed(url, cost)
    
    def request(self, url: str, cost: float = 1.0) -> bool:
        """
        Make a request to a URL, consuming tokens.
        
        Args:
            url: URL to request
            cost: Cost of the request in tokens
            
        Returns:
            True if the request was allowed, False otherwise
        """
        # Check robots.txt first
        if not self.can_fetch(url):
            return False
        
        # Then check rate limits
        return self.domain_limiter.request(url, cost)


# Singleton instance
_crawler_rate_limiter = None

def get_crawler_rate_limiter(
    default_rate: float = 1.0,
    default_per: float = 1.0,
    default_burst: float = 3.0,
    user_agent: str = "NewsCrawler"
) -> CrawlerRateLimiter:
    """
    Get or create a singleton instance of the CrawlerRateLimiter.
    
    Args:
        default_rate: Default rate limit (requests per time period)
        default_per: Default time period in seconds
        default_burst: Default burst capacity
        user_agent: User agent to use for robots.txt requests
        
    Returns:
        CrawlerRateLimiter instance
    """
    global _crawler_rate_limiter
    if _crawler_rate_limiter is None:
        _crawler_rate_limiter = CrawlerRateLimiter(
            default_rate=default_rate,
            default_per=default_per,
            default_burst=default_burst,
            user_agent=user_agent
        )
    return _crawler_rate_limiter 