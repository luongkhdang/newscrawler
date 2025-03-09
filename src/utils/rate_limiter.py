"""
Rate limiter utility for the API.
This module provides a simple in-memory rate limiter to prevent abuse of the API.
"""

import time
from typing import Dict, Tuple, Optional
import logging
from collections import defaultdict
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Simple in-memory rate limiter.
    
    This class implements a token bucket algorithm for rate limiting.
    Each client (identified by IP address) has a bucket of tokens that refills over time.
    Each request consumes a token, and if the bucket is empty, the request is rejected.
    """
    
    def __init__(self, rate: int = 60, per: int = 60, burst: int = 100):
        """
        Initialize the rate limiter.
        
        Args:
            rate: Number of tokens to add per time period
            per: Time period in seconds
            burst: Maximum number of tokens that can be accumulated
        """
        self.rate = rate  # tokens per time period
        self.per = per    # time period in seconds
        self.burst = burst  # maximum bucket size
        self.tokens = defaultdict(lambda: (self.burst, time.time()))  # (tokens, last_update_time)
        self.refill_rate = self.rate / self.per  # tokens per second
    
    def _get_tokens(self, key: str) -> Tuple[float, float]:
        """
        Get the current number of tokens and last update time for a key.
        
        Args:
            key: Client identifier (usually IP address)
            
        Returns:
            Tuple of (tokens, last_update_time)
        """
        return self.tokens.get(key, (self.burst, time.time()))
    
    def _update_tokens(self, key: str, tokens: float, last_time: float) -> None:
        """
        Update the tokens and last update time for a key.
        
        Args:
            key: Client identifier
            tokens: New token count
            last_time: New last update time
        """
        self.tokens[key] = (tokens, last_time)
    
    def is_allowed(self, key: str, cost: float = 1.0) -> bool:
        """
        Check if a request is allowed based on rate limits.
        
        Args:
            key: Client identifier
            cost: Cost of the request in tokens
            
        Returns:
            True if the request is allowed, False otherwise
        """
        current_tokens, last_update_time = self._get_tokens(key)
        now = time.time()
        
        # Calculate token refill since last update
        time_passed = now - last_update_time
        new_tokens = current_tokens + time_passed * self.refill_rate
        
        # Cap tokens at the burst limit
        new_tokens = min(new_tokens, self.burst)
        
        # Check if we have enough tokens
        if new_tokens < cost:
            # Not enough tokens, request is rejected
            self._update_tokens(key, new_tokens, now)
            return False
        
        # Enough tokens, consume them and allow the request
        self._update_tokens(key, new_tokens - cost, now)
        return True
    
    def get_retry_after(self, key: str, cost: float = 1.0) -> float:
        """
        Get the time in seconds until the next request would be allowed.
        
        Args:
            key: Client identifier
            cost: Cost of the request in tokens
            
        Returns:
            Time in seconds until the next request would be allowed
        """
        current_tokens, last_update_time = self._get_tokens(key)
        now = time.time()
        
        # Calculate token refill since last update
        time_passed = now - last_update_time
        new_tokens = current_tokens + time_passed * self.refill_rate
        
        # If we already have enough tokens, return 0
        if new_tokens >= cost:
            return 0
        
        # Calculate how much time is needed to refill enough tokens
        tokens_needed = cost - new_tokens
        seconds_needed = tokens_needed / self.refill_rate
        
        return max(0, seconds_needed)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting API requests.
    """
    
    def __init__(self, app, rate_limiter: Optional[RateLimiter] = None):
        """
        Initialize the middleware.
        
        Args:
            app: FastAPI application
            rate_limiter: Rate limiter instance (creates a default one if None)
        """
        super().__init__(app)
        self.rate_limiter = rate_limiter or RateLimiter()
    
    async def dispatch(self, request: Request, call_next):
        """
        Process a request and apply rate limiting.
        
        Args:
            request: FastAPI request
            call_next: Next middleware in the chain
            
        Returns:
            Response from the next middleware
        """
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check if the request is allowed
        if not self.rate_limiter.is_allowed(client_ip):
            retry_after = int(self.rate_limiter.get_retry_after(client_ip))
            logger.warning(f"Rate limit exceeded for {client_ip}, retry after {retry_after}s")
            
            # Return 429 Too Many Requests
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after)}
            )
        
        # Process the request
        return await call_next(request) 