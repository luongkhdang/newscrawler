"""
Caching utility for the API.
This module provides a simple in-memory cache to improve performance.
"""

import time
import hashlib
import json
import logging
from typing import Dict, Any, Optional, Callable, TypeVar, Generic
from functools import wraps
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')

class Cache(Generic[T]):
    """
    Simple in-memory cache.
    
    This class implements a simple in-memory cache with expiration.
    """
    
    def __init__(self, expiration_seconds: int = 3600, max_size: int = 1000):
        """
        Initialize the cache.
        
        Args:
            expiration_seconds: Time in seconds before a cache entry expires
            max_size: Maximum number of items to store in the cache
        """
        self.expiration_seconds = expiration_seconds
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def _generate_key(self, *args, **kwargs) -> str:
        """
        Generate a cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cache key as a string
        """
        # Convert args and kwargs to a string representation
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_str = ":".join(key_parts)
        
        # Hash the string to get a fixed-length key
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[T]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check if the entry has expired
        if time.time() > entry["expires_at"]:
            # Remove expired entry
            del self.cache[key]
            return None
        
        # Update access time
        entry["last_accessed"] = time.time()
        
        return entry["value"]
    
    def set(self, key: str, value: T) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Check if we need to evict entries
        if len(self.cache) >= self.max_size:
            self._evict()
        
        # Add new entry
        self.cache[key] = {
            "value": value,
            "created_at": time.time(),
            "last_accessed": time.time(),
            "expires_at": time.time() + self.expiration_seconds
        }
    
    def _evict(self) -> None:
        """
        Evict entries from the cache when it's full.
        
        This method removes the least recently used entries.
        """
        # Sort entries by last accessed time
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1]["last_accessed"]
        )
        
        # Remove the oldest 10% of entries
        num_to_remove = max(1, int(len(self.cache) * 0.1))
        for i in range(num_to_remove):
            if i < len(sorted_entries):
                del self.cache[sorted_entries[i][0]]
    
    def clear(self) -> None:
        """Clear the entire cache."""
        self.cache.clear()
    
    def remove(self, key: str) -> None:
        """
        Remove a specific entry from the cache.
        
        Args:
            key: Cache key to remove
        """
        if key in self.cache:
            del self.cache[key]
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "expiration_seconds": self.expiration_seconds,
            "oldest_entry": min(entry["created_at"] for entry in self.cache.values()) if self.cache else None,
            "newest_entry": max(entry["created_at"] for entry in self.cache.values()) if self.cache else None,
        }


def cached(cache: Cache, key_func: Optional[Callable] = None):
    """
    Decorator to cache function results.
    
    Args:
        cache: Cache instance to use
        key_func: Optional function to generate cache keys
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                # Use function name and arguments as key
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Check cache
            cached_result = cache.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Cache miss, call function
            logger.debug(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            
            # Cache result
            cache.set(key, result)
            
            return result
        return wrapper
    return decorator


# Create global cache instances
vector_cache = Cache[Any](expiration_seconds=3600)  # 1 hour expiration
search_cache = Cache[Any](expiration_seconds=300)   # 5 minutes expiration 