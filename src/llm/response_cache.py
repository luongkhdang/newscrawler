"""
Response cache for the RAG system.
This module provides caching for LLM responses to similar queries.
"""

import logging
import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.cache import Cache
from src.vector.processor import generate_embedding

logger = logging.getLogger(__name__)

# Global cache instance
rag_response_cache = Cache(expiration_seconds=3600, max_size=1000)  # 1 hour expiration


class RAGResponseCache:
    """
    Cache for RAG responses with semantic similarity matching.
    """
    
    def __init__(self, cache: Cache = rag_response_cache, similarity_threshold: float = 0.92):
        """
        Initialize the RAG response cache.
        
        Args:
            cache: Cache instance to use
            similarity_threshold: Threshold for considering queries similar (0-1)
        """
        self.cache = cache
        self.similarity_threshold = similarity_threshold
        self.query_embeddings: Dict[str, List[float]] = {}
    
    def _get_query_hash(self, query: str) -> str:
        """
        Get a hash for a query string.
        
        Args:
            query: Query string
            
        Returns:
            Hash string
        """
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for a query, with caching.
        
        Args:
            query: Query string
            
        Returns:
            Query embedding
        """
        query_hash = self._get_query_hash(query)
        
        if query_hash not in self.query_embeddings:
            self.query_embeddings[query_hash] = generate_embedding(query)
        
        return self.query_embeddings[query_hash]
    
    def _find_similar_query(self, query: str) -> Optional[str]:
        """
        Find a similar query in the cache.
        
        Args:
            query: Query string
            
        Returns:
            Similar query key if found, None otherwise
        """
        query_embedding = self._get_query_embedding(query)
        query_embedding_reshaped = np.array(query_embedding).reshape(1, -1)
        
        # Check all cached queries for similarity
        for cached_key in self.cache.keys():
            # Skip non-query keys
            if not cached_key.startswith('query:'):
                continue
            
            cached_query = cached_key[6:]  # Remove 'query:' prefix
            cached_embedding = self._get_query_embedding(cached_query)
            cached_embedding_reshaped = np.array(cached_embedding).reshape(1, -1)
            
            similarity = cosine_similarity(query_embedding_reshaped, cached_embedding_reshaped)[0][0]
            
            if similarity >= self.similarity_threshold:
                logger.info(f"Found similar query in cache: '{cached_query}' (similarity: {similarity:.4f})")
                return cached_query
        
        return None
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached response for a query.
        
        Args:
            query: Query string
            
        Returns:
            Cached response if found, None otherwise
        """
        # Try exact match first
        exact_key = f"query:{query}"
        exact_match = self.cache.get(exact_key)
        
        if exact_match:
            logger.info(f"Cache hit (exact match): '{query}'")
            return exact_match
        
        # Try semantic similarity match
        similar_query = self._find_similar_query(query)
        
        if similar_query:
            similar_key = f"query:{similar_query}"
            similar_match = self.cache.get(similar_key)
            
            if similar_match:
                logger.info(f"Cache hit (similar match): '{query}' -> '{similar_query}'")
                
                # Clone the response and add metadata about the cache hit
                response = similar_match.copy()
                response['cache_metadata'] = {
                    'cache_hit_type': 'semantic_similarity',
                    'original_query': similar_query,
                    'current_query': query
                }
                
                return response
        
        logger.info(f"Cache miss: '{query}'")
        return None
    
    def set(self, query: str, response: Dict[str, Any]) -> None:
        """
        Cache a response for a query.
        
        Args:
            query: Query string
            response: Response to cache
        """
        key = f"query:{query}"
        
        # Store the embedding for future similarity checks
        self._get_query_embedding(query)
        
        # Cache the response
        self.cache.set(key, response)
        logger.info(f"Cached response for query: '{query}'")
    
    def clear(self) -> None:
        """Clear the cache and embeddings."""
        self.cache.clear()
        self.query_embeddings.clear()
        logger.info("Cleared RAG response cache")
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_stats = self.cache.stats()
        cache_stats['embedding_count'] = len(self.query_embeddings)
        return cache_stats


# Singleton instance
_rag_cache = None

def get_rag_cache() -> RAGResponseCache:
    """
    Get or create a singleton instance of the RAGResponseCache.
    
    Returns:
        RAGResponseCache instance
    """
    global _rag_cache
    if _rag_cache is None:
        _rag_cache = RAGResponseCache()
    return _rag_cache 