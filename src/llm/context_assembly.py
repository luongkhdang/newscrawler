"""
Context assembly module for the RAG system.
This module provides different strategies for assembling context from retrieved articles.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.vector.processor import generate_embedding

logger = logging.getLogger(__name__)

class ContextAssembler:
    """
    Base class for context assembly strategies.
    """
    
    def assemble(self, articles: List[Dict[str, Any]], query: str) -> str:
        """
        Assemble context from retrieved articles.
        
        Args:
            articles: List of relevant articles
            query: The user's query
            
        Returns:
            Assembled context string
        """
        raise NotImplementedError("Subclasses must implement this method")


class SimpleContextAssembler(ContextAssembler):
    """
    Simple context assembly strategy that concatenates articles.
    """
    
    def assemble(self, articles: List[Dict[str, Any]], query: str) -> str:
        """
        Assemble context by simply concatenating articles.
        
        Args:
            articles: List of relevant articles
            query: The user's query
            
        Returns:
            Assembled context string
        """
        if not articles:
            return "No relevant articles found."
        
        context_parts = []
        for i, article in enumerate(articles, 1):
            context_parts.append(f"Article {i}:")
            context_parts.append(f"Title: {article['title']}")
            context_parts.append(f"Source: {article['source_domain']}")
            if article.get('published_date'):
                context_parts.append(f"Date: {article['published_date']}")
            context_parts.append(f"Content: {article['content']}")
            context_parts.append("")
        
        return "\n".join(context_parts)


class ChunkedContextAssembler(ContextAssembler):
    """
    Context assembly strategy that chunks articles and selects the most relevant chunks.
    """
    
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50, max_chunks: int = 10):
        """
        Initialize the chunked context assembler.
        
        Args:
            chunk_size: Size of each chunk in words
            chunk_overlap: Overlap between chunks in words
            max_chunks: Maximum number of chunks to include in the context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks = max_chunks
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def _get_chunk_similarity(self, chunk: str, query: str) -> float:
        """
        Calculate similarity between a chunk and the query.
        
        Args:
            chunk: Text chunk
            query: Query text
            
        Returns:
            Similarity score
        """
        try:
            chunk_embedding = generate_embedding(chunk)
            query_embedding = generate_embedding(query)
            
            # Reshape embeddings for cosine_similarity
            chunk_embedding_reshaped = np.array(chunk_embedding).reshape(1, -1)
            query_embedding_reshaped = np.array(query_embedding).reshape(1, -1)
            
            similarity = cosine_similarity(chunk_embedding_reshaped, query_embedding_reshaped)[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating chunk similarity: {e}")
            return 0.0
    
    def assemble(self, articles: List[Dict[str, Any]], query: str) -> str:
        """
        Assemble context by chunking articles and selecting the most relevant chunks.
        
        Args:
            articles: List of relevant articles
            query: The user's query
            
        Returns:
            Assembled context string
        """
        if not articles:
            return "No relevant articles found."
        
        # Create chunks from all articles
        all_chunks = []
        for article_idx, article in enumerate(articles):
            article_chunks = self._chunk_text(article['content'])
            
            for chunk_idx, chunk in enumerate(article_chunks):
                all_chunks.append({
                    'text': chunk,
                    'article_idx': article_idx,
                    'chunk_idx': chunk_idx,
                    'article_title': article['title'],
                    'source_domain': article['source_domain'],
                    'published_date': article.get('published_date'),
                    'similarity': self._get_chunk_similarity(chunk, query)
                })
        
        # Sort chunks by similarity
        all_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Select top chunks
        selected_chunks = all_chunks[:self.max_chunks]
        
        # Sort selected chunks by article and chunk index for coherence
        selected_chunks.sort(key=lambda x: (x['article_idx'], x['chunk_idx']))
        
        # Assemble context
        context_parts = []
        current_article_idx = -1
        
        for chunk in selected_chunks:
            # Add article header if this is a new article
            if chunk['article_idx'] != current_article_idx:
                current_article_idx = chunk['article_idx']
                context_parts.append(f"Article {current_article_idx + 1}:")
                context_parts.append(f"Title: {chunk['article_title']}")
                context_parts.append(f"Source: {chunk['source_domain']}")
                if chunk.get('published_date'):
                    context_parts.append(f"Date: {chunk['published_date']}")
            
            # Add chunk
            context_parts.append(f"Excerpt: {chunk['text']}")
            context_parts.append("")
        
        return "\n".join(context_parts)


class SemanticContextAssembler(ContextAssembler):
    """
    Context assembly strategy that uses semantic similarity to rerank and filter content.
    """
    
    def __init__(self, max_tokens: int = 3000):
        """
        Initialize the semantic context assembler.
        
        Args:
            max_tokens: Maximum number of tokens to include in the context
        """
        self.max_tokens = max_tokens
        # Rough approximation: 1 token ≈ 4 characters in English
        self.chars_per_token = 4
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        return len(text) // self.chars_per_token
    
    def _truncate_to_max_tokens(self, text: str) -> str:
        """
        Truncate text to maximum token limit.
        
        Args:
            text: Input text
            
        Returns:
            Truncated text
        """
        if self._estimate_tokens(text) <= self.max_tokens:
            return text
        
        # Truncate to max_tokens
        max_chars = self.max_tokens * self.chars_per_token
        truncated = text[:max_chars]
        
        # Try to truncate at a sentence boundary
        last_period = truncated.rfind('.')
        if last_period > max_chars * 0.8:  # Only truncate at sentence if we're not losing too much
            truncated = truncated[:last_period + 1]
        
        return truncated
    
    def assemble(self, articles: List[Dict[str, Any]], query: str) -> str:
        """
        Assemble context using semantic similarity to rerank and filter content.
        
        Args:
            articles: List of relevant articles
            query: The user's query
            
        Returns:
            Assembled context string
        """
        if not articles:
            return "No relevant articles found."
        
        # Sort articles by similarity
        sorted_articles = sorted(articles, key=lambda x: x.get('similarity', 0), reverse=True)
        
        # Assemble context
        context_parts = []
        total_tokens = 0
        
        for i, article in enumerate(sorted_articles, 1):
            article_parts = []
            article_parts.append(f"Article {i}:")
            article_parts.append(f"Title: {article['title']}")
            article_parts.append(f"Source: {article['source_domain']}")
            if article.get('published_date'):
                article_parts.append(f"Date: {article['published_date']}")
            
            # Add content
            content = article['content']
            article_parts.append(f"Content: {content}")
            article_parts.append("")
            
            article_text = "\n".join(article_parts)
            article_tokens = self._estimate_tokens(article_text)
            
            # Check if adding this article would exceed token limit
            if total_tokens + article_tokens > self.max_tokens:
                # If this is the first article, truncate it
                if not context_parts:
                    truncated_article = self._truncate_to_max_tokens(article_text)
                    context_parts.append(truncated_article)
                # Otherwise, skip this article
                break
            
            # Add article to context
            context_parts.append(article_text)
            total_tokens += article_tokens
        
        return "\n".join(context_parts)


class TemporalContextAssembler(ContextAssembler):
    """
    Context assembly strategy that prioritizes recent articles.
    """
    
    def __init__(self, recency_weight: float = 0.3, similarity_weight: float = 0.7):
        """
        Initialize the temporal context assembler.
        
        Args:
            recency_weight: Weight for recency in scoring (0-1)
            similarity_weight: Weight for similarity in scoring (0-1)
        """
        self.recency_weight = recency_weight
        self.similarity_weight = similarity_weight
    
    def _calculate_recency_score(self, published_date: Optional[str]) -> float:
        """
        Calculate recency score for an article.
        
        Args:
            published_date: Article publication date string
            
        Returns:
            Recency score (0-1, higher is more recent)
        """
        if not published_date:
            return 0.5  # Default score for articles without dates
        
        try:
            # Parse date string
            if isinstance(published_date, str):
                pub_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            else:
                pub_date = published_date
            
            # Calculate days since publication
            days_ago = (datetime.now() - pub_date).days
            
            # Convert to score (exponential decay)
            # 0 days -> 1.0, 30 days -> 0.5, 365 days -> 0.05
            recency_score = max(0.01, min(1.0, 2 ** (-days_ago / 30)))
            return recency_score
        
        except Exception as e:
            logger.error(f"Error calculating recency score: {e}")
            return 0.5
    
    def assemble(self, articles: List[Dict[str, Any]], query: str) -> str:
        """
        Assemble context prioritizing recent articles.
        
        Args:
            articles: List of relevant articles
            query: The user's query
            
        Returns:
            Assembled context string
        """
        if not articles:
            return "No relevant articles found."
        
        # Calculate combined scores
        for article in articles:
            similarity = article.get('similarity', 0.5)
            recency = self._calculate_recency_score(article.get('published_date'))
            
            # Combined score
            article['combined_score'] = (
                self.similarity_weight * similarity + 
                self.recency_weight * recency
            )
        
        # Sort by combined score
        sorted_articles = sorted(articles, key=lambda x: x.get('combined_score', 0), reverse=True)
        
        # Assemble context
        context_parts = []
        for i, article in enumerate(sorted_articles, 1):
            context_parts.append(f"Article {i}:")
            context_parts.append(f"Title: {article['title']}")
            context_parts.append(f"Source: {article['source_domain']}")
            if article.get('published_date'):
                context_parts.append(f"Date: {article['published_date']}")
            context_parts.append(f"Content: {article['content']}")
            context_parts.append("")
        
        return "\n".join(context_parts)


def get_context_assembler(strategy: str = "simple") -> ContextAssembler:
    """
    Factory function to get a context assembler by strategy name.
    
    Args:
        strategy: Name of the context assembly strategy
        
    Returns:
        ContextAssembler instance
    """
    assemblers = {
        "simple": SimpleContextAssembler(),
        "chunked": ChunkedContextAssembler(),
        "semantic": SemanticContextAssembler(),
        "temporal": TemporalContextAssembler()
    }
    
    if strategy not in assemblers:
        logger.warning(f"Unknown context assembly strategy: {strategy}. Using simple strategy.")
        return assemblers["simple"]
    
    return assemblers[strategy] 