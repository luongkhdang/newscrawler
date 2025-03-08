"""
Base scraper interface for the multi-strategy scraper system.
All scraper implementations must inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

from src.models.article import Article, ArticleMetadata, ArticleImage


class BaseScraper(ABC):
    """
    Abstract base class for all scraper implementations.
    Defines the common interface that all scrapers must implement.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the scraper with optional configuration.
        
        Args:
            config: Configuration options for the scraper
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def can_fetch(self, url: str) -> bool:
        """
        Check if the URL can be fetched according to robots.txt.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the URL can be fetched, False otherwise
        """
        pass
    
    @abstractmethod
    def scrape(self, url: str) -> Optional[Article]:
        """
        Scrape an article from the given URL.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Standardized Article object or None if scraping failed
            
        Raises:
            ScraperError: For scraping errors
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Get the name of this scraper strategy.
        
        Returns:
            The strategy name as a string
        """
        pass
    
    def calculate_quality_score(self, article: Article) -> float:
        """
        Calculate a quality score for the extracted article.
        
        Args:
            article: The extracted article
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        total_weight = 0.0
        
        # Content length score (30%)
        if len(article.content) > 0:
            content_score = min(1.0, len(article.content) / 2000)
            score += content_score * 0.3
            total_weight += 0.3
        
        # Metadata completeness score (30%)
        metadata_score = 0.0
        if article.metadata:
            if article.metadata.authors:
                metadata_score += 0.25
            if article.metadata.published_date:
                metadata_score += 0.25
            if article.metadata.tags:
                metadata_score += 0.25
            if article.metadata.summary:
                metadata_score += 0.25
            score += metadata_score * 0.3
            total_weight += 0.3
        
        # Title quality score (20%)
        if article.title and len(article.title) > 10:
            title_score = min(1.0, len(article.title) / 50)
            score += title_score * 0.2
            total_weight += 0.2
        
        # Images score (20%)
        if article.images:
            images_score = min(1.0, len(article.images) / 3)
            score += images_score * 0.2
            total_weight += 0.2
        
        # Normalize score if we have any weights
        if total_weight > 0:
            return score / total_weight
        
        return 0.0 