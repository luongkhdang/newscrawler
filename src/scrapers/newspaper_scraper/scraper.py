"""
Newspaper4k-based scraper implementation.
"""

import logging
from typing import List, Optional
from urllib.parse import urlparse
import datetime

import newspaper
import requests
from newspaper import Article as NewspaperArticle
from newspaper.article import ArticleException

from .models import Article, ArticleMetadata, ArticleImage, ScraperConfig
from .exceptions import (
    ScraperError,
    RobotsExclusionError,
    RateLimitError,
    ExtractionError,
    NetworkError,
)
from .rate_limiter import RateLimiter
from .robots_cache import RobotsCache


class NewspaperScraper:
    """
    A wrapper around Newspaper4k that adds robots.txt compliance,
    rate limiting, and standardized output.
    """
    
    def __init__(self, config: Optional[ScraperConfig] = None):
        """
        Initialize the scraper with optional configuration.
        
        Args:
            config: Configuration options for the scraper
        """
        self.config = config or ScraperConfig()
        self.robots_cache = RobotsCache(user_agent=self.config.user_agent)
        self.rate_limiter = RateLimiter(self.config.rate_limits)
        self.logger = logging.getLogger(__name__)
    
    def can_fetch(self, url: str) -> bool:
        """
        Check if the URL can be fetched according to robots.txt.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the URL can be fetched, False otherwise
        """
        return self.robots_cache.can_fetch(url)
    
    def scrape(self, url: str) -> Article:
        """
        Scrape an article from the given URL.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Standardized Article object
            
        Raises:
            RobotsExclusionError: If robots.txt disallows access
            RateLimitError: If rate limit is exceeded
            ExtractionError: For content extraction errors
            NetworkError: For network-related errors
            ScraperError: For general scraping errors
        """
        try:
            # Check robots.txt
            if not self.can_fetch(url):
                raise RobotsExclusionError(f"robots.txt disallows access to {url}")
            
            # Apply rate limiting
            domain = urlparse(url).netloc
            self.rate_limiter.wait(domain)
            
            # Configure newspaper article
            article = NewspaperArticle(
                url,
                language=self.config.language,
                fetch_images=True,
                memoize_articles=True,
                request_timeout=self.config.timeout,
            )
            
            # Download and parse the article
            article.download()
            article.parse()
            
            # Extract natural language processing features if needed
            # article.nlp()
            
            # Process the article into our standardized format
            return self._process_article(article, url)
        
        except ArticleException as e:
            self.logger.error(f"Extraction error for {url}: {str(e)}")
            raise ExtractionError(f"Failed to extract article: {str(e)}")
        
        except requests.RequestException as e:
            self.logger.error(f"Network error for {url}: {str(e)}")
            raise NetworkError(f"Network error: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Unexpected error for {url}: {str(e)}")
            raise ScraperError(f"Unexpected error: {str(e)}")
    
    def _process_article(self, article: NewspaperArticle, url: str) -> Article:
        """
        Process a newspaper Article object into our standardized Article format.
        
        Args:
            article: newspaper.Article object
            url: Original URL
            
        Returns:
            Standardized Article object
        """
        # Extract metadata
        metadata = self.extract_metadata(article)
        
        # Extract images
        images = self.extract_images(article)
        
        # Clean content
        clean_content = self.clean_content(article.text)
        
        # Calculate quality score
        quality_score = self.calculate_quality_score(article, clean_content)
        
        # Create standardized article
        return Article(
            url=url,
            title=article.title,
            content=clean_content,
            html=article.html,
            metadata=metadata,
            images=images,
            language=article.meta_lang or self.config.language or "en",
            quality_score=quality_score,
        )
    
    def extract_metadata(self, article: NewspaperArticle) -> ArticleMetadata:
        """
        Extract metadata from a newspaper Article object.
        
        Args:
            article: newspaper.Article object
            
        Returns:
            Standardized ArticleMetadata object
        """
        # Extract authors
        authors = article.authors or []
        
        # Extract dates
        published_date = None
        if article.publish_date:
            if isinstance(article.publish_date, str):
                try:
                    published_date = datetime.datetime.fromisoformat(article.publish_date)
                except ValueError:
                    self.logger.warning(f"Could not parse publish date: {article.publish_date}")
            else:
                published_date = article.publish_date
        
        # Extract domain
        source_domain = urlparse(article.url).netloc
        
        # Extract tags/keywords
        tags = article.keywords or []
        
        # Create metadata object
        return ArticleMetadata(
            authors=authors,
            published_date=published_date,
            modified_date=None,  # Newspaper doesn't extract modified date
            section=None,  # Newspaper doesn't extract section
            tags=tags,
            summary=article.summary or "",
            source_domain=source_domain,
        )
    
    def extract_images(self, article: NewspaperArticle) -> List[ArticleImage]:
        """
        Extract images from a newspaper Article object.
        
        Args:
            article: newspaper.Article object
            
        Returns:
            List of ArticleImage objects
        """
        images = []
        
        # Process top image if available
        if article.top_image:
            images.append(
                ArticleImage(
                    url=article.top_image,
                    caption=None,  # Newspaper doesn't extract captions
                    local_path=None,
                    width=None,
                    height=None,
                )
            )
        
        # Process other images
        for img_url in article.images:
            if img_url != article.top_image:  # Avoid duplicating top image
                images.append(
                    ArticleImage(
                        url=img_url,
                        caption=None,
                        local_path=None,
                        width=None,
                        height=None,
                    )
                )
        
        return images
    
    def clean_content(self, content: str) -> str:
        """
        Clean and normalize article content.
        
        Args:
            content: Raw article content
            
        Returns:
            Cleaned and normalized content
        """
        # Basic cleaning - remove extra whitespace
        cleaned = " ".join(content.split())
        
        # Additional cleaning could be implemented here
        # - Remove boilerplate text
        # - Fix common encoding issues
        # - Normalize quotes and dashes
        
        return cleaned
    
    def calculate_quality_score(self, article: NewspaperArticle, content: str) -> float:
        """
        Calculate a quality score for the article.
        
        Args:
            article: newspaper.Article object
            content: Cleaned content
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        max_score = 5.0
        points = 0.0
        
        # Check content length (1 point)
        if len(content) > 200:
            points += 1.0
        
        # Check title (1 point)
        if article.title and len(article.title) > 10:
            points += 1.0
        
        # Check metadata (1 point)
        if article.authors and article.publish_date:
            points += 1.0
        
        # Check images (1 point)
        if article.top_image:
            points += 1.0
        
        # Check for keywords/tags (1 point)
        if article.keywords and len(article.keywords) > 0:
            points += 1.0
        
        # Calculate final score
        score = points / max_score
        
        return score 