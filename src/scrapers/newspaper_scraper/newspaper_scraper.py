"""
Newspaper4k-based scraper implementation.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import urllib.parse
import urllib.robotparser
from dataclasses import dataclass, field

import newspaper
from newspaper import Article as NewspaperArticle
from newspaper import Config as NewspaperConfig

from src.scrapers.base_scraper import BaseScraper
from src.models.article import Article, ArticleMetadata, ArticleImage
from src.utils.exceptions import (
    ScraperError, 
    RobotsExclusionError, 
    ExtractionError, 
    NetworkError,
    ParsingError
)


@dataclass
class NewspaperScraperConfig:
    """Configuration for the Newspaper4k-based scraper."""
    user_agent: str = "NewsCrawler/1.0 (https://example.com/bot)"
    browser_user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    request_timeout: int = 30
    follow_redirects: bool = True
    number_threads: int = 1
    memoize_articles: bool = True
    language: str = "en"
    fetch_images: bool = True
    minimum_content_length: int = 300
    robots_cache_timeout: int = 3600  # 1 hour
    rate_limits: Dict[str, float] = field(default_factory=lambda: {"default": 1.0})
    max_retries: int = 3
    retry_backoff: float = 2.0
    verify_ssl: bool = True


class NewspaperScraper(BaseScraper):
    """
    Newspaper4k-based scraper implementation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the scraper with optional configuration.
        
        Args:
            config: Configuration options for the scraper
        """
        super().__init__(config)
        
        # Convert generic config to typed config
        self.config = NewspaperScraperConfig(**self.config) if self.config else NewspaperScraperConfig()
        
        # Initialize Newspaper configuration
        self.newspaper_config = NewspaperConfig()
        self.newspaper_config.browser_user_agent = self.config.browser_user_agent
        self.newspaper_config.request_timeout = self.config.request_timeout
        self.newspaper_config.follow_meta_refresh = self.config.follow_redirects
        self.newspaper_config.number_threads = self.config.number_threads
        self.newspaper_config.memoize_articles = self.config.memoize_articles
        self.newspaper_config.fetch_images = self.config.fetch_images
        self.newspaper_config.language = self.config.language
        
        # Initialize robots.txt cache
        self.robots_cache = {}
        self.robots_cache_timestamps = {}
        
        # Initialize rate limiting
        self.last_request_time = {}
    
    def get_strategy_name(self) -> str:
        """
        Get the name of this scraper strategy.
        
        Returns:
            The strategy name as a string
        """
        return "newspaper"
    
    def can_fetch(self, url: str) -> bool:
        """
        Check if the URL can be fetched according to robots.txt.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the URL can be fetched, False otherwise
        """
        parsed_url = urllib.parse.urlparse(url)
        domain = parsed_url.netloc
        
        # Check if robots.txt is cached and not expired
        current_time = time.time()
        if domain in self.robots_cache:
            cache_time = self.robots_cache_timestamps.get(domain, 0)
            if current_time - cache_time < self.config.robots_cache_timeout:
                rp = self.robots_cache[domain]
                return rp.can_fetch(self.config.user_agent, url)
        
        # Fetch and parse robots.txt
        try:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{parsed_url.scheme}://{domain}/robots.txt")
            rp.read()
            
            # Cache the result
            self.robots_cache[domain] = rp
            self.robots_cache_timestamps[domain] = current_time
            
            return rp.can_fetch(self.config.user_agent, url)
        except Exception as e:
            self.logger.warning(f"Error fetching robots.txt for {domain}: {str(e)}")
            # If we can't fetch robots.txt, assume we can fetch the URL
            return True
    
    def _apply_rate_limiting(self, domain: str):
        """
        Apply rate limiting for a domain.
        
        Args:
            domain: The domain to apply rate limiting for
        """
        current_time = time.time()
        if domain in self.last_request_time:
            last_time = self.last_request_time[domain]
            rate_limit = self.config.rate_limits.get(domain, self.config.rate_limits.get("default", 1.0))
            
            # Calculate the time to wait
            time_to_wait = rate_limit - (current_time - last_time)
            if time_to_wait > 0:
                self.logger.debug(f"Rate limiting: waiting {time_to_wait:.2f} seconds for {domain}")
                time.sleep(time_to_wait)
        
        # Update the last request time
        self.last_request_time[domain] = time.time()
    
    def scrape(self, url: str) -> Optional[Article]:
        """
        Scrape an article from the given URL.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Standardized Article object or None if scraping failed
            
        Raises:
            RobotsExclusionError: If robots.txt disallows access
            ExtractionError: If content extraction fails
            NetworkError: If network operations fail
            ParsingError: If parsing operations fail
            ScraperError: For general scraping errors
        """
        # Check robots.txt
        if not self.can_fetch(url):
            raise RobotsExclusionError(f"robots.txt disallows access to {url}")
        
        # Apply rate limiting
        domain = urllib.parse.urlparse(url).netloc
        self._apply_rate_limiting(domain)
        
        # Initialize retry counter
        retries = 0
        
        while retries <= self.config.max_retries:
            try:
                # Create and download the article
                article = NewspaperArticle(url, config=self.newspaper_config)
                article.download()
                article.parse()
                
                # Extract natural language
                try:
                    article.nlp()
                except Exception as e:
                    self.logger.warning(f"NLP processing failed for {url}: {str(e)}")
                
                # Process the article
                processed_article = self._process_article(article, url)
                
                # Check if the article has enough content
                if len(processed_article.content) < self.config.minimum_content_length:
                    self.logger.warning(f"Article content too short: {len(processed_article.content)} characters")
                    if retries < self.config.max_retries:
                        retries += 1
                        backoff_time = self.config.retry_backoff ** retries
                        self.logger.info(f"Retrying in {backoff_time:.2f} seconds (attempt {retries}/{self.config.max_retries})")
                        time.sleep(backoff_time)
                        continue
                    else:
                        self.logger.error(f"Failed to extract sufficient content after {retries} retries")
                        return None
                
                # Calculate quality score
                processed_article.quality_score = self.calculate_quality_score(processed_article)
                
                return processed_article
                
            except newspaper.ArticleException as e:
                if "Download failed" in str(e):
                    if retries < self.config.max_retries:
                        retries += 1
                        backoff_time = self.config.retry_backoff ** retries
                        self.logger.info(f"Download failed, retrying in {backoff_time:.2f} seconds (attempt {retries}/{self.config.max_retries})")
                        time.sleep(backoff_time)
                        continue
                    else:
                        raise NetworkError(f"Failed to download article after {retries} retries: {str(e)}")
                else:
                    raise ExtractionError(f"Article extraction failed: {str(e)}")
            except Exception as e:
                if retries < self.config.max_retries:
                    retries += 1
                    backoff_time = self.config.retry_backoff ** retries
                    self.logger.info(f"Error occurred, retrying in {backoff_time:.2f} seconds (attempt {retries}/{self.config.max_retries})")
                    time.sleep(backoff_time)
                    continue
                else:
                    raise ScraperError(f"Unexpected error: {str(e)}")
        
        # If we get here, all retries failed
        raise ScraperError(f"Failed to scrape article after {self.config.max_retries} retries")
    
    def _process_article(self, article: NewspaperArticle, url: str) -> Article:
        """
        Process a newspaper Article object into our standardized Article format.
        
        Args:
            article: newspaper.Article object
            url: The original URL
            
        Returns:
            Standardized Article object
        """
        # Extract metadata
        metadata = self._extract_metadata(article, url)
        
        # Extract images
        images = self._extract_images(article)
        
        # Create standardized article
        return Article(
            url=url,
            title=article.title,
            content=article.text,
            html=article.html,
            metadata=metadata,
            images=images,
            quality_score=0.0  # Will be calculated later
        )
    
    def _extract_metadata(self, article: NewspaperArticle, url: str) -> ArticleMetadata:
        """
        Extract metadata from a newspaper Article object.
        
        Args:
            article: newspaper.Article object
            url: The original URL
            
        Returns:
            ArticleMetadata object
        """
        # Parse domain from URL
        domain = urllib.parse.urlparse(url).netloc
        
        # Convert publish date
        published_date = None
        if article.publish_date:
            if isinstance(article.publish_date, str):
                try:
                    published_date = datetime.fromisoformat(article.publish_date)
                except ValueError:
                    self.logger.warning(f"Could not parse publish date: {article.publish_date}")
            else:
                published_date = article.publish_date
        
        # Create metadata
        return ArticleMetadata(
            source_domain=domain,
            authors=article.authors,
            published_date=published_date,
            modified_date=None,  # Newspaper doesn't extract this
            section=None,  # Newspaper doesn't extract this
            tags=article.keywords if hasattr(article, 'keywords') else [],
            summary=article.summary if hasattr(article, 'summary') else "",
            language=article.meta_lang or self.config.language,
            extraction_date=datetime.now()
        )
    
    def _extract_images(self, article: NewspaperArticle) -> List[ArticleImage]:
        """
        Extract images from a newspaper Article object.
        
        Args:
            article: newspaper.Article object
            
        Returns:
            List of ArticleImage objects
        """
        images = []
        
        # Top image
        if article.top_image:
            images.append(ArticleImage(
                url=article.top_image,
                caption=None,
                local_path=None,
                width=None,
                height=None
            ))
        
        # Other images
        if hasattr(article, 'images') and article.images:
            for img_url in article.images:
                if img_url != article.top_image:  # Avoid duplicates
                    images.append(ArticleImage(
                        url=img_url,
                        caption=None,
                        local_path=None,
                        width=None,
                        height=None
                    ))
        
        return images 