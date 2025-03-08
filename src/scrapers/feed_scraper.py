import logging
import datetime
import time
import feedparser
import requests
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
from dataclasses import dataclass, field

from src.models.article import Article, ArticleMetadata
from src.scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

@dataclass
class FeedScraperConfig:
    """Configuration for the FeedScraper."""
    user_agent: str = "NewsCrawler/1.0 FeedScraper"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 2
    respect_robots_txt: bool = True
    rate_limit: float = 1.0  # Requests per second
    feed_cache_time: int = 3600  # Cache feed data for 1 hour (in seconds)
    
    # Additional feed-specific settings
    max_entries_per_feed: int = 20
    min_content_length: int = 100
    
    # Fields with default factory
    headers: Dict[str, str] = field(default_factory=lambda: {
        "User-Agent": "NewsCrawler/1.0 FeedScraper"
    })


class FeedScraper(BaseScraper):
    """
    A scraper that extracts articles from RSS/Atom feeds.
    """
    
    def __init__(self, config: Optional[FeedScraperConfig] = None):
        """Initialize the FeedScraper with the given configuration."""
        super().__init__()  # Initialize the base class
        self.config = config or FeedScraperConfig()
        self.last_request_time = 0
        self._feed_cache = {}  # Cache for feed data
        
        # For robots.txt compliance
        self._robots_cache = {}
    
    def get_strategy_name(self) -> str:
        """
        Get the name of this scraper strategy.
        
        Returns:
            The strategy name as a string
        """
        return "feed"
    
    def can_fetch(self, url: str) -> bool:
        """
        Check if the URL can be fetched according to robots.txt.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the URL can be fetched, False otherwise
        """
        if not self.config.respect_robots_txt:
            return True
            
        try:
            from urllib.robotparser import RobotFileParser
            from urllib.parse import urlparse
            
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Check if we have a cached robots.txt parser for this domain
            if domain in self._robots_cache:
                return self._robots_cache[domain].can_fetch(self.config.user_agent, url)
            
            # Create a new parser
            robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"
            rp = RobotFileParser()
            rp.set_url(robots_url)
            
            # Fetch and parse robots.txt
            self._rate_limit()
            rp.read()
            
            # Cache the parser
            self._robots_cache[domain] = rp
            
            # Check if we can fetch the URL
            return rp.can_fetch(self.config.user_agent, url)
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {e}")
            # If there's an error, assume we can fetch
            return True
        
    def _rate_limit(self):
        """Implement rate limiting to avoid overloading servers."""
        if self.config.rate_limit > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < (1.0 / self.config.rate_limit):
                time.sleep((1.0 / self.config.rate_limit) - elapsed)
        self.last_request_time = time.time()
    
    def _get_feed_url(self, url: str) -> str:
        """
        If the URL is already a feed URL, return it.
        Otherwise, try to discover the feed URL from the website.
        """
        parsed_url = urlparse(url)
        
        # If URL ends with common feed extensions, assume it's a feed URL
        if parsed_url.path.endswith(('.xml', '.rss', '.atom')):
            return url
            
        # Check if URL is a domain or homepage
        if not parsed_url.path or parsed_url.path == '/':
            # Try common feed URLs
            common_feed_paths = [
                '/feed', '/rss', '/atom', '/rss.xml', '/atom.xml', 
                '/feed.xml', '/index.xml', '/feeds/posts/default',
                '/rss/all.xml'
            ]
            
            for path in common_feed_paths:
                feed_url = f"{parsed_url.scheme}://{parsed_url.netloc}{path}"
                try:
                    self._rate_limit()
                    response = requests.head(feed_url, 
                                           headers=self.config.headers, 
                                           timeout=self.config.timeout)
                    if response.status_code == 200:
                        logger.info(f"Found feed at {feed_url}")
                        return feed_url
                except Exception as e:
                    logger.debug(f"Error checking feed URL {feed_url}: {e}")
                    continue
        
        # If we couldn't find a feed URL, try to parse the HTML to find feed links
        try:
            self._rate_limit()
            response = requests.get(url, headers=self.config.headers, timeout=self.config.timeout)
            if response.status_code == 200:
                # Use feedparser's auto-discovery
                feed = feedparser.parse(response.text)
                if feed.feed and hasattr(feed.feed, 'link'):
                    return feed.feed.link
                
                # Simple regex-based approach as fallback
                import re
                feed_urls = re.findall(r'<link[^>]*type=["\']application/(rss|atom)\+xml["\'][^>]*href=["\']([^"\']+)["\']', response.text)
                if feed_urls:
                    for _, href in feed_urls:
                        if href.startswith('/'):
                            return f"{parsed_url.scheme}://{parsed_url.netloc}{href}"
                        return href
        except Exception as e:
            logger.error(f"Error discovering feed URL for {url}: {e}")
        
        # If all else fails, return the original URL and hope feedparser can handle it
        return url
    
    def _parse_feed(self, url: str) -> Dict[str, Any]:
        """Parse the feed and return the feed data."""
        # Check if we have a cached version
        now = time.time()
        if url in self._feed_cache:
            cache_time, feed_data = self._feed_cache[url]
            if now - cache_time < self.config.feed_cache_time:
                return feed_data
        
        # Get the feed URL
        feed_url = self._get_feed_url(url)
        
        # Parse the feed
        self._rate_limit()
        feed = feedparser.parse(feed_url)
        
        # Cache the feed data
        self._feed_cache[url] = (now, feed)
        
        return feed
    
    def _extract_article_from_entry(self, entry: Dict[str, Any], feed_url: str) -> Optional[Article]:
        """Extract article data from a feed entry."""
        try:
            # Get the article URL
            article_url = entry.get('link', '')
            if not article_url:
                return None
            
            # Get the article title
            title = entry.get('title', 'Unknown Title')
            
            # Get the article content
            content = ''
            if 'content' in entry:
                for content_item in entry.content:
                    if 'value' in content_item:
                        content += content_item.value
            elif 'summary' in entry:
                content = entry.summary
            elif 'description' in entry:
                content = entry.description
                
            # If content is too short, skip this entry
            if len(content) < self.config.min_content_length:
                logger.debug(f"Content too short for {article_url}: {len(content)} characters")
                return None
                
            # Get the article metadata
            authors = []
            if 'author' in entry:
                authors.append(entry.author)
            elif 'authors' in entry:
                authors.extend([author.name for author in entry.authors])
                
            published_date = None
            if 'published_parsed' in entry and entry.published_parsed:
                published_date = datetime.datetime(*entry.published_parsed[:6])
            elif 'updated_parsed' in entry and entry.updated_parsed:
                published_date = datetime.datetime(*entry.updated_parsed[:6])
                
            # Extract images
            images = []
            if 'media_content' in entry:
                for media in entry.media_content:
                    if 'url' in media and media.get('medium') == 'image':
                        images.append(media.url)
            
            # Get the domain from the URL
            domain = urlparse(article_url).netloc
            
            # Create article metadata with the correct parameter names
            metadata = ArticleMetadata(
                source_domain=domain,
                authors=authors,
                published_date=published_date,
                modified_date=None,
                section=None,
                tags=[tag.term for tag in entry.get('tags', [])] if 'tags' in entry else [],
                summary=entry.get('summary', '')[:200] if 'summary' in entry else '',
                language=entry.get('language', 'en')
            )
            
            # Store feed URL in extra_data
            metadata.extra_data = {"feed_url": feed_url}
            
            # Create and return the article
            article = Article(
                url=article_url,
                title=title,
                content=content,
                html=None,  # Feed entries don't typically include full HTML
                images=images,
                metadata=metadata,
                quality_score=0.8  # Default quality score for feed articles
            )
            
            # Calculate quality score
            article.quality_score = self.calculate_quality_score(article)
            
            return article
            
        except Exception as e:
            logger.error(f"Error extracting article from feed entry: {e}")
            return None
    
    def scrape(self, url: str) -> Optional[Article]:
        """
        Scrape an article from the given URL.
        If the URL is a feed URL, extract the first article from the feed.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Standardized Article object or None if scraping failed
        """
        # Check if we can fetch this URL according to robots.txt
        if not self.can_fetch(url):
            logger.warning(f"robots.txt disallows access to {url}")
            raise Exception(f"robots.txt disallows access to {url}")
            
        try:
            # Parse the feed
            feed = self._parse_feed(url)
            
            # If no entries, return None
            if not feed.entries:
                logger.warning(f"No entries found in feed at {url}")
                return None
                
            # Get the first entry
            entry = feed.entries[0]
            
            # Extract and return the article
            return self._extract_article_from_entry(entry, url)
            
        except Exception as e:
            logger.error(f"Error scraping feed at {url}: {e}")
            return None
    
    def scrape_feed(self, url: str) -> List[Article]:
        """
        Scrape all articles from the feed at the given URL.
        Returns a list of Article objects.
        
        Args:
            url: The URL of the feed to scrape
            
        Returns:
            List of Article objects
        """
        articles = []
        
        # Check if we can fetch this URL according to robots.txt
        if not self.can_fetch(url):
            logger.warning(f"robots.txt disallows access to {url}")
            return articles
            
        try:
            # Parse the feed
            feed = self._parse_feed(url)
            
            # Process each entry
            for entry in feed.entries[:self.config.max_entries_per_feed]:
                article = self._extract_article_from_entry(entry, url)
                if article:
                    articles.append(article)
                    
            logger.info(f"Extracted {len(articles)} articles from feed at {url}")
            
        except Exception as e:
            logger.error(f"Error scraping feed at {url}: {e}")
            
        return articles 