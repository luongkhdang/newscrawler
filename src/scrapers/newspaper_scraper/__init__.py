"""
Newspaper4k-based scraper package.
"""

from .newspaper_scraper import NewspaperScraper, NewspaperScraperConfig

__all__ = ['NewspaperScraper', 'NewspaperScraperConfig']

from .models import Article, ArticleMetadata, ArticleImage, ScraperConfig
from .exceptions import (
    ScraperError,
    RobotsExclusionError,
    RateLimitError,
    ExtractionError,
    NetworkError,
) 