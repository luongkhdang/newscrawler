"""
Newspaper4k-based scraper for NewsCrawler.
"""

from .scraper import NewspaperScraper
from .models import Article, ArticleMetadata, ArticleImage, ScraperConfig
from .exceptions import (
    ScraperError,
    RobotsExclusionError,
    RateLimitError,
    ExtractionError,
    NetworkError,
) 