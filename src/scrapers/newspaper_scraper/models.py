"""
Data models for the Newspaper4k scraper.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class ScraperConfig:
    """Configuration options for the scraper."""
    user_agent: str = "NewsCrawler/1.0"
    timeout: int = 30
    rate_limits: Dict[str, float] = field(default_factory=lambda: {"default": 1.0})
    max_retries: int = 3
    follow_redirects: bool = True
    verify_ssl: bool = True
    custom_selectors: Dict[str, Dict[str, str]] = field(default_factory=dict)
    language: Optional[str] = None


@dataclass
class ArticleMetadata:
    """Article metadata."""
    authors: List[str]
    published_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    section: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    summary: str = ""
    source_domain: str = ""


@dataclass
class ArticleImage:
    """Article image representation."""
    url: str
    caption: Optional[str] = None
    local_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class Article:
    """Standardized article representation."""
    url: str
    title: str
    content: str
    html: str
    metadata: ArticleMetadata
    images: List[ArticleImage] = field(default_factory=list)
    language: str = "en"
    quality_score: float = 0.0 