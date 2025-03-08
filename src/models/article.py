"""
Article model classes for standardized article representation.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class ArticleImage:
    """Article image representation."""
    url: str
    caption: Optional[str] = None
    local_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class ArticleMetadata:
    """Article metadata."""
    source_domain: str
    authors: List[str] = field(default_factory=list)
    published_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    section: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    summary: str = ""
    language: str = "en"
    extraction_date: datetime = field(default_factory=datetime.now)


@dataclass
class Article:
    """Standardized article representation."""
    url: str
    title: str
    content: str
    html: str
    metadata: ArticleMetadata
    images: List[ArticleImage] = field(default_factory=list)
    quality_score: float = 0.0
    
    def to_dict(self):
        """Convert the article to a dictionary for serialization."""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "metadata": {
                "source_domain": self.metadata.source_domain,
                "authors": self.metadata.authors,
                "published_date": self.metadata.published_date.isoformat() if self.metadata.published_date else None,
                "modified_date": self.metadata.modified_date.isoformat() if self.metadata.modified_date else None,
                "section": self.metadata.section,
                "tags": self.metadata.tags,
                "summary": self.metadata.summary,
                "language": self.metadata.language,
                "extraction_date": self.metadata.extraction_date.isoformat(),
            },
            "images": [
                {
                    "url": img.url,
                    "caption": img.caption,
                    "local_path": img.local_path,
                    "width": img.width,
                    "height": img.height,
                }
                for img in self.images
            ],
            "quality_score": self.quality_score,
        } 