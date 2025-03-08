"""
Test script to scrape the first article from url.csv using Newspaper4k directly.
"""

import sys
import json
import logging
from datetime import datetime
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import urllib.parse

import newspaper
from newspaper import Article as NewspaperArticle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


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


def get_first_article_url():
    """Get the URL of the first article from url.csv."""
    try:
        with open('url.csv', 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            first_row = next(csv_reader)
            return first_row['url']
    except Exception as e:
        logger.error(f"Error reading url.csv: {e}")
        return None


def calculate_quality_score(article: NewspaperArticle, content: str) -> float:
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


def clean_content(content: str) -> str:
    """
    Clean and normalize article content.
    
    Args:
        content: Raw article content
        
    Returns:
        Cleaned and normalized content
    """
    # Basic cleaning - remove extra whitespace
    cleaned = " ".join(content.split())
    
    return cleaned


def extract_metadata(article: NewspaperArticle) -> ArticleMetadata:
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
                published_date = datetime.fromisoformat(article.publish_date)
            except ValueError:
                logger.warning(f"Could not parse publish date: {article.publish_date}")
        else:
            published_date = article.publish_date
    
    # Extract domain
    source_domain = urllib.parse.urlparse(article.url).netloc
    
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


def extract_images(article: NewspaperArticle) -> List[ArticleImage]:
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


def process_article(article: NewspaperArticle, url: str) -> Article:
    """
    Process a newspaper Article object into our standardized Article format.
    
    Args:
        article: newspaper.Article object
        url: Original URL
        
    Returns:
        Standardized Article object
    """
    # Extract metadata
    metadata = extract_metadata(article)
    
    # Extract images
    images = extract_images(article)
    
    # Clean content
    clean_text = clean_content(article.text)
    
    # Calculate quality score
    quality_score = calculate_quality_score(article, clean_text)
    
    # Create standardized article
    return Article(
        url=url,
        title=article.title,
        content=clean_text,
        html=article.html,
        metadata=metadata,
        images=images,
        language=article.meta_lang or "en",
        quality_score=quality_score,
    )


def test_scraper(url):
    """
    Test scraping the given URL using Newspaper4k directly.
    
    Args:
        url: The URL to scrape
    """
    try:
        # Configure newspaper article
        article = NewspaperArticle(
            url,
            language='en',
            fetch_images=True,
            memoize_articles=True,
            request_timeout=30,
        )
        
        # Download and parse the article
        logger.info(f"Downloading article from {url}...")
        article.download()
        
        logger.info("Parsing article...")
        article.parse()
        
        # Extract natural language processing features
        logger.info("Extracting NLP features...")
        article.nlp()
        
        # Process the article into our standardized format
        processed_article = process_article(article, url)
        
        # Print article information
        logger.info(f"Title: {processed_article.title}")
        logger.info(f"Authors: {', '.join(processed_article.metadata.authors)}")
        if processed_article.metadata.published_date:
            logger.info(f"Published: {processed_article.metadata.published_date.isoformat()}")
        logger.info(f"Content length: {len(processed_article.content)} characters")
        logger.info(f"Number of images: {len(processed_article.images)}")
        logger.info(f"Quality score: {processed_article.quality_score:.2f}")
        
        # Convert to JSON
        article_dict = {
            "url": processed_article.url,
            "title": processed_article.title,
            "content": processed_article.content[:500] + "..." if len(processed_article.content) > 500 else processed_article.content,
            "metadata": {
                "authors": processed_article.metadata.authors,
                "published_date": processed_article.metadata.published_date,
                "tags": processed_article.metadata.tags,
                "summary": processed_article.metadata.summary,
                "source_domain": processed_article.metadata.source_domain,
            },
            "images": [{"url": img.url} for img in processed_article.images],
            "language": processed_article.language,
            "quality_score": processed_article.quality_score,
        }
        
        # Print JSON representation
        logger.info("Article JSON:")
        print(json.dumps(article_dict, indent=2, cls=DateTimeEncoder))
        
        return processed_article
        
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        return None


def main():
    """Main function to run the test."""
    # Get the first article URL from url.csv
    url = get_first_article_url()
    
    if not url:
        logger.error("Failed to get URL from url.csv")
        return
    
    logger.info(f"Testing scraper on URL: {url}")
    test_scraper(url)


if __name__ == "__main__":
    main() 