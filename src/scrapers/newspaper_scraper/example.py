"""
Example usage of the NewspaperScraper.
"""

import logging
import json
from datetime import datetime
import sys

from .scraper import NewspaperScraper
from .models import ScraperConfig
from .exceptions import ScraperError


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


def scrape_url(url: str) -> None:
    """
    Scrape an article from the given URL and print the results.
    
    Args:
        url: The URL to scrape
    """
    # Configure the scraper
    config = ScraperConfig(
        user_agent="NewsCrawler/1.0 (Example)",
        timeout=30,
        rate_limits={"default": 2.0},  # 2 second delay between requests
        max_retries=3,
        follow_redirects=True,
        verify_ssl=True,
    )
    
    # Create the scraper
    scraper = NewspaperScraper(config)
    
    try:
        # Check if we can fetch the URL
        if not scraper.can_fetch(url):
            logger.warning(f"robots.txt disallows access to {url}")
            return
        
        # Scrape the article
        logger.info(f"Scraping {url}...")
        article = scraper.scrape(url)
        
        # Print article information
        logger.info(f"Title: {article.title}")
        logger.info(f"Authors: {', '.join(article.metadata.authors)}")
        if article.metadata.published_date:
            logger.info(f"Published: {article.metadata.published_date.isoformat()}")
        logger.info(f"Content length: {len(article.content)} characters")
        logger.info(f"Number of images: {len(article.images)}")
        logger.info(f"Quality score: {article.quality_score:.2f}")
        
        # Convert to JSON
        article_dict = {
            "url": article.url,
            "title": article.title,
            "content": article.content[:200] + "..." if len(article.content) > 200 else article.content,
            "metadata": {
                "authors": article.metadata.authors,
                "published_date": article.metadata.published_date,
                "tags": article.metadata.tags,
                "summary": article.metadata.summary,
                "source_domain": article.metadata.source_domain,
            },
            "images": [{"url": img.url} for img in article.images],
            "language": article.language,
            "quality_score": article.quality_score,
        }
        
        # Print JSON representation
        logger.info("Article JSON:")
        print(json.dumps(article_dict, indent=2, cls=DateTimeEncoder))
        
    except ScraperError as e:
        logger.error(f"Error scraping {url}: {str(e)}")


def main():
    """Main function to run the example."""
    if len(sys.argv) < 2:
        logger.error("Please provide a URL to scrape")
        logger.info("Usage: python -m src.scrapers.newspaper_scraper.example <url>")
        return
    
    url = sys.argv[1]
    scrape_url(url)


if __name__ == "__main__":
    main() 