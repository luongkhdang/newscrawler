"""
Simple test script to verify the Newspaper4k scraper implementation.
"""

import logging
import json
import sys
import os
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

from src.scrapers.newspaper_scraper import NewspaperScraper, NewspaperScraperConfig
from src.utils.exceptions import ScraperError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_scraper(url):
    """
    Test the Newspaper4k scraper with a single URL.
    
    Args:
        url: The URL to test
    """
    logger.info(f"Testing Newspaper4k scraper with URL: {url}")
    
    # Create scraper configuration
    config = NewspaperScraperConfig(
        user_agent="NewsCrawler-Test/1.0",
        browser_user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        request_timeout=30,
        minimum_content_length=100,  # Lower threshold for testing
        rate_limits={"default": 1.0}
    )
    
    # Create scraper
    scraper = NewspaperScraper(config.__dict__)
    
    try:
        # Check if we can fetch the URL
        if not scraper.can_fetch(url):
            logger.error(f"robots.txt disallows access to {url}")
            return
        
        # Scrape the URL
        article = scraper.scrape(url)
        
        if article:
            # Print article information
            logger.info(f"Title: {article.title}")
            logger.info(f"Content length: {len(article.content)} characters")
            logger.info(f"Quality score: {article.quality_score:.2f}")
            logger.info(f"Authors: {', '.join(article.metadata.authors) if article.metadata.authors else 'Unknown'}")
            logger.info(f"Published date: {article.metadata.published_date}")
            logger.info(f"Number of images: {len(article.images)}")
            
            # Save article to file
            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            domain = article.metadata.source_domain
            filename = f"{domain}_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(article.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved article to {filepath}")
            
            # Print content preview
            content_preview = article.content[:500] + "..." if len(article.content) > 500 else article.content
            logger.info(f"Content preview:\n{content_preview}")
        else:
            logger.error("Failed to scrape article")
    
    except ScraperError as e:
        logger.error(f"Scraper error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")


def main():
    """Main function."""
    # Test with a sample URL
    test_url = "https://www.bbc.com/news/world-us-canada-68465048"
    
    # Check if a URL was provided as a command-line argument
    if len(sys.argv) > 1:
        test_url = sys.argv[1]
    
    test_scraper(test_url)


if __name__ == "__main__":
    main() 