"""
Tests for the NewspaperScraper.
"""

import unittest
from unittest.mock import patch, MagicMock
import datetime

from src.scrapers.newspaper_scraper.scraper import NewspaperScraper
from src.scrapers.newspaper_scraper.models import ScraperConfig, Article, ArticleMetadata
from src.scrapers.newspaper_scraper.exceptions import (
    RobotsExclusionError,
    ExtractionError,
    NetworkError,
)


class TestNewspaperScraper(unittest.TestCase):
    """Test cases for the NewspaperScraper class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ScraperConfig(
            user_agent="TestAgent/1.0",
            timeout=10,
            rate_limits={"default": 0.1},  # Fast for testing
            max_retries=1,
        )
        self.scraper = NewspaperScraper(self.config)
        self.test_url = "https://example.com/article"

    @patch("src.scrapers.newspaper_scraper.scraper.NewspaperArticle")
    @patch("src.scrapers.newspaper_scraper.scraper.RobotsCache")
    @patch("src.scrapers.newspaper_scraper.scraper.RateLimiter")
    def test_scrape_success(self, mock_rate_limiter, mock_robots_cache, mock_newspaper_article):
        """Test successful article scraping."""
        # Configure mocks
        mock_robots_cache_instance = mock_robots_cache.return_value
        mock_robots_cache_instance.can_fetch.return_value = True

        mock_rate_limiter_instance = mock_rate_limiter.return_value
        
        mock_article = MagicMock()
        mock_article.title = "Test Article"
        mock_article.text = "This is a test article content."
        mock_article.html = "<html><body>This is a test article content.</body></html>"
        mock_article.authors = ["Test Author"]
        mock_article.publish_date = datetime.datetime(2025, 3, 8)
        mock_article.keywords = ["test", "article"]
        mock_article.summary = "Test article summary"
        mock_article.top_image = "https://example.com/image.jpg"
        mock_article.images = ["https://example.com/image.jpg", "https://example.com/image2.jpg"]
        mock_article.meta_lang = "en"
        mock_article.url = self.test_url
        
        mock_newspaper_article.return_value = mock_article
        
        # Call the method under test
        result = self.scraper.scrape(self.test_url)
        
        # Verify the result
        self.assertIsInstance(result, Article)
        self.assertEqual(result.url, self.test_url)
        self.assertEqual(result.title, "Test Article")
        self.assertEqual(result.content, "This is a test article content.")
        self.assertEqual(result.language, "en")
        
        # Verify metadata
        self.assertEqual(result.metadata.authors, ["Test Author"])
        self.assertEqual(result.metadata.published_date, datetime.datetime(2025, 3, 8))
        self.assertEqual(result.metadata.tags, ["test", "article"])
        
        # Verify images
        self.assertEqual(len(result.images), 2)
        self.assertEqual(result.images[0].url, "https://example.com/image.jpg")
        
        # Verify method calls
        mock_robots_cache_instance.can_fetch.assert_called_once_with(self.test_url)
        mock_article.download.assert_called_once()
        mock_article.parse.assert_called_once()

    @patch("src.scrapers.newspaper_scraper.scraper.RobotsCache")
    def test_robots_exclusion(self, mock_robots_cache):
        """Test robots.txt exclusion handling."""
        # Configure mock
        mock_robots_cache_instance = mock_robots_cache.return_value
        mock_robots_cache_instance.can_fetch.return_value = False
        
        # Verify exception is raised
        with self.assertRaises(RobotsExclusionError):
            self.scraper.scrape(self.test_url)
        
        # Verify method calls
        mock_robots_cache_instance.can_fetch.assert_called_once_with(self.test_url)

    @patch("src.scrapers.newspaper_scraper.scraper.NewspaperArticle")
    @patch("src.scrapers.newspaper_scraper.scraper.RobotsCache")
    @patch("src.scrapers.newspaper_scraper.scraper.RateLimiter")
    def test_extraction_error(self, mock_rate_limiter, mock_robots_cache, mock_newspaper_article):
        """Test handling of extraction errors."""
        # Configure mocks
        mock_robots_cache_instance = mock_robots_cache.return_value
        mock_robots_cache_instance.can_fetch.return_value = True
        
        mock_article = MagicMock()
        mock_article.download.side_effect = Exception("Download failed")
        mock_newspaper_article.return_value = mock_article
        
        # Verify exception is raised
        with self.assertRaises(ExtractionError):
            self.scraper.scrape(self.test_url)

    def test_clean_content(self):
        """Test content cleaning functionality."""
        # Test with extra whitespace
        content = "This   is \n a \t test   content."
        expected = "This is a test content."
        result = self.scraper.clean_content(content)
        self.assertEqual(result, expected)

    def test_calculate_quality_score(self):
        """Test quality score calculation."""
        # Create a mock article with good quality
        mock_article = MagicMock()
        mock_article.title = "Test Article Title"
        mock_article.authors = ["Author 1", "Author 2"]
        mock_article.publish_date = datetime.datetime.now()
        mock_article.top_image = "https://example.com/image.jpg"
        mock_article.keywords = ["test", "article", "quality"]
        
        # Calculate score for good content
        good_content = "This is a high-quality article with sufficient length to pass the content length check."
        score = self.scraper.calculate_quality_score(mock_article, good_content)
        self.assertEqual(score, 1.0)  # Should get full score
        
        # Calculate score for poor content
        poor_content = "Too short"
        mock_article.title = ""
        mock_article.authors = []
        mock_article.publish_date = None
        mock_article.top_image = ""
        mock_article.keywords = []
        
        score = self.scraper.calculate_quality_score(mock_article, poor_content)
        self.assertEqual(score, 0.0)  # Should get zero score


if __name__ == "__main__":
    unittest.main() 