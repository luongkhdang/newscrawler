"""
Factory for creating the appropriate scraper for a given URL.
"""

import logging
from typing import Dict, Any, Optional
from urllib.parse import urlparse

from src.scrapers.base_scraper import BaseScraper
from src.utils.url_classifier import URLClassifier
from src.utils.exceptions import ConfigurationError
from src.scrapers.newspaper_scraper import NewspaperScraper
from src.scrapers.feed_scraper import FeedScraper

logger = logging.getLogger(__name__)

class ScraperFactory:
    """
    Factory for creating the appropriate scraper for a given URL.
    """
    
    def __init__(self, scrapers: Dict[str, BaseScraper] = None, config: Dict[str, Any] = None):
        """
        Initialize the factory with scrapers and configuration.
        
        Args:
            scrapers: Dictionary of scraper instances keyed by strategy name
            config: Configuration options
        """
        self.config = config or {}
        self.scrapers = scrapers or {}
        self.url_classifier = URLClassifier()
        self.logger = logging.getLogger(__name__)
        
        # Register default scrapers if none provided
        if not self.scrapers:
            self.register_scraper("newspaper", NewspaperScraper())
            self.register_scraper("feed", FeedScraper())
    
    def register_scraper(self, strategy_name: str, scraper: BaseScraper):
        """
        Register a scraper for a strategy.
        
        Args:
            strategy_name: The name of the strategy
            scraper: The scraper instance
        """
        self.scrapers[strategy_name] = scraper
        self.logger.debug(f"Registered scraper for strategy: {strategy_name}")
    
    def get_scraper(self, url: str) -> BaseScraper:
        """
        Get the appropriate scraper for the given URL.
        
        Args:
            url: The URL to scrape
            
        Returns:
            The appropriate scraper instance
            
        Raises:
            ConfigurationError: If no scraper is available for the strategy
        """
        strategy = self.url_classifier.classify_url(url)
        self.logger.debug(f"Selected strategy for {url}: {strategy}")
        
        if strategy in self.scrapers:
            return self.scrapers[strategy]
        else:
            # If no specific scraper is available, use the newspaper scraper as fallback
            if "newspaper" in self.scrapers:
                self.logger.warning(f"No scraper available for strategy '{strategy}', using newspaper as fallback")
                return self.scrapers["newspaper"]
            else:
                raise ConfigurationError(f"No scraper available for strategy: {strategy}")
    
    def get_available_strategies(self) -> Dict[str, str]:
        """
        Get the available scraper strategies.
        
        Returns:
            Dictionary of strategy names and descriptions
        """
        return {
            name: scraper.__class__.__doc__.strip() if scraper.__class__.__doc__ else "No description available"
            for name, scraper in self.scrapers.items()
        }

    @staticmethod
    def create_scraper(strategy: str) -> Optional[BaseScraper]:
        """Create a scraper based on the strategy."""
        if strategy == "newspaper":
            return NewspaperScraper()
        elif strategy == "feed":
            return FeedScraper()
        else:
            logger.error(f"No scraper available for strategy: {strategy}")
            return None 