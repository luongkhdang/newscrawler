"""
Utilities package for the scraper system.
"""

from .exceptions import (
    ScraperError, 
    RobotsExclusionError, 
    RateLimitError, 
    ExtractionError, 
    NetworkError,
    ParsingError,
    ConfigurationError,
    ValidationError
)

__all__ = [
    'ScraperError', 
    'RobotsExclusionError', 
    'RateLimitError', 
    'ExtractionError', 
    'NetworkError',
    'ParsingError',
    'ConfigurationError',
    'ValidationError'
] 