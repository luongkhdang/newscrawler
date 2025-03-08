"""
Exception classes for the scraper system.
"""


class ScraperError(Exception):
    """Base class for scraper exceptions."""
    pass


class RobotsExclusionError(ScraperError):
    """Raised when robots.txt disallows access to a URL."""
    pass


class RateLimitError(ScraperError):
    """Raised when rate limit is exceeded."""
    pass


class ExtractionError(ScraperError):
    """Raised when content extraction fails."""
    pass


class NetworkError(ScraperError):
    """Raised when network operations fail."""
    pass


class ParsingError(ScraperError):
    """Raised when parsing operations fail."""
    pass


class ConfigurationError(ScraperError):
    """Raised when there is a configuration error."""
    pass


class ValidationError(ScraperError):
    """Raised when validation fails."""
    pass 