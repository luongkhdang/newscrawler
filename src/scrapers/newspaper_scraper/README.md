# Newspaper4k Scraper

A robust wrapper around the Newspaper4k library for extracting articles from news websites.

## Features

- **Robots.txt Compliance**: Respects robots.txt rules for ethical scraping
- **Rate Limiting**: Implements domain-specific rate limiting to avoid overloading servers
- **Error Handling**: Comprehensive error handling with retry mechanisms
- **Standardized Output**: Converts Newspaper4k output to a standardized Article format
- **Quality Assessment**: Calculates a quality score for extracted articles
- **Metadata Extraction**: Extracts and normalizes article metadata
- **Image Extraction**: Extracts and processes article images

## Usage

```python
from src.scrapers.newspaper_scraper import NewspaperScraper, NewspaperScraperConfig

# Create configuration
config = NewspaperScraperConfig(
    user_agent="YourBot/1.0",
    browser_user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    request_timeout=30,
    minimum_content_length=300,
    rate_limits={"default": 1.0, "example.com": 5.0}  # Domain-specific rate limits in seconds
)

# Create scraper
scraper = NewspaperScraper(config.__dict__)

# Check if we can fetch the URL according to robots.txt
if scraper.can_fetch("https://example.com/article"):
    # Scrape the article
    article = scraper.scrape("https://example.com/article")
    
    # Access article properties
    print(f"Title: {article.title}")
    print(f"Content: {article.content[:100]}...")
    print(f"Authors: {', '.join(article.metadata.authors)}")
    print(f"Published date: {article.metadata.published_date}")
    print(f"Quality score: {article.quality_score}")
```

## Configuration Options

| Option | Description | Default |
| ------ | ----------- | ------- |
| `user_agent` | User agent for robots.txt requests | `"NewsCrawler/1.0"` |
| `browser_user_agent` | User agent for article requests | `"Mozilla/5.0..."` |
| `request_timeout` | Request timeout in seconds | `30` |
| `follow_redirects` | Whether to follow redirects | `True` |
| `number_threads` | Number of threads for Newspaper4k | `1` |
| `memoize_articles` | Whether to memoize articles | `True` |
| `language` | Default language | `"en"` |
| `fetch_images` | Whether to fetch images | `True` |
| `minimum_content_length` | Minimum content length in characters | `300` |
| `robots_cache_timeout` | Robots.txt cache timeout in seconds | `3600` |
| `rate_limits` | Domain-specific rate limits in seconds | `{"default": 1.0}` |
| `max_retries` | Maximum number of retries | `3` |
| `retry_backoff` | Retry backoff factor | `2.0` |
| `verify_ssl` | Whether to verify SSL certificates | `True` |

## Error Handling

The scraper can raise the following exceptions:

- `RobotsExclusionError`: When robots.txt disallows access to a URL
- `ExtractionError`: When content extraction fails
- `NetworkError`: When network operations fail
- `ParsingError`: When parsing operations fail
- `ScraperError`: For general scraping errors

## Dependencies

- newspaper4k
- lxml[html_clean]
- nltk
- requests
- python-dateutil

## Data Models

### Article

The main article representation with the following properties:

- `url`: The URL of the article
- `title`: The article title
- `content`: The cleaned article content
- `html`: The raw HTML content
- `metadata`: An ArticleMetadata object
- `images`: A list of ArticleImage objects
- `language`: The article language code
- `quality_score`: A score between 0.0 and 1.0 indicating article quality

### ArticleMetadata

Contains metadata about the article:

- `authors`: List of author names
- `published_date`: Publication date (datetime)
- `modified_date`: Last modification date (datetime)
- `section`: Article section or category
- `tags`: List of article tags or keywords
- `summary`: Brief summary of the article
- `source_domain`: Domain of the source website

### ArticleImage

Represents an image in the article:

- `url`: The URL of the image
- `caption`: Image caption (if available)
- `local_path`: Path to locally stored image (if downloaded)
- `width`: Image width (if available)
- `height`: Image height (if available)

## Running Tests

```bash
# Run all tests
pytest tests/scrapers/newspaper_scraper

# Run specific test
pytest tests/scrapers/newspaper_scraper/test_scraper.py::TestNewspaperScraper::test_scrape_success
```

## Example Script

An example script is provided to demonstrate usage:

```bash
# Run the example script with a URL
python -m src.scrapers.newspaper_scraper.example https://example.com/article
``` 