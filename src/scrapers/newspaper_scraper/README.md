# Newspaper4k Scraper

A robust wrapper around the Newspaper4k library for extracting articles from news websites with additional features:

- Robots.txt compliance
- Rate limiting
- Standardized output format
- Error handling and resilience
- Content quality assessment

## Features

- **Robots.txt Compliance**: Respects robots.txt directives to ensure ethical scraping
- **Rate Limiting**: Configurable per-domain rate limiting to avoid overloading servers
- **Standardized Output**: Consistent article format regardless of source
- **Error Handling**: Comprehensive error handling with appropriate exception types
- **Content Quality Assessment**: Automatic quality scoring of extracted content
- **Metadata Extraction**: Extracts and normalizes article metadata
- **Image Extraction**: Extracts and processes article images

## Usage

### Basic Usage

```python
from src.scrapers.newspaper_scraper import NewspaperScraper, ScraperConfig

# Create a scraper with default configuration
scraper = NewspaperScraper()

# Scrape an article
article = scraper.scrape("https://example.com/article")

# Access article properties
print(f"Title: {article.title}")
print(f"Content: {article.content[:100]}...")
print(f"Authors: {', '.join(article.metadata.authors)}")
print(f"Published: {article.metadata.published_date}")
print(f"Images: {len(article.images)}")
```

### Custom Configuration

```python
from src.scrapers.newspaper_scraper import NewspaperScraper, ScraperConfig

# Create a custom configuration
config = ScraperConfig(
    user_agent="MyNewsBot/1.0 (contact@example.com)",
    timeout=30,
    rate_limits={
        "example.com": 5.0,  # 5 seconds between requests to example.com
        "default": 2.0,      # 2 seconds for all other domains
    },
    max_retries=3,
    follow_redirects=True,
    verify_ssl=True,
    language="en",  # Default language for articles
)

# Create a scraper with custom configuration
scraper = NewspaperScraper(config)

# Scrape an article
article = scraper.scrape("https://example.com/article")
```

### Error Handling

```python
from src.scrapers.newspaper_scraper import (
    NewspaperScraper,
    RobotsExclusionError,
    RateLimitError,
    ExtractionError,
    NetworkError,
    ScraperError,
)

scraper = NewspaperScraper()

try:
    article = scraper.scrape("https://example.com/article")
    # Process article...
except RobotsExclusionError:
    print("Access to this URL is disallowed by robots.txt")
except RateLimitError:
    print("Rate limit exceeded")
except ExtractionError as e:
    print(f"Failed to extract article: {e}")
except NetworkError as e:
    print(f"Network error: {e}")
except ScraperError as e:
    print(f"General scraper error: {e}")
```

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