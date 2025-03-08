# NewsCrawler

A comprehensive web scraping and article retrieval system designed to collect news articles from a predefined list of URLs, process and clean the content, and store it in a structured database for future RAG (Retrieval-Augmented Generation) chatbot use.

## Project Overview

NewsCrawler is designed to:

1. Crawl and extract articles from diverse news sources
2. Process and clean the content to extract relevant information
3. Store articles in a PostgreSQL database with appropriate schema design
4. Provide a RESTful API using FastAPI for data retrieval and management
5. Run in containerized environments using Docker for easy deployment
6. Integrate with LangChain for future RAG capabilities

## Features

- **Multi-Strategy Web Scraping**: Uses multiple libraries (Newspaper4k, feedparser, BeautifulSoup4, Puppeteer) to handle different website structures
- **Ethical Crawling**: Implements robots.txt compliance, rate limiting, and politeness protocols
- **Content Processing**: Extracts and normalizes article content, metadata, and images
- **Standardized Output**: Provides consistent article format regardless of source
- **Error Handling**: Implements comprehensive error handling and retry mechanisms
- **Database Storage**: Stores articles in a PostgreSQL database with optimized schema
- **API Access**: Provides RESTful API endpoints for data retrieval and management
- **Containerization**: Runs in Docker containers for easy deployment and scaling

## Project Structure

```
newscrawler/
├── src/                        # Source code
│   ├── scrapers/               # Web scraping modules
│   │   ├── newspaper_scraper/  # Newspaper4k-based scraper
│   │   ├── feed_scraper/       # Feedparser-based scraper
│   │   ├── bs4_scraper/        # BeautifulSoup4-based scraper
│   │   └── puppeteer_scraper/  # Puppeteer-based scraper
│   ├── processors/             # Content processing modules
│   ├── database/               # Database modules
│   ├── api/                    # FastAPI implementation
│   └── utils/                  # Utility modules
├── tests/                      # Test suite
├── docs/                       # Documentation
├── docker/                     # Docker configuration
├── scripts/                    # Utility scripts
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

## Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (for containerized deployment)
- PostgreSQL (for database storage)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/newscrawler.git
   cd newscrawler
   ```

2. Create and activate a virtual environment:
   ```bash
   # Create a virtual environment
   python -m venv venv
   
   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package in development mode:
   ```bash
   pip install -e .
   ```

5. When you're done working on the project, deactivate the virtual environment:
   ```bash
   deactivate
   ```

### Usage

#### Running the Scrapers

```python
# Make sure your virtual environment is activated before running any code
from src.scrapers.newspaper_scraper import NewspaperScraper

# Create a scraper
scraper = NewspaperScraper()

# Scrape an article
article = scraper.scrape("https://example.com/article")

# Access article properties
print(f"Title: {article.title}")
print(f"Content: {article.content[:100]}...")
print(f"Authors: {', '.join(article.metadata.authors)}")
```

#### Running Tests

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/scrapers/newspaper_scraper/
```

#### Docker Deployment

```bash
# Build and start containers
docker-compose up -d

# View logs
docker-compose logs -f

# Stop containers
docker-compose down
```

> **Note**: When using Docker, the virtual environment is handled within the container itself. You don't need to activate a local virtual environment when running the application via Docker, as the container provides its own isolated environment.

## Documentation

For more detailed documentation, see the following:

- [Newspaper4k Scraper](src/scrapers/newspaper_scraper/README.md)
- [Database Schema](docs/database_schema.md)
- [API Documentation](docs/api.md)
- [Docker Deployment](docs/docker_deployment.md)

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Newspaper4k](https://newspaper4k.readthedocs.io/) for article extraction
- [feedparser](https://feedparser.readthedocs.io/) for RSS feed processing
- [BeautifulSoup4](https://pypi.org/project/beautifulsoup4/) for HTML parsing
- [Puppeteer](https://pptr.dev/guides/) for JavaScript-heavy sites
- [FastAPI](https://fastapi.tiangolo.com/) for API development
- [PostgreSQL](https://www.postgresql.org/) for database storage
- [LangChain](https://python.langchain.com/) for RAG integration

## Simple Test Script

For testing purposes, we've created a simple script that can extract content from news articles using only built-in Python libraries. This is useful when you encounter dependency issues with more complex libraries like Newspaper4k.

### Features

- Extracts article title, content, and metadata
- Uses multiple strategies for title extraction
- Implements robust content cleaning
- Saves results to both JSON and text files
- Handles errors gracefully with comprehensive logging

### Usage

```bash
python test_simple.py
```

The script will:
1. Read the first URL from `url.csv`
2. Fetch the article content
3. Extract the title, content, and metadata
4. Save the results to both JSON and text files

### Output Files

The script generates two output files:
- `extracted_[domain]_[timestamp].json`: Contains the full extraction result in JSON format
- `extracted_[domain]_[timestamp].txt`: Contains the article content in plain text format

### Limitations

- Limited to basic HTML parsing (no JavaScript rendering)
- May not work well with all news sites
- No robots.txt compliance
- No rate limiting

For more complex scraping needs, use the full Newspaper4k implementation. 