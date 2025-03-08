# NewsCrawler

A comprehensive system for collecting, processing, storing, and retrieving news articles from various sources. The system employs web scraping techniques to gather articles, processes them using natural language processing, stores them in a structured database, and provides a RESTful API for accessing the data. Additionally, the system includes a retrieval-augmented generation (RAG) component that enhances large language model capabilities with the collected news data.

## Features

- **Multi-strategy web scraping**: Supports various scraping methods (Newspaper4k, BeautifulSoup4, Feed parsing) to handle different website structures
- **Content processing**: Extracts and cleans article content, metadata, and generates summaries
- **Vector embeddings**: Generates vector embeddings for semantic search capabilities
- **PostgreSQL storage**: Stores articles and vector embeddings in a PostgreSQL database with pgvector extension
- **FastAPI endpoints**: Provides RESTful API for data retrieval and management
- **RAG integration**: Enhances LLM responses with relevant news articles using Groq API
- **Docker support**: Containerized deployment for easy setup and scaling

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL with pgvector extension
- Docker and Docker Compose (optional)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/newscrawler.git
   cd newscrawler
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file with the following variables:
   ```
   DATABASE_URL=postgresql://newscrawler:newscrawler_password@localhost:5432/newscrawler
   GROQ_API_KEY=your_groq_api_key
   ```

5. Start the PostgreSQL database:
   ```bash
   docker-compose up -d postgres
   ```

6. Run database migrations:
   ```bash
   alembic upgrade head
   ```

## Usage

### Command-Line Interface

The NewsCrawler provides a command-line interface for testing functionality:

1. Crawl a URL:
   ```bash
   python -m src.cli crawl https://example.com/article --save --embed
   ```

2. Ask a question using RAG:
   ```bash
   python -m src.cli ask "What is the latest news about climate change?"
   ```

3. Process articles (generate embeddings):
   ```bash
   python -m src.cli process --batch-size 10
   ```

### API Server

Start the API server:

```bash
uvicorn src.api.main:app --reload
```

The API will be available at http://localhost:8000, with documentation at http://localhost:8000/docs.

### Docker Deployment

Deploy the entire system using Docker Compose:

```bash
docker-compose up -d
```

This will start the following services:
- PostgreSQL database
- FastAPI application
- Crawler service
- Vector processing service

## API Endpoints

### Articles

- `GET /articles/`: List articles with pagination and filtering
- `GET /articles/{article_id}`: Get a specific article by ID
- `POST /articles/`: Create a new article
- `PUT /articles/{article_id}`: Update an article
- `DELETE /articles/{article_id}`: Delete an article

### Sources

- `GET /sources/`: List sources with pagination and filtering
- `GET /sources/{source_id}`: Get a specific source by ID
- `POST /sources/`: Create a new source
- `PUT /sources/{source_id}`: Update a source
- `DELETE /sources/{source_id}`: Delete a source

### Crawling

- `POST /crawl/url`: Crawl a specific URL
- `POST /crawl/source/{source_id}`: Crawl all articles from a source

### Search

- `GET /search/text`: Perform text-based search on articles
- `GET /search/semantic`: Perform semantic search using vector embeddings

### LLM

- `GET /llm/models`: Get available LLM models
- `POST /llm/generate`: Generate text using the Groq API
- `POST /llm/summarize`: Summarize text using the Groq API
- `POST /llm/analyze-sentiment`: Analyze sentiment of text
- `POST /llm/extract-entities`: Extract entities from text
- `POST /llm/ask`: Answer questions using RAG

## Project Structure

```
newscrawler/
├── alembic/                  # Database migrations
├── src/                      # Source code
│   ├── api/                  # FastAPI application
│   │   ├── routers/          # API routers
│   │   └── main.py           # Main API application
│   ├── database/             # Database models and utilities
│   │   ├── models.py         # SQLAlchemy models
│   │   └── session.py        # Database session management
│   ├── scrapers/             # Web scraping components
│   │   ├── base_scraper.py   # Base scraper class
│   │   ├── newspaper_scraper/ # Newspaper4k scraper
│   │   ├── bs4_scraper/      # BeautifulSoup4 scraper
│   │   ├── feed_scraper/     # Feed scraper
│   │   └── scraper_factory.py # Scraper factory
│   ├── vector/               # Vector processing components
│   │   └── processor.py      # Vector embedding generation
│   ├── llm/                  # LLM integration components
│   │   └── rag.py            # RAG implementation
│   └── utils/                # Common utilities
│       ├── groq_client.py    # Groq API client
│       └── url_classifier.py # URL classification
├── tests/                    # Test suite
├── docker-compose.yml        # Docker Compose configuration
├── Dockerfile.api            # API Dockerfile
├── Dockerfile.crawler        # Crawler Dockerfile
├── Dockerfile.vector         # Vector processor Dockerfile
└── requirements.txt          # Python dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 