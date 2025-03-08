# NewsCrawler Implementation Plan - Updated

## Overview
This plan outlines the essential steps to complete the NewsCrawler project with core functionality. The plan has been updated to reflect the current state of the project and identify remaining tasks.

## Current Status

The project has made significant progress with the following components already implemented:

- Project structure and environment setup
- Core scraping functionality with multiple strategies
- Database models and session management
- FastAPI application with routers for articles, sources, search, and LLM
- Vector embedding generation and storage
- RAG (Retrieval-Augmented Generation) implementation with Groq API
- Command-line interface for testing
- Docker configuration for containerization

## Phase 1: Database Setup and Migrations (In Progress)

### 1.1 Database Configuration
- [x] Define database models for articles, sources, and logs
- [x] Set up SQLAlchemy ORM models
- [ ] Complete database migrations with Alembic
- [ ] Create initial migration script
- [ ] Test database schema with sample data

### 1.2 Database Optimization
- [ ] Implement efficient indexing for common queries
- [ ] Configure pgvector extension for vector search
- [ ] Set up connection pooling for improved performance
- [ ] Implement database sharding strategy (if needed for scale)

## Phase 2: API Enhancement and Testing (In Progress)

### 2.1 API Endpoint Completion
- [x] Implement article retrieval endpoints
- [x] Create source management endpoints
- [x] Implement search functionality (text and semantic)
- [x] Add LLM integration endpoints
- [ ] Implement comprehensive error handling
- [ ] Add pagination and filtering to all list endpoints
- [ ] Implement rate limiting for API endpoints

### 2.2 API Testing
- [ ] Create unit tests for API endpoints
- [ ] Implement integration tests for API workflows
- [ ] Test API performance under load
- [ ] Document API endpoints with examples

## Phase 3: Vector Search and RAG Optimization (In Progress)

### 3.1 Vector Search Enhancement
- [x] Implement basic vector similarity search
- [x] Add vector indexing for performance
- [ ] Implement hybrid search combining vector and keyword search
- [ ] Optimize vector search performance
- [ ] Add caching for frequent queries

### 3.2 RAG System Enhancement
- [x] Implement basic RAG functionality
- [x] Integrate with Groq API
- [ ] Implement more sophisticated context assembly
- [ ] Add support for different embedding models
- [ ] Implement response caching for similar queries
- [ ] Add evaluation metrics for RAG quality

## Phase 4: Crawler Enhancement and Scheduling (Planned)

### 4.1 Crawler Optimization
- [x] Implement multi-strategy scraping
- [ ] Add support for JavaScript-heavy sites
- [ ] Implement robots.txt compliance
- [ ] Add rate limiting and politeness protocols
- [ ] Implement proxy rotation for improved throughput
- [ ] Add support for authentication for protected content

### 4.2 Scheduling and Monitoring
- [ ] Implement scheduled crawling of sources
- [ ] Add crawl job queue with priority
- [ ] Implement crawl status monitoring
- [ ] Add failure recovery mechanisms
- [ ] Create dashboard for crawl statistics

## Phase 5: Monitoring, Logging, and Documentation (Planned)

### 5.1 Monitoring and Logging
- [ ] Implement comprehensive logging
- [ ] Add metrics collection for performance monitoring
- [ ] Create health check endpoints
- [ ] Implement alerting for critical issues
- [ ] Set up log aggregation

### 5.2 Documentation
- [x] Update README.md with usage instructions
- [ ] Create API documentation with examples
- [ ] Document system architecture
- [ ] Create deployment guides
- [ ] Add code documentation and comments

## Implementation Details

### Project Structure
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

### Core Dependencies
```
# Web Scraping
newspaper4k==0.9.3.1
beautifulsoup4==4.13.3
feedparser==6.0.11
requests==2.32.3

# Database
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.5
pgvector>=0.2.0
alembic>=1.10.0

# API
fastapi>=0.95.0
uvicorn>=0.17.6
pydantic>=2.0.0

# Vector Processing
sentence-transformers>=2.2.2
langchain>=0.0.267
langchain-community>=0.0.1

# LLM Integration
groq>=0.18.0
langchain-openai>=0.0.1
langchain-huggingface>=0.0.1

# Utilities
python-dotenv>=1.0.0
tenacity>=8.2.0
```

### Database Schema

#### Articles Table
```python
class Article(Base):
    """Article model."""
    __tablename__ = "articles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url = Column(Text, unique=True, nullable=False)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    published_date = Column(DateTime, nullable=True)
    author = Column(Text, nullable=True)
    source_domain = Column(Text, nullable=False)
    category = Column(Text, nullable=True)
    keywords = Column(ARRAY(Text), nullable=True)
    vector_embedding = Column(Vector(1536), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

#### Sources Table
```python
class Source(Base):
    """Source model."""
    __tablename__ = "sources"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    domain = Column(Text, unique=True, nullable=False)
    name = Column(Text, nullable=False)
    base_url = Column(Text, nullable=False)
    scraper_type = Column(Text, nullable=False)
    active = Column(Boolean, default=True)
    last_crawled = Column(DateTime, nullable=True)
    crawl_frequency = Column(Integer, default=24)  # in hours
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

#### Crawl Logs Table
```python
class CrawlLog(Base):
    """Crawl log model."""
    __tablename__ = "crawl_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(UUID(as_uuid=True), ForeignKey("sources.id"))
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    articles_found = Column(Integer, default=0)
    articles_added = Column(Integer, default=0)
    articles_updated = Column(Integer, default=0)
    status = Column(Text, default="in_progress")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
```

### RAG Implementation

The RAG system has been implemented with the following components:

```python
class RAGSystem:
    """
    Retrieval-Augmented Generation system for enhancing LLM responses with news articles.
    """
    
    def __init__(self, db_session: Optional[Session] = None):
        """Initialize the RAG system."""
        self.db = db_session or SessionLocal()
        self.groq_client = get_groq_client()
        self.logger = logging.getLogger(__name__)
    
    def retrieve_relevant_articles(
        self, 
        query: str, 
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Retrieve articles relevant to the query using vector similarity."""
        # Implementation details...
    
    def generate_context(self, articles: List[Dict[str, Any]]) -> str:
        """Generate context from retrieved articles."""
        # Implementation details...
    
    def answer_question(
        self, 
        query: str,
        model: str = "llama3-70b-8192",
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Answer a question using RAG."""
        # Implementation details...
```

### CLI Implementation

A command-line interface has been implemented for testing:

```python
def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="NewsCrawler CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Crawl command
    crawl_parser = subparsers.add_parser("crawl", help="Crawl a URL")
    crawl_parser.add_argument("url", help="URL to crawl")
    crawl_parser.add_argument("--save", action="store_true", help="Save the article to the database")
    crawl_parser.add_argument("--embed", action="store_true", help="Generate embedding for the article")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question using RAG")
    ask_parser.add_argument("question", help="Question to ask")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process articles")
    process_parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
```

## Next Steps

The following tasks should be prioritized to complete the implementation:

1. **Complete Database Migrations**:
   - Finish setting up Alembic
   - Create initial migration script
   - Test database schema with sample data

2. **Enhance API Endpoints**:
   - Implement comprehensive error handling
   - Add pagination and filtering to all list endpoints
   - Create unit and integration tests

3. **Optimize Vector Search and RAG**:
   - Implement hybrid search
   - Add caching for frequent queries
   - Enhance context assembly for better RAG responses

4. **Implement Crawler Scheduling**:
   - Add scheduled crawling of sources
   - Implement crawl job queue
   - Add monitoring and failure recovery

5. **Add Monitoring and Documentation**:
   - Implement comprehensive logging
   - Add metrics collection
   - Complete API documentation
   - Create deployment guides

## Getting Started Guide

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/newscrawler.git
   cd newscrawler
   ```

2. **Set up the virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file with the following variables:
   ```
   DATABASE_URL=postgresql://newscrawler:newscrawler_password@localhost:5432/newscrawler
   GROQ_API_KEY=your_groq_api_key
   ```

4. **Start the database**
   ```bash
   docker-compose up -d postgres
   ```

5. **Run database migrations**
   ```bash
   alembic upgrade head
   ```

6. **Start the API server**
   ```bash
   uvicorn src.api.main:app --reload
   ```

7. **Test the CLI**
   ```bash
   # Crawl a URL
   python -m src.cli crawl https://example.com/article --save --embed
   
   # Ask a question
   python -m src.cli ask "What is the latest news about climate change?"
   
   # Process articles
   python -m src.cli process --batch-size 10
   ```

This implementation plan provides a roadmap for completing the NewsCrawler project, building on the significant progress already made and focusing on the remaining tasks to deliver a fully functional system.
