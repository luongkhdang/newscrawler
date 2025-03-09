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
- Topic classification and entity extraction modules

## Phase 1: Database Setup and Migrations (In Progress)

### 1.1 Database Configuration
- [x] Define database models for articles, sources, and logs
- [x] Set up SQLAlchemy ORM models
- [x] Complete database migrations with Alembic
- [x] Create initial migration script
- [x] Test database schema with sample data

### 1.2 Database Optimization
- [ ] Implement efficient indexing for common queries
  - [x] Create indexes for frequently queried fields (published_date, source_domain, category)
  - [ ] Implement partial indexes for filtered queries
  - [ ] Add composite indexes for multi-field queries
- [x] Configure pgvector extension for vector search
- [ ] Set up connection pooling for improved performance
  - [ ] Implement PgBouncer for connection pooling
  - [ ] Configure optimal pool sizes based on workload
  - [ ] Add connection timeout and retry logic
- [ ] Implement database sharding strategy (if needed for scale)
  - [ ] Evaluate sharding requirements based on data volume projections
  - [ ] Design time-based partitioning for historical data

## Phase 2: API Enhancement and Testing (In Progress)

### 2.1 API Endpoint Completion
- [x] Implement article retrieval endpoints
- [x] Create source management endpoints
- [x] Implement search functionality (text and semantic)
- [x] Add LLM integration endpoints
- [x] Implement comprehensive error handling
- [x] Add pagination and filtering to all list endpoints
- [x] Implement rate limiting for API endpoints

### 2.2 API Testing
- [x] Create unit tests for API endpoints
- [x] Implement integration tests for API workflows
- [ ] Test API performance under load
  - [ ] Set up load testing environment with Locust or JMeter
  - [ ] Create test scenarios for common API usage patterns
  - [ ] Establish performance baselines and thresholds
  - [ ] Identify and address performance bottlenecks
- [ ] Document API endpoints with examples
  - [ ] Create OpenAPI documentation with detailed examples
  - [ ] Add Postman collection for API testing
  - [ ] Create usage tutorials for common workflows

## Phase 3: Vector Search and RAG Optimization (In Progress)

### 3.1 Vector Search Enhancement
- [x] Implement basic vector similarity search
- [x] Add vector indexing for performance
- [x] Implement hybrid search combining vector and keyword search
- [x] Optimize vector search performance
- [x] Add caching for frequent queries

### 3.2 RAG System Enhancement
- [x] Implement basic RAG functionality
- [x] Integrate with Groq API
- [x] Implement more sophisticated context assembly
- [x] Add support for different embedding models
- [x] Implement response caching for similar queries
- [x] Add evaluation metrics for RAG quality

## Phase 4: Crawler Enhancement and Scheduling (Planned)

### 4.1 Crawler Optimization
- [x] Implement multi-strategy scraping
- [x] Add support for JavaScript-heavy sites with Puppeteer
- [x] Implement robots.txt compliance
- [x] Add rate limiting and politeness protocols
- [ ] Implement proxy rotation for improved throughput
  - [ ] Set up proxy management system
  - [ ] Implement proxy health checking
  - [ ] Add automatic proxy rotation based on response codes
  - [ ] Configure proxy selection based on target domain
- [ ] Add support for authentication for protected content
  - [x] Implement credential storage with encryption
  - [ ] Create authentication strategies for different site types
  - [ ] Add session management for authenticated crawling

### 4.2 Scheduling and Monitoring
- [x] Implement scheduled crawling of sources
- [x] Add crawl job queue with priority
- [x] Implement crawl status monitoring
- [ ] Add failure recovery mechanisms
  - [ ] Implement automatic retry with exponential backoff
  - [ ] Create circuit breaker pattern for failing sources
  - [ ] Add dead letter queue for failed crawl jobs
  - [ ] Implement notification system for persistent failures
- [ ] Create dashboard for crawl statistics
  - [ ] Design and implement dashboard UI
  - [ ] Add real-time crawl status visualization
  - [ ] Create historical performance charts
  - [ ] Implement source health monitoring

## Phase 5: Monitoring, Logging, and Documentation (Planned)

### 5.1 Monitoring and Logging
- [x] Implement comprehensive logging
- [x] Add metrics collection for performance monitoring
- [x] Create health check endpoints
- [ ] Implement alerting for critical issues
  - [ ] Set up alerting thresholds for system metrics
  - [ ] Configure notification channels (email, Slack, etc.)
  - [ ] Implement alert aggregation to prevent alert storms
  - [ ] Create escalation policies for different alert severities
- [ ] Set up log aggregation
  - [ ] Configure centralized logging with ELK stack or similar
  - [ ] Implement structured logging format
  - [ ] Create log retention and rotation policies
  - [ ] Add log search and visualization capabilities

### 5.2 Documentation
- [x] Update README.md with usage instructions
- [ ] Create API documentation with examples
  - [ ] Document all API endpoints with request/response examples
  - [ ] Add authentication and authorization details
  - [ ] Create API usage tutorials for common scenarios
  - [ ] Document error codes and troubleshooting steps
- [ ] Document system architecture
  - [ ] Create architecture diagrams
  - [ ] Document component interactions
  - [ ] Add sequence diagrams for key workflows
  - [ ] Document scaling considerations
- [ ] Create deployment guides
  - [ ] Document Docker deployment process
  - [ ] Add Kubernetes deployment configuration
  - [ ] Create cloud deployment guides (AWS, GCP, Azure)
  - [ ] Document backup and recovery procedures
- [ ] Add code documentation and comments
  - [ ] Ensure all modules have docstrings
  - [ ] Add type hints throughout the codebase
  - [ ] Document complex algorithms and business logic
  - [ ] Create developer onboarding guide

## Phase 6: Content Filtering and Topic Focus (In Progress)

### 6.1 Topic Classification System
- [x] Implement NLP-based topic classification
  - [x] Create model-based classification with fallback to keyword-based approach
  - [x] Implement topic scoring and threshold filtering
  - [x] Add multi-label classification for articles spanning multiple topics
  - [ ] Create feedback loop for classification improvement
- [x] Add entity extraction for countries and organizations
  - [x] Implement named entity recognition for key entities
  - [x] Create entity linking to knowledge base
  - [x] Add entity relationship mapping
  - [x] Implement entity-based filtering

### 6.2 Content Relevance Filtering
- [x] Implement relevance scoring system
  - [x] Create scoring algorithm based on topic match, entity presence, and content quality
  - [x] Add configurable thresholds for different topics
  - [x] Implement content quality assessment
  - [ ] Create feedback mechanism for relevance scoring
- [ ] Add storage optimization based on relevance
  - [x] Update database schema with relevance fields
  - [x] Create migration for new fields
  - [ ] Create retention policies tied to relevance scores
  - [ ] Add automatic content pruning for low-relevance articles
  - [ ] Implement content summarization for storage efficiency

## Implementation Timeline

### Week 1-2: Database Optimization and API Testing
- Complete database indexing for common queries
- Set up connection pooling with PgBouncer
- Implement API load testing and address performance issues
- Complete API documentation with examples

### Week 3-4: Crawler Enhancement
- Implement proxy rotation system
- Add authentication support for protected content
- Create failure recovery mechanisms
- Begin dashboard development for crawl statistics

### Week 5-6: Monitoring, Alerting, and Documentation
- Set up alerting for critical issues
- Implement log aggregation with ELK stack
- Create system architecture documentation
- Develop deployment guides for different environments

### Week 7-8: Content Filtering and Topic Focus
- ✅ Implement topic classification system
- ✅ Add entity extraction and relationship mapping
- ✅ Create relevance scoring algorithm
- Implement storage optimization based on relevance

### Week 9-10: Final Integration and Testing
- Complete dashboard for crawl statistics
- Finalize all documentation
- Perform end-to-end testing
- Address any remaining issues

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
│   ├── topics/               # Topic classification components
│   │   ├── classifier.py     # Topic classifier
│   │   ├── entities.py       # Entity extraction
│   │   └── relevance.py      # Relevance scoring
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

# Topic Classification
spacy>=3.5.0
transformers>=4.28.0
scikit-learn>=1.2.2

# Monitoring and Logging
prometheus-client>=0.16.0
python-json-logger>=2.0.7
elasticsearch>=8.8.0

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
    # New fields for topic classification and relevance
    topics = Column(ARRAY(Text), nullable=True)
    entities = Column(JSONB, nullable=True)
    relevance_score = Column(Float, nullable=True)
    is_relevant = Column(Boolean, default=True)
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
    # New fields for authentication and proxy settings
    requires_auth = Column(Boolean, default=False)
    auth_config = Column(JSONB, nullable=True)
    proxy_settings = Column(JSONB, nullable=True)
```

## Implementation Progress Summary

We have made significant progress on the NewsCrawler implementation plan. Here's a summary of what we've accomplished and what's left to do:

### Completed Tasks

1. **Database Setup and Migrations**
   - Defined database models for articles, sources, and logs
   - Set up SQLAlchemy ORM models
   - Created and executed database migrations
   - Configured pgvector extension for vector search
   - Added new fields for topic classification and relevance

2. **API Enhancement**
   - Implemented comprehensive article retrieval endpoints with pagination and filtering
   - Enhanced search functionality with text, semantic, and hybrid search options
   - Added robust error handling throughout the API
   - Implemented rate limiting to prevent abuse
   - Added LLM integration endpoints

3. **Vector Search Optimization**
   - Implemented basic vector similarity search
   - Added vector indexing for performance
   - Created a hybrid search combining vector and keyword search
   - Implemented caching for embeddings and search results
   - Optimized vector search performance

4. **RAG System Enhancement**
   - Implemented sophisticated context assembly strategies
   - Added support for different embedding models
   - Implemented response caching for similar queries
   - Added evaluation metrics for RAG quality

5. **Crawler Scheduling**
   - Implemented scheduled crawling of sources
   - Added crawl job queue with priority
   - Implemented crawl status monitoring
   - Added CLI and API interfaces for crawler management

6. **Crawler Enhancement**
   - Implemented multi-strategy scraping
   - Added support for JavaScript-heavy sites with Puppeteer
   - Implemented robots.txt compliance
   - Added rate limiting and politeness protocols

7. **Monitoring and Logging**
   - Implemented comprehensive logging with different output formats
   - Added metrics collection for performance monitoring
   - Created health check and metrics endpoints
   - Added Prometheus-compatible metrics export

8. **Topic Classification and Content Filtering**
   - Implemented topic classification system with model-based and keyword-based approaches
   - Added entity extraction for countries, organizations, and people
   - Created relevance scoring algorithm based on topic match, entity presence, and content quality
   - Updated database schema with fields for topics, entities, and relevance

### Current Focus Areas

1. **Database Optimization**
   - Implementing efficient indexing for common queries
   - Setting up connection pooling for improved performance

2. **API Testing and Documentation**
   - Testing API performance under load
   - Creating comprehensive API documentation with examples

3. **Crawler Enhancement**
   - Implementing proxy rotation for improved throughput
   - Adding support for authentication for protected content
   - Creating failure recovery mechanisms

4. **Content Filtering Integration**
   - Integrating topic classification and relevance scoring into the article processing pipeline
   - Implementing storage optimization based on relevance scores
   - Creating retention policies for content management

### Next Steps

The immediate next steps are:

1. **Integrate Topic Classification into Processing Pipeline**
   - Update article processing to include topic classification and relevance scoring
   - Implement filtering based on relevance scores
   - Add API endpoints for topic-based search and filtering

2. **Complete Database Optimization**
   - Implement indexes for frequently queried fields
   - Set up PgBouncer for connection pooling

3. **Finalize API Testing**
   - Set up load testing environment
   - Create test scenarios and establish performance baselines
   - Address any performance bottlenecks

4. **Begin Crawler Enhancement**
   - Implement proxy rotation system
   - Add authentication support for protected content

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