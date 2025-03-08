# System Design Documentation

This document contains system architecture documentation, diagrams, and performance considerations for NewsCrawler.

## Architecture Overview

NewsCrawler is designed as a modular, containerized application with several key components:

1. **URL Manager**: Manages the queue of URLs to be crawled, tracks crawl history, and prioritizes crawling tasks.
2. **Multi-Strategy Crawler**: Implements different scraping strategies using various libraries (Newspaper4k, feedparser, BeautifulSoup4, Puppeteer) based on website characteristics.
3. **Content Processor**: Cleans and processes extracted content, extracts metadata, and prepares it for storage.
4. **Database**: PostgreSQL database for storing articles, metadata, and crawl history.
5. **API Layer**: FastAPI-based RESTful API for data retrieval and system management.
6. **Scheduler**: Manages periodic crawling tasks and handles retry logic.

The system follows a microservices architecture pattern, with each component running in its own Docker container, communicating through well-defined APIs. This approach allows for independent scaling of components based on load and resource requirements.

## Architecture Diagrams

### Component Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   URL Manager   │────▶│ Multi-Strategy  │────▶│    Content      │
│                 │     │    Crawler      │     │   Processor     │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
        ▲                                                 │
        │                                                 ▼
┌───────┴───────┐                               ┌─────────────────┐
│               │                               │                 │
│   Scheduler   │                               │    Database     │
│               │                               │                 │
└───────────────┘                               └────────┬────────┘
        ▲                                                │
        │                                                ▼
┌───────┴───────┐                               ┌─────────────────┐
│               │                               │                 │
│  API Layer    │◀──────────────────────────────│  LangChain      │
│               │                               │  Integration    │
└───────────────┘                               └─────────────────┘
```

### Database Schema ER Diagram

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│     sources     │       │     articles    │       │     metadata    │
├─────────────────┤       ├─────────────────┤       ├─────────────────┤
│ id              │       │ id              │       │ id              │
│ name            │       │ source_id       │◀──────│ article_id      │
│ url             │       │ url             │       │ key             │
│ feed_url        │       │ title           │       │ value           │
│ scraper_type    │       │ content         │       │ type            │
│ last_crawled    │       │ html_content    │       └─────────────────┘
│ crawl_frequency │       │ published_date  │
│ is_active       │       │ created_at      │
└──────┬──────────┘       │ updated_at      │
       │                  └──────┬──────────┘
       │                         │
       └─────────────────────────┘

┌─────────────────┐       ┌─────────────────┐
│  crawl_history  │       │     images      │
├─────────────────┤       ├─────────────────┤
│ id              │       │ id              │
│ source_id       │       │ article_id      │
│ url             │       │ url             │
│ status          │       │ local_path      │
│ error_message   │       │ caption         │
│ crawled_at      │       │ width           │
└─────────────────┘       │ height          │
                          └─────────────────┘
```

### Sequence Diagram for Article Crawling

```
┌─────────┐  ┌────────────┐  ┌────────┐  ┌─────────────┐  ┌────────┐  ┌──────────┐
│Scheduler│  │URL Manager │  │Crawler │  │Content      │  │Database│  │API Layer │
└────┬────┘  └─────┬──────┘  └───┬────┘  │Processor    │  └───┬────┘  └────┬─────┘
     │            │              │       └──────┬──────┘      │            │
     │ Trigger Crawl             │              │             │            │
     │───────────>│              │              │             │            │
     │            │ Get Next URLs│              │             │            │
     │            │<─────────────│              │             │            │
     │            │              │              │             │            │
     │            │ Return URLs  │              │             │            │
     │            │──────────────>              │             │            │
     │            │              │              │             │            │
     │            │              │ Fetch Content│             │            │
     │            │              │─────────────>│             │            │
     │            │              │              │             │            │
     │            │              │              │ Store Article            │
     │            │              │              │────────────>│            │
     │            │              │              │             │            │
     │            │ Update Crawl │              │             │            │
     │            │ History      │              │             │            │
     │            │<─────────────│              │             │            │
     │            │              │              │             │ Notify New │
     │            │              │              │             │ Content    │
     │            │              │              │             │───────────>│
     │            │              │              │             │            │
┌────┴────┐  ┌─────┴──────┐  ┌───┴────┐  ┌──────┴──────┐  ┌───┴────┐  ┌────┴─────┐
│Scheduler│  │URL Manager │  │Crawler │  │Content      │  │Database│  │API Layer │
└─────────┘  └────────────┘  └────────┘  │Processor    │  └────────┘  └──────────┘
                                         └─────────────┘
```

## Data Flow

1. **URL Acquisition**:
   - URLs are loaded from url.csv and news-sources.md
   - Each URL is categorized and assigned a scraper type
   - URLs are stored in the database with metadata

2. **Crawling Process**:
   - The scheduler triggers crawling tasks based on configured frequencies
   - The URL manager selects URLs for crawling based on priority and last crawl time
   - The multi-strategy crawler selects the appropriate scraping method for each URL
   - Content is extracted using the selected library (Newspaper4k, feedparser, BeautifulSoup4, Puppeteer)
   - Crawl history is updated with status and timestamp

3. **Content Processing**:
   - Raw content is cleaned and normalized
   - Metadata (publication date, author, title) is extracted
   - Images are downloaded and processed
   - Content is categorized and tagged

4. **Storage**:
   - Processed articles are stored in the PostgreSQL database
   - Metadata is stored in a structured format
   - Images are stored with references to their articles

5. **Retrieval**:
   - The API layer provides endpoints for querying articles
   - LangChain integration enables RAG capabilities
   - Search functionality allows for content discovery

## Performance Considerations

### Complexity Analysis

- **URL Queue Management**: O(log n) for priority-based selection using a heap data structure
- **Content Extraction**: O(n) where n is the size of the HTML content
- **Database Operations**: 
  - Article insertion: O(1)
  - Article retrieval by ID: O(1)
  - Article search by content: O(log n) with proper indexing

### Bottleneck Identification

1. **Network I/O**: Web scraping is inherently I/O bound
   - Solution: Asynchronous crawling with configurable concurrency
   
2. **JavaScript-heavy Sites**: Rendering JS-heavy sites with Puppeteer is resource-intensive
   - Solution: Dedicated container with resource limits for Puppeteer operations
   
3. **Database Write Operations**: High volume of article insertions can strain the database
   - Solution: Batch insertions and proper indexing

### Optimization Strategies

1. **Crawling Optimization**:
   - Implement conditional GET requests with ETag and Last-Modified headers
   - Use HTTP/2 where supported for parallel requests
   - Implement connection pooling for HTTP requests

2. **Content Processing Optimization**:
   - Use multiprocessing for CPU-bound tasks
   - Implement caching for frequently accessed data
   - Use streaming processing for large articles

3. **Database Optimization**:
   - Implement proper indexing for common query patterns
   - Use connection pooling for database connections
   - Implement query optimization and monitoring

### Scalability Considerations

1. **Horizontal Scaling**:
   - Each component can be scaled independently based on load
   - Stateless design allows for easy replication

2. **Vertical Scaling**:
   - Resource allocation can be adjusted per container
   - CPU and memory limits can be configured based on workload

3. **Load Distribution**:
   - Implement work queue pattern for distributing crawling tasks
   - Use load balancing for API requests

## Technology Stack

- **Programming Languages**:
  - Python for core crawling and processing logic
  - JavaScript/Node.js for Puppeteer integration

- **Web Scraping**:
  - Newspaper4k for article extraction
  - feedparser for RSS feed processing
  - BeautifulSoup4 for custom HTML parsing
  - Puppeteer for JavaScript-heavy sites

- **Backend**:
  - FastAPI for RESTful API
  - PostgreSQL for data storage
  - LangChain for RAG integration

- **Containerization**:
  - Docker for containerization
  - Docker Compose for orchestration

## Integration Points

1. **External News Sources**:
   - HTTP/HTTPS connections to target websites
   - RSS feed consumption
   - Respect for robots.txt and rate limits

2. **Database**:
   - PostgreSQL connection for data storage and retrieval
   - Connection pooling for efficient resource usage

3. **LangChain Integration**:
   - Document loading from PostgreSQL
   - Vector embeddings for similarity search
   - Integration with LLM for RAG capabilities

4. **API Consumers**:
   - RESTful endpoints for data retrieval
   - Authentication and authorization for secure access
   - Rate limiting for fair usage

## Deployment Architecture

The system will be deployed using Docker Compose with the following containers:

1. **url-manager**: Manages URL queue and crawl history
2. **newspaper-crawler**: Handles Newspaper4k-based crawling
3. **feed-crawler**: Handles feedparser-based crawling
4. **custom-crawler**: Handles BeautifulSoup4-based crawling
5. **puppeteer-crawler**: Handles Puppeteer-based crawling
6. **content-processor**: Processes and cleans extracted content
7. **postgres**: PostgreSQL database
8. **api**: FastAPI application
9. **scheduler**: Manages crawling schedules

Each container will have appropriate resource limits and health checks. The system can be deployed on a single server for development or distributed across multiple servers for production. 