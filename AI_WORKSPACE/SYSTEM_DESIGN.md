# System Design Document

## System Overview

NewsCrawler is a comprehensive system designed to collect, process, store, and retrieve news articles from various sources. The system employs web scraping techniques to gather articles, processes them using natural language processing, stores them in a structured database, and provides a RESTful API for accessing the data. Additionally, the system includes a retrieval-augmented generation (RAG) component that enhances large language model capabilities with the collected news data.

## Architecture

The system follows a microservices architecture, with each component responsible for a specific function. The components communicate through well-defined APIs and message queues.

### High-Level Components

1. **Web Scraping Service**
   - Responsible for collecting news articles from various sources
   - Implements multiple scraping strategies based on source characteristics
   - Handles rate limiting, error recovery, and content extraction

2. **Processing Service**
   - Cleans and normalizes article content
   - Extracts metadata (publication date, author, etc.)
   - Generates embeddings for semantic search
   - Creates article summaries

3. **Storage Service**
   - Manages the PostgreSQL database
   - Handles data persistence and retrieval
   - Implements efficient indexing for various query types

4. **API Service**
   - Provides RESTful endpoints for accessing the system
   - Implements authentication and authorization
   - Handles request validation and response formatting

5. **RAG Service**
   - Implements retrieval-augmented generation capabilities
   - Integrates with large language models
   - Provides context-aware responses based on news data

### Component Interactions

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Web Scraping   │────▶│   Processing    │────▶│    Storage      │
│    Service      │     │    Service      │     │    Service      │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  RAG Service    │◀────│   API Service   │◀────│    Storage      │
│                 │     │                 │     │    Service      │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Vector Search Architecture

Based on our research findings, the vector search component of the system is designed to provide efficient semantic search capabilities for news articles. This architecture leverages the power of vector embeddings to find semantically similar content.

### Embedding Generation Pipeline

1. **Text Preprocessing**
   - Clean and normalize article text
   - Remove HTML tags, special characters, and irrelevant content
   - Standardize formatting and encoding

2. **Chunking Strategy**
   - Implement RecursiveCharacterTextSplitter with 500 character chunks and 100 character overlap
   - Preserve article metadata with each chunk
   - Maintain references to the original article

3. **Embedding Model**
   - Use BAAI/bge-small-en-v1.5 (384 dimensions) as the primary embedding model
   - Implement batch processing for efficient embedding generation
   - Cache embeddings to avoid redundant computation

4. **Vector Storage**
   - Store embeddings in PostgreSQL using the pgvector extension
   - Create HNSW indexes for efficient similarity search
   - Implement periodic index maintenance for optimal performance

### Search Implementation

1. **Query Processing**
   - Convert search queries to vector embeddings using the same model
   - Apply the same preprocessing steps used for article content
   - Support both natural language queries and keyword-based searches

2. **Hybrid Search**
   - Perform vector similarity search using cosine similarity
   - Combine with keyword-based search for improved precision
   - Implement weighted scoring to balance semantic and lexical relevance

3. **Result Ranking**
   - Score results based on a combination of similarity metrics
   - Apply recency and relevance boosting
   - Filter results based on user preferences and query context

4. **Performance Optimization**
   - Implement query caching for common searches
   - Use approximate nearest neighbor search for large-scale retrieval
   - Balance precision and recall based on use case requirements

### Vector Search Flow Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  User Query     │────▶│  Query Vector   │────▶│  Vector Search  │
│                 │     │   Embedding     │     │                 │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Result         │◀────│  Result Ranking │◀────│  Keyword Search │
│  Presentation   │     │  & Filtering    │     │  (Optional)     │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Detailed Component Design

### Web Scraping Service

#### Components

1. **Crawler Manager**
   - Schedules and coordinates crawling jobs
   - Maintains a list of news sources and their crawling configurations
   - Monitors crawling performance and adjusts parameters accordingly

2. **Source-Specific Crawlers**
   - Implements custom crawling logic for different types of news sources
   - Handles site-specific challenges (JavaScript rendering, authentication, etc.)
   - Extracts article URLs from various page types (homepages, section pages, etc.)

3. **Article Extractor**
   - Extracts article content, title, author, publication date, etc.
   - Uses Newspaper4k for general extraction
   - Falls back to custom extraction logic for problematic sites

4. **Rate Limiter**
   - Enforces polite crawling behavior
   - Implements adaptive rate limiting based on server responses
   - Respects robots.txt and site-specific crawling policies

5. **Error Handler**
   - Manages retries for failed requests
   - Logs and reports crawling errors
   - Implements circuit breaker pattern for problematic sources

#### Data Flow

1. Crawler Manager selects sources due for crawling
2. Source-Specific Crawlers extract article URLs
3. Article Extractor processes each URL to extract content
4. Rate Limiter controls request frequency
5. Error Handler manages failures and retries
6. Extracted articles are sent to the Processing Service

### Processing Service

#### Components

1. **Text Cleaner**
   - Removes HTML tags, advertisements, and irrelevant content
   - Normalizes text formatting and encoding
   - Corrects common extraction errors

2. **Metadata Extractor**
   - Extracts and validates publication date, author, category, etc.
   - Infers missing metadata when possible
   - Standardizes metadata formats

3. **Text Splitter**
   - Implements RecursiveCharacterTextSplitter with 500 character chunks and 100 character overlap
   - Preserves metadata with each chunk
   - Maintains references to the original article

4. **Embedding Generator**
   - Generates vector embeddings using BAAI/bge-small-en-v1.5
   - Processes chunks in batches for efficiency
   - Validates embedding quality

5. **Summarizer**
   - Generates concise summaries of articles
   - Uses extractive or abstractive summarization techniques
   - Adapts summary length based on article characteristics

#### Data Flow

1. Text Cleaner processes raw article content
2. Metadata Extractor identifies and validates article metadata
3. Text Splitter breaks articles into chunks for embedding
4. Embedding Generator creates vector representations
5. Summarizer generates article summaries
6. Processed articles are sent to the Storage Service

### Storage Service

#### Components

1. **Database Manager**
   - Handles database connections and transactions
   - Implements connection pooling and query optimization
   - Manages database migrations and schema updates

2. **Article Repository**
   - Provides CRUD operations for articles
   - Implements efficient querying patterns
   - Handles data validation and integrity checks

3. **Vector Store**
   - Manages vector embeddings in PostgreSQL with pgvector
   - Creates and maintains HNSW indexes for efficient similarity search
   - Implements vector search operations

4. **Cache Manager**
   - Implements caching for frequently accessed data
   - Manages cache invalidation and updates
   - Optimizes cache usage based on access patterns

5. **Backup Manager**
   - Schedules and manages database backups
   - Implements point-in-time recovery capabilities
   - Monitors backup integrity and performance

#### Data Flow

1. Database Manager establishes and manages connections
2. Article Repository stores processed articles and metadata
3. Vector Store indexes and manages article embeddings
4. Cache Manager optimizes data access
5. Backup Manager ensures data durability

### API Service

#### Components

1. **Request Handler**
   - Processes incoming HTTP requests
   - Validates request parameters and authentication
   - Routes requests to appropriate controllers

2. **Controllers**
   - Implement business logic for different API endpoints
   - Coordinate with other services to fulfill requests
   - Handle error conditions and edge cases

3. **Response Formatter**
   - Formats API responses according to defined standards
   - Implements pagination, filtering, and sorting
   - Optimizes response size and structure

4. **Authentication Manager**
   - Validates user credentials and tokens
   - Manages user sessions and permissions
   - Implements security best practices

5. **Rate Limiter**
   - Enforces API usage limits
   - Prevents abuse and ensures fair resource allocation
   - Implements graduated rate limiting based on user tiers

#### Endpoints

1. **Article Endpoints**
   - `GET /articles`: List articles with filtering and pagination
   - `GET /articles/{id}`: Get a specific article by ID
   - `POST /articles`: Create a new article (admin only)
   - `PUT /articles/{id}`: Update an existing article (admin only)
   - `DELETE /articles/{id}`: Delete an article (admin only)

2. **Search Endpoints**
   - `GET /search`: Search articles by keyword or query
   - `GET /search/semantic`: Semantic search using vector embeddings
   - `GET /search/hybrid`: Combined keyword and semantic search

3. **Source Endpoints**
   - `GET /sources`: List news sources
   - `GET /sources/{id}`: Get a specific source by ID
   - `POST /sources`: Add a new source (admin only)
   - `PUT /sources/{id}`: Update an existing source (admin only)
   - `DELETE /sources/{id}`: Delete a source (admin only)

4. **Crawl Endpoints**
   - `POST /crawl`: Trigger a crawling job (admin only)
   - `GET /crawl/status`: Check crawling job status
   - `GET /crawl/logs`: Get crawling logs

5. **User Endpoints**
   - `POST /users/register`: Register a new user
   - `POST /users/login`: User login
   - `GET /users/profile`: Get user profile
   - `PUT /users/profile`: Update user profile

### RAG Service

#### Components

1. **Query Processor**
   - Analyzes and processes user queries
   - Extracts key entities and intents
   - Formulates retrieval strategy

2. **Retriever**
   - Implements vector-based retrieval using LangChain
   - Combines keyword and semantic search for improved results
   - Filters and ranks retrieved documents

3. **Context Builder**
   - Constructs context from retrieved documents
   - Formats context for LLM consumption
   - Optimizes context length and relevance

4. **LLM Interface**
   - Connects to GroqAI for primary LLM capabilities
   - Implements local LLM fallback for reliability
   - Manages prompt engineering and response generation

5. **Response Generator**
   - Processes LLM output
   - Formats responses for end users
   - Adds citations and references to source articles

#### Data Flow

1. Query Processor analyzes the user query
2. Retriever finds relevant articles and chunks
3. Context Builder constructs context from retrieved content
4. LLM Interface generates a response using the context
5. Response Generator formats and returns the final response

## Database Schema

### Articles Table

| Column           | Type          | Description                                |
|------------------|--------------|--------------------------------------------|
| id               | UUID          | Primary key                                |
| url              | TEXT          | Article URL (unique)                       |
| title            | TEXT          | Article title                              |
| content          | TEXT          | Full article text                          |
| summary          | TEXT          | Auto-generated summary                     |
| published_date   | TIMESTAMP     | Publication date                           |
| author           | TEXT          | Author name(s)                             |
| source_domain    | TEXT          | Domain of the source                       |
| category         | TEXT          | Article category/section                   |
| keywords         | TEXT[]        | Array of keywords                          |
| created_at       | TIMESTAMP     | Record creation timestamp                  |
| updated_at       | TIMESTAMP     | Record update timestamp                    |
| vector_embedding | VECTOR(384)   | Article vector embedding                   |

### Article_Chunks Table

| Column           | Type          | Description                                |
|------------------|--------------|--------------------------------------------|
| id               | UUID          | Primary key                                |
| article_id       | UUID          | Foreign key to Articles table              |
| chunk_index      | INTEGER       | Position of the chunk in the article       |
| content          | TEXT          | Chunk text content                         |
| vector_embedding | VECTOR(384)   | Chunk vector embedding                     |
| created_at       | TIMESTAMP     | Record creation timestamp                  |

### Sources Table

| Column           | Type          | Description                                |
|------------------|--------------|--------------------------------------------|
| id               | UUID          | Primary key                                |
| domain           | TEXT          | Domain name (unique)                       |
| name             | TEXT          | Source name                                |
| base_url         | TEXT          | Base URL                                   |
| scraper_type     | TEXT          | Type of scraper to use                     |
| active           | BOOLEAN       | Whether the source is active               |
| last_crawled     | TIMESTAMP     | Timestamp of last crawl                    |
| crawl_frequency  | INTEGER       | How often to crawl (in hours)              |

### Crawl_Logs Table

| Column           | Type          | Description                                |
|------------------|--------------|--------------------------------------------|
| id               | UUID          | Primary key                                |
| source_id        | UUID          | Foreign key to Sources table               |
| start_time       | TIMESTAMP     | Start timestamp                            |
| end_time         | TIMESTAMP     | End timestamp                              |
| articles_found   | INTEGER       | Number of articles found                   |
| articles_added   | INTEGER       | Number of new articles added               |
| articles_updated | INTEGER       | Number of articles updated                 |
| status           | TEXT          | Success/failure status                     |
| error_message    | TEXT          | Error message if failed                    |

### Users Table

| Column           | Type          | Description                                |
|------------------|--------------|--------------------------------------------|
| id               | UUID          | Primary key                                |
| username         | TEXT          | Username (unique)                          |
| email            | TEXT          | Email address (unique)                     |
| password_hash    | TEXT          | Hashed password                            |
| role             | TEXT          | User role (user, admin)                    |
| created_at       | TIMESTAMP     | Record creation timestamp                  |
| updated_at       | TIMESTAMP     | Record update timestamp                    |

## Deployment Architecture

The system is deployed using Docker containers orchestrated with Docker Compose. Each component runs in its own container, with appropriate networking and volume configurations.

### Container Configuration

1. **Database Container**
   - PostgreSQL with pgvector extension
   - Persistent volume for data storage
   - Optimized configuration for vector operations

2. **API Container**
   - FastAPI application
   - Gunicorn/Uvicorn for production serving
   - Connection to database container

3. **Crawler Container**
   - Scheduled crawling jobs
   - Connection to database container
   - Volume for log storage

4. **Processing Container**
   - Text processing and embedding generation
   - Connection to database container
   - GPU support (optional)

5. **RAG Container**
   - LangChain and LLM integration
   - Connection to database container
   - Cache for frequent queries

### Networking

- Internal network for container communication
- Exposed ports for API access
- Secure communication between containers

### Scaling Strategy

- Horizontal scaling for API and Processing containers
- Vertical scaling for Database container
- Load balancing for API requests
- Distributed crawling for improved performance

## Security Considerations

1. **Authentication and Authorization**
   - JWT-based authentication
   - Role-based access control
   - Secure password storage

2. **Data Protection**
   - Encrypted database connections
   - Secure API endpoints (HTTPS)
   - Input validation and sanitization

3. **Rate Limiting and Abuse Prevention**
   - API rate limiting
   - Crawler rate limiting
   - IP-based blocking for abuse

4. **Monitoring and Logging**
   - Comprehensive logging for all components
   - Real-time monitoring and alerting
   - Regular security audits

## Performance Considerations

1. **Database Optimization**
   - Efficient indexing for common queries
   - HNSW indexes for vector search
   - Query optimization and caching

2. **API Performance**
   - Response caching
   - Pagination and result limiting
   - Asynchronous processing for long-running operations

3. **Crawler Efficiency**
   - Parallel crawling with controlled concurrency
   - Incremental crawling to minimize resource usage
   - Adaptive rate limiting based on server responses

4. **Vector Search Optimization**
   - Batch processing for embedding generation
   - Approximate nearest neighbor search for large datasets
   - Caching for frequent queries

## Monitoring and Maintenance

1. **Health Checks**
   - Regular health checks for all components
   - Automated recovery procedures
   - Alerting for critical issues

2. **Performance Monitoring**
   - Resource usage tracking
   - Query performance analysis
   - Bottleneck identification

3. **Database Maintenance**
   - Regular backups
   - Index maintenance
   - Data archiving for older articles

4. **System Updates**
   - Scheduled maintenance windows
   - Rolling updates to minimize downtime
   - Version control for all components

## Future Enhancements

1. **Advanced NLP Features**
   - Named entity recognition
   - Sentiment analysis
   - Topic modeling

2. **Improved Search Capabilities**
   - Multi-language support
   - Image search
   - Temporal search

3. **User Personalization**
   - User preferences and interests
   - Personalized recommendations
   - Reading history and bookmarks

4. **Integration Capabilities**
   - API integrations with other systems
   - Webhook support
   - Export capabilities for various formats 