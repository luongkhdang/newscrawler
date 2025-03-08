# Knowledge Base

## Purpose
This document serves as the central reference point for all AI agents working on NewsCrawler. When encountering unfamiliar concepts, theories, vocabulary, or structures, AI agents should consult this document first.

## Core Concepts

### Web Scraping
Web scraping is the process of extracting data from websites. It involves making HTTP requests to web servers, downloading HTML content, and parsing that content to extract structured data. In the context of NewsCrawler, web scraping is used to collect news articles from various sources.

Key components:
- **HTTP Requests**: Communication with web servers to retrieve HTML content
- **HTML Parsing**: Converting raw HTML into structured data
- **DOM Manipulation**: Navigating and extracting data from the Document Object Model
- **Rate Limiting**: Controlling request frequency to avoid overloading servers
- **User-Agent Management**: Identifying the scraper to websites appropriately
- **Error Handling**: Managing failed requests, timeouts, and other issues

### Retrieval-Augmented Generation (RAG)
RAG is an AI framework that enhances large language models by providing them with external knowledge. Instead of relying solely on information learned during training, RAG systems retrieve relevant documents from a knowledge base and use them to generate more accurate, up-to-date responses.

Key components:
- **Document Retrieval**: Finding relevant documents from a corpus
- **Vector Embeddings**: Converting text into numerical representations for similarity search
- **Context Integration**: Incorporating retrieved information into prompt context
- **Knowledge Base Management**: Organizing and updating the external knowledge source

### Database Management
Database management involves designing, implementing, and maintaining a database system to store and retrieve data efficiently. For NewsCrawler, this includes creating a schema for news articles, optimizing queries, and ensuring data integrity.

Key components:
- **Schema Design**: Defining the structure of database tables
- **Indexing**: Creating indexes to speed up queries
- **Query Optimization**: Writing efficient database queries
- **Data Integrity**: Ensuring data consistency and accuracy
- **Backup and Recovery**: Protecting against data loss

### Containerization
Containerization is the process of packaging an application and its dependencies into a standardized unit (container) that can run consistently across different computing environments. Docker is the primary containerization technology used in NewsCrawler.

Key components:
- **Container Images**: Lightweight, standalone packages containing software and dependencies
- **Container Orchestration**: Managing multiple containers as a single application
- **Environment Variables**: Configuring containers for different environments
- **Volume Management**: Persisting data outside of containers
- **Networking**: Enabling communication between containers

## Technical Terminology

### Web Scraping
- **HTML**: HyperText Markup Language, the standard markup language for documents designed to be displayed in a web browser.
- **CSS Selectors**: Patterns used to select HTML elements for data extraction.
- **XPath**: XML Path Language, used to navigate through elements in an XML document (including HTML).
- **DOM**: Document Object Model, a programming interface for HTML and XML documents.
- **Headless Browser**: A web browser without a graphical user interface, used for automated testing and web scraping.
- **Robots.txt**: A file that indicates which parts of a website should not be accessed by web crawlers.
- **Rate Limiting**: Restricting the number of requests a client can make to a server in a given timeframe.
- **User-Agent**: A string that identifies the client software making a request to a web server.

### Data Processing
- **NLP**: Natural Language Processing, a field of AI focused on the interaction between computers and human language.
- **Tokenization**: Breaking text into words, phrases, or other meaningful elements.
- **Stemming/Lemmatization**: Reducing words to their base or root form.
- **Named Entity Recognition (NER)**: Identifying and classifying named entities in text.
- **Text Cleaning**: Removing unwanted characters, formatting, and noise from text data.
- **Metadata**: Data that provides information about other data, such as publication date or author of an article.

### Database
- **SQL**: Structured Query Language, used for managing relational databases.
- **Schema**: The structure that defines how data is organized in a database.
- **Index**: A data structure that improves the speed of data retrieval operations.
- **Query**: A request for data from a database.
- **Transaction**: A sequence of database operations that are treated as a single unit.
- **ACID**: Atomicity, Consistency, Isolation, Durability - properties that guarantee database transactions are processed reliably.
- **ORM**: Object-Relational Mapping, a technique for converting data between incompatible type systems in object-oriented programming languages.

### API Development
- **REST**: Representational State Transfer, an architectural style for designing networked applications.
- **Endpoint**: A specific URL where an API can be accessed.
- **HTTP Methods**: GET, POST, PUT, DELETE, etc., used to perform different operations on resources.
- **JSON**: JavaScript Object Notation, a lightweight data interchange format.
- **Authentication**: Verifying the identity of a user or system.
- **Authorization**: Determining whether a user or system has permission to access a resource.
- **Rate Limiting**: Restricting the number of API requests a client can make in a given timeframe.

### Containerization
- **Docker**: A platform for developing, shipping, and running applications in containers.
- **Docker Image**: A lightweight, standalone, executable package that includes everything needed to run an application.
- **Docker Container**: A running instance of a Docker image.
- **Dockerfile**: A text document that contains all the commands needed to build a Docker image.
- **Docker Compose**: A tool for defining and running multi-container Docker applications.
- **Volume**: A persistent data storage mechanism for Docker containers.
- **Environment Variable**: A variable whose value is set outside the program, typically through functionality built into the operating system or microservice.

## Data Sources

### News Websites
- **Traditional News Sites**: Established news organizations with structured content (e.g., NYTimes, WSJ, The Guardian).
- **Business News Sites**: Specialized in business and financial news (e.g., Bloomberg, Business Insider).
- **Regional News Sites**: Focus on specific geographic regions (e.g., VietnamNet).
- **RSS Feeds**: Standardized format for distributing regularly updated content.
- **News APIs**: Programmatic access to news content provided by news organizations or aggregators.

## Implementation Technologies

### Web Scraping Tools
- **Newspaper4k**: Python library for extracting and parsing newspaper articles, built on top of newspaper3k with Python 3.10+ support.
  - Key features: Article extraction, natural language processing, multi-language support
  - Best for: Clean article text extraction from news sites
  - Documentation: https://newspaper4k.readthedocs.io/

- **feedparser**: Python library for parsing RSS and Atom feeds.
  - Key features: Feed parsing, handling various feed formats
  - Best for: Processing RSS feeds from news sources
  - Documentation: https://feedparser.readthedocs.io/

- **gnews**: Python library for searching and retrieving news articles from Google News.
  - Key features: Search functionality, topic filtering, language support
  - Best for: Discovering news articles by topic or keyword
  - Documentation: https://pypi.org/project/gnews/

- **BeautifulSoup4**: Python library for pulling data out of HTML and XML files.
  - Key features: HTML parsing, navigation, search, modification
  - Best for: Custom scraping of websites with complex structures
  - Documentation: https://pypi.org/project/beautifulsoup4/

- **Puppeteer**: Node.js library for controlling headless Chrome or Chromium.
  - Key features: Full browser automation, JavaScript execution, screenshot capture
  - Best for: Scraping JavaScript-heavy websites that require browser rendering
  - Documentation: https://pptr.dev/guides/

### Backend Technologies
- **LangChain**: Framework for developing applications powered by language models.
  - Key features: Document loading, text splitting, embeddings, vector stores, retrieval
  - Best for: Building RAG applications with LLMs
  - Documentation: https://python.langchain.com/docs/get_started/introduction

- **FastAPI**: Modern, fast web framework for building APIs with Python.
  - Key features: Fast performance, automatic documentation, data validation
  - Best for: Building high-performance RESTful APIs
  - Documentation: https://fastapi.tiangolo.com/

- **PostgreSQL**: Open-source relational database system.
  - Key features: ACID compliance, JSON support, full-text search
  - Best for: Storing structured data with complex relationships
  - Documentation: https://www.postgresql.org/docs/

- **Docker**: Platform for developing, shipping, and running applications in containers.
  - Key features: Containerization, isolation, portability
  - Best for: Consistent deployment across environments
  - Documentation: https://docs.docker.com/

## Methodological Approaches

### Multi-Strategy Scraping
A methodological approach that employs different scraping techniques based on the structure and behavior of the target website. This approach recognizes that no single scraping method works for all websites and adapts accordingly.

Implementation steps:
1. Analyze website structure and behavior
2. Select appropriate scraping library/method
3. Implement site-specific extraction logic
4. Validate and clean extracted data
5. Handle errors and edge cases

### Incremental Crawling
A methodological approach that focuses on retrieving only new or updated content since the last crawl, rather than re-crawling all content. This approach minimizes resource usage and reduces load on target websites.

Implementation steps:
1. Track last crawl timestamp for each source
2. Identify new or updated content based on publication date
3. Crawl only the identified content
4. Update the crawl timestamp after successful completion
5. Periodically perform full crawls to catch missed updates

### Content Deduplication
A methodological approach to identify and eliminate duplicate or near-duplicate content in the database. This ensures that the knowledge base contains unique information and reduces storage requirements.

Implementation steps:
1. Generate content fingerprints or hash values
2. Compare new content against existing entries
3. Implement similarity detection for near-duplicates
4. Merge or update existing entries when appropriate
5. Maintain version history for significant updates

## Glossary of Terms

- **API (Application Programming Interface)**: A set of rules that allows different software applications to communicate with each other.
- **CRUD (Create, Read, Update, Delete)**: The four basic operations of persistent storage.
- **CSS (Cascading Style Sheets)**: A style sheet language used for describing the presentation of a document written in HTML.
- **CSV (Comma-Separated Values)**: A simple file format used to store tabular data.
- **Docker**: A platform for developing, shipping, and running applications in containers.
- **DOM (Document Object Model)**: A programming interface for HTML and XML documents.
- **FastAPI**: A modern, fast web framework for building APIs with Python.
- **HTML (HyperText Markup Language)**: The standard markup language for documents designed to be displayed in a web browser.
- **HTTP (HyperText Transfer Protocol)**: An application protocol for distributed, collaborative, hypermedia information systems.
- **JSON (JavaScript Object Notation)**: A lightweight data interchange format.
- **LangChain**: A framework for developing applications powered by language models.
- **LLM (Large Language Model)**: A type of AI model trained on vast amounts of text data to generate human-like text.
- **NLP (Natural Language Processing)**: A field of AI focused on the interaction between computers and human language.
- **ORM (Object-Relational Mapping)**: A technique for converting data between incompatible type systems in object-oriented programming languages.
- **PostgreSQL**: An open-source relational database system.
- **Puppeteer**: A Node.js library for controlling headless Chrome or Chromium.
- **RAG (Retrieval-Augmented Generation)**: An AI framework that enhances language models by providing them with external knowledge.
- **REST (Representational State Transfer)**: An architectural style for designing networked applications.
- **RSS (Really Simple Syndication)**: A web feed that allows users and applications to access updates to websites in a standardized format.
- **SQL (Structured Query Language)**: A domain-specific language used for managing relational databases.
- **URL (Uniform Resource Locator)**: A reference to a web resource that specifies its location on a computer network.
- **Web Scraping**: The process of extracting data from websites.
- **XML (eXtensible Markup Language)**: A markup language that defines a set of rules for encoding documents in a format that is both human-readable and machine-readable. 


### 1. PostgreSQL Database Setup

#### Database Schema Design
- **Articles Table**:
  - `id`: Primary key (UUID)
  - `url`: URL of the article (unique)
  - `title`: Article title
  - `content`: Full article text
  - `summary`: Auto-generated summary
  - `published_date`: Publication date
  - `author`: Author name(s)
  - `source_domain`: Domain of the source
  - `category`: Article category/section
  - `keywords`: Array of keywords
  - `created_at`: Timestamp of when the record was created
  - `updated_at`: Timestamp of when the record was last updated
  - `vector_embedding`: Vector representation of the article (for semantic search)

- **Sources Table**:
  - `id`: Primary key (UUID)
  - `domain`: Domain name (unique)
  - `name`: Source name
  - `base_url`: Base URL
  - `scraper_type`: Type of scraper to use
  - `active`: Boolean indicating if the source is active
  - `last_crawled`: Timestamp of last crawl
  - `crawl_frequency`: How often to crawl (in hours)

- **Crawl_Logs Table**:
  - `id`: Primary key (UUID)
  - `source_id`: Foreign key to Sources table
  - `start_time`: Start timestamp
  - `end_time`: End timestamp
  - `articles_found`: Number of articles found
  - `articles_added`: Number of new articles added
  - `articles_updated`: Number of articles updated
  - `status`: Success/failure status
  - `error_message`: Error message if failed

#### Database Migration Scripts
- Create SQL migration scripts for initial schema creation
- Implement SQLAlchemy models matching the database schema
- Create database upgrade/downgrade scripts for future schema changes

### 2. FastAPI Implementation

#### API Structure
- **Core API Module**:
  - Application factory pattern
  - Configuration management
  - Dependency injection setup
  - Authentication middleware (JWT-based)
  - CORS configuration
  - Rate limiting

- **Endpoints**:
  - `/articles`: CRUD operations for articles
  - `/sources`: CRUD operations for news sources
  - `/crawl`: Endpoints to trigger and monitor crawling jobs
  - `/search`: Text and semantic search capabilities
  - `/stats`: System statistics and metrics
  - `/health`: Health check endpoint

- **Documentation**:
  - Automatic Swagger/OpenAPI documentation
  - Custom documentation for complex endpoints
  - Authentication documentation

#### API Features
- Pagination for list endpoints
- Filtering and sorting capabilities
- Full-text search integration
- Vector search for semantic queries (using LangChain)
- Background task handling for long-running operations
- WebSocket support for real-time updates

### 3. LangChain Integration

#### Vector Database Setup
- Configure PostgreSQL with pgvector extension for vector storage
- Implement vector embedding generation using LangChain's embedding models
- Create indexing system for efficient vector search

#### RAG Components
- **Document Loaders**:
  - Implement custom document loaders for different article formats
  - Configure metadata extraction during loading

- **Text Splitters**:
  - Implement semantic text splitting for articles
  - Configure chunk size and overlap for optimal retrieval

- **Retrievers**:
  - Implement vector-based retrieval using LangChain
  - Create hybrid retrieval combining keyword and semantic search
  - Implement relevance filtering and re-ranking

- **LLM Integration**:
  - Configure LangChain to use specified LLM models
  - Implement prompt templates for different query types
  - Create output parsers for structured responses

#### LangChain Agents
- Design specialized agents for different types of news analysis
- Implement tool integration for external data enrichment
- Create agent orchestration for complex queries

### 4. Docker Containerization

#### Container Architecture
- **Database Container**:
  - PostgreSQL with pgvector extension
  - Persistent volume configuration
  - Optimized PostgreSQL configuration

- **API Container**:
  - FastAPI application
  - Gunicorn/Uvicorn for production serving
  - Health check configuration

- **Crawler Container**:
  - Scheduled crawling jobs
  - Configurable parallelism
  - Resource limitation

- **Vector Processing Container**:
  - Batch processing for vector embeddings
  - GPU support configuration (optional)