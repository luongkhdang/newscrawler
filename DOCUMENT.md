# NewsCrawler

## !IMPORTANT – REFERENCE SYSTEM
- **KNOWLEDGE_BASE.md**: Reference point for AI agents when they need to understand concepts, theories, vocabulary, or structures.
- **DOCUMENT.md**: Main project document (this file).
- **AI_WORKSPACE/FINDINGS.md**: Contains research findings and insights discovered by AI agents.
- **AI_WORKSPACE/PROTOCOLS.md**: Contains rules and protocols for all AI agents working on this project.
- **AI_WORKSPACE/QUESTIONS.md**: Repository for user questions directed to AI agents.
- **AI_WORKSPACE/PLANS/**: Directory containing all research and implementation plans created by AI agents.
- **AI_WORKSPACE/SYSTEM_DESIGN.md**: Contains system architecture documentation, diagrams, and performance considerations.
- **AI_WORKSPACE/CODE_QUALITY.md**: Defines engineering standards and code quality guidelines.
- **AI_WORKSPACE/SECURITY_COMPLIANCE.md**: Outlines security practices and compliance requirements.

## !IMPORTANT – AI AGENT RULES
1. When encountering an unfamiliar concept, ALWAYS check KNOWLEDGE_BASE.md first before responding.
2. All research findings must be documented in AI_WORKSPACE/FINDINGS.md using the specified format.
3. Follow all protocols specified in AI_WORKSPACE/PROTOCOLS.md.
4. When uncertain about a user request, refer to AI_WORKSPACE/QUESTIONS.md for clarification.
5. When asked to create a plan, follow the Plan Creation Workflow in AI_WORKSPACE/PROTOCOLS.md and store the plan in AI_WORKSPACE/PLANS/ directory.
6. Optimize content in AI_WORKSPACE/ directory for AI agent consumption (token efficiency/impact ratio) and content outside AI_WORKSPACE/ for human consumption.
7. DO NOT REPEAT INFORMATION. Use cross-references between documents whenever possible to avoid redundancy.
8. Track progress on plans by checking off completed tasks and updating plan status.
9. Use dynamic tagging instead of fixed categories for questions and findings to improve adaptability.
10. Before implementing any code, document the system design in AI_WORKSPACE/SYSTEM_DESIGN.md.
11. Follow code quality guidelines in AI_WORKSPACE/CODE_QUALITY.md for all implementations.
12. Ensure all code adheres to security and compliance standards in AI_WORKSPACE/SECURITY_COMPLIANCE.md.
13. ALWAYS use Python virtual environments (venv) instead of global environments for all Python development. Never install packages globally.

## Project Overview
NewsCrawler is a comprehensive web scraping and article retrieval system designed to collect news articles from a predefined list of URLs, process and clean the content, and store it in a structured database. The collected data will serve as the knowledge base for a future Retrieval-Augmented Generation (RAG) chatbot, enabling it to provide up-to-date information and insights based on news content.

## Project Scope

### Included:
- Development of a robust web scraping system using multiple libraries (Newspaper4k, feedparser, gnews, BeautifulSoup4, Puppeteer)
- Processing and cleaning of article content to extract relevant information
- Storage of articles in a PostgreSQL database with appropriate schema design
- Implementation of a RESTful API using FastAPI for data retrieval and management
- Containerization of the entire system using Docker for easy deployment
- Integration with LangChain for future RAG capabilities

### Excluded:
- Development of the actual RAG chatbot interface (future project)
- Real-time news alerting or notification systems
- Sentiment analysis or advanced NLP processing of articles (may be added in future iterations)
- User authentication and authorization systems (unless required for API access)
- **Image scraping and storage**: The system should NEVER scrape, process, or store images from articles to optimize performance, reduce bandwidth usage, and avoid potential copyright issues

## Key Components

### 1. Data Collection Module
- Multi-strategy web scraper using various libraries to handle different website structures
- URL management system to track crawled URLs and avoid duplicates
- Rate limiting and politeness protocols to avoid overloading news sites
- Error handling and retry mechanisms for failed requests
- Configuration to explicitly disable image extraction in all scraper implementations

### 2. Content Processing Module
- Article extraction and cleaning
- Metadata extraction (publication date, author, title, etc.)
- Content categorization and tagging
- **No image processing**: All scrapers must be configured to skip image extraction and processing

### 3. Database Module
- PostgreSQL database with optimized schema for article storage
- Indexing for efficient retrieval
- Version control for updated articles
- Backup and recovery mechanisms
- Schema design that excludes image storage fields

### 4. API Module
- FastAPI implementation for RESTful endpoints
- Query capabilities for article retrieval
- Admin endpoints for system management
- Documentation using Swagger/OpenAPI

### 5. Containerization and Deployment
- Docker configuration for all system components
- Docker Compose for multi-container orchestration
- Environment configuration management
- Logging and monitoring setup

## Implementation Strategy

### Phase 1: Research and Design (2 weeks)
- Evaluate and test different web scraping libraries for various news sources
- Design database schema and API endpoints
- Create system architecture documentation

### Phase 2: Core Implementation (3 weeks)
- Develop the data collection module with support for multiple scraping strategies
- Implement the database module and schema
- Create basic content processing functionality

### Phase 3: API and Integration (2 weeks)
- Develop the FastAPI endpoints
- Integrate with the database module
- Implement advanced content processing features

### Phase 4: Containerization and Testing (2 weeks)
- Create Docker configurations
- Implement comprehensive testing
- Optimize performance and resource usage

### Phase 5: Documentation and Deployment (1 week)
- Complete system documentation
- Prepare deployment guides
- Finalize monitoring and maintenance procedures

## Timeline
- **Weeks 1-2**: Research and Design
- **Weeks 3-5**: Core Implementation
- **Weeks 6-7**: API and Integration
- **Weeks 8-9**: Containerization and Testing
- **Week 10**: Documentation and Deployment
- **Total Project Duration**: 10 weeks

## Success Metrics
- Successfully crawl and process at least 95% of provided URLs
- Database query response time under 100ms for standard queries
- API response time under 200ms for standard requests
- System capable of processing at least 10,000 articles per day
- Less than 1% duplicate articles in the database
- Docker containers using less than 2GB RAM in total
- Complete documentation coverage for all system components

## Resources

### Libraries and Tools
- **Web Scraping**:
  - Newspaper4k: https://newspaper4k.readthedocs.io/
  - feedparser: https://feedparser.readthedocs.io/
  - gnews: https://pypi.org/project/gnews/
  - beautifulsoup4: https://pypi.org/project/beautifulsoup4/
  - Puppeteer: https://pptr.dev/guides/

- **Backend and Database**:
  - FastAPI: https://fastapi.tiangolo.com/
  - PostgreSQL: https://www.postgresql.org/
  - SQLAlchemy: https://www.sqlalchemy.org/
  - pgvector: https://github.com/pgvector/pgvector

- **Vector Processing**:
  - LangChain: https://python.langchain.com/
  - HuggingFace Embeddings: https://huggingface.co/models

- **Containerization**:
  - Docker: https://www.docker.com/
  - Docker Compose: https://docs.docker.com/compose/

### Data Sources
- url.csv file containing target URLs
- Additional news sources from news-sources.txt

### Development Guidelines

#### Virtual Environment Setup
All Python development for this project MUST use virtual environments (venv) instead of global environments. This ensures dependency isolation, reproducibility, and prevents conflicts between packages.

```bash
# Create a virtual environment in the project directory
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies within the virtual environment
pip install -r requirements.txt

# When finished, deactivate the virtual environment
deactivate
```

#### Virtual Environment Best Practices
1. Never install packages globally for this project
2. Always activate the virtual environment before working on the project
3. Keep requirements.txt updated with all dependencies
4. Include venv/ in .gitignore to avoid committing the virtual environment
5. Document any new dependencies added to the project

