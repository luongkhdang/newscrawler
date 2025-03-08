# PostgreSQL Database Schema Analysis

## Overview
This document analyzes the current PostgreSQL database schema for the NewsCrawler project, focusing on its structure, relationships, indexing strategies, and potential optimization opportunities.

## Current Schema

The database consists of three main tables:

### 1. Articles Table

```sql
CREATE TABLE IF NOT EXISTS articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    published_date TIMESTAMP,
    author TEXT,
    source_domain TEXT NOT NULL,
    category TEXT,
    keywords TEXT[],
    vector_embedding VECTOR(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Purpose**: Stores the extracted articles from various news sources.

**Key Fields**:
- `id`: UUID primary key
- `url`: Unique URL of the article
- `title`: Article title
- `content`: Full article content
- `source_domain`: Domain of the source website
- `vector_embedding`: 1536-dimensional vector for semantic search (using pgvector)

**Indexes**:
- Primary key on `id`
- Unique constraint on `url`
- Index on `source_domain`
- Index on `published_date`
- Vector index on `vector_embedding` using IVFFlat algorithm

### 2. Sources Table

```sql
CREATE TABLE IF NOT EXISTS sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    domain TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    base_url TEXT NOT NULL,
    scraper_type TEXT NOT NULL,
    active BOOLEAN DEFAULT TRUE,
    last_crawled TIMESTAMP WITH TIME ZONE,
    crawl_frequency INTEGER DEFAULT 24, -- in hours
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Purpose**: Manages the news sources to be crawled.

**Key Fields**:
- `id`: UUID primary key
- `domain`: Unique domain of the source
- `name`: Human-readable name of the source
- `base_url`: Base URL for crawling
- `scraper_type`: Type of scraper to use (e.g., 'newspaper')
- `active`: Whether the source is currently active
- `last_crawled`: When the source was last crawled
- `crawl_frequency`: How often to crawl the source (in hours)

**Indexes**:
- Primary key on `id`
- Unique constraint on `domain`
- Index on `domain`

### 3. Crawl Logs Table

```sql
CREATE TABLE IF NOT EXISTS crawl_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID REFERENCES sources(id),
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    articles_found INTEGER DEFAULT 0,
    articles_added INTEGER DEFAULT 0,
    articles_updated INTEGER DEFAULT 0,
    status TEXT DEFAULT 'in_progress',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Purpose**: Tracks crawling operations and their results.

**Key Fields**:
- `id`: UUID primary key
- `source_id`: Foreign key to the sources table
- `start_time`: When the crawl started
- `end_time`: When the crawl ended
- `articles_found`: Number of articles found
- `articles_added`: Number of new articles added
- `articles_updated`: Number of existing articles updated
- `status`: Status of the crawl (e.g., 'in_progress', 'completed', 'failed')
- `error_message`: Error message if the crawl failed

**Indexes**:
- Primary key on `id`
- Foreign key on `source_id`
- Index on `source_id`
- Index on `status`

## Relationships

1. **Sources to Crawl Logs**: One-to-many relationship
   - A source can have multiple crawl logs
   - Each crawl log belongs to exactly one source

2. **Sources to Articles**: Implicit one-to-many relationship through `source_domain`
   - A source can have multiple articles
   - Each article belongs to one source (through `source_domain`)

## Extensions

1. **pgvector**: Enables vector operations for semantic search
   - Used for storing and querying article embeddings
   - Configured with IVFFlat index for efficient similarity search

2. **uuid-ossp**: Provides UUID generation functions
   - Used for generating primary keys

## SQLAlchemy ORM Implementation

The schema is implemented using SQLAlchemy ORM with the following models:

1. **Article**: Maps to the `articles` table
2. **Source**: Maps to the `sources` table
3. **CrawlLog**: Maps to the `crawl_logs` table

The models use SQLAlchemy's relationship features to establish the connections between tables.

## Analysis

### Strengths

1. **Well-structured schema**: Clear separation of concerns between articles, sources, and crawl logs
2. **UUID primary keys**: Provides globally unique identifiers without coordination
3. **Vector search support**: Built-in support for semantic search using pgvector
4. **Appropriate indexing**: Indexes on frequently queried fields
5. **Timestamps**: Proper tracking of creation and update times
6. **Crawl management**: Comprehensive tracking of crawl operations

### Limitations and Improvement Opportunities

1. **Missing direct relationship**: No direct foreign key relationship between articles and sources
   - Currently relies on `source_domain` text field
   - Could benefit from a proper foreign key relationship

2. **Limited metadata**: The articles table could benefit from additional metadata fields
   - Reading time estimation
   - Content language
   - Content type (news, opinion, etc.)
   - Image URLs (if needed in the future)

3. **Vector indexing**: The current IVFFlat index with 100 lists may need tuning
   - Performance depends on the size of the dataset
   - May need to adjust the number of lists as the dataset grows

4. **Text search capabilities**: No full-text search indexes
   - Could benefit from PostgreSQL's full-text search capabilities
   - Would enable hybrid search (combining vector and keyword search)

5. **Chunking strategy**: No support for storing article chunks for RAG
   - May need additional tables for storing article chunks
   - Would enable more granular retrieval

6. **Versioning**: No versioning of article content
   - Could benefit from tracking article updates
   - Important for maintaining historical data

## Recommendations

1. **Add Foreign Key Relationship**:
   ```sql
   ALTER TABLE articles ADD COLUMN source_id UUID REFERENCES sources(id);
   UPDATE articles SET source_id = (SELECT id FROM sources WHERE domain = articles.source_domain);
   CREATE INDEX idx_articles_source_id ON articles(source_id);
   ```

2. **Enhance Metadata**:
   ```sql
   ALTER TABLE articles ADD COLUMN reading_time_minutes INTEGER;
   ALTER TABLE articles ADD COLUMN language TEXT;
   ALTER TABLE articles ADD COLUMN content_type TEXT;
   ```

3. **Add Full-Text Search**:
   ```sql
   ALTER TABLE articles ADD COLUMN content_tsvector TSVECTOR;
   CREATE INDEX idx_articles_content_tsvector ON articles USING GIN (content_tsvector);
   CREATE TRIGGER tsvector_update_trigger BEFORE INSERT OR UPDATE ON articles
   FOR EACH ROW EXECUTE FUNCTION tsvector_update_trigger(content_tsvector, 'pg_catalog.english', content, title);
   ```

4. **Create Chunks Table**:
   ```sql
   CREATE TABLE article_chunks (
       id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
       article_id UUID REFERENCES articles(id) ON DELETE CASCADE,
       chunk_index INTEGER NOT NULL,
       content TEXT NOT NULL,
       vector_embedding VECTOR(1536),
       created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
       UNIQUE(article_id, chunk_index)
   );
   CREATE INDEX idx_article_chunks_article_id ON article_chunks(article_id);
   CREATE INDEX idx_article_chunks_vector_embedding ON article_chunks USING ivfflat (vector_embedding vector_cosine_ops) WITH (lists = 100);
   ```

5. **Optimize Vector Indexing**:
   - Benchmark different index types (IVFFlat, HNSW)
   - Tune parameters based on dataset size and query patterns
   - Consider partitioning for very large datasets

6. **Add Article Versioning**:
   ```sql
   CREATE TABLE article_versions (
       id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
       article_id UUID REFERENCES articles(id) ON DELETE CASCADE,
       content TEXT NOT NULL,
       version INTEGER NOT NULL,
       created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
       UNIQUE(article_id, version)
   );
   CREATE INDEX idx_article_versions_article_id ON article_versions(article_id);
   ```

## Next Steps

1. Implement and test the recommended schema changes
2. Benchmark performance with different indexing strategies
3. Test pgvector extension with various embedding models
4. Develop migration scripts using Alembic
5. Create database maintenance procedures
6. Document the optimized schema 