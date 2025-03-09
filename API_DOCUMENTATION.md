# NewsCrawler API Documentation

This document provides comprehensive documentation for the NewsCrawler API, including endpoint descriptions, request parameters, response formats, and examples.

## Table of Contents

- [Authentication](#authentication)
- [Articles API](#articles-api)
- [Search API](#search-api)
- [LLM API](#llm-api)
- [Crawler API](#crawler-api)
- [Health API](#health-api)
- [Metrics API](#metrics-api)

## Authentication

The NewsCrawler API uses API key authentication for securing endpoints. Include your API key in the request headers as follows:

```
X-API-Key: your_api_key_here
```

### Rate Limiting

The API implements rate limiting to prevent abuse. By default, clients are limited to:
- 60 requests per minute
- Burst capacity of 100 requests

When rate limits are exceeded, the API will return a 429 Too Many Requests status code with a Retry-After header indicating when to retry.

### Error Responses

All API endpoints follow a consistent error response format:

```json
{
  "detail": "Error message describing what went wrong",
  "status_code": 400,
  "error_type": "ValidationError",
  "timestamp": "2023-03-15T12:00:00Z",
  "path": "/api/endpoint"
}
```

Common error status codes:
- 400: Bad Request - Invalid input parameters
- 401: Unauthorized - Missing or invalid API key
- 403: Forbidden - Insufficient permissions
- 404: Not Found - Resource not found
- 429: Too Many Requests - Rate limit exceeded
- 500: Internal Server Error - Server-side error

## Articles API

The Articles API provides endpoints for retrieving, creating, updating, and deleting news articles.

### Get Articles

Retrieves a paginated list of articles with optional filtering and sorting.

**Endpoint:** `GET /articles`

**Query Parameters:**
- `page` (integer, default: 1): Page number (1-indexed)
- `size` (integer, default: 10): Page size (1-100)
- `source_domain` (string, optional): Filter by source domain
- `category` (string, optional): Filter by category
- `author` (string, optional): Filter by author
- `published_after` (datetime, optional): Filter by publication date (after)
- `published_before` (datetime, optional): Filter by publication date (before)
- `search` (string, optional): Search in title and content
- `sort_by` (string, default: "published_date"): Field to sort by
- `sort_order` (string, default: "desc"): Sort order (asc or desc)

**Response:**
```json
{
  "items": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "url": "https://example.com/article1",
      "title": "Example Article Title",
      "content": "Article content...",
      "summary": "Article summary...",
      "published_date": "2023-03-15T12:00:00Z",
      "author": "John Doe",
      "source_domain": "example.com",
      "category": "Technology",
      "keywords": ["tech", "news", "example"],
      "created_at": "2023-03-15T12:30:00Z",
      "updated_at": "2023-03-15T12:30:00Z"
    }
  ],
  "total": 100,
  "page": 1,
  "size": 10,
  "pages": 10
}
```

### Get Article by ID

Retrieves a specific article by its ID.

**Endpoint:** `GET /articles/{article_id}`

**Path Parameters:**
- `article_id` (string, required): The unique identifier of the article

**Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "url": "https://example.com/article1",
  "title": "Example Article Title",
  "content": "Article content...",
  "summary": "Article summary...",
  "published_date": "2023-03-15T12:00:00Z",
  "author": "John Doe",
  "source_domain": "example.com",
  "category": "Technology",
  "keywords": ["tech", "news", "example"],
  "created_at": "2023-03-15T12:30:00Z",
  "updated_at": "2023-03-15T12:30:00Z"
}
```

### Create Article

Creates a new article.

**Endpoint:** `POST /articles`

**Request Body:**
```json
{
  "url": "https://example.com/article1",
  "title": "Example Article Title",
  "content": "Article content...",
  "summary": "Article summary...",
  "published_date": "2023-03-15T12:00:00Z",
  "author": "John Doe",
  "source_domain": "example.com",
  "category": "Technology",
  "keywords": ["tech", "news", "example"]
}
```

**Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "url": "https://example.com/article1",
  "title": "Example Article Title",
  "content": "Article content...",
  "summary": "Article summary...",
  "published_date": "2023-03-15T12:00:00Z",
  "author": "John Doe",
  "source_domain": "example.com",
  "category": "Technology",
  "keywords": ["tech", "news", "example"],
  "created_at": "2023-03-15T12:30:00Z",
  "updated_at": "2023-03-15T12:30:00Z"
}
```

## Search API

The Search API provides endpoints for searching articles using text-based, semantic, and hybrid search methods.

### Text Search

Performs a text-based search on articles with pagination and filtering.

**Endpoint:** `GET /search/text`

**Query Parameters:**
- `q` (string, required): Search query
- `page` (integer, default: 1): Page number (1-indexed)
- `size` (integer, default: 10): Page size (1-100)
- `source_domain` (string, optional): Filter by source domain
- `category` (string, optional): Filter by category
- `author` (string, optional): Filter by author
- `published_after` (datetime, optional): Filter by publication date (after)
- `published_before` (datetime, optional): Filter by publication date (before)

**Response:**
```json
{
  "items": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "url": "https://example.com/article1",
      "title": "Example Article Title",
      "summary": "Article summary...",
      "published_date": "2023-03-15T12:00:00Z",
      "author": "John Doe",
      "source_domain": "example.com",
      "category": "Technology",
      "score": null
    }
  ],
  "total": 100,
  "page": 1,
  "size": 10,
  "pages": 10,
  "query": "example search"
}
```

### Semantic Search

Performs a semantic search on articles using vector embeddings with pagination and filtering.

**Endpoint:** `GET /search/semantic`

**Query Parameters:**
- `q` (string, required): Search query
- `page` (integer, default: 1): Page number (1-indexed)
- `size` (integer, default: 10): Page size (1-100)
- `source_domain` (string, optional): Filter by source domain
- `category` (string, optional): Filter by category
- `author` (string, optional): Filter by author
- `published_after` (datetime, optional): Filter by publication date (after)
- `published_before` (datetime, optional): Filter by publication date (before)
- `min_similarity` (float, default: 0.5): Minimum similarity score (0-1)

**Response:**
```json
{
  "items": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "url": "https://example.com/article1",
      "title": "Example Article Title",
      "summary": "Article summary...",
      "published_date": "2023-03-15T12:00:00Z",
      "author": "John Doe",
      "source_domain": "example.com",
      "category": "Technology",
      "score": 0.85
    }
  ],
  "total": 100,
  "page": 1,
  "size": 10,
  "pages": 10,
  "query": "example search"
}
```

### Hybrid Search

Performs a hybrid search combining text and semantic search with pagination and filtering.

**Endpoint:** `GET /search/hybrid`

**Query Parameters:**
- `q` (string, required): Search query
- `page` (integer, default: 1): Page number (1-indexed)
- `size` (integer, default: 10): Page size (1-100)
- `source_domain` (string, optional): Filter by source domain
- `category` (string, optional): Filter by category
- `author` (string, optional): Filter by author
- `published_after` (datetime, optional): Filter by publication date (after)
- `published_before` (datetime, optional): Filter by publication date (before)
- `semantic_weight` (float, default: 0.7): Weight for semantic search (0-1)
- `min_similarity` (float, default: 0.3): Minimum similarity score (0-1)

**Response:**
```json
{
  "items": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "url": "https://example.com/article1",
      "title": "Example Article Title",
      "summary": "Article summary...",
      "published_date": "2023-03-15T12:00:00Z",
      "author": "John Doe",
      "source_domain": "example.com",
      "category": "Technology",
      "score": 0.92
    }
  ],
  "total": 100,
  "page": 1,
  "size": 10,
  "pages": 10,
  "query": "example search"
}
```

## LLM API

The LLM API provides endpoints for text generation, summarization, question answering, and other natural language processing tasks using Groq's LLM models.

### Get Available Models

Retrieves a list of available LLM models.

**Endpoint:** `GET /llm/models`

**Response:**
```json
{
  "models": [
    {
      "id": "llama3-8b-8192",
      "name": "Llama 3 8B",
      "description": "Smaller, faster model suitable for most tasks",
      "context_length": 8192,
      "is_default": true
    },
    {
      "id": "llama3-70b-8192",
      "name": "Llama 3 70B",
      "description": "Larger, more capable model for complex tasks",
      "context_length": 8192,
      "is_default": false
    }
  ]
}
```

### Text Generation

Generates text based on a prompt.

**Endpoint:** `POST /llm/generate`

**Request Body:**
```json
{
  "prompt": "Write a short story about a robot learning to paint",
  "model": "llama3-8b-8192",
  "max_tokens": 1024,
  "temperature": 0.7,
  "system_message": "You are a creative storyteller."
}
```

**Response:**
```json
{
  "generated_text": "In a small studio apartment overlooking the city...",
  "model": "llama3-8b-8192"
}
```

### Summarize Text

Summarizes a given text.

**Endpoint:** `POST /llm/summarize`

**Request Body:**
```json
{
  "text": "Long text to be summarized...",
  "max_length": 200
}
```

**Response:**
```json
{
  "summary": "Concise summary of the provided text..."
}
```

### Sentiment Analysis

Analyzes the sentiment of a given text.

**Endpoint:** `POST /llm/sentiment`

**Request Body:**
```json
{
  "text": "Text to analyze sentiment..."
}
```

**Response:**
```json
{
  "analysis": "The text expresses a positive sentiment with themes of optimism and satisfaction..."
}
```

### Entity Extraction

Extracts entities from a given text.

**Endpoint:** `POST /llm/entities`

**Request Body:**
```json
{
  "text": "Text to extract entities from..."
}
```

**Response:**
```json
{
  "entities": "Extracted entities in structured format..."
}
```

### Question Answering (RAG)

Answers a question using Retrieval-Augmented Generation.

**Endpoint:** `POST /llm/ask`

**Request Body:**
```json
{
  "question": "What are the latest developments in renewable energy?",
  "model": "llama3-70b-8192",
  "max_tokens": 1024,
  "temperature": 0.7,
  "context_strategy": "semantic",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "source_domain": "example.com",
  "category": "Technology",
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-12-31T23:59:59Z"
}
```

**Response:**
```json
{
  "answer": "The latest developments in renewable energy include...",
  "sources": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "title": "New Solar Technology Breakthrough",
      "url": "https://example.com/article1",
      "published_date": "2023-03-15T12:00:00Z",
      "source_domain": "example.com",
      "similarity": 0.92
    }
  ],
  "evaluation": {
    "relevance_score": 0.85,
    "factual_accuracy": 0.9,
    "response_time_ms": 1250
  }
}
```

### Question Answering with Specific Context

Answers a question using specific articles as context.

**Endpoint:** `POST /llm/ask-with-context`

**Request Body:**
```json
{
  "question": "What are the key points from these articles?",
  "context_article_ids": ["123e4567-e89b-12d3-a456-426614174000", "223e4567-e89b-12d3-a456-426614174001"],
  "model": "llama3-70b-8192",
  "max_tokens": 1024,
  "temperature": 0.7,
  "context_strategy": "semantic"
}
```

**Response:**
```json
{
  "answer": "The key points from these articles include...",
  "sources": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "title": "Article Title 1",
      "url": "https://example.com/article1",
      "published_date": "2023-03-15T12:00:00Z",
      "source_domain": "example.com"
    },
    {
      "id": "223e4567-e89b-12d3-a456-426614174001",
      "title": "Article Title 2",
      "url": "https://example.com/article2",
      "published_date": "2023-03-16T12:00:00Z",
      "source_domain": "example.com"
    }
  ],
  "evaluation": {
    "relevance_score": 0.88,
    "factual_accuracy": 0.92,
    "response_time_ms": 1150
  }
}
```

### Article Summarization

Summarizes a specific article.

**Endpoint:** `POST /llm/summarize-article`

**Request Body:**
```json
{
  "article_id": "123e4567-e89b-12d3-a456-426614174000",
  "model": "llama3-70b-8192",
  "max_tokens": 512,
  "temperature": 0.3
}
```

**Response:**
```json
{
  "summary": "Concise summary of the article...",
  "article_id": "123e4567-e89b-12d3-a456-426614174000",
  "title": "Article Title",
  "url": "https://example.com/article",
  "published_date": "2023-03-15T12:00:00Z",
  "source_domain": "example.com"
}
```

## Crawler API

The Crawler API provides endpoints for managing news sources, scheduling crawl jobs, and monitoring crawl status.

### Get Sources

Retrieves a list of news sources.

**Endpoint:** `GET /crawler/sources`

**Query Parameters:**
- `active` (boolean, optional): Filter by active status

**Response:**
```json
{
  "items": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "name": "Example News",
      "base_url": "https://example.com",
      "scraper_type": "newspaper",
      "active": true,
      "last_crawled": "2023-03-15T12:00:00Z",
      "crawl_frequency": 24
    }
  ],
  "total": 1
}
```

### Get Source by ID

Retrieves a specific news source by its ID.

**Endpoint:** `GET /crawler/sources/{source_id}`

**Path Parameters:**
- `source_id` (string, required): The unique identifier of the source

**Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "name": "Example News",
  "base_url": "https://example.com",
  "scraper_type": "newspaper",
  "active": true,
  "last_crawled": "2023-03-15T12:00:00Z",
  "crawl_frequency": 24
}
```

### Get Jobs

Retrieves the status of all crawl jobs.

**Endpoint:** `GET /crawler/jobs`

**Response:**
```json
{
  "pending": [
    {
      "job_id": "job-123",
      "source_id": "123e4567-e89b-12d3-a456-426614174000",
      "source_name": "Example News",
      "source_url": "https://example.com",
      "scraper_type": "newspaper",
      "priority": 2,
      "status": "pending",
      "created_at": 1678888800.0,
      "scheduled_time": 1678889400.0,
      "start_time": null,
      "end_time": null,
      "error_message": null,
      "articles_found": 0,
      "articles_added": 0,
      "articles_updated": 0
    }
  ],
  "running": [],
  "completed": [],
  "failed": []
}
```

### Schedule Job

Schedules a new crawl job.

**Endpoint:** `POST /crawler/jobs`

**Request Body:**
```json
{
  "source_id": "123e4567-e89b-12d3-a456-426614174000",
  "priority": "high",
  "max_urls": 200,
  "respect_robots_txt": true,
  "crawl_delay": 2
}
```

**Response:**
```json
{
  "job_id": "job-124",
  "source_id": "123e4567-e89b-12d3-a456-426614174000",
  "source_name": "Example News",
  "status": "pending"
}
```

### Get Job by ID

Retrieves a specific crawl job by its ID.

**Endpoint:** `GET /crawler/jobs/{job_id}`

**Path Parameters:**
- `job_id` (string, required): The unique identifier of the job

**Response:**
```json
{
  "job_id": "job-124",
  "source_id": "123e4567-e89b-12d3-a456-426614174000",
  "source_name": "Example News",
  "source_url": "https://example.com",
  "scraper_type": "newspaper",
  "priority": 3,
  "status": "running",
  "created_at": 1678888900.0,
  "scheduled_time": 1678889000.0,
  "start_time": 1678889000.0,
  "end_time": null,
  "error_message": null,
  "articles_found": 15,
  "articles_added": 10,
  "articles_updated": 2
}
```

### Cancel Job

Cancels a pending or running crawl job.

**Endpoint:** `DELETE /crawler/jobs/{job_id}`

**Path Parameters:**
- `job_id` (string, required): The unique identifier of the job

**Response:**
204 No Content

### Get Crawl Logs

Retrieves crawl logs for a specific source.

**Endpoint:** `GET /crawler/logs`

**Query Parameters:**
- `source_id` (string, optional): Filter by source ID
- `status` (string, optional): Filter by status (in_progress, completed, failed)
- `page` (integer, default: 1): Page number
- `size` (integer, default: 10): Page size

**Response:**
```json
{
  "items": [
    {
      "id": "log-123",
      "source_id": "123e4567-e89b-12d3-a456-426614174000",
      "start_time": "2023-03-15T12:00:00Z",
      "end_time": "2023-03-15T12:15:00Z",
      "articles_found": 50,
      "articles_added": 30,
      "articles_updated": 5,
      "status": "completed",
      "error_message": null
    }
  ],
  "total": 1
}
```

## Health API

The Health API provides endpoints for checking the health and status of the NewsCrawler system.

### Health Check

Retrieves the current health status of the system.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime": 3600.5,
  "memory_usage": {
    "rss": 156483584,
    "vms": 287309824,
    "rss_mb": 149.24,
    "vms_mb": 274.0
  },
  "environment": "development",
  "cache_stats": {
    "vector_cache": {
      "hits": 1250,
      "misses": 320,
      "size": 450,
      "max_size": 1000,
      "hit_rate": 0.796
    },
    "search_cache": {
      "hits": 2340,
      "misses": 560,
      "size": 780,
      "max_size": 1000,
      "hit_rate": 0.807
    }
  }
}
```

## Metrics API

The Metrics API provides endpoints for retrieving performance metrics, cache statistics, and system information.

### Get All Metrics

Retrieves all metrics, cache statistics, and system information.

**Endpoint:** `GET /metrics`

**Response:**
```json
{
  "metrics": {
    "api.request_duration": {
      "type": "Histogram",
      "description": "Duration of API requests",
      "count": 5280,
      "sum": 1250.45,
      "mean": 0.237,
      "p50": 0.185,
      "p90": 0.452,
      "p95": 0.687,
      "p99": 1.245,
      "buckets": {
        "0.01": 120,
        "0.05": 980,
        "0.1": 1540,
        "0.5": 2100,
        "1.0": 350,
        "5.0": 180,
        "10.0": 10
      }
    },
    "api.requests_total": {
      "type": "Counter",
      "description": "Total number of API requests",
      "value": 5280
    },
    "crawler.articles_processed": {
      "type": "Counter",
      "description": "Number of articles processed by the crawler",
      "value": 12450
    },
    "vector.embeddings_generated": {
      "type": "Counter",
      "description": "Number of embeddings generated",
      "value": 10280
    }
  },
  "cache_stats": {
    "vector_cache": {
      "hits": 1250,
      "misses": 320,
      "size": 450,
      "max_size": 1000,
      "hit_rate": 0.796
    },
    "search_cache": {
      "hits": 2340,
      "misses": 560,
      "size": 780,
      "max_size": 1000,
      "hit_rate": 0.807
    },
    "rag_cache": {
      "hits": 890,
      "misses": 210,
      "size": 320,
      "max_size": 500,
      "hit_rate": 0.809
    }
  },
  "system_info": {
    "cpu_percent": 35.2,
    "memory_usage": {
      "rss": 149.24,
      "vms": 274.0
    },
    "threads": 12,
    "open_files": 8,
    "connections": 24
  }
}
```

### Get Prometheus Metrics

Retrieves metrics in Prometheus format.

**Endpoint:** `GET /metrics/prometheus`

**Response:**
```
# HELP api.request_duration Duration of API requests
# TYPE api.request_duration histogram
api.request_duration_sum 1250.45
api.request_duration_count 5280
api.request_duration_bucket{le="0.01"} 120
api.request_duration_bucket{le="0.05"} 980
api.request_duration_bucket{le="0.1"} 1540
api.request_duration_bucket{le="0.5"} 2100
api.request_duration_bucket{le="1.0"} 350
api.request_duration_bucket{le="5.0"} 180
api.request_duration_bucket{le="10.0"} 10
api.request_duration_bucket{le="+Inf"} 5280

# HELP api.requests_total Total number of API requests
# TYPE api.requests_total counter
api.requests_total 5280

# HELP crawler.articles_processed Number of articles processed by the crawler
# TYPE crawler.articles_processed counter
crawler.articles_processed 12450

# HELP vector.embeddings_generated Number of embeddings generated
# TYPE vector.embeddings_generated counter
vector.embeddings_generated 10280
```

### Reset Metrics

Resets all metrics counters.

**Endpoint:** `POST /metrics/reset`

**Response:**
204 No Content

### Get Cache Statistics

Retrieves statistics for all caches.

**Endpoint:** `GET /metrics/cache`

**Response:**
```json
{
  "vector_cache": {
    "hits": 1250,
    "misses": 320,
    "size": 450,
    "max_size": 1000,
    "hit_rate": 0.796
  },
  "search_cache": {
    "hits": 2340,
    "misses": 560,
    "size": 780,
    "max_size": 1000,
    "hit_rate": 0.807
  },
  "rag_cache": {
    "hits": 890,
    "misses": 210,
    "size": 320,
    "max_size": 500,
    "hit_rate": 0.809
  }
}
```

### Clear Caches

Clears all caches.

**Endpoint:** `POST /metrics/cache/clear`

**Response:**
204 No Content

### Get System Information

Retrieves system information.

**Endpoint:** `GET /metrics/system`

**Response:**
```json
{
  "cpu_percent": 35.2,
  "memory_usage": {
    "rss": 149.24,
    "vms": 274.0
  },
  "threads": 12,
  "open_files": 8,
  "connections": 24,
  "uptime": 86400.5,
  "os_info": {
    "system": "Windows",
    "release": "10",
    "version": "10.0.19045"
  }
}
``` 