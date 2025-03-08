# User Interface and Monitoring Implementation Plan

## Overview
This plan outlines the implementation of a user interface for exploring the collected data and a comprehensive monitoring system for the NewsCrawler project.

## 1. Web Dashboard Development (Week 1-2)

### Tasks:
- [ ] Design and implement a React-based dashboard
- [ ] Create data visualization components for content analysis
- [ ] Develop search interface with advanced filtering
- [ ] Implement user authentication and authorization

### Implementation Details:
```jsx
// frontend/src/components/Dashboard.jsx
import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card } from 'react-bootstrap';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import axios from 'axios';

import StatCard from './StatCard';
import RecentArticles from './RecentArticles';
import SourceDistribution from './SourceDistribution';
import TimelineChart from './TimelineChart';

const Dashboard = () => {
  const [stats, setStats] = useState({
    totalArticles: 0,
    totalSources: 0,
    articlesLast24h: 0,
    crawlSuccess: 0
  });
  
  const [sourceData, setSourceData] = useState([]);
  const [timelineData, setTimelineData] = useState([]);
  const [recentArticles, setRecentArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        
        // Fetch overall stats
        const statsResponse = await axios.get('/api/stats/overview');
        setStats(statsResponse.data);
        
        // Fetch source distribution
        const sourcesResponse = await axios.get('/api/stats/sources');
        setSourceData(sourcesResponse.data);
        
        // Fetch timeline data
        const timelineResponse = await axios.get('/api/stats/timeline');
        setTimelineData(timelineResponse.data);
        
        // Fetch recent articles
        const articlesResponse = await axios.get('/api/articles/recent');
        setRecentArticles(articlesResponse.data);
        
        setLoading(false);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        setLoading(false);
      }
    };
    
    fetchDashboardData();
    
    // Refresh data every 5 minutes
    const interval = setInterval(fetchDashboardData, 5 * 60 * 1000);
    
    return () => clearInterval(interval);
  }, []);
  
  if (loading) {
    return <div className="text-center p-5">Loading dashboard data...</div>;
  }
  
  return (
    <Container fluid className="dashboard-container">
      <h1 className="dashboard-title mb-4">NewsCrawler Dashboard</h1>
      
      <Row className="mb-4">
        <Col md={3}>
          <StatCard 
            title="Total Articles" 
            value={stats.totalArticles.toLocaleString()} 
            icon="newspaper" 
            color="primary" 
          />
        </Col>
        <Col md={3}>
          <StatCard 
            title="News Sources" 
            value={stats.totalSources.toLocaleString()} 
            icon="globe" 
            color="success" 
          />
        </Col>
        <Col md={3}>
          <StatCard 
            title="Last 24 Hours" 
            value={stats.articlesLast24h.toLocaleString()} 
            icon="clock" 
            color="info" 
          />
        </Col>
        <Col md={3}>
          <StatCard 
            title="Crawl Success Rate" 
            value={`${stats.crawlSuccess}%`} 
            icon="check-circle" 
            color="warning" 
          />
        </Col>
      </Row>
      
      <Row className="mb-4">
        <Col md={8}>
          <Card className="shadow-sm">
            <Card.Header>Articles Collected Over Time</Card.Header>
            <Card.Body>
              <TimelineChart data={timelineData} />
            </Card.Body>
          </Card>
        </Col>
        <Col md={4}>
          <Card className="shadow-sm">
            <Card.Header>Source Distribution</Card.Header>
            <Card.Body>
              <SourceDistribution data={sourceData} />
            </Card.Body>
          </Card>
        </Col>
      </Row>
      
      <Row>
        <Col md={12}>
          <Card className="shadow-sm">
            <Card.Header>Recently Collected Articles</Card.Header>
            <Card.Body>
              <RecentArticles articles={recentArticles} />
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default Dashboard;
```

```jsx
// frontend/src/components/Search.jsx
import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Form, Button, Card, Spinner } from 'react-bootstrap';
import axios from 'axios';

import ArticleList from './ArticleList';
import SearchFilters from './SearchFilters';

const Search = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [totalResults, setTotalResults] = useState(0);
  const [page, setPage] = useState(1);
  const [filters, setFilters] = useState({
    sources: [],
    dateFrom: null,
    dateTo: null,
    categories: []
  });
  
  const handleSearch = async (e) => {
    e?.preventDefault();
    
    if (!query.trim() && Object.values(filters).every(f => !f || (Array.isArray(f) && f.length === 0))) {
      return;
    }
    
    try {
      setLoading(true);
      
      const response = await axios.get('/api/search', {
        params: {
          q: query,
          page,
          limit: 10,
          sources: filters.sources.join(','),
          date_from: filters.dateFrom,
          date_to: filters.dateTo,
          categories: filters.categories.join(',')
        }
      });
      
      setResults(response.data.results);
      setTotalResults(response.data.total);
      setLoading(false);
    } catch (error) {
      console.error('Error performing search:', error);
      setLoading(false);
    }
  };
  
  useEffect(() => {
    if (query || Object.values(filters).some(f => f && (!Array.isArray(f) || f.length > 0))) {
      handleSearch();
    }
  }, [page, filters]);
  
  const handleFilterChange = (newFilters) => {
    setFilters(newFilters);
    setPage(1); // Reset to first page when filters change
  };
  
  return (
    <Container className="search-container py-4">
      <h1 className="mb-4">Search Articles</h1>
      
      <Row className="mb-4">
        <Col>
          <Form onSubmit={handleSearch}>
            <Form.Group className="d-flex">
              <Form.Control
                type="text"
                placeholder="Search for articles..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="me-2"
              />
              <Button variant="primary" type="submit" disabled={loading}>
                {loading ? <Spinner animation="border" size="sm" /> : 'Search'}
              </Button>
            </Form.Group>
          </Form>
        </Col>
      </Row>
      
      <Row>
        <Col md={3}>
          <SearchFilters onChange={handleFilterChange} />
        </Col>
        <Col md={9}>
          {loading ? (
            <div className="text-center p-5">
              <Spinner animation="border" />
              <p className="mt-2">Searching articles...</p>
            </div>
          ) : (
            <>
              <p className="search-results-count">
                {totalResults > 0 ? `Found ${totalResults} results` : 'No results found'}
              </p>
              <ArticleList 
                articles={results} 
                totalResults={totalResults}
                page={page}
                onPageChange={setPage}
              />
            </>
          )}
        </Col>
      </Row>
    </Container>
  );
};

export default Search;
```

## 2. System Monitoring Dashboard (Week 2-3)

### Tasks:
- [ ] Implement Prometheus for metrics collection
- [ ] Set up Grafana for visualization and alerting
- [ ] Create custom metrics for crawler performance
- [ ] Develop alerting rules for system health

### Implementation Details:
```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: newscrawler-prometheus
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    ports:
      - "9090:9090"
    networks:
      - monitoring-network
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: newscrawler-grafana
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin_password
      - GF_USERS_ALLOW_SIGN_UP=false
    ports:
      - "3000:3000"
    networks:
      - monitoring-network
    depends_on:
      - prometheus
    restart: unless-stopped

  node-exporter:
    image: prom/node-exporter:latest
    container_name: newscrawler-node-exporter
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    ports:
      - "9100:9100"
    networks:
      - monitoring-network
    restart: unless-stopped

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    container_name: newscrawler-cadvisor
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    ports:
      - "8080:8080"
    networks:
      - monitoring-network
    restart: unless-stopped

networks:
  monitoring-network:
    driver: bridge

volumes:
  prometheus_data:
  grafana_data:
```

```yaml
# monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          # - alertmanager:9093

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'api'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['api:8000']

  - job_name: 'crawler'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['crawler:8001']

  - job_name: 'vector_processor'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['vector_processor:8002']
```

```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Gauge, Histogram, Summary
import time
import functools

# Define metrics
ARTICLES_SCRAPED = Counter(
    'articles_scraped_total', 
    'Total number of articles scraped',
    ['source_domain', 'status']
)

SCRAPING_DURATION = Histogram(
    'scraping_duration_seconds', 
    'Time spent scraping articles',
    ['source_domain'],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600)
)

ACTIVE_CRAWLS = Gauge(
    'active_crawls', 
    'Number of currently active crawling tasks',
    ['source_domain']
)

EMBEDDING_GENERATION_TIME = Histogram(
    'embedding_generation_seconds', 
    'Time spent generating embeddings',
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
)

EMBEDDING_BATCH_SIZE = Gauge(
    'embedding_batch_size', 
    'Current batch size for embedding generation'
)

API_REQUEST_DURATION = Histogram(
    'api_request_duration_seconds', 
    'API request duration in seconds',
    ['endpoint', 'method', 'status_code'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0)
)

API_REQUESTS_TOTAL = Counter(
    'api_requests_total', 
    'Total count of API requests',
    ['endpoint', 'method', 'status_code']
)

CACHE_HITS = Counter(
    'cache_hits_total', 
    'Total number of cache hits',
    ['cache_type']
)

CACHE_MISSES = Counter(
    'cache_misses_total', 
    'Total number of cache misses',
    ['cache_type']
)

DATABASE_QUERY_DURATION = Histogram(
    'database_query_duration_seconds', 
    'Database query duration in seconds',
    ['query_type'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)
)

def track_time(metric, labels=None):
    """Decorator to track execution time of a function with a Prometheus metric."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            if labels:
                label_values = {k: v(*args, **kwargs) if callable(v) else v for k, v in labels.items()}
                metric.labels(**label_values).observe(duration)
            else:
                metric.observe(duration)
                
            return result
        return wrapper
    return decorator

def track_scraping(source_domain):
    """Context manager to track scraping metrics."""
    class ScrapingTracker:
        def __init__(self, domain):
            self.domain = domain
            
        def __enter__(self):
            ACTIVE_CRAWLS.labels(source_domain=self.domain).inc()
            self.start_time = time.time()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.start_time
            SCRAPING_DURATION.labels(source_domain=self.domain).observe(duration)
            ACTIVE_CRAWLS.labels(source_domain=self.domain).dec()
            
            status = 'error' if exc_type else 'success'
            ARTICLES_SCRAPED.labels(source_domain=self.domain, status=status).inc()
            
    return ScrapingTracker(source_domain)
```

```python
# src/api/middleware.py
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.monitoring.metrics import API_REQUEST_DURATION, API_REQUESTS_TOTAL

class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to collect Prometheus metrics for API requests."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process the request
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        
        # Extract endpoint and method
        endpoint = request.url.path
        method = request.method
        status_code = str(response.status_code)
        
        # Record request duration
        API_REQUEST_DURATION.labels(
            endpoint=endpoint,
            method=method,
            status_code=status_code
        ).observe(duration)
        
        # Increment request counter
        API_REQUESTS_TOTAL.labels(
            endpoint=endpoint,
            method=method,
            status_code=status_code
        ).inc()
        
        return response
```

## 3. Performance Monitoring and Alerting (Week 3)

### Tasks:
- [ ] Implement health check endpoints for all services
- [ ] Create alerting rules for critical system conditions
- [ ] Set up notification channels (email, Slack)
- [ ] Develop performance dashboards for key metrics

### Implementation Details:
```yaml
# monitoring/prometheus/rules/alerts.yml
groups:
  - name: newscrawler_alerts
    rules:
      # System-level alerts
      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 80% for 5 minutes on {{ $labels.instance }}"

      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 85% for 5 minutes on {{ $labels.instance }}"

      - alert: DiskSpaceRunningOut
        expr: (node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes * 100 > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Disk space running out"
          description: "Disk usage is above 85% for 5 minutes on {{ $labels.instance }}"

      # Application-specific alerts
      - alert: HighErrorRate
        expr: sum(rate(api_requests_total{status_code=~"5.."}[5m])) / sum(rate(api_requests_total[5m])) * 100 > 5
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High API error rate"
          description: "Error rate is above 5% for 2 minutes"

      - alert: SlowAPIResponses
        expr: histogram_quantile(0.95, sum(rate(api_request_duration_seconds_bucket[5m])) by (le)) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow API responses"
          description: "95th percentile of API response time is above 1 second for 5 minutes"

      - alert: CrawlerFailures
        expr: sum(rate(articles_scraped_total{status="error"}[15m])) / sum(rate(articles_scraped_total[15m])) * 100 > 20
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "High crawler failure rate"
          description: "Crawler failure rate is above 20% for 15 minutes"

      - alert: NoArticlesScraped
        expr: sum(increase(articles_scraped_total{status="success"}[1h])) == 0
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "No articles scraped"
          description: "No articles have been successfully scraped in the last hour"

      - alert: SlowEmbeddingGeneration
        expr: histogram_quantile(0.95, sum(rate(embedding_generation_seconds_bucket[15m])) by (le)) > 5
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Slow embedding generation"
          description: "95th percentile of embedding generation time is above 5 seconds for 15 minutes"
```

```python
# src/api/routers/health.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import psutil
import os
import redis
import time
from typing import Dict, Any

from src.database.session import get_db
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/")
async def health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Perform a basic health check of the API and its dependencies.
    
    Returns:
        Dict with health status information
    """
    health_data = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": os.getenv("APP_VERSION", "unknown"),
        "components": {}
    }
    
    # Check database connection
    try:
        db.execute("SELECT 1").fetchall()
        health_data["components"]["database"] = {"status": "healthy"}
    except Exception as e:
        health_data["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_data["status"] = "degraded"
    
    # Check Redis connection if used
    try:
        redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        redis_client.ping()
        health_data["components"]["redis"] = {"status": "healthy"}
    except Exception as e:
        health_data["components"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_data["status"] = "degraded"
    
    # Check system resources
    health_data["components"]["system"] = {
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent
    }
    
    # If any component is unhealthy, return a 503 status code
    if health_data["status"] != "healthy":
        raise HTTPException(status_code=503, detail=health_data)
    
    return health_data

@router.get("/metrics")
async def metrics():
    """
    Expose Prometheus metrics.
    
    Returns:
        Prometheus metrics in text format
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@router.get("/readiness")
async def readiness_check(db: Session = Depends(get_db)) -> Dict[str, str]:
    """
    Check if the service is ready to accept traffic.
    
    Returns:
        Dict with readiness status
    """
    # Check database connection
    try:
        db.execute("SELECT 1").fetchall()
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail={"status": "not ready", "error": str(e)})

@router.get("/liveness")
async def liveness_check() -> Dict[str, str]:
    """
    Check if the service is alive.
    
    Returns:
        Dict with liveness status
    """
    return {"status": "alive"}
```

## Integration Plan

1. Set up the React frontend project structure
2. Implement the dashboard and search components
3. Configure Prometheus and Grafana for monitoring
4. Integrate metrics collection into all services

## Testing Strategy

1. Develop end-to-end tests for the web interface
2. Test monitoring system with simulated failure scenarios
3. Verify alerting functionality with test alerts
4. Conduct usability testing of the dashboard interface 