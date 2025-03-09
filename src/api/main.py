from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
from typing import List, Optional

from src.api.routers import articles, sources, crawl, search, health, crawler, metrics
from src.api.routers import llm  # Import the new LLM router
from src.database.session import get_db, init_db
from src.database.optimization import create_indexes, optimize_vector_search
from src.utils.rate_limiter import RateLimitMiddleware, RateLimiter
from src.utils.logging_config import configure_logging
from src.utils.metrics import increment_counter, timed

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="NewsCrawler API",
    description="API for the NewsCrawler system",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
rate_limiter = RateLimiter(
    rate=int(os.getenv("RATE_LIMIT_RATE", "60")),  # 60 requests
    per=int(os.getenv("RATE_LIMIT_PER", "60")),    # per 60 seconds
    burst=int(os.getenv("RATE_LIMIT_BURST", "100"))  # with a burst of 100
)
app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(articles.router, prefix="/articles", tags=["articles"])
app.include_router(sources.router, prefix="/sources", tags=["sources"])
app.include_router(crawl.router, prefix="/crawl", tags=["crawl"])
app.include_router(search.router, prefix="/search", tags=["search"])
app.include_router(llm.router, prefix="/llm", tags=["llm"])  # Include the new LLM router
app.include_router(crawler.router, tags=["crawler"])
app.include_router(metrics.router, tags=["metrics"])

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting NewsCrawler API")
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized")
        
        # Apply database optimizations
        logger.info("Applying database optimizations")
        db = next(get_db())
        try:
            # Create indexes
            create_indexes(db)
            logger.info("Database indexes created")
            
            # Optimize vector search
            optimize_vector_search()
            logger.info("Vector search optimized")
        except Exception as e:
            logger.error(f"Error applying database optimizations: {e}")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
    
    # Initialize metrics
    try:
        metrics.init_metrics()
        logger.info("Metrics initialized")
    except Exception as e:
        logger.error(f"Error initializing metrics: {e}")
    
    # Log configuration
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"Log level: {os.getenv('LOG_LEVEL', 'INFO')}")
    logger.info(f"Rate limit: {rate_limiter.rate} requests per {rate_limiter.per} seconds with burst of {rate_limiter.burst}")
    
    # Log Groq API key status
    if os.getenv("GROQ_API_KEY"):
        logger.info("Groq API key found in environment variables")
    else:
        logger.warning("Groq API key not found in environment variables. LLM features will not work.")
    
    logger.info("NewsCrawler API started successfully")
    increment_counter("api.startup_count", description="Number of API server startups")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the API")

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    return {
        "message": "Welcome to the NewsCrawler API",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
    }

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

@app.middleware("http")
@timed("api.request_duration", description="Duration of API requests")
async def metrics_middleware(request: Request, call_next):
    """Middleware to collect metrics for API requests."""
    # Increment request counter
    increment_counter("api.request_count", description="Total number of API requests")
    
    # Get path for path-specific metrics
    path = request.url.path
    method = request.method
    
    # Increment path-specific counter
    increment_counter(f"api.request.{method}.{path}", description=f"Requests to {method} {path}")
    
    # Process request
    try:
        response = await call_next(request)
        
        # Increment status code counter
        increment_counter(f"api.response.{response.status_code}", description=f"Responses with status code {response.status_code}")
        
        return response
    except Exception as e:
        # Increment error counter
        increment_counter("api.error_count", description="Total number of API errors")
        raise 