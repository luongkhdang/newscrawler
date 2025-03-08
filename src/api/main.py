from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
from typing import List, Optional

from src.api.routers import articles, sources, crawl, search, health
from src.database.session import get_db, init_db

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

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(articles.router, prefix="/articles", tags=["articles"])
app.include_router(sources.router, prefix="/sources", tags=["sources"])
app.include_router(crawl.router, prefix="/crawl", tags=["crawl"])
app.include_router(search.router, prefix="/search", tags=["search"])

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the API")
    init_db()

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