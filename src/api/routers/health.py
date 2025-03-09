from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, Any
import logging
import os
import psutil
import time

from src.database.session import get_db
from src.utils.cache import vector_cache, search_cache

router = APIRouter(prefix="/health", tags=["health"])
logger = logging.getLogger(__name__)

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    uptime: float
    memory_usage: Dict[str, Any]
    environment: str
    cache_stats: Dict[str, Any]

start_time = time.time()

@router.get("", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse: Health check information
    """
    # Get memory usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Get cache statistics
    cache_stats = {
        "vector_cache": vector_cache.stats(),
        "search_cache": search_cache.stats()
    }
    
    return {
        "status": "healthy",
        "version": "0.1.0",
        "uptime": time.time() - start_time,
        "memory_usage": {
            "rss": memory_info.rss,
            "vms": memory_info.vms,
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024)
        },
        "environment": os.getenv("ENVIRONMENT", "development"),
        "cache_stats": cache_stats
    } 