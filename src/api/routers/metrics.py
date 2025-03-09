"""
API endpoints for metrics and monitoring.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from src.utils.metrics import get_all_metrics
from src.utils.cache import vector_cache, search_cache
from src.llm.response_cache import get_rag_cache

router = APIRouter(prefix="/metrics", tags=["metrics"])


class MetricsResponse(BaseModel):
    """Response model for metrics."""
    metrics: Dict[str, Dict[str, Any]]
    cache_stats: Dict[str, Dict[str, Any]]
    system_info: Dict[str, Any]


@router.get("", response_model=MetricsResponse)
async def get_metrics():
    """Get all metrics."""
    try:
        # Get metrics
        metrics = get_all_metrics()
        
        # Get cache stats
        cache_stats = {
            "vector_cache": vector_cache.stats(),
            "search_cache": search_cache.stats(),
            "rag_cache": get_rag_cache().stats()
        }
        
        # Get system info
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        system_info = {
            "cpu_percent": process.cpu_percent(),
            "memory_usage": {
                "rss": memory_info.rss / (1024 * 1024),  # MB
                "vms": memory_info.vms / (1024 * 1024),  # MB
            },
            "threads": process.num_threads(),
            "open_files": len(process.open_files()),
            "connections": len(process.connections()),
            "uptime": process.create_time(),
            "system": {
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory": {
                    "total": psutil.virtual_memory().total / (1024 * 1024 * 1024),  # GB
                    "available": psutil.virtual_memory().available / (1024 * 1024 * 1024),  # GB
                    "percent": psutil.virtual_memory().percent
                },
                "disk": {
                    "total": psutil.disk_usage('/').total / (1024 * 1024 * 1024),  # GB
                    "free": psutil.disk_usage('/').free / (1024 * 1024 * 1024),  # GB
                    "percent": psutil.disk_usage('/').percent
                }
            }
        }
        
        return {
            "metrics": metrics,
            "cache_stats": cache_stats,
            "system_info": system_info
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")


@router.get("/prometheus")
async def get_prometheus_metrics():
    """Get metrics in Prometheus format."""
    try:
        # Get metrics
        metrics = get_all_metrics()
        
        # Convert to Prometheus format
        lines = []
        
        for name, data in metrics.items():
            metric_type = data.get("type", "unknown")
            description = data.get("description", "")
            
            # Add metric type comment
            lines.append(f"# HELP {name} {description}")
            
            if metric_type == "Counter":
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {data.get('value', 0)}")
            
            elif metric_type == "Gauge":
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {data.get('value', 0)}")
            
            elif metric_type == "Histogram":
                lines.append(f"# TYPE {name} histogram")
                
                # Add sum
                lines.append(f"{name}_sum {data.get('sum', 0)}")
                
                # Add count
                lines.append(f"{name}_count {data.get('count', 0)}")
                
                # Add buckets
                buckets = data.get("buckets", {})
                for bucket, count in sorted(buckets.items()):
                    if bucket == float('inf'):
                        lines.append(f'{name}_bucket{{le="+Inf"}} {count}')
                    else:
                        lines.append(f'{name}_bucket{{le="{bucket}"}} {count}')
        
        return "\n".join(lines)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting Prometheus metrics: {str(e)}")


@router.post("/reset", status_code=204)
async def reset_metrics():
    """Reset all metrics."""
    try:
        from src.utils.metrics import reset_all_metrics
        reset_all_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting metrics: {str(e)}")


@router.get("/cache")
async def get_cache_stats():
    """Get cache statistics."""
    try:
        return {
            "vector_cache": vector_cache.stats(),
            "search_cache": search_cache.stats(),
            "rag_cache": get_rag_cache().stats()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cache stats: {str(e)}")


@router.post("/cache/clear", status_code=204)
async def clear_caches():
    """Clear all caches."""
    try:
        vector_cache.clear()
        search_cache.clear()
        get_rag_cache().clear()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing caches: {str(e)}")


@router.get("/system")
async def get_system_info():
    """Get system information."""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "process": {
                "pid": process.pid,
                "cpu_percent": process.cpu_percent(),
                "memory_usage": {
                    "rss": memory_info.rss / (1024 * 1024),  # MB
                    "vms": memory_info.vms / (1024 * 1024),  # MB
                },
                "threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "connections": len(process.connections()),
                "uptime": process.create_time()
            },
            "system": {
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory": {
                    "total": psutil.virtual_memory().total / (1024 * 1024 * 1024),  # GB
                    "available": psutil.virtual_memory().available / (1024 * 1024 * 1024),  # GB
                    "percent": psutil.virtual_memory().percent
                },
                "disk": {
                    "total": psutil.disk_usage('/').total / (1024 * 1024 * 1024),  # GB
                    "free": psutil.disk_usage('/').free / (1024 * 1024 * 1024),  # GB
                    "percent": psutil.disk_usage('/').percent
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system info: {str(e)}") 