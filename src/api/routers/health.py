from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import time

from src.database.session import get_db

router = APIRouter()

@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint.
    
    Returns:
        dict: Health status information
    """
    # Check database connection
    db_status = "healthy"
    db_response_time = 0
    try:
        start_time = time.time()
        db.execute("SELECT 1")
        db_response_time = time.time() - start_time
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "ok",
        "timestamp": time.time(),
        "database": {
            "status": db_status,
            "response_time_ms": round(db_response_time * 1000, 2)
        },
        "version": "0.1.0"
    } 