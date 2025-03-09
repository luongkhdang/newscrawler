"""
API endpoints for managing the crawler.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from src.database.session import get_db
from src.database.models import Source, CrawlLog
from src.scrapers.scheduler import get_scheduler, JobPriority, JobStatus

router = APIRouter(prefix="/crawler", tags=["crawler"])


class SourceResponse(BaseModel):
    """Response model for a source."""
    id: str
    name: str
    base_url: str
    scraper_type: str
    active: bool
    last_crawled: Optional[datetime] = None
    crawl_frequency: int


class SourceListResponse(BaseModel):
    """Response model for a list of sources."""
    items: List[SourceResponse]
    total: int


class JobResponse(BaseModel):
    """Response model for a job."""
    job_id: str
    source_id: str
    source_name: str
    source_url: str
    scraper_type: str
    priority: int
    status: str
    created_at: float
    scheduled_time: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    articles_found: int
    articles_added: int
    articles_updated: int


class JobListResponse(BaseModel):
    """Response model for a list of jobs."""
    pending: List[JobResponse]
    running: List[JobResponse]
    completed: List[JobResponse]
    failed: List[JobResponse]


class ScheduleJobRequest(BaseModel):
    """Request model for scheduling a job."""
    source_id: str = Field(..., description="ID of the source to crawl")
    priority: str = Field("medium", description="Priority of the job (high, medium, low)")
    max_urls: int = Field(100, description="Maximum number of URLs to crawl")
    respect_robots_txt: bool = Field(True, description="Whether to respect robots.txt")
    crawl_delay: int = Field(1, description="Delay between requests in seconds")


class ScheduleJobResponse(BaseModel):
    """Response model for scheduling a job."""
    job_id: str
    source_id: str
    source_name: str
    status: str


class CrawlLogResponse(BaseModel):
    """Response model for a crawl log."""
    id: str
    source_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    articles_found: int
    articles_added: int
    articles_updated: int
    status: str
    error_message: Optional[str] = None


class CrawlLogListResponse(BaseModel):
    """Response model for a list of crawl logs."""
    items: List[CrawlLogResponse]
    total: int


@router.get("/sources", response_model=SourceListResponse)
async def get_sources(
    active: Optional[bool] = Query(None, description="Filter by active status"),
    db: Session = Depends(get_db)
):
    """Get a list of sources."""
    query = db.query(Source)
    
    if active is not None:
        query = query.filter(Source.active == active)
    
    sources = query.all()
    
    return {
        "items": [
            {
                "id": str(source.id),
                "name": source.name,
                "base_url": source.base_url,
                "scraper_type": source.scraper_type,
                "active": source.active,
                "last_crawled": source.last_crawled,
                "crawl_frequency": source.crawl_frequency
            }
            for source in sources
        ],
        "total": len(sources)
    }


@router.get("/sources/{source_id}", response_model=SourceResponse)
async def get_source(
    source_id: str = Path(..., description="ID of the source"),
    db: Session = Depends(get_db)
):
    """Get a source by ID."""
    source = db.query(Source).filter(Source.id == source_id).first()
    
    if not source:
        raise HTTPException(status_code=404, detail=f"Source with ID {source_id} not found")
    
    return {
        "id": str(source.id),
        "name": source.name,
        "base_url": source.base_url,
        "scraper_type": source.scraper_type,
        "active": source.active,
        "last_crawled": source.last_crawled,
        "crawl_frequency": source.crawl_frequency
    }


@router.get("/jobs", response_model=JobListResponse)
async def get_jobs():
    """Get all jobs."""
    scheduler = get_scheduler()
    jobs = scheduler.get_all_jobs()
    
    return jobs


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str = Path(..., description="ID of the job")
):
    """Get a job by ID."""
    scheduler = get_scheduler()
    job = scheduler.get_job_status(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
    
    return job


@router.post("/jobs", response_model=ScheduleJobResponse)
async def schedule_job(
    request: ScheduleJobRequest,
    db: Session = Depends(get_db)
):
    """Schedule a new crawl job."""
    # Get source from database
    source = db.query(Source).filter(Source.id == request.source_id).first()
    
    if not source:
        raise HTTPException(status_code=404, detail=f"Source with ID {request.source_id} not found")
    
    # Map priority string to enum
    priority_map = {
        "high": JobPriority.HIGH,
        "medium": JobPriority.MEDIUM,
        "low": JobPriority.LOW
    }
    
    if request.priority not in priority_map:
        raise HTTPException(status_code=400, detail=f"Invalid priority: {request.priority}")
    
    # Schedule the job
    scheduler = get_scheduler()
    job_id = scheduler.schedule_job(
        source_id=str(source.id),
        source_name=source.name,
        source_url=source.base_url,
        scraper_type=source.scraper_type,
        priority=priority_map[request.priority],
        max_urls=request.max_urls,
        respect_robots_txt=request.respect_robots_txt,
        crawl_delay=request.crawl_delay
    )
    
    return {
        "job_id": job_id,
        "source_id": str(source.id),
        "source_name": source.name,
        "status": "pending"
    }


@router.delete("/jobs/{job_id}", status_code=204)
async def cancel_job(
    job_id: str = Path(..., description="ID of the job to cancel")
):
    """Cancel a job."""
    scheduler = get_scheduler()
    success = scheduler.cancel_job(job_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found or could not be canceled")


@router.get("/logs", response_model=CrawlLogListResponse)
async def get_crawl_logs(
    source_id: Optional[str] = Query(None, description="Filter by source ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of logs to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db)
):
    """Get crawl logs."""
    query = db.query(CrawlLog)
    
    if source_id:
        query = query.filter(CrawlLog.source_id == source_id)
    
    if status:
        query = query.filter(CrawlLog.status == status)
    
    total = query.count()
    logs = query.order_by(CrawlLog.start_time.desc()).offset(offset).limit(limit).all()
    
    return {
        "items": [
            {
                "id": str(log.id),
                "source_id": str(log.source_id),
                "start_time": log.start_time,
                "end_time": log.end_time,
                "articles_found": log.articles_found,
                "articles_added": log.articles_added,
                "articles_updated": log.articles_updated,
                "status": log.status,
                "error_message": log.error_message
            }
            for log in logs
        ],
        "total": total
    }


@router.get("/logs/{log_id}", response_model=CrawlLogResponse)
async def get_crawl_log(
    log_id: str = Path(..., description="ID of the crawl log"),
    db: Session = Depends(get_db)
):
    """Get a crawl log by ID."""
    log = db.query(CrawlLog).filter(CrawlLog.id == log_id).first()
    
    if not log:
        raise HTTPException(status_code=404, detail=f"Crawl log with ID {log_id} not found")
    
    return {
        "id": str(log.id),
        "source_id": str(log.source_id),
        "start_time": log.start_time,
        "end_time": log.end_time,
        "articles_found": log.articles_found,
        "articles_added": log.articles_added,
        "articles_updated": log.articles_updated,
        "status": log.status,
        "error_message": log.error_message
    }


@router.post("/start", status_code=204)
async def start_scheduler():
    """Start the crawler scheduler."""
    scheduler = get_scheduler()
    scheduler.start()


@router.post("/stop", status_code=204)
async def stop_scheduler():
    """Stop the crawler scheduler."""
    scheduler = get_scheduler()
    scheduler.stop() 