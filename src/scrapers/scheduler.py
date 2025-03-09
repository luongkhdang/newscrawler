"""
Scheduler for the crawler system.
This module provides functionality for scheduling crawl jobs with priority and monitoring.
"""

import logging
import time
import threading
import heapq
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import os
import signal
from sqlalchemy.orm import Session

from src.database.session import SessionLocal
from src.database.models import Source, CrawlLog
from src.scrapers.scraper_factory import ScraperFactory
from src.models.article import Article

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status of a crawl job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class JobPriority(int, Enum):
    """Priority levels for crawl jobs."""
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass(order=True)
class CrawlJob:
    """
    Represents a crawl job in the scheduler.
    """
    priority: int
    created_at: float = field(compare=False)
    job_id: str = field(compare=False)
    source_id: str = field(compare=False)
    source_name: str = field(compare=False)
    source_url: str = field(compare=False)
    scraper_type: str = field(compare=False)
    status: JobStatus = field(compare=False, default=JobStatus.PENDING)
    scheduled_time: Optional[float] = field(compare=False, default=None)
    start_time: Optional[float] = field(compare=False, default=None)
    end_time: Optional[float] = field(compare=False, default=None)
    error_message: Optional[str] = field(compare=False, default=None)
    articles_found: int = field(compare=False, default=0)
    articles_added: int = field(compare=False, default=0)
    articles_updated: int = field(compare=False, default=0)
    max_urls: int = field(compare=False, default=100)
    respect_robots_txt: bool = field(compare=False, default=True)
    crawl_delay: int = field(compare=False, default=1)  # seconds between requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "job_id": self.job_id,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "scraper_type": self.scraper_type,
            "priority": self.priority,
            "status": self.status,
            "created_at": self.created_at,
            "scheduled_time": self.scheduled_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error_message": self.error_message,
            "articles_found": self.articles_found,
            "articles_added": self.articles_added,
            "articles_updated": self.articles_updated,
            "max_urls": self.max_urls,
            "respect_robots_txt": self.respect_robots_txt,
            "crawl_delay": self.crawl_delay
        }


class CrawlerScheduler:
    """
    Scheduler for the crawler system.
    Manages crawl jobs, scheduling, and execution.
    """
    
    def __init__(
        self,
        max_workers: int = 3,
        poll_interval: int = 5,
        job_timeout: int = 3600,
        state_file: str = "crawler_state.json"
    ):
        """
        Initialize the crawler scheduler.
        
        Args:
            max_workers: Maximum number of concurrent crawl jobs
            poll_interval: Interval in seconds to check for new jobs
            job_timeout: Maximum time in seconds for a job to complete
            state_file: File to save scheduler state
        """
        self.max_workers = max_workers
        self.poll_interval = poll_interval
        self.job_timeout = job_timeout
        self.state_file = state_file
        
        # Job queues and tracking
        self.job_queue = []  # Priority queue of pending jobs
        self.running_jobs: Dict[str, Tuple[CrawlJob, threading.Thread]] = {}
        self.completed_jobs: Dict[str, CrawlJob] = {}
        self.failed_jobs: Dict[str, CrawlJob] = {}
        
        # Scheduler state
        self.is_running = False
        self.scheduler_thread = None
        self.lock = threading.RLock()
        
        # Load state if available
        self._load_state()
    
    def start(self):
        """Start the scheduler."""
        with self.lock:
            if self.is_running:
                logger.warning("Scheduler is already running")
                return
            
            self.is_running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()
            logger.info("Crawler scheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        with self.lock:
            if not self.is_running:
                logger.warning("Scheduler is not running")
                return
            
            self.is_running = False
            
            # Wait for scheduler thread to finish
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=10)
            
            # Cancel all running jobs
            for job_id, (job, thread) in list(self.running_jobs.items()):
                self._cancel_job(job_id)
            
            # Save state
            self._save_state()
            
            logger.info("Crawler scheduler stopped")
    
    def schedule_job(
        self,
        source_id: str,
        source_name: str,
        source_url: str,
        scraper_type: str,
        priority: JobPriority = JobPriority.MEDIUM,
        scheduled_time: Optional[datetime] = None,
        max_urls: int = 100,
        respect_robots_txt: bool = True,
        crawl_delay: int = 1
    ) -> str:
        """
        Schedule a new crawl job.
        
        Args:
            source_id: ID of the source to crawl
            source_name: Name of the source
            source_url: URL of the source
            scraper_type: Type of scraper to use
            priority: Priority of the job
            scheduled_time: Time to schedule the job for (None for immediate)
            max_urls: Maximum number of URLs to crawl
            respect_robots_txt: Whether to respect robots.txt
            crawl_delay: Delay between requests in seconds
            
        Returns:
            Job ID
        """
        with self.lock:
            job_id = str(uuid.uuid4())
            scheduled_timestamp = scheduled_time.timestamp() if scheduled_time else None
            
            job = CrawlJob(
                priority=priority.value,
                created_at=time.time(),
                job_id=job_id,
                source_id=source_id,
                source_name=source_name,
                source_url=source_url,
                scraper_type=scraper_type,
                scheduled_time=scheduled_timestamp,
                max_urls=max_urls,
                respect_robots_txt=respect_robots_txt,
                crawl_delay=crawl_delay
            )
            
            heapq.heappush(self.job_queue, job)
            logger.info(f"Scheduled job {job_id} for source {source_name} with priority {priority.name}")
            
            # Save state
            self._save_state()
            
            return job_id
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.
        
        Args:
            job_id: ID of the job to cancel
            
        Returns:
            True if the job was canceled, False otherwise
        """
        with self.lock:
            return self._cancel_job(job_id)
    
    def _cancel_job(self, job_id: str) -> bool:
        """
        Internal method to cancel a job.
        
        Args:
            job_id: ID of the job to cancel
            
        Returns:
            True if the job was canceled, False otherwise
        """
        # Check if job is running
        if job_id in self.running_jobs:
            job, thread = self.running_jobs[job_id]
            job.status = JobStatus.CANCELED
            
            # Thread will notice the status change and exit
            logger.info(f"Canceled running job {job_id}")
            
            # Move to completed jobs
            self.completed_jobs[job_id] = job
            del self.running_jobs[job_id]
            
            return True
        
        # Check if job is in queue
        for i, job in enumerate(self.job_queue):
            if job.job_id == job_id:
                job.status = JobStatus.CANCELED
                self.completed_jobs[job_id] = job
                
                # Remove from queue (will be rebuilt on next operation)
                self.job_queue[i] = self.job_queue[-1]
                self.job_queue.pop()
                heapq.heapify(self.job_queue)
                
                logger.info(f"Canceled pending job {job_id}")
                return True
        
        logger.warning(f"Job {job_id} not found for cancellation")
        return False
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a job.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Job status dictionary or None if not found
        """
        with self.lock:
            # Check running jobs
            if job_id in self.running_jobs:
                job, _ = self.running_jobs[job_id]
                return job.to_dict()
            
            # Check completed jobs
            if job_id in self.completed_jobs:
                return self.completed_jobs[job_id].to_dict()
            
            # Check failed jobs
            if job_id in self.failed_jobs:
                return self.failed_jobs[job_id].to_dict()
            
            # Check pending jobs
            for job in self.job_queue:
                if job.job_id == job_id:
                    return job.to_dict()
            
            return None
    
    def get_all_jobs(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all jobs grouped by status.
        
        Returns:
            Dictionary with jobs grouped by status
        """
        with self.lock:
            result = {
                "pending": [job.to_dict() for job in self.job_queue],
                "running": [job.to_dict() for job, _ in self.running_jobs.values()],
                "completed": list(self.completed_jobs.values()),
                "failed": list(self.failed_jobs.values())
            }
            
            return result
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.is_running:
            try:
                self._process_jobs()
                self._check_running_jobs()
                self._schedule_recurring_jobs()
                
                # Save state periodically
                self._save_state()
                
                # Sleep for poll interval
                time.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
    
    def _process_jobs(self):
        """Process pending jobs."""
        with self.lock:
            # Check if we can start more jobs
            while len(self.running_jobs) < self.max_workers and self.job_queue:
                # Get the highest priority job
                job = heapq.heappop(self.job_queue)
                
                # Check if job is scheduled for the future
                current_time = time.time()
                if job.scheduled_time and job.scheduled_time > current_time:
                    # Put it back in the queue
                    heapq.heappush(self.job_queue, job)
                    break
                
                # Start the job
                self._start_job(job)
    
    def _start_job(self, job: CrawlJob):
        """
        Start a crawl job.
        
        Args:
            job: The job to start
        """
        job.status = JobStatus.RUNNING
        job.start_time = time.time()
        
        # Create a thread for the job
        thread = threading.Thread(
            target=self._run_job,
            args=(job,),
            daemon=True
        )
        
        # Add to running jobs
        self.running_jobs[job.job_id] = (job, thread)
        
        # Start the thread
        thread.start()
        
        logger.info(f"Started job {job.job_id} for source {job.source_name}")
    
    def _run_job(self, job: CrawlJob):
        """
        Run a crawl job.
        
        Args:
            job: The job to run
        """
        db = SessionLocal()
        try:
            # Create crawl log
            crawl_log = CrawlLog(
                source_id=job.source_id,
                start_time=datetime.fromtimestamp(job.start_time),
                status="in_progress"
            )
            db.add(crawl_log)
            db.commit()
            
            # Create scraper
            scraper_factory = ScraperFactory()
            scraper = scraper_factory.create_scraper(job.scraper_type)
            
            if not scraper:
                raise ValueError(f"No scraper available for type: {job.scraper_type}")
            
            # Run the crawl
            self._crawl_source(job, scraper, db)
            
            # Update job status
            with self.lock:
                job.status = JobStatus.COMPLETED
                job.end_time = time.time()
                
                # Move to completed jobs
                if job.job_id in self.running_jobs:
                    self.completed_jobs[job.job_id] = job
                    del self.running_jobs[job.job_id]
            
            # Update crawl log
            crawl_log.end_time = datetime.fromtimestamp(job.end_time)
            crawl_log.articles_found = job.articles_found
            crawl_log.articles_added = job.articles_added
            crawl_log.articles_updated = job.articles_updated
            crawl_log.status = "completed"
            db.commit()
            
            logger.info(f"Completed job {job.job_id} for source {job.source_name}")
        
        except Exception as e:
            # Update job status
            with self.lock:
                job.status = JobStatus.FAILED
                job.end_time = time.time()
                job.error_message = str(e)
                
                # Move to failed jobs
                if job.job_id in self.running_jobs:
                    self.failed_jobs[job.job_id] = job
                    del self.running_jobs[job.job_id]
            
            # Update crawl log
            try:
                crawl_log.end_time = datetime.fromtimestamp(job.end_time)
                crawl_log.status = "failed"
                crawl_log.error_message = str(e)
                db.commit()
            except Exception as db_error:
                logger.error(f"Error updating crawl log: {db_error}")
            
            logger.error(f"Failed job {job.job_id} for source {job.source_name}: {e}")
        
        finally:
            db.close()
    
    def _crawl_source(self, job: CrawlJob, scraper, db: Session):
        """
        Crawl a source.
        
        Args:
            job: The crawl job
            scraper: The scraper to use
            db: Database session
        """
        # Get source from database
        source = db.query(Source).filter(Source.id == job.source_id).first()
        
        if not source:
            raise ValueError(f"Source not found: {job.source_id}")
        
        # TODO: Implement actual crawling logic
        # This is a placeholder for the actual crawling logic
        # In a real implementation, this would:
        # 1. Get URLs from the source
        # 2. Filter URLs based on robots.txt if enabled
        # 3. Crawl each URL with the appropriate scraper
        # 4. Process and store the results
        
        # Simulate crawling
        job.articles_found = 10
        job.articles_added = 5
        job.articles_updated = 2
        
        # Update source last crawled time
        source.last_crawled = datetime.now()
        db.commit()
    
    def _check_running_jobs(self):
        """Check for timed out or completed jobs."""
        with self.lock:
            current_time = time.time()
            
            for job_id, (job, thread) in list(self.running_jobs.items()):
                # Check if job has timed out
                if job.start_time and (current_time - job.start_time) > self.job_timeout:
                    logger.warning(f"Job {job_id} timed out after {self.job_timeout} seconds")
                    self._cancel_job(job_id)
                
                # Check if thread has completed
                elif not thread.is_alive():
                    # Thread completed but job status wasn't updated
                    if job.status == JobStatus.RUNNING:
                        logger.warning(f"Job {job_id} thread completed but status wasn't updated")
                        job.status = JobStatus.FAILED
                        job.end_time = current_time
                        job.error_message = "Job thread completed unexpectedly"
                        
                        # Move to failed jobs
                        self.failed_jobs[job_id] = job
                        del self.running_jobs[job_id]
    
    def _schedule_recurring_jobs(self):
        """Schedule recurring jobs based on sources in the database."""
        try:
            db = SessionLocal()
            
            # Get sources that need to be crawled
            current_time = datetime.now()
            sources = db.query(Source).filter(
                Source.active == True,
                (Source.last_crawled == None) | 
                (current_time - Source.last_crawled > timedelta(hours=Source.crawl_frequency))
            ).all()
            
            for source in sources:
                # Check if source is already in queue or running
                source_id = str(source.id)
                if self._is_source_scheduled(source_id):
                    continue
                
                # Schedule the job
                self.schedule_job(
                    source_id=source_id,
                    source_name=source.name,
                    source_url=source.base_url,
                    scraper_type=source.scraper_type,
                    priority=JobPriority.MEDIUM
                )
            
            db.close()
        
        except Exception as e:
            logger.error(f"Error scheduling recurring jobs: {e}")
    
    def _is_source_scheduled(self, source_id: str) -> bool:
        """
        Check if a source is already scheduled or running.
        
        Args:
            source_id: ID of the source
            
        Returns:
            True if the source is already scheduled or running
        """
        # Check running jobs
        for job, _ in self.running_jobs.values():
            if job.source_id == source_id:
                return True
        
        # Check pending jobs
        for job in self.job_queue:
            if job.source_id == source_id:
                return True
        
        return False
    
    def _save_state(self):
        """Save scheduler state to file."""
        try:
            state = {
                "pending_jobs": [job.to_dict() for job in self.job_queue],
                "running_jobs": [job.to_dict() for job, _ in self.running_jobs.values()],
                "completed_jobs": [job.to_dict() for job in self.completed_jobs.values()],
                "failed_jobs": [job.to_dict() for job in self.failed_jobs.values()]
            }
            
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving scheduler state: {e}")
    
    def _load_state(self):
        """Load scheduler state from file."""
        if not os.path.exists(self.state_file):
            return
        
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            
            # Load pending jobs
            for job_dict in state.get("pending_jobs", []):
                job = self._dict_to_job(job_dict)
                if job:
                    heapq.heappush(self.job_queue, job)
            
            # Load completed jobs
            for job_dict in state.get("completed_jobs", []):
                job = self._dict_to_job(job_dict)
                if job:
                    self.completed_jobs[job.job_id] = job
            
            # Load failed jobs
            for job_dict in state.get("failed_jobs", []):
                job = self._dict_to_job(job_dict)
                if job:
                    self.failed_jobs[job.job_id] = job
            
            logger.info(f"Loaded scheduler state: {len(self.job_queue)} pending, "
                       f"{len(self.completed_jobs)} completed, {len(self.failed_jobs)} failed")
        
        except Exception as e:
            logger.error(f"Error loading scheduler state: {e}")
    
    def _dict_to_job(self, job_dict: Dict[str, Any]) -> Optional[CrawlJob]:
        """
        Convert a dictionary to a CrawlJob.
        
        Args:
            job_dict: Dictionary representation of a job
            
        Returns:
            CrawlJob instance or None if conversion failed
        """
        try:
            return CrawlJob(
                priority=job_dict["priority"],
                created_at=job_dict["created_at"],
                job_id=job_dict["job_id"],
                source_id=job_dict["source_id"],
                source_name=job_dict["source_name"],
                source_url=job_dict["source_url"],
                scraper_type=job_dict["scraper_type"],
                status=JobStatus(job_dict["status"]),
                scheduled_time=job_dict.get("scheduled_time"),
                start_time=job_dict.get("start_time"),
                end_time=job_dict.get("end_time"),
                error_message=job_dict.get("error_message"),
                articles_found=job_dict.get("articles_found", 0),
                articles_added=job_dict.get("articles_added", 0),
                articles_updated=job_dict.get("articles_updated", 0),
                max_urls=job_dict.get("max_urls", 100),
                respect_robots_txt=job_dict.get("respect_robots_txt", True),
                crawl_delay=job_dict.get("crawl_delay", 1)
            )
        except Exception as e:
            logger.error(f"Error converting job dictionary to CrawlJob: {e}")
            return None


# Singleton instance
_scheduler = None

def get_scheduler() -> CrawlerScheduler:
    """
    Get or create a singleton instance of the CrawlerScheduler.
    
    Returns:
        CrawlerScheduler instance
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = CrawlerScheduler()
    return _scheduler 