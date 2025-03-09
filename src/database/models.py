import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy import Column, String, Text, Boolean, Integer, DateTime, ForeignKey, ARRAY, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from src.database.session import Base

class Article(Base):
    """Article model."""
    __tablename__ = "articles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url = Column(Text, unique=True, nullable=False)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    published_date = Column(DateTime, nullable=True)
    author = Column(Text, nullable=True)
    source_domain = Column(Text, nullable=False)
    category = Column(Text, nullable=True)
    keywords = Column(ARRAY(Text), nullable=True)
    vector_embedding = Column(Vector(1536), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # New fields for topic classification and relevance
    topics = Column(ARRAY(Text), nullable=True)
    entities = Column(JSONB, nullable=True)
    relevance_score = Column(Float, nullable=True)
    is_relevant = Column(Boolean, default=True)

    def __repr__(self):
        return f"<Article(id={self.id}, title={self.title})>"

class Source(Base):
    """Source model."""
    __tablename__ = "sources"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    domain = Column(Text, unique=True, nullable=False)
    name = Column(Text, nullable=False)
    base_url = Column(Text, nullable=False)
    scraper_type = Column(Text, nullable=False)
    active = Column(Boolean, default=True)
    last_crawled = Column(DateTime, nullable=True)
    crawl_frequency = Column(Integer, default=24)  # in hours
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # New fields for authentication and proxy settings
    requires_auth = Column(Boolean, default=False)
    auth_config = Column(JSONB, nullable=True)
    proxy_settings = Column(JSONB, nullable=True)

    crawl_logs = relationship("CrawlLog", back_populates="source")

    def __repr__(self):
        return f"<Source(id={self.id}, domain={self.domain})>"

class CrawlLog(Base):
    """Crawl log model."""
    __tablename__ = "crawl_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(UUID(as_uuid=True), ForeignKey("sources.id"))
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    articles_found = Column(Integer, default=0)
    articles_added = Column(Integer, default=0)
    articles_updated = Column(Integer, default=0)
    status = Column(Text, default="in_progress")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    source = relationship("Source", back_populates="crawl_logs")

    def __repr__(self):
        return f"<CrawlLog(id={self.id}, source_id={self.source_id}, status={self.status})>" 