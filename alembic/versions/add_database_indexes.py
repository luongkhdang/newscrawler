"""Add database indexes for performance optimization

Revision ID: 7a9d8c5e3b12
Revises: 5a8d7b9e4c21
Create Date: 2025-03-09 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '7a9d8c5e3b12'
down_revision = '5a8d7b9e4c21'
branch_labels = None
depends_on = None


def upgrade():
    # Create indexes for Article table
    op.create_index('ix_articles_published_date', 'articles', ['published_date'], unique=False)
    op.create_index('ix_articles_source_domain', 'articles', ['source_domain'], unique=False)
    op.create_index('ix_articles_category', 'articles', ['category'], unique=False)
    op.create_index('ix_articles_created_at', 'articles', ['created_at'], unique=False)
    op.create_index('ix_articles_updated_at', 'articles', ['updated_at'], unique=False)
    
    # Create partial index for relevant articles
    op.execute("""
        CREATE INDEX ix_articles_relevant ON articles (published_date, relevance_score)
        WHERE is_relevant = true;
    """)
    
    # Create indexes for Source table
    op.create_index('ix_sources_active', 'sources', ['active'], unique=False)
    op.create_index('ix_sources_last_crawled', 'sources', ['last_crawled'], unique=False)
    op.create_index('ix_sources_crawl_frequency', 'sources', ['crawl_frequency'], unique=False)
    
    # Create indexes for CrawlLog table
    op.create_index('ix_crawl_logs_start_time', 'crawl_logs', ['start_time'], unique=False)
    op.create_index('ix_crawl_logs_status', 'crawl_logs', ['status'], unique=False)
    
    # Create composite indexes for common query patterns
    op.create_index('ix_articles_domain_date', 'articles', ['source_domain', 'published_date'], unique=False)
    op.create_index('ix_articles_domain_relevance', 'articles', ['source_domain', 'relevance_score'], unique=False)
    
    # Create HNSW index for vector_embedding column
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_articles_vector_hnsw ON articles 
        USING hnsw (vector_embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)


def downgrade():
    # Drop HNSW index
    op.execute("DROP INDEX IF EXISTS ix_articles_vector_hnsw;")
    
    # Drop composite indexes
    op.drop_index('ix_articles_domain_relevance', table_name='articles')
    op.drop_index('ix_articles_domain_date', table_name='articles')
    
    # Drop CrawlLog indexes
    op.drop_index('ix_crawl_logs_status', table_name='crawl_logs')
    op.drop_index('ix_crawl_logs_start_time', table_name='crawl_logs')
    
    # Drop Source indexes
    op.drop_index('ix_sources_crawl_frequency', table_name='sources')
    op.drop_index('ix_sources_last_crawled', table_name='sources')
    op.drop_index('ix_sources_active', table_name='sources')
    
    # Drop partial index
    op.execute("DROP INDEX IF EXISTS ix_articles_relevant;")
    
    # Drop Article indexes
    op.drop_index('ix_articles_updated_at', table_name='articles')
    op.drop_index('ix_articles_created_at', table_name='articles')
    op.drop_index('ix_articles_category', table_name='articles')
    op.drop_index('ix_articles_source_domain', table_name='articles')
    op.drop_index('ix_articles_published_date', table_name='articles') 