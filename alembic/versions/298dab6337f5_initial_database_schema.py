"""Initial database schema

Revision ID: 298dab6337f5
Revises: 
Create Date: 2025-03-08 13:29:10.333204

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '298dab6337f5'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Enable uuid-ossp extension
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    
    # Create articles table
    op.create_table(
        'articles',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('url', sa.Text(), nullable=False),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('published_date', sa.DateTime(), nullable=True),
        sa.Column('author', sa.Text(), nullable=True),
        sa.Column('source_domain', sa.Text(), nullable=False),
        sa.Column('category', sa.Text(), nullable=True),
        sa.Column('keywords', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('vector_embedding', Vector(1536), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
        sa.UniqueConstraint('url')
    )
    
    # Create sources table
    op.create_table(
        'sources',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('domain', sa.Text(), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('base_url', sa.Text(), nullable=False),
        sa.Column('scraper_type', sa.Text(), nullable=False),
        sa.Column('active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column('last_crawled', sa.DateTime(), nullable=True),
        sa.Column('crawl_frequency', sa.Integer(), server_default=sa.text('24'), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
        sa.UniqueConstraint('domain')
    )
    
    # Create crawl_logs table
    op.create_table(
        'crawl_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('source_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('sources.id'), nullable=False),
        sa.Column('start_time', sa.DateTime(), nullable=False),
        sa.Column('end_time', sa.DateTime(), nullable=True),
        sa.Column('articles_found', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('articles_added', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('articles_updated', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('status', sa.Text(), server_default=sa.text("'in_progress'"), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False)
    )
    
    # Create indexes
    op.create_index('idx_articles_source_domain', 'articles', ['source_domain'])
    op.create_index('idx_articles_published_date', 'articles', ['published_date'])
    op.create_index('idx_sources_domain', 'sources', ['domain'])
    op.create_index('idx_crawl_logs_source_id', 'crawl_logs', ['source_id'])
    op.create_index('idx_crawl_logs_status', 'crawl_logs', ['status'])
    
    # Create vector index for semantic search
    op.execute('CREATE INDEX idx_articles_vector_embedding ON articles USING ivfflat (vector_embedding vector_cosine_ops) WITH (lists = 100)')


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes
    op.execute('DROP INDEX IF EXISTS idx_articles_vector_embedding')
    op.drop_index('idx_crawl_logs_status')
    op.drop_index('idx_crawl_logs_source_id')
    op.drop_index('idx_sources_domain')
    op.drop_index('idx_articles_published_date')
    op.drop_index('idx_articles_source_domain')
    
    # Drop tables
    op.drop_table('crawl_logs')
    op.drop_table('sources')
    op.drop_table('articles')
    
    # Disable pgvector extension
    op.execute('DROP EXTENSION IF EXISTS vector')
