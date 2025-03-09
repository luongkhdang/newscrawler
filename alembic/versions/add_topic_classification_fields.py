"""Add topic classification and relevance fields

Revision ID: 5a8d7b9e4c21
Revises: 298dab6337f5
Create Date: 2025-03-08 15:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '5a8d7b9e4c21'
down_revision = '298dab6337f5'
branch_labels = None
depends_on = None


def upgrade():
    # Add topic classification and relevance fields to articles table
    op.add_column('articles', sa.Column('topics', sa.ARRAY(sa.Text()), nullable=True))
    op.add_column('articles', sa.Column('entities', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('articles', sa.Column('relevance_score', sa.Float(), nullable=True))
    op.add_column('articles', sa.Column('is_relevant', sa.Boolean(), server_default='true', nullable=False))
    
    # Add authentication and proxy settings to sources table
    op.add_column('sources', sa.Column('requires_auth', sa.Boolean(), server_default='false', nullable=False))
    op.add_column('sources', sa.Column('auth_config', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('sources', sa.Column('proxy_settings', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    
    # Create indexes for efficient querying
    op.create_index(op.f('ix_articles_topics'), 'articles', ['topics'], unique=False)
    op.create_index(op.f('ix_articles_relevance_score'), 'articles', ['relevance_score'], unique=False)
    op.create_index(op.f('ix_articles_is_relevant'), 'articles', ['is_relevant'], unique=False)


def downgrade():
    # Drop indexes
    op.drop_index(op.f('ix_articles_is_relevant'), table_name='articles')
    op.drop_index(op.f('ix_articles_relevance_score'), table_name='articles')
    op.drop_index(op.f('ix_articles_topics'), table_name='articles')
    
    # Drop columns from sources table
    op.drop_column('sources', 'proxy_settings')
    op.drop_column('sources', 'auth_config')
    op.drop_column('sources', 'requires_auth')
    
    # Drop columns from articles table
    op.drop_column('articles', 'is_relevant')
    op.drop_column('articles', 'relevance_score')
    op.drop_column('articles', 'entities')
    op.drop_column('articles', 'topics') 