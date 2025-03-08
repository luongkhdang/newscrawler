# Database Migration Strategies Using Alembic

## Overview
This document analyzes database migration strategies for the NewsCrawler project using Alembic, a lightweight database migration tool for SQLAlchemy. Proper migration management is essential for evolving the database schema while maintaining data integrity and minimizing downtime, especially as the project scales.

## Current State

The NewsCrawler project currently lacks a formal migration system. Schema changes are managed through:
1. Direct SQL scripts in `init-scripts/01-init.sql` for initial setup
2. SQLAlchemy model definitions in `src/database/models.py`
3. Manual schema modifications in development and production

This approach has several limitations:
- No version control for schema changes
- Difficult to track the history of schema modifications
- Risk of data loss during schema updates
- Challenging to coordinate changes across development and production environments
- No automated way to upgrade or downgrade the schema

## Alembic Introduction

Alembic is the recommended migration tool for SQLAlchemy-based applications. It provides:
- Version control for database schemas
- Automated generation of migration scripts
- Support for upgrading and downgrading schemas
- Integration with SQLAlchemy models
- Branching and merging capabilities for complex migration paths

## Implementation Strategy

### 1. Initial Setup

```python
# Install Alembic
pip install alembic

# Initialize Alembic in the project
alembic init migrations
```

### 2. Configure Alembic

Edit `alembic.ini`:
```ini
# alembic.ini
[alembic]
script_location = migrations
prepend_sys_path = .
sqlalchemy.url = postgresql://newscrawler:newscrawler_password@localhost:5432/newscrawler
```

Edit `migrations/env.py`:
```python
# migrations/env.py
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import the SQLAlchemy models
from src.database.models import Base
from src.database.session import engine

# This is the Alembic Config object
config = context.config

# Set the SQLAlchemy URL from the engine
config.set_main_option("sqlalchemy.url", str(engine.url))

# Interpret the config file for Python logging
fileConfig(config.config_file_name)

# Set target metadata
target_metadata = Base.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### 3. Create Initial Migration

Since the database schema already exists, we'll create a "baseline" migration:

```bash
# Create a baseline migration
alembic revision --autogenerate -m "baseline"
```

This will generate a migration script that represents the current state of the database schema.

### 4. Migration Workflow

For future schema changes, follow this workflow:

1. Update SQLAlchemy models in `src/database/models.py`
2. Generate a migration script:
   ```bash
   alembic revision --autogenerate -m "description_of_changes"
   ```
3. Review and edit the generated migration script if necessary
4. Apply the migration:
   ```bash
   alembic upgrade head
   ```

## Migration Strategies for Different Types of Changes

### 1. Adding New Tables

Adding new tables is straightforward with Alembic:

```python
# 1. Add the model to src/database/models.py
class ArticleChunk(Base):
    """Article chunk model for RAG."""
    __tablename__ = "article_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    article_id = Column(UUID(as_uuid=True), ForeignKey("articles.id", ondelete="CASCADE"))
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    vector_embedding = Column(VECTOR(1536), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    article = relationship("Article", back_populates="chunks")

# 2. Update the Article model to include the relationship
class Article(Base):
    # ... existing fields ...
    chunks = relationship("ArticleChunk", back_populates="article", cascade="all, delete-orphan")

# 3. Generate and apply the migration
# $ alembic revision --autogenerate -m "add_article_chunks_table"
# $ alembic upgrade head
```

### 2. Adding Columns to Existing Tables

Adding columns requires careful consideration, especially for non-nullable columns:

```python
# 1. Add the column to the model
class Article(Base):
    # ... existing fields ...
    reading_time_minutes = Column(Integer, nullable=True)

# 2. Generate the migration
# $ alembic revision --autogenerate -m "add_reading_time_column"

# 3. Edit the migration to handle existing data
"""add_reading_time_column

Revision ID: abc123def456
Revises: previous_revision
Create Date: 2025-03-15

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'abc123def456'
down_revision = 'previous_revision'
branch_labels = None
depends_on = None

def upgrade():
    # Add the column as nullable first
    op.add_column('articles', sa.Column('reading_time_minutes', sa.Integer(), nullable=True))
    
    # Update existing rows with calculated values
    op.execute("""
        UPDATE articles 
        SET reading_time_minutes = GREATEST(1, ROUND(LENGTH(content) / 1000))
    """)
    
    # Optional: Make the column non-nullable after populating it
    # op.alter_column('articles', 'reading_time_minutes', nullable=False)

def downgrade():
    op.drop_column('articles', 'reading_time_minutes')
```

### 3. Modifying Columns

Modifying columns can be risky, especially for large tables:

```python
# 1. Update the model
class Article(Base):
    # ... existing fields ...
    # Change content from Text to JSONB for structured storage
    content = Column(JSONB, nullable=False)

# 2. Generate the migration
# $ alembic revision --autogenerate -m "convert_content_to_jsonb"

# 3. Edit the migration for a safe conversion
"""convert_content_to_jsonb

Revision ID: abc123def456
Revises: previous_revision
Create Date: 2025-03-15

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers
revision = 'abc123def456'
down_revision = 'previous_revision'
branch_labels = None
depends_on = None

def upgrade():
    # Add a new column
    op.add_column('articles', sa.Column('content_jsonb', JSONB(), nullable=True))
    
    # Convert data in batches to avoid locking the table for too long
    connection = op.get_bind()
    
    # Get total number of articles
    result = connection.execute("SELECT COUNT(*) FROM articles")
    total_articles = result.scalar()
    
    # Process in batches of 1000
    batch_size = 1000
    for offset in range(0, total_articles, batch_size):
        # Convert text content to JSON
        connection.execute(f"""
            UPDATE articles 
            SET content_jsonb = jsonb_build_object('text', content)
            WHERE id IN (
                SELECT id FROM articles ORDER BY id LIMIT {batch_size} OFFSET {offset}
            )
        """)
    
    # Rename columns
    op.alter_column('articles', 'content', new_column_name='content_text')
    op.alter_column('articles', 'content_jsonb', new_column_name='content')
    
    # Optional: Drop the old column if no longer needed
    # op.drop_column('articles', 'content_text')

def downgrade():
    # Add a new column for text content
    op.add_column('articles', sa.Column('content_text', sa.Text(), nullable=True))
    
    # Convert JSON back to text
    connection = op.get_bind()
    connection.execute("""
        UPDATE articles 
        SET content_text = content->>'text'
    """)
    
    # Rename columns
    op.alter_column('articles', 'content', new_column_name='content_jsonb')
    op.alter_column('articles', 'content_text', new_column_name='content')
    
    # Drop the JSON column
    op.drop_column('articles', 'content_jsonb')
```

### 4. Adding Indexes

Adding indexes is a common operation that can be done without downtime:

```python
# 1. Generate a migration
# $ alembic revision -m "add_full_text_search_index"

# 2. Edit the migration
"""add_full_text_search_index

Revision ID: abc123def456
Revises: previous_revision
Create Date: 2025-03-15

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = 'abc123def456'
down_revision = 'previous_revision'
branch_labels = None
depends_on = None

def upgrade():
    # Add tsvector column
    op.add_column('articles', sa.Column('content_tsvector', postgresql.TSVECTOR))
    
    # Create index concurrently to avoid locking the table
    op.execute(
        "CREATE INDEX CONCURRENTLY idx_articles_content_tsvector ON articles USING GIN (content_tsvector)"
    )
    
    # Create trigger for automatic updates
    op.execute("""
        CREATE TRIGGER tsvector_update_trigger BEFORE INSERT OR UPDATE ON articles
        FOR EACH ROW EXECUTE FUNCTION tsvector_update_trigger(content_tsvector, 'pg_catalog.english', content, title)
    """)
    
    # Populate existing data
    op.execute(
        "UPDATE articles SET content_tsvector = to_tsvector('english', content || ' ' || title)"
    )

def downgrade():
    # Drop trigger
    op.execute("DROP TRIGGER IF EXISTS tsvector_update_trigger ON articles")
    
    # Drop index
    op.execute("DROP INDEX IF EXISTS idx_articles_content_tsvector")
    
    # Drop column
    op.drop_column('articles', 'content_tsvector')
```

### 5. Vector Index Management

Managing vector indexes requires special attention:

```python
# Generate a migration
# $ alembic revision -m "optimize_vector_index"

"""optimize_vector_index

Revision ID: abc123def456
Revises: previous_revision
Create Date: 2025-03-15

"""
from alembic import op

# revision identifiers
revision = 'abc123def456'
down_revision = 'previous_revision'
branch_labels = None
depends_on = None

def upgrade():
    # Drop existing index
    op.execute("DROP INDEX IF EXISTS idx_articles_vector_embedding")
    
    # Create optimized HNSW index
    # Note: This can take a long time for large datasets
    op.execute("""
        CREATE INDEX idx_articles_vector_embedding ON articles 
        USING hnsw (vector_embedding vector_cosine_ops) 
        WITH (m = 16, ef_construction = 64)
    """)

def downgrade():
    # Drop HNSW index
    op.execute("DROP INDEX IF EXISTS idx_articles_vector_embedding")
    
    # Recreate original IVFFlat index
    op.execute("""
        CREATE INDEX idx_articles_vector_embedding ON articles 
        USING ivfflat (vector_embedding vector_cosine_ops) 
        WITH (lists = 100)
    """)
```

## Best Practices for Database Migrations

### 1. Zero-Downtime Migrations

For production environments, zero-downtime migrations are essential:

1. **Use CREATE CONCURRENTLY for indexes**:
   ```sql
   CREATE INDEX CONCURRENTLY idx_name ON table_name (column_name);
   ```

2. **Add columns as nullable first**:
   ```sql
   ALTER TABLE table_name ADD COLUMN column_name data_type NULL;
   -- Update data
   UPDATE table_name SET column_name = default_value;
   -- Then make it non-nullable if needed
   ALTER TABLE table_name ALTER COLUMN column_name SET NOT NULL;
   ```

3. **Process large tables in batches**:
   ```python
   def process_in_batches(connection, table, batch_size=1000):
       """Process a large table in batches to avoid long locks."""
       result = connection.execute(f"SELECT COUNT(*) FROM {table}")
       total_rows = result.scalar()
       
       for offset in range(0, total_rows, batch_size):
           connection.execute(f"""
               UPDATE {table} 
               SET column = new_value
               WHERE id IN (
                   SELECT id FROM {table} ORDER BY id LIMIT {batch_size} OFFSET {offset}
               )
           """)
   ```

### 2. Testing Migrations

Always test migrations in a staging environment before applying to production:

```python
# Create a migration test script
def test_migration(revision):
    """Test a migration by applying it and then rolling it back."""
    # Create a test database
    engine = create_engine("postgresql://user:pass@localhost/test_db")
    
    # Apply migrations up to the previous revision
    alembic_config = Config("alembic.ini")
    command.upgrade(alembic_config, revision.down_revision)
    
    # Apply the migration being tested
    command.upgrade(alembic_config, revision.revision)
    
    # Verify the schema matches expectations
    inspector = inspect(engine)
    # ... perform assertions on the schema ...
    
    # Roll back the migration
    command.downgrade(alembic_config, revision.down_revision)
    
    # Verify the rollback was successful
    # ... perform assertions on the schema ...
```

### 3. Handling Failed Migrations

Have a plan for handling failed migrations:

```python
# Add to your deployment script
try:
    # Apply migration
    subprocess.run(["alembic", "upgrade", "head"], check=True)
except subprocess.CalledProcessError:
    # Roll back to the last known good state
    subprocess.run(["alembic", "downgrade", "-1"], check=True)
    
    # Notify the team
    send_alert("Migration failed, rolled back to previous version")
    
    # Exit with error
    sys.exit(1)
```

### 4. Monitoring Migrations

Monitor the progress and performance of migrations:

```python
# Add to your migration script
def upgrade():
    # Start timing
    start_time = time.time()
    
    # Log the start of the migration
    logger.info(f"Starting migration {revision}")
    
    # Execute the migration
    op.add_column('articles', sa.Column('new_column', sa.String()))
    
    # Log completion and timing
    elapsed = time.time() - start_time
    logger.info(f"Migration {revision} completed in {elapsed:.2f} seconds")
```

## Deployment Strategy

### 1. Development Environment

For development, migrations can be applied directly:

```bash
# Apply all pending migrations
alembic upgrade head

# Roll back the last migration
alembic downgrade -1

# Generate a new migration
alembic revision --autogenerate -m "description"
```

### 2. CI/CD Pipeline

Integrate migrations into your CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
name: Database Migration

on:
  push:
    branches: [ main ]
    paths:
      - 'src/database/models.py'
      - 'migrations/**'

jobs:
  migrate:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Test migrations
      run: |
        # Create test database
        psql -c "CREATE DATABASE test_db" -U postgres
        
        # Run migration tests
        python tests/test_migrations.py
    
    - name: Apply migrations to staging
      if: success()
      run: |
        # Set environment variables for staging
        export DATABASE_URL=postgresql://user:pass@staging-db/newscrawler
        
        # Apply migrations
        alembic upgrade head
```

### 3. Production Deployment

For production, use a more cautious approach:

1. **Create a deployment plan**:
   - Schedule the migration during low-traffic periods
   - Notify stakeholders about potential impact
   - Prepare rollback procedures

2. **Backup the database**:
   ```bash
   pg_dump -U username -d newscrawler -f pre_migration_backup.sql
   ```

3. **Apply migrations with monitoring**:
   ```bash
   # Apply migrations with timing
   time alembic upgrade head
   
   # Verify the migration was successful
   python scripts/verify_migration.py
   ```

4. **Have a rollback plan**:
   ```bash
   # If verification fails, roll back
   alembic downgrade -1
   
   # If alembic rollback fails, restore from backup
   psql -U username -d newscrawler -f pre_migration_backup.sql
   ```

## Recommended Migration Strategy for NewsCrawler

Based on our analysis, we recommend the following migration strategy for the NewsCrawler project:

### 1. Initial Setup

```bash
# Install Alembic
pip install alembic

# Initialize Alembic
alembic init migrations

# Configure Alembic (as described above)

# Create baseline migration
alembic revision --autogenerate -m "baseline"

# Apply baseline migration
alembic upgrade head
```

### 2. Development Workflow

1. Make changes to SQLAlchemy models
2. Generate migration: `alembic revision --autogenerate -m "description"`
3. Review and edit the migration script
4. Test the migration in a development environment
5. Commit the migration script to version control

### 3. Deployment Workflow

1. Run automated tests for migrations in CI
2. Deploy to staging and verify
3. Schedule production deployment during low-traffic period
4. Backup production database
5. Apply migration to production
6. Verify the migration was successful
7. Monitor application performance

### 4. Recommended Migrations

Based on our analysis of the current schema, we recommend the following migrations:

1. **Add Full-Text Search Capabilities**:
   ```bash
   alembic revision -m "add_full_text_search"
   # Edit the migration as described in the "Adding Indexes" section
   ```

2. **Create Article Chunks Table for RAG**:
   ```bash
   alembic revision --autogenerate -m "add_article_chunks_table"
   # Edit the migration as needed
   ```

3. **Optimize Vector Indexes**:
   ```bash
   alembic revision -m "optimize_vector_indexes"
   # Edit the migration as described in the "Vector Index Management" section
   ```

4. **Add Foreign Key Relationship Between Articles and Sources**:
   ```bash
   alembic revision -m "add_source_id_to_articles"
   # Edit the migration to add the column and populate it based on source_domain
   ```

## Conclusion

Implementing a robust database migration strategy with Alembic is essential for the NewsCrawler project as it evolves. By following the best practices outlined in this document, the team can safely make schema changes while minimizing downtime and risk of data loss.

The recommended approach emphasizes:
- Version control for database schema changes
- Zero-downtime migrations for production
- Comprehensive testing before deployment
- Clear rollback procedures for failed migrations
- Batch processing for large tables
- Monitoring and verification of migrations

With these practices in place, the NewsCrawler project can confidently evolve its database schema to support new features and optimizations while maintaining data integrity and system availability. 