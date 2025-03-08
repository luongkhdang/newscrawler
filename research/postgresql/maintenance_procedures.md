# PostgreSQL Database Maintenance Procedures

## Overview
This document outlines recommended maintenance procedures for the PostgreSQL database in the NewsCrawler project. Regular maintenance is essential for ensuring optimal performance, data integrity, and system reliability, especially as the database grows with millions of articles and vector embeddings.

## Maintenance Goals

The primary goals of database maintenance for the NewsCrawler project are:

1. **Performance Optimization**: Ensure queries remain fast as the database grows
2. **Data Integrity**: Prevent corruption and ensure consistency
3. **Storage Efficiency**: Manage disk space and reduce bloat
4. **Availability**: Minimize downtime during maintenance operations
5. **Disaster Recovery**: Prepare for and mitigate potential failures

## Regular Maintenance Tasks

### 1. VACUUM and ANALYZE

PostgreSQL's VACUUM process reclaims storage occupied by dead tuples, while ANALYZE updates statistics used by the query planner.

#### Recommended Schedule

| Database Size | VACUUM ANALYZE | VACUUM FULL |
|---------------|----------------|-------------|
| < 10GB        | Weekly         | Monthly     |
| 10GB - 100GB  | Daily          | Quarterly   |
| > 100GB       | Daily          | Bi-annually |

#### Implementation

**Automated VACUUM (autovacuum)**

Configure PostgreSQL's autovacuum for optimal performance:

```ini
# postgresql.conf
autovacuum = on
autovacuum_vacuum_threshold = 50
autovacuum_analyze_threshold = 50
autovacuum_vacuum_scale_factor = 0.1
autovacuum_analyze_scale_factor = 0.05
autovacuum_vacuum_cost_delay = 20ms
autovacuum_vacuum_cost_limit = 2000
```

**Manual VACUUM**

Create a maintenance script for manual VACUUM operations:

```bash
#!/bin/bash
# maintenance.sh

# Regular VACUUM ANALYZE
psql -U newscrawler -d newscrawler -c "VACUUM ANALYZE articles;"
psql -U newscrawler -d newscrawler -c "VACUUM ANALYZE sources;"
psql -U newscrawler -d newscrawler -c "VACUUM ANALYZE crawl_logs;"

# Log the completion
echo "$(date): VACUUM ANALYZE completed" >> /var/log/postgres_maintenance.log
```

**VACUUM FULL (with downtime)**

For operations requiring downtime, schedule during low-traffic periods:

```bash
#!/bin/bash
# full_vacuum.sh

# Notify monitoring system
curl -X POST "https://monitoring.example.com/api/maintenance/start"

# Perform VACUUM FULL
psql -U newscrawler -d newscrawler -c "VACUUM FULL articles;"
psql -U newscrawler -d newscrawler -c "VACUUM FULL sources;"
psql -U newscrawler -d newscrawler -c "VACUUM FULL crawl_logs;"

# Log the completion
echo "$(date): VACUUM FULL completed" >> /var/log/postgres_maintenance.log

# Notify monitoring system
curl -X POST "https://monitoring.example.com/api/maintenance/end"
```

### 2. Index Maintenance

Indexes can become bloated over time, especially with frequent updates.

#### Recommended Schedule

| Operation | Frequency |
|-----------|-----------|
| Index Statistics Update | Weekly |
| Index Rebuilding | Monthly or when bloat exceeds 30% |
| Vector Index Optimization | Quarterly |

#### Implementation

**Index Statistics**

```bash
#!/bin/bash
# update_index_stats.sh

# Update statistics for all indexes
psql -U newscrawler -d newscrawler -c "ANALYZE;"

# Log the completion
echo "$(date): Index statistics updated" >> /var/log/postgres_maintenance.log
```

**Identify Bloated Indexes**

```sql
-- bloated_indexes.sql
SELECT
    schemaname || '.' || tablename as table_name,
    indexname as index_name,
    pg_size_pretty(pg_relation_size(schemaname || '.' || indexname::text)) as index_size,
    pg_size_pretty(pg_relation_size(schemaname || '.' || tablename::text)) as table_size,
    idx_scan as index_scans
FROM pg_stat_user_indexes
JOIN pg_indexes ON pg_stat_user_indexes.indexrelname = pg_indexes.indexname
ORDER BY pg_relation_size(schemaname || '.' || indexname::text) DESC
LIMIT 20;
```

**Rebuild Bloated Indexes**

```bash
#!/bin/bash
# rebuild_indexes.sh

# Rebuild B-tree indexes concurrently (no downtime)
psql -U newscrawler -d newscrawler -c "REINDEX INDEX CONCURRENTLY idx_articles_published_date;"
psql -U newscrawler -d newscrawler -c "REINDEX INDEX CONCURRENTLY idx_articles_source_domain;"
psql -U newscrawler -d newscrawler -c "REINDEX INDEX CONCURRENTLY idx_crawl_logs_source_id;"

# Log the completion
echo "$(date): Indexes rebuilt" >> /var/log/postgres_maintenance.log
```

**Vector Index Optimization**

Vector indexes (pgvector) require special handling:

```bash
#!/bin/bash
# optimize_vector_indexes.sh

# Backup the database first
pg_dump -U newscrawler -d newscrawler -t articles -f articles_backup.sql

# Drop and recreate the vector index with optimized parameters
psql -U newscrawler -d newscrawler << EOF
DROP INDEX idx_articles_vector_embedding;
CREATE INDEX idx_articles_vector_embedding ON articles 
USING hnsw (vector_embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);
EOF

# Log the completion
echo "$(date): Vector indexes optimized" >> /var/log/postgres_maintenance.log
```

### 3. Database Backups

Regular backups are essential for disaster recovery.

#### Recommended Schedule

| Backup Type | Frequency | Retention |
|-------------|-----------|-----------|
| Full Backup | Daily     | 30 days   |
| WAL Archiving | Continuous | 7 days    |
| Offsite Backup | Weekly   | 90 days   |

#### Implementation

**Full Database Backup**

```bash
#!/bin/bash
# daily_backup.sh

# Set variables
BACKUP_DIR="/var/backups/postgres"
DATE=$(date +%Y%m%d)
BACKUP_FILE="$BACKUP_DIR/newscrawler_$DATE.sql.gz"

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

# Perform the backup
pg_dump -U newscrawler -d newscrawler | gzip > $BACKUP_FILE

# Set permissions
chmod 600 $BACKUP_FILE

# Log the backup
echo "$(date): Full backup completed to $BACKUP_FILE" >> /var/log/postgres_backup.log

# Clean up old backups (keep last 30 days)
find $BACKUP_DIR -name "newscrawler_*.sql.gz" -mtime +30 -delete
```

**WAL Archiving (Point-in-Time Recovery)**

Configure PostgreSQL for WAL archiving:

```ini
# postgresql.conf
wal_level = replica
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/wal_archive/%f'
```

**Offsite Backup**

```bash
#!/bin/bash
# offsite_backup.sh

# Set variables
BACKUP_DIR="/var/backups/postgres"
DATE=$(date +%Y%m%d)
BACKUP_FILE="$BACKUP_DIR/newscrawler_$DATE.sql.gz"
S3_BUCKET="s3://newscrawler-backups"

# Perform the backup
pg_dump -U newscrawler -d newscrawler | gzip > $BACKUP_FILE

# Upload to S3 (or your preferred cloud storage)
aws s3 cp $BACKUP_FILE $S3_BUCKET/

# Log the backup
echo "$(date): Offsite backup completed to $S3_BUCKET/$(basename $BACKUP_FILE)" >> /var/log/postgres_backup.log
```

### 4. Database Monitoring

Proactive monitoring helps identify issues before they become critical.

#### Key Metrics to Monitor

| Metric Category | Specific Metrics |
|-----------------|------------------|
| Performance | Query execution time, Index usage, Cache hit ratio |
| Resource Usage | CPU, Memory, Disk I/O, Disk space |
| Database Size | Table growth, Index size, WAL size |
| Connections | Active connections, Connection time |
| Errors | Failed queries, Deadlocks, Replication errors |

#### Implementation

**Setup PostgreSQL Monitoring**

```bash
# Install monitoring tools
apt-get install prometheus postgresql-prometheus-exporter grafana

# Configure PostgreSQL exporter
cat > /etc/postgresql-exporter/postgresql-exporter.conf << EOF
DATA_SOURCE_NAME="postgresql://postgres_exporter:password@localhost:5432/postgres?sslmode=disable"
EOF

# Start the exporter
systemctl enable postgresql-exporter
systemctl start postgresql-exporter
```

**Custom Monitoring Queries**

Create a monitoring script with custom queries:

```bash
#!/bin/bash
# monitor_database.sh

# Check database size
psql -U newscrawler -d newscrawler -c "
SELECT pg_size_pretty(pg_database_size('newscrawler')) as database_size;
"

# Check table sizes
psql -U newscrawler -d newscrawler -c "
SELECT relname as table_name, 
       pg_size_pretty(pg_total_relation_size(relid)) as total_size,
       pg_size_pretty(pg_relation_size(relid)) as table_size,
       pg_size_pretty(pg_total_relation_size(relid) - pg_relation_size(relid)) as index_size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC
LIMIT 10;
"

# Check index usage
psql -U newscrawler -d newscrawler -c "
SELECT indexrelname as index_name,
       relname as table_name,
       idx_scan as index_scans,
       idx_tup_read as tuples_read,
       idx_tup_fetch as tuples_fetched
FROM pg_stat_user_indexes
JOIN pg_statio_user_tables ON pg_stat_user_indexes.relid = pg_statio_user_tables.relid
ORDER BY idx_scan DESC
LIMIT 10;
"

# Check for slow queries
psql -U newscrawler -d newscrawler -c "
SELECT query, calls, total_time, mean_time, rows
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
"
```

**Alerting**

Set up alerts for critical conditions:

```yaml
# prometheus/alerts.yml
groups:
- name: PostgreSQL
  rules:
  - alert: HighDatabaseLoad
    expr: rate(pg_stat_database_xact_commit{datname="newscrawler"}[5m]) > 1000
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High database load"
      description: "Database is experiencing high load (> 1000 transactions per second)"

  - alert: LowCacheHitRatio
    expr: pg_stat_database_blks_hit{datname="newscrawler"} / (pg_stat_database_blks_hit{datname="newscrawler"} + pg_stat_database_blks_read{datname="newscrawler"}) < 0.9
    for: 15m
    labels:
      severity: warning
    annotations:
      summary: "Low cache hit ratio"
      description: "Database cache hit ratio is below 90%"

  - alert: HighDiskUsage
    expr: node_filesystem_avail_bytes{mountpoint="/var/lib/postgresql"} / node_filesystem_size_bytes{mountpoint="/var/lib/postgresql"} < 0.2
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Low disk space"
      description: "PostgreSQL disk has less than 20% free space"
```

### 5. Query Optimization

Regularly review and optimize slow queries.

#### Implementation

**Identify Slow Queries**

Enable query logging and analysis:

```ini
# postgresql.conf
log_min_duration_statement = 1000  # Log queries taking more than 1 second
shared_preload_libraries = 'pg_stat_statements'
pg_stat_statements.track = all
```

**Query Analysis Script**

```bash
#!/bin/bash
# analyze_queries.sh

# Find slow queries from the past day
psql -U newscrawler -d newscrawler -c "
SELECT query, calls, total_time, mean_time, rows
FROM pg_stat_statements
WHERE mean_time > 1000  -- queries taking more than 1 second on average
ORDER BY total_time DESC
LIMIT 20;
"

# Reset statistics after analysis (optional)
# psql -U newscrawler -d newscrawler -c "SELECT pg_stat_statements_reset();"
```

**Query Optimization Process**

1. Identify slow queries using pg_stat_statements
2. Analyze with EXPLAIN ANALYZE:
   ```sql
   EXPLAIN ANALYZE SELECT * FROM articles WHERE published_date > '2025-01-01';
   ```
3. Optimize through indexing, query rewriting, or schema changes
4. Verify improvement with EXPLAIN ANALYZE again
5. Document optimizations in a query optimization log

## Specialized Maintenance for Vector Data

The pgvector extension requires specialized maintenance procedures.

### 1. Vector Index Tuning

Vector indexes should be periodically tuned based on dataset size and query patterns.

#### IVFFlat Index Tuning

```sql
-- Tune IVFFlat index
DROP INDEX idx_articles_vector_embedding;

-- Calculate optimal lists parameter (approximately sqrt of row count)
CREATE INDEX idx_articles_vector_embedding ON articles 
USING ivfflat (vector_embedding vector_cosine_ops) 
WITH (lists = (SELECT GREATEST(100, FLOOR(SQRT(COUNT(*))) FROM articles)));
```

#### HNSW Index Tuning

```sql
-- Tune HNSW index
DROP INDEX idx_articles_vector_embedding;

-- Adjust parameters based on dataset size
CREATE INDEX idx_articles_vector_embedding ON articles 
USING hnsw (vector_embedding vector_cosine_ops) 
WITH (
    m = 16,                                                -- More connections for larger datasets
    ef_construction = (SELECT GREATEST(64, COUNT(*) / 1000) FROM articles)  -- Scale with dataset size
);
```

### 2. Vector Quality Assessment

Periodically assess the quality of vector embeddings:

```sql
-- Check for null embeddings
SELECT COUNT(*) FROM articles WHERE vector_embedding IS NULL;

-- Check for zero-magnitude embeddings (potential issues)
SELECT COUNT(*) FROM articles WHERE vector_embedding <=> vector_embedding::vector != 0;

-- Sample random vectors to check dimensionality
SELECT 
    array_length(string_to_array(vector_embedding::text, ','), 1) as dimensions,
    COUNT(*) as count
FROM articles
GROUP BY dimensions;
```

### 3. Vector Reindexing

For large-scale changes to vector embeddings:

```bash
#!/bin/bash
# reindex_vectors.sh

# Export articles that need reindexing
psql -U newscrawler -d newscrawler -c "
COPY (
    SELECT id, content 
    FROM articles 
    WHERE vector_embedding IS NULL OR updated_at > vector_updated_at
) TO '/tmp/articles_to_reindex.csv' WITH CSV HEADER;
"

# Process with Python script to generate embeddings
python scripts/generate_embeddings.py --input /tmp/articles_to_reindex.csv --output /tmp/new_embeddings.csv

# Import updated embeddings
psql -U newscrawler -d newscrawler -c "
CREATE TEMP TABLE temp_embeddings (
    id UUID,
    vector_embedding vector(1536)
);

COPY temp_embeddings FROM '/tmp/new_embeddings.csv' WITH CSV HEADER;

UPDATE articles
SET 
    vector_embedding = te.vector_embedding,
    vector_updated_at = NOW()
FROM temp_embeddings te
WHERE articles.id = te.id;
"

# Log completion
echo "$(date): Vector reindexing completed" >> /var/log/postgres_maintenance.log
```

## Maintenance Automation

### 1. Scheduled Maintenance with cron

Set up cron jobs for regular maintenance tasks:

```bash
# /etc/cron.d/postgres-maintenance

# Daily VACUUM ANALYZE (at 2 AM)
0 2 * * * postgres /usr/local/bin/maintenance.sh

# Weekly index statistics update (Sunday at 3 AM)
0 3 * * 0 postgres /usr/local/bin/update_index_stats.sh

# Monthly index rebuilding (1st of month at 4 AM)
0 4 1 * * postgres /usr/local/bin/rebuild_indexes.sh

# Daily backup (at 1 AM)
0 1 * * * postgres /usr/local/bin/daily_backup.sh

# Weekly offsite backup (Saturday at 1 AM)
0 1 * * 6 postgres /usr/local/bin/offsite_backup.sh

# Hourly monitoring
0 * * * * postgres /usr/local/bin/monitor_database.sh
```

### 2. Maintenance Dashboard

Create a maintenance dashboard to track database health:

```python
# app.py (Flask application for maintenance dashboard)
from flask import Flask, render_template
import psycopg2
import json

app = Flask(__name__)

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="newscrawler",
        user="newscrawler",
        password="newscrawler_password"
    )

@app.route('/')
def index():
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Get database size
    cur.execute("SELECT pg_size_pretty(pg_database_size('newscrawler')) as size")
    db_size = cur.fetchone()[0]
    
    # Get table counts
    cur.execute("""
        SELECT 
            (SELECT COUNT(*) FROM articles) as articles_count,
            (SELECT COUNT(*) FROM sources) as sources_count,
            (SELECT COUNT(*) FROM crawl_logs) as logs_count
    """)
    counts = cur.fetchone()
    
    # Get last maintenance times
    cur.execute("""
        SELECT relname, last_vacuum, last_analyze
        FROM pg_stat_user_tables
        WHERE relname IN ('articles', 'sources', 'crawl_logs')
    """)
    maintenance_times = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return render_template(
        'index.html', 
        db_size=db_size, 
        counts=counts, 
        maintenance_times=maintenance_times
    )

if __name__ == '__main__':
    app.run(debug=True)
```

## Disaster Recovery Procedures

### 1. Database Corruption Recovery

In case of database corruption:

```bash
#!/bin/bash
# recover_from_corruption.sh

# Stop PostgreSQL
systemctl stop postgresql

# Check data directory for corruption
pg_resetwal --dry-run /var/lib/postgresql/14/main

# If corruption is detected, restore from backup
pg_ctl stop -D /var/lib/postgresql/14/main
mv /var/lib/postgresql/14/main /var/lib/postgresql/14/main.corrupt
mkdir -p /var/lib/postgresql/14/main
chmod 700 /var/lib/postgresql/14/main
chown postgres:postgres /var/lib/postgresql/14/main

# Restore base backup
gunzip -c /var/backups/postgres/newscrawler_latest.sql.gz | psql -U postgres -d postgres

# Start PostgreSQL
systemctl start postgresql

# Log the recovery
echo "$(date): Recovered from database corruption" >> /var/log/postgres_recovery.log
```

### 2. Point-in-Time Recovery

For recovering to a specific point in time:

```bash
#!/bin/bash
# point_in_time_recovery.sh

# Set recovery target time
RECOVERY_TIME="2025-03-15 14:30:00"

# Stop PostgreSQL
systemctl stop postgresql

# Create recovery.conf
cat > /var/lib/postgresql/14/main/recovery.conf << EOF
restore_command = 'cp /var/lib/postgresql/wal_archive/%f %p'
recovery_target_time = '$RECOVERY_TIME'
EOF

# Start PostgreSQL in recovery mode
systemctl start postgresql

# Monitor recovery progress
tail -f /var/log/postgresql/postgresql-14-main.log

# After recovery completes, reset to normal operation
psql -U postgres -c "SELECT pg_wal_replay_resume();"

# Log the recovery
echo "$(date): Completed point-in-time recovery to $RECOVERY_TIME" >> /var/log/postgres_recovery.log
```

## Maintenance Procedures for Different Environments

### Development Environment

For development environments, focus on:
- Simplified backup procedures (daily backups only)
- More aggressive VACUUM to clean up after testing
- Regular database resets for testing

```bash
#!/bin/bash
# dev_maintenance.sh

# VACUUM ANALYZE
psql -U newscrawler -d newscrawler -c "VACUUM ANALYZE;"

# Backup development database
pg_dump -U newscrawler -d newscrawler > /tmp/newscrawler_dev_backup.sql

# Reset test data if needed
# psql -U newscrawler -d newscrawler -c "TRUNCATE articles, crawl_logs;"
# psql -U newscrawler -d newscrawler -c "INSERT INTO articles SELECT * FROM articles_sample;"
```

### Staging Environment

For staging environments, mirror production procedures but with:
- More frequent index rebuilding for testing
- Regular restoration from production backups for testing
- Performance testing with production-like data volumes

```bash
#!/bin/bash
# staging_refresh.sh

# Stop application
systemctl stop newscrawler-api

# Drop and recreate database
psql -U postgres -c "DROP DATABASE IF EXISTS newscrawler_staging;"
psql -U postgres -c "CREATE DATABASE newscrawler_staging OWNER newscrawler;"

# Restore from production backup
gunzip -c /var/backups/postgres/newscrawler_production_latest.sql.gz | psql -U newscrawler -d newscrawler_staging

# Anonymize sensitive data if needed
psql -U newscrawler -d newscrawler_staging -c "UPDATE users SET email = 'user' || id || '@example.com';"

# Start application
systemctl start newscrawler-api

# Log the refresh
echo "$(date): Staging environment refreshed from production" >> /var/log/postgres_maintenance.log
```

### Production Environment

For production, prioritize:
- Zero-downtime maintenance whenever possible
- Comprehensive backup strategy
- Careful scheduling of maintenance tasks during low-traffic periods
- Thorough testing of all procedures in staging first

## Recommended Maintenance Schedule for NewsCrawler

Based on our analysis, we recommend the following maintenance schedule for the NewsCrawler project:

| Task | Frequency | Environment | Downtime Required |
|------|-----------|-------------|-------------------|
| VACUUM ANALYZE | Daily | All | No |
| Full Backup | Daily | Staging, Production | No |
| Index Statistics Update | Weekly | All | No |
| Offsite Backup | Weekly | Production | No |
| Index Rebuilding | Monthly | All | No (with CONCURRENTLY) |
| VACUUM FULL | Quarterly | All | Yes |
| Vector Index Optimization | Quarterly | All | Yes |
| Database Monitoring | Continuous | All | No |
| Query Performance Review | Monthly | All | No |

## Implementation Plan

To implement these maintenance procedures:

1. **Setup Phase (Week 1)**
   - Install monitoring tools
   - Configure PostgreSQL parameters
   - Create maintenance scripts
   - Set up backup procedures

2. **Testing Phase (Week 2)**
   - Test all maintenance scripts in development
   - Verify backup and restore procedures
   - Test monitoring and alerting

3. **Deployment Phase (Week 3)**
   - Deploy maintenance scripts to all environments
   - Set up cron jobs for automation
   - Document all procedures

4. **Verification Phase (Week 4)**
   - Verify all maintenance tasks are running correctly
   - Test recovery procedures
   - Train team members on maintenance procedures

## Conclusion

A comprehensive database maintenance strategy is essential for the long-term health and performance of the NewsCrawler system. By implementing the procedures outlined in this document, the team can ensure:

1. Optimal query performance even as the database grows to millions of articles
2. Data integrity and protection against corruption or loss
3. Efficient use of storage resources
4. Minimal downtime for maintenance operations
5. Quick recovery from potential failures

Regular maintenance should be viewed as an essential part of the system's operation, not an afterthought. By automating most maintenance tasks and establishing clear procedures for manual interventions, the team can maintain a healthy and performant database with minimal effort. 