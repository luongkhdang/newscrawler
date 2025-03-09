"""
Script to initialize database optimizations.
This script applies database optimizations such as creating indexes and configuring connection pooling.
"""

import logging
import sys
import os
from sqlalchemy.orm import Session

from src.database.session import SessionLocal
from src.database.optimization import create_indexes, optimize_vector_search

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Initialize database optimizations."""
    logger.info("Starting database optimization initialization")
    
    # Create database session
    db = SessionLocal()
    try:
        # Create indexes
        logger.info("Creating database indexes")
        create_indexes(db)
        
        # Optimize vector search
        logger.info("Optimizing vector search")
        optimize_vector_search()
        
        logger.info("Database optimizations completed successfully")
    except Exception as e:
        logger.error(f"Error initializing database optimizations: {e}")
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    main() 