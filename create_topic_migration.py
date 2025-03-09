"""
Script to create a migration for adding topic classification and relevance fields.
"""

import os
import sys
from alembic import command
from alembic.config import Config

def create_migration(message):
    """Create a new migration with the given message."""
    # Get the directory of this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create an Alembic configuration
    alembic_cfg = Config(os.path.join(base_dir, "alembic.ini"))
    
    # Create a new migration
    command.revision(alembic_cfg, message=message, autogenerate=True)
    
    print(f"Migration created with message: {message}")

if __name__ == "__main__":
    message = "Add topic classification and relevance fields"
    create_migration(message) 