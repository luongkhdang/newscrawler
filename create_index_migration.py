"""
Script to create a new Alembic migration for adding database indexes.
"""

import os
import sys
import datetime
import subprocess

# Get the current timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# Create the migration file name
migration_name = f"add_database_indexes_{timestamp}"

# Run the alembic revision command
try:
    subprocess.run(
        [
            "alembic", 
            "revision", 
            "--autogenerate", 
            "-m", 
            "Add database indexes for performance optimization"
        ],
        check=True
    )
    print(f"Successfully created migration: {migration_name}")
except subprocess.CalledProcessError as e:
    print(f"Error creating migration: {e}")
    sys.exit(1)

print("Migration file created. Please review the migration file before applying it.")
print("To apply the migration, run: alembic upgrade head") 