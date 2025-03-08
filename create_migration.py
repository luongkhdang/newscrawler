import os
import sys
from alembic import command
from alembic.config import Config

# Get the directory of this script
dir_path = os.path.dirname(os.path.realpath(__file__))

# Create an Alembic configuration object
alembic_cfg = Config(os.path.join(dir_path, "alembic.ini"))

# Create a revision with autogenerate
command.revision(alembic_cfg, "Initial database schema", autogenerate=True)

print("Migration created successfully!") 