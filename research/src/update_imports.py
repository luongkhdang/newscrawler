#!/usr/bin/env python
"""
Script to update imports in the Python files to reflect the new directory structure.
This script modifies the import statements in the Python files to use relative imports
and updates file paths to reflect the new directory structure.
"""

import os
import re
import sys

def update_file(file_path):
    """Update imports and file paths in a Python file."""
    print(f"Updating {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update file paths
    content = re.sub(r'open\([\'"]([^\'"]*)\.csv[\'"]', r'open\(os.path.join\(RESULTS_DIR, "\1.csv"\)', content)
    content = re.sub(r'open\([\'"]([^\'"]*)\.txt[\'"]', r'open\(os.path.join\(DATA_DIR, "\1.txt"\)', content)
    content = re.sub(r'open\([\'"]([^\'"]*)\.log[\'"]', r'open\(os.path.join\(LOGS_DIR, "\1.log"\)', content)
    
    # Add directory constants at the beginning of the file
    if 'SCRIPT_DIR' not in content:
        imports_end = content.find('\n\n', content.find('import'))
        if imports_end == -1:
            imports_end = content.find('\n', content.find('import'))
        
        dir_constants = '\n\n# Directory paths\nSCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))\n'
        dir_constants += 'PROJECT_DIR = os.path.dirname(SCRIPT_DIR)\n'
        dir_constants += 'DATA_DIR = os.path.join(PROJECT_DIR, "data")\n'
        dir_constants += 'LOGS_DIR = os.path.join(PROJECT_DIR, "logs")\n'
        dir_constants += 'RESULTS_DIR = os.path.join(PROJECT_DIR, "results")\n\n'
        
        content = content[:imports_end] + dir_constants + content[imports_end:]
    
    # Ensure os is imported
    if 'import os' not in content and 'from os import' not in content:
        first_import = content.find('import')
        content = content[:first_import] + 'import os\n' + content[first_import:]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Updated {file_path}")

def main():
    """Main function to update all Python files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Update all Python files in the src directory
    for filename in os.listdir(script_dir):
        if filename.endswith('.py') and filename != 'update_imports.py' and filename != 'main.py':
            update_file(os.path.join(script_dir, filename))
    
    print("All files updated successfully!")

if __name__ == '__main__':
    main() 