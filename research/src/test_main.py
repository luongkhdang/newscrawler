#!/usr/bin/env python
"""
Test script for main.py to verify that it works correctly.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the src directory to the path so we can import main
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main module
import main

class TestMain(unittest.TestCase):
    """Test cases for the main.py script."""

    @patch('main.run_article_extraction')
    def test_extract_command(self, mock_extract):
        """Test that the extract command calls the correct function."""
        # Mock sys.argv
        with patch('sys.argv', ['main.py', 'extract']):
            main.main()
            mock_extract.assert_called_once()

    @patch('main.run_threading_test')
    def test_threading_command(self, mock_threading):
        """Test that the threading command calls the correct function."""
        # Mock sys.argv
        with patch('sys.argv', ['main.py', 'threading']):
            main.main()
            mock_threading.assert_called_once()

    @patch('main.run_db_integration')
    def test_db_command(self, mock_db):
        """Test that the db command calls the correct function."""
        # Mock sys.argv
        with patch('sys.argv', ['main.py', 'db']):
            main.main()
            mock_db.assert_called_once()

    def test_setup_paths(self):
        """Test that setup_paths creates the necessary directories."""
        # Call setup_paths
        paths = main.setup_paths()
        
        # Check that the paths dictionary contains the expected keys
        self.assertIn('base_dir', paths)
        self.assertIn('data_dir', paths)
        self.assertIn('logs_dir', paths)
        self.assertIn('results_dir', paths)
        
        # Check that the directories exist
        self.assertTrue(os.path.exists(paths['data_dir']))
        self.assertTrue(os.path.exists(paths['logs_dir']))
        self.assertTrue(os.path.exists(paths['results_dir']))

if __name__ == '__main__':
    unittest.main() 