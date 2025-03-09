#!/usr/bin/env python
"""
Script to run the NewsCrawler tests.
"""
import os
import sys
import pytest


def main():
    """Run the tests."""
    # Add the current directory to the path so that the tests can import the src module
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Run the tests
    args = [
        "-v",  # Verbose output
        "--cov=src",  # Measure code coverage for the src directory
        "--cov-report=term",  # Output coverage report to the terminal
        "--cov-report=html:coverage_html",  # Output coverage report to HTML
        "tests/",  # Run tests in the tests directory
    ]
    
    # Add any additional arguments passed to this script
    args.extend(sys.argv[1:])
    
    # Run pytest with the arguments
    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(main()) 