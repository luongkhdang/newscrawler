"""
Script to download required NLTK data.
"""

import nltk

def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    print("NLTK data download complete.")

if __name__ == "__main__":
    download_nltk_data() 