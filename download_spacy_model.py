"""
Script to download the spaCy model needed for entity extraction.
"""

import subprocess
import sys

def download_spacy_model():
    """Download the spaCy model needed for entity extraction."""
    print("Downloading spaCy model for entity extraction...")
    
    # Try to import spacy to check if it's installed
    try:
        import spacy
        print(f"spaCy version: {spacy.__version__}")
    except ImportError:
        print("spaCy is not installed. Please install it first with:")
        print("pip install spacy")
        sys.exit(1)
    
    # Download the English model
    print("Downloading en_core_web_lg model...")
    result = subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_lg"], 
                           capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Successfully downloaded en_core_web_lg model.")
    else:
        print("Failed to download en_core_web_lg model.")
        print("Error:", result.stderr)
        print("You can try downloading it manually with:")
        print("python -m spacy download en_core_web_lg")
        sys.exit(1)
    
    # Download a smaller model as a fallback
    print("Downloading en_core_web_sm model as a fallback...")
    result = subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                           capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Successfully downloaded en_core_web_sm model.")
    else:
        print("Failed to download en_core_web_sm model.")
        print("Error:", result.stderr)
        print("You can try downloading it manually with:")
        print("python -m spacy download en_core_web_sm")
        sys.exit(1)
    
    print("All spaCy models downloaded successfully.")

if __name__ == "__main__":
    download_spacy_model() 