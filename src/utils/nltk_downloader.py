import nltk
import logging
import os
import sys

logger = logging.getLogger(__name__)

def download_nltk_data():
    """
    Download required NLTK data packages.
    This should be called during application initialization.
    """
    required_packages = [
        'punkt',
        'stopwords',
        'averaged_perceptron_tagger'
    ]
    
    # Create a directory for NLTK data if it doesn't exist
    nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Set the NLTK data path
    nltk.data.path.append(nltk_data_dir)
    
    # Download required packages
    for package in required_packages:
        try:
            logger.info(f"Downloading NLTK package: {package}")
            nltk.download(package, download_dir=nltk_data_dir, quiet=True)
            logger.info(f"Successfully downloaded NLTK package: {package}")
        except Exception as e:
            logger.error(f"Failed to download NLTK package {package}: {e}")
            
    logger.info(f"NLTK data path: {nltk.data.path}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Download NLTK data
    download_nltk_data()
    
    # Print success message
    print("NLTK data download complete.") 