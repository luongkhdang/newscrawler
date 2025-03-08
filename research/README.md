# Newspaper4k Research

This directory contains research scripts and findings related to the Newspaper4k library for the NewsCrawler project.

## Directory Structure

```
research/
├── data/           # URL lists and other input data
├── logs/           # Log files from script execution
├── results/        # Output files and databases
├── src/            # Source code for research scripts
│   ├── main.py     # Main entry point
│   ├── newspaper4k_poc.py           # Article extraction proof of concept
│   ├── newspaper4k_threading.py     # Threading performance testing
│   ├── newspaper4k_db_integration.py # Database integration testing
└── venv/           # Python virtual environment
```

## Setup

1. Create and activate a virtual environment:

```bash
# Create a virtual environment in the project directory
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install NLTK data:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Usage

The research code can be run using the main.py script, which provides a unified interface to the different research components.

### Article Extraction Research

```bash
# Run with default test URLs
python src/main.py extract

# Run with a single URL
python src/main.py extract --url https://example.com/article

# Run with a file containing URLs
python src/main.py extract --urls current_urls.txt

# Specify output file
python src/main.py extract --output custom_results.csv
```

### Threading Performance Research

```bash
# Run with default test URLs
python src/main.py threading

# Run with a single URL
python src/main.py threading --url https://example.com

# Run with specific thread counts
python src/main.py threading --threads 1,2,4,8

# Specify output file
python src/main.py threading --output threading_results.csv
```

### Database Integration Research

```bash
# Process a single URL and store in database
python src/main.py db --url https://example.com/article

# Process URLs from a file
python src/main.py db --urls current_urls.txt

# Query articles in the database
python src/main.py db --query "keyword"

# Specify database file
python src/main.py db --db custom_database.db
```

## Features

- Tests article extraction with and without image fetching
- Monitors network requests to verify no image requests are made
- Measures processing time and memory usage
- Evaluates multi-threading performance
- Tests database integration for article storage
- Extracts article content, metadata, and keywords

## Interpreting Results

The scripts generate CSV files with detailed metrics for each test:

### Article Extraction Metrics

- `url`: The URL of the article
- `success`: Whether extraction was successful
- `processing_time`: Time taken to process the article (seconds)
- `memory_increase`: Memory used during processing (MB)
- `image_requests_count`: Number of image-related network requests
- `has_images`: Whether the article object contains images
- `images_disabled`: Whether image extraction was disabled
- `domain`: The domain of the article
- `title`: Article title
- `authors`: Article authors
- `publish_date`: Publication date
- `text_length`: Length of extracted text
- `keywords`: Keywords extracted from the article

### Threading Performance Metrics

- `url`: The URL of the news source
- `thread_count`: Number of threads used
- `success`: Whether processing was successful
- `processing_time`: Time taken to process the source (seconds)
- `memory_increase`: Memory used during processing (MB)
- `article_count`: Number of articles found in the source
- `processed_articles`: Number of articles processed
- `successful_articles`: Number of articles successfully processed

## Research Findings

The key findings from this research are documented in the `AI_WORKSPACE/FINDINGS.md` file under the "Newspaper4k Configuration Research" and "Newspaper4k Practical Testing Results" sections.

## Next Steps

1. Expand testing to a larger and more diverse set of news sources
2. Implement multi-threading optimization research
3. Develop integration with other components (URL management, database storage)
4. Create a production-ready implementation based on research findings 