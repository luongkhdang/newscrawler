#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Newspaper4k Benchmarking Script

This script benchmarks the performance of Newspaper4k across different news sources.
It measures:
1. Success rate (percentage of articles successfully extracted)
2. Extraction time
3. Content quality metrics
4. Memory usage

Usage:
    python benchmark_script.py --sources sources.txt --output results.json
"""

import argparse
import csv
import json
import logging
import os
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import newspaper
import pandas as pd
import requests
from newspaper import Article, Config
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("newspaper_benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define news source categories for analysis
SOURCE_CATEGORIES = {
    "mainstream": [
        "https://www.cnn.com",
        "https://www.bbc.com",
        "https://www.nytimes.com",
        "https://www.washingtonpost.com",
        "https://www.theguardian.com",
    ],
    "tech": [
        "https://techcrunch.com",
        "https://www.wired.com",
        "https://www.theverge.com",
        "https://arstechnica.com",
        "https://www.engadget.com",
    ],
    "blog": [
        "https://medium.com",
        "https://www.blogspot.com",
        "https://wordpress.com",
        "https://substack.com",
        "https://dev.to",
    ],
    "javascript_heavy": [
        "https://www.bloomberg.com",
        "https://www.wsj.com",
        "https://www.economist.com",
        "https://www.forbes.com",
        "https://www.businessinsider.com",
    ],
}


class NewspaperBenchmark:
    """Benchmark Newspaper4k library performance across different news sources."""

    def __init__(self, sources_file: Optional[str] = None, max_articles: int = 5):
        """
        Initialize the benchmark.

        Args:
            sources_file: Path to a file containing news source URLs (one per line)
            max_articles: Maximum number of articles to extract per source
        """
        self.sources = []
        self.max_articles = max_articles
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "library_version": newspaper.__version__,
            "sources": {},
            "summary": {},
        }

        # Load sources from file if provided
        if sources_file and os.path.exists(sources_file):
            with open(sources_file, "r") as f:
                self.sources = [line.strip() for line in f if line.strip()]
        else:
            # Use default sources from categories
            for category, urls in SOURCE_CATEGORIES.items():
                self.sources.extend(urls)

        # Configure newspaper
        self.config = Config()
        self.config.browser_user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
        self.config.request_timeout = 10
        self.config.fetch_images = False  # Disable image fetching as per project requirements
        self.config.memoize_articles = False

    def extract_articles(self, source_url: str) -> Dict:
        """
        Extract articles from a news source and measure performance.

        Args:
            source_url: URL of the news source

        Returns:
            Dict containing benchmark results for the source
        """
        logger.info(f"Benchmarking source: {source_url}")
        source_results = {
            "url": source_url,
            "articles_attempted": 0,
            "articles_succeeded": 0,
            "total_time": 0,
            "avg_time_per_article": 0,
            "avg_article_length": 0,
            "memory_usage": 0,
            "errors": [],
            "articles": [],
        }

        try:
            # Start memory tracking
            tracemalloc.start()

            # Build newspaper source
            start_time = time.time()
            source = newspaper.build(source_url, config=self.config)
            build_time = time.time() - start_time
            source_results["build_time"] = build_time

            # Get article URLs
            article_urls = source.article_urls()
            source_results["total_articles_found"] = len(article_urls)

            # Limit the number of articles to process
            article_urls = article_urls[:self.max_articles]
            source_results["articles_attempted"] = len(article_urls)

            # Process each article
            for url in article_urls:
                article_result = self._process_article(url)
                source_results["articles"].append(article_result)

                if article_result["success"]:
                    source_results["articles_succeeded"] += 1
                    source_results["total_time"] += article_result["extraction_time"]
                    source_results["avg_article_length"] += len(article_result["text"])
                else:
                    source_results["errors"].append(
                        {"url": url, "error": article_result["error"]}
                    )

            # Calculate averages
            if source_results["articles_succeeded"] > 0:
                source_results["avg_time_per_article"] = (
                    source_results["total_time"] / source_results["articles_succeeded"]
                )
                source_results["avg_article_length"] = (
                    source_results["avg_article_length"]
                    / source_results["articles_succeeded"]
                )

            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()
            source_results["memory_usage"] = peak / 1024 / 1024  # Convert to MB
            tracemalloc.stop()

            # Calculate success rate
            source_results["success_rate"] = (
                source_results["articles_succeeded"] / source_results["articles_attempted"]
                if source_results["articles_attempted"] > 0
                else 0
            )

        except Exception as e:
            logger.error(f"Error benchmarking source {source_url}: {str(e)}")
            source_results["error"] = str(e)
            source_results["success_rate"] = 0

        return source_results

    def _process_article(self, article_url: str) -> Dict:
        """
        Process a single article and measure performance.

        Args:
            article_url: URL of the article to process

        Returns:
            Dict containing benchmark results for the article
        """
        article_result = {
            "url": article_url,
            "success": False,
            "extraction_time": 0,
            "title": "",
            "authors": [],
            "publish_date": None,
            "text": "",
            "error": None,
        }

        try:
            # Download and parse article
            start_time = time.time()
            article = Article(article_url, config=self.config)
            article.download()
            article.parse()
            article_result["extraction_time"] = time.time() - start_time

            # Extract article data
            article_result["title"] = article.title
            article_result["authors"] = article.authors
            article_result["publish_date"] = (
                article.publish_date.isoformat() if article.publish_date else None
            )
            article_result["text"] = article.text
            article_result["success"] = True

        except Exception as e:
            article_result["error"] = str(e)
            logger.error(f"Error processing article {article_url}: {str(e)}")

        return article_result

    def run_benchmark(self) -> Dict:
        """
        Run the benchmark on all sources.

        Returns:
            Dict containing all benchmark results
        """
        logger.info(f"Starting benchmark with {len(self.sources)} sources")

        # Process each source
        with ThreadPoolExecutor(max_workers=5) as executor:
            source_results = list(
                tqdm(
                    executor.map(self.extract_articles, self.sources),
                    total=len(self.sources),
                    desc="Benchmarking sources",
                )
            )

        # Store results
        for result in source_results:
            source_url = result["url"]
            self.results["sources"][source_url] = result

        # Calculate summary statistics
        self._calculate_summary()

        return self.results

    def _calculate_summary(self):
        """Calculate summary statistics across all sources."""
        sources = self.results["sources"].values()
        
        if not sources:
            self.results["summary"] = {
                "total_sources": 0,
                "total_articles_attempted": 0,
                "total_articles_succeeded": 0,
                "overall_success_rate": 0,
                "avg_time_per_article": 0,
                "avg_article_length": 0,
                "avg_memory_usage": 0,
            }
            return

        total_articles_attempted = sum(s.get("articles_attempted", 0) for s in sources)
        total_articles_succeeded = sum(s.get("articles_succeeded", 0) for s in sources)
        
        self.results["summary"] = {
            "total_sources": len(sources),
            "total_articles_attempted": total_articles_attempted,
            "total_articles_succeeded": total_articles_succeeded,
            "overall_success_rate": (
                total_articles_succeeded / total_articles_attempted
                if total_articles_attempted > 0
                else 0
            ),
            "avg_time_per_article": (
                sum(s.get("avg_time_per_article", 0) for s in sources) / len(sources)
                if sources
                else 0
            ),
            "avg_article_length": (
                sum(s.get("avg_article_length", 0) for s in sources) / len(sources)
                if sources
                else 0
            ),
            "avg_memory_usage": (
                sum(s.get("memory_usage", 0) for s in sources) / len(sources)
                if sources
                else 0
            ),
        }

        # Calculate category-specific statistics
        self._calculate_category_stats()

    def _calculate_category_stats(self):
        """Calculate statistics by source category."""
        category_stats = {}
        
        for category, urls in SOURCE_CATEGORIES.items():
            category_sources = [
                self.results["sources"].get(url, {})
                for url in urls
                if url in self.results["sources"]
            ]
            
            if not category_sources:
                continue
                
            total_attempted = sum(s.get("articles_attempted", 0) for s in category_sources)
            total_succeeded = sum(s.get("articles_succeeded", 0) for s in category_sources)
            
            category_stats[category] = {
                "sources_count": len(category_sources),
                "articles_attempted": total_attempted,
                "articles_succeeded": total_succeeded,
                "success_rate": (
                    total_succeeded / total_attempted if total_attempted > 0 else 0
                ),
                "avg_time_per_article": (
                    sum(s.get("avg_time_per_article", 0) for s in category_sources)
                    / len(category_sources)
                ),
                "avg_memory_usage": (
                    sum(s.get("memory_usage", 0) for s in category_sources)
                    / len(category_sources)
                ),
            }
            
        self.results["category_stats"] = category_stats

    def save_results(self, output_file: str):
        """
        Save benchmark results to a file.

        Args:
            output_file: Path to the output file
        """
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        # Also save a CSV summary for easy analysis
        csv_file = output_file.replace(".json", ".csv")
        self._save_csv_summary(csv_file)

    def _save_csv_summary(self, csv_file: str):
        """
        Save a CSV summary of the results.

        Args:
            csv_file: Path to the CSV file
        """
        rows = []
        for source_url, data in self.results["sources"].items():
            rows.append({
                "source_url": source_url,
                "articles_attempted": data.get("articles_attempted", 0),
                "articles_succeeded": data.get("articles_succeeded", 0),
                "success_rate": data.get("success_rate", 0),
                "avg_time_per_article": data.get("avg_time_per_article", 0),
                "avg_article_length": data.get("avg_article_length", 0),
                "memory_usage_mb": data.get("memory_usage", 0),
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        logger.info(f"CSV summary saved to {csv_file}")


def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark Newspaper4k library")
    parser.add_argument(
        "--sources", type=str, help="Path to a file containing news source URLs"
    )
    parser.add_argument(
        "--output", type=str, default="newspaper_benchmark_results.json",
        help="Path to the output file"
    )
    parser.add_argument(
        "--max-articles", type=int, default=5,
        help="Maximum number of articles to extract per source"
    )
    args = parser.parse_args()

    benchmark = NewspaperBenchmark(args.sources, args.max_articles)
    results = benchmark.run_benchmark()
    benchmark.save_results(args.output)

    # Print summary
    print("\nBenchmark Summary:")
    print(f"Total sources: {results['summary']['total_sources']}")
    print(f"Total articles attempted: {results['summary']['total_articles_attempted']}")
    print(f"Total articles succeeded: {results['summary']['total_articles_succeeded']}")
    print(f"Overall success rate: {results['summary']['overall_success_rate']:.2%}")
    print(f"Average time per article: {results['summary']['avg_time_per_article']:.2f} seconds")
    print(f"Average article length: {results['summary']['avg_article_length']:.0f} characters")
    print(f"Average memory usage: {results['summary']['avg_memory_usage']:.2f} MB")

    if "category_stats" in results:
        print("\nCategory Statistics:")
        for category, stats in results["category_stats"].items():
            print(f"\n{category.upper()}:")
            print(f"  Success rate: {stats['success_rate']:.2%}")
            print(f"  Avg time per article: {stats['avg_time_per_article']:.2f} seconds")
            print(f"  Avg memory usage: {stats['avg_memory_usage']:.2f} MB")


if __name__ == "__main__":
    main() 