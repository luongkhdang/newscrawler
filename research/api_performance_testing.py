"""
API Performance and Scalability Testing Script

This script tests the performance and scalability of the FastAPI application:
1. Measures response times under various load conditions
2. Tests concurrent request handling
3. Evaluates API throughput and latency
4. Identifies bottlenecks and optimization opportunities
"""

import os
import sys
import time
import logging
import json
import asyncio
import aiohttp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('research/api_performance_testing.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API configuration
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"

# Test endpoints
TEST_ENDPOINTS = [
    {
        "name": "Root",
        "url": "/",
        "method": "GET",
        "data": None
    },
    {
        "name": "Health Check",
        "url": "/health",
        "method": "GET",
        "data": None
    },
    {
        "name": "Articles List",
        "url": "/articles",
        "method": "GET",
        "data": None
    },
    {
        "name": "Text Search",
        "url": "/search/text",
        "method": "GET",
        "data": {"q": "climate change", "limit": 10}
    },
    {
        "name": "Semantic Search",
        "url": "/search/semantic",
        "method": "GET",
        "data": {"q": "artificial intelligence", "limit": 10}
    },
    {
        "name": "Document Retrieval",
        "url": "/rag/retrieve",
        "method": "POST",
        "data": {"query": "renewable energy", "use_hybrid_search": True, "num_documents": 5}
    }
]

# Load test configurations
LOAD_TEST_CONFIGS = [
    {"name": "Light Load", "concurrent_requests": 5, "total_requests": 50},
    {"name": "Medium Load", "concurrent_requests": 20, "total_requests": 100},
    {"name": "Heavy Load", "concurrent_requests": 50, "total_requests": 200}
]

async def make_request(session, endpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Make a request to an endpoint and measure response time."""
    url = f"{API_URL}{endpoint['url']}"
    method = endpoint["method"].lower()
    data = endpoint.get("data")
    
    start_time = time.time()
    
    try:
        if method == "get":
            async with session.get(url, params=data) as response:
                status_code = response.status
                response_size = len(await response.read())
        else:  # POST
            async with session.post(url, json=data) as response:
                status_code = response.status
                response_size = len(await response.read())
        
        end_time = time.time()
        response_time = end_time - start_time
        
        return {
            "endpoint": endpoint["name"],
            "url": endpoint["url"],
            "method": endpoint["method"],
            "response_time": response_time,
            "status_code": status_code,
            "response_size": response_size,
            "success": 200 <= status_code < 300
        }
    
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        
        return {
            "endpoint": endpoint["name"],
            "url": endpoint["url"],
            "method": endpoint["method"],
            "response_time": response_time,
            "status_code": 0,
            "response_size": 0,
            "success": False,
            "error": str(e)
        }

async def run_load_test(endpoint: Dict[str, Any], concurrent_requests: int, total_requests: int) -> List[Dict[str, Any]]:
    """Run a load test on an endpoint with a specific concurrency level."""
    logger.info(f"Running load test on {endpoint['name']} with {concurrent_requests} concurrent requests...")
    
    results = []
    
    # Create a connection pool with limited connections
    connector = aiohttp.TCPConnector(limit=concurrent_requests)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Create tasks for all requests
        tasks = []
        for _ in range(total_requests):
            tasks.append(make_request(session, endpoint))
        
        # Execute tasks in batches to avoid overwhelming the server
        batch_size = concurrent_requests
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
            
            # Small delay between batches
            if i + batch_size < len(tasks):
                await asyncio.sleep(0.1)
    
    return results

def analyze_load_test_results(results: List[Dict[str, Any]], endpoint_name: str, config_name: str) -> Dict[str, Any]:
    """Analyze the results of a load test."""
    # Filter successful requests
    successful_results = [r for r in results if r["success"]]
    
    # Calculate statistics
    total_requests = len(results)
    successful_requests = len(successful_results)
    success_rate = successful_requests / total_requests if total_requests > 0 else 0
    
    if successful_results:
        response_times = [r["response_time"] for r in successful_results]
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        avg_response_time = sum(response_times) / len(response_times)
        p50_response_time = np.percentile(response_times, 50)
        p90_response_time = np.percentile(response_times, 90)
        p95_response_time = np.percentile(response_times, 95)
        p99_response_time = np.percentile(response_times, 99)
        
        response_sizes = [r["response_size"] for r in successful_results]
        avg_response_size = sum(response_sizes) / len(response_sizes)
        
        # Calculate throughput (requests per second)
        total_time = sum(response_times)
        throughput = successful_requests / total_time if total_time > 0 else 0
    else:
        min_response_time = 0
        max_response_time = 0
        avg_response_time = 0
        p50_response_time = 0
        p90_response_time = 0
        p95_response_time = 0
        p99_response_time = 0
        avg_response_size = 0
        throughput = 0
    
    return {
        "endpoint": endpoint_name,
        "config": config_name,
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "success_rate": success_rate * 100,
        "min_response_time": min_response_time,
        "max_response_time": max_response_time,
        "avg_response_time": avg_response_time,
        "p50_response_time": p50_response_time,
        "p90_response_time": p90_response_time,
        "p95_response_time": p95_response_time,
        "p99_response_time": p99_response_time,
        "avg_response_size": avg_response_size,
        "throughput": throughput
    }

async def test_api_performance():
    """Test API performance under various load conditions."""
    logger.info("Starting API performance testing...")
    
    all_results = []
    summary_results = []
    
    # Test each endpoint under different load conditions
    for endpoint in TEST_ENDPOINTS:
        endpoint_results = []
        
        for config in LOAD_TEST_CONFIGS:
            # Skip heavy load for expensive endpoints
            if config["name"] == "Heavy Load" and endpoint["name"] in ["Semantic Search", "Document Retrieval"]:
                logger.info(f"Skipping heavy load test for {endpoint['name']} to avoid overloading the server")
                continue
            
            # Run load test
            results = await run_load_test(
                endpoint=endpoint,
                concurrent_requests=config["concurrent_requests"],
                total_requests=config["total_requests"]
            )
            
            endpoint_results.extend(results)
            
            # Analyze results
            summary = analyze_load_test_results(results, endpoint["name"], config["name"])
            summary_results.append(summary)
            
            logger.info(f"Load test completed for {endpoint['name']} under {config['name']}:")
            logger.info(f"  Success rate: {summary['success_rate']:.2f}%")
            logger.info(f"  Avg response time: {summary['avg_response_time']:.4f}s")
            logger.info(f"  P95 response time: {summary['p95_response_time']:.4f}s")
            logger.info(f"  Throughput: {summary['throughput']:.2f} requests/second")
        
        all_results.extend(endpoint_results)
    
    # Save detailed results to CSV
    df = pd.DataFrame(all_results)
    df.to_csv('research/results/api_performance_detailed.csv', index=False)
    
    # Save summary results to CSV
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv('research/results/api_performance_summary.csv', index=False)
    
    # Create visualizations
    create_performance_visualizations(summary_df)
    
    logger.info("API performance testing completed")
    return all_results, summary_results

def create_performance_visualizations(summary_df: pd.DataFrame):
    """Create visualizations of API performance results."""
    logger.info("Creating performance visualizations...")
    
    # Response time by endpoint and load
    plt.figure(figsize=(12, 8))
    
    # Group by endpoint and config
    pivot_df = summary_df.pivot(index='endpoint', columns='config', values='avg_response_time')
    
    # Plot grouped bar chart
    pivot_df.plot(kind='bar', ax=plt.gca())
    
    plt.title('Average Response Time by Endpoint and Load')
    plt.xlabel('Endpoint')
    plt.ylabel('Response Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('research/results/response_time_by_endpoint.png')
    
    # Throughput by endpoint and load
    plt.figure(figsize=(12, 8))
    
    # Group by endpoint and config
    pivot_df = summary_df.pivot(index='endpoint', columns='config', values='throughput')
    
    # Plot grouped bar chart
    pivot_df.plot(kind='bar', ax=plt.gca())
    
    plt.title('Throughput by Endpoint and Load')
    plt.xlabel('Endpoint')
    plt.ylabel('Throughput (requests/second)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('research/results/throughput_by_endpoint.png')
    
    # Success rate by endpoint and load
    plt.figure(figsize=(12, 8))
    
    # Group by endpoint and config
    pivot_df = summary_df.pivot(index='endpoint', columns='config', values='success_rate')
    
    # Plot grouped bar chart
    pivot_df.plot(kind='bar', ax=plt.gca())
    
    plt.title('Success Rate by Endpoint and Load')
    plt.xlabel('Endpoint')
    plt.ylabel('Success Rate (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('research/results/success_rate_by_endpoint.png')
    
    logger.info("Performance visualizations created")

async def test_api_scalability():
    """Test API scalability by gradually increasing load."""
    logger.info("Starting API scalability testing...")
    
    # Select a subset of endpoints for scalability testing
    scalability_endpoints = [
        next(e for e in TEST_ENDPOINTS if e["name"] == "Health Check"),
        next(e for e in TEST_ENDPOINTS if e["name"] == "Articles List"),
        next(e for e in TEST_ENDPOINTS if e["name"] == "Text Search")
    ]
    
    # Define concurrency levels for scalability testing
    concurrency_levels = [1, 5, 10, 20, 30, 40, 50, 75, 100]
    
    scalability_results = []
    
    for endpoint in scalability_endpoints:
        endpoint_results = []
        
        for concurrency in concurrency_levels:
            # Run load test with fixed number of requests
            total_requests = concurrency * 2  # Each connection handles 2 requests
            
            results = await run_load_test(
                endpoint=endpoint,
                concurrent_requests=concurrency,
                total_requests=total_requests
            )
            
            # Analyze results
            summary = analyze_load_test_results(results, endpoint["name"], f"Concurrency-{concurrency}")
            summary["concurrency"] = concurrency
            endpoint_results.append(summary)
            
            logger.info(f"Scalability test completed for {endpoint['name']} with concurrency {concurrency}:")
            logger.info(f"  Success rate: {summary['success_rate']:.2f}%")
            logger.info(f"  Avg response time: {summary['avg_response_time']:.4f}s")
            logger.info(f"  Throughput: {summary['throughput']:.2f} requests/second")
        
        scalability_results.extend(endpoint_results)
    
    # Save scalability results to CSV
    scalability_df = pd.DataFrame(scalability_results)
    scalability_df.to_csv('research/results/api_scalability_results.csv', index=False)
    
    # Create scalability visualizations
    create_scalability_visualizations(scalability_df)
    
    logger.info("API scalability testing completed")
    return scalability_results

def create_scalability_visualizations(scalability_df: pd.DataFrame):
    """Create visualizations of API scalability results."""
    logger.info("Creating scalability visualizations...")
    
    # Group by endpoint
    endpoints = scalability_df['endpoint'].unique()
    
    # Response time vs. concurrency
    plt.figure(figsize=(10, 6))
    
    for endpoint in endpoints:
        endpoint_data = scalability_df[scalability_df['endpoint'] == endpoint]
        plt.plot(endpoint_data['concurrency'], endpoint_data['avg_response_time'], marker='o', label=endpoint)
    
    plt.title('Response Time vs. Concurrency')
    plt.xlabel('Concurrency Level')
    plt.ylabel('Average Response Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig('research/results/response_time_vs_concurrency.png')
    
    # Throughput vs. concurrency
    plt.figure(figsize=(10, 6))
    
    for endpoint in endpoints:
        endpoint_data = scalability_df[scalability_df['endpoint'] == endpoint]
        plt.plot(endpoint_data['concurrency'], endpoint_data['throughput'], marker='o', label=endpoint)
    
    plt.title('Throughput vs. Concurrency')
    plt.xlabel('Concurrency Level')
    plt.ylabel('Throughput (requests/second)')
    plt.legend()
    plt.grid(True)
    plt.savefig('research/results/throughput_vs_concurrency.png')
    
    # Success rate vs. concurrency
    plt.figure(figsize=(10, 6))
    
    for endpoint in endpoints:
        endpoint_data = scalability_df[scalability_df['endpoint'] == endpoint]
        plt.plot(endpoint_data['concurrency'], endpoint_data['success_rate'], marker='o', label=endpoint)
    
    plt.title('Success Rate vs. Concurrency')
    plt.xlabel('Concurrency Level')
    plt.ylabel('Success Rate (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('research/results/success_rate_vs_concurrency.png')
    
    logger.info("Scalability visualizations created")

async def main():
    """Main function to run all tests."""
    logger.info("Starting API performance and scalability testing...")
    
    # Create results directory if it doesn't exist
    os.makedirs('research/results', exist_ok=True)
    
    # Run performance tests
    await test_api_performance()
    
    # Run scalability tests
    await test_api_scalability()
    
    logger.info("API performance and scalability testing completed")

if __name__ == "__main__":
    asyncio.run(main()) 