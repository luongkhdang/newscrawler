"""
FastAPI Implementation Analysis Script

This script analyzes the current FastAPI implementation, focusing on:
1. Endpoint design and RESTful practices
2. API performance and scalability
3. Documentation quality
4. Error handling and validation
5. Authentication and rate limiting
"""

import os
import sys
import time
import logging
import json
import requests
from typing import Dict, List, Any, Optional
import pandas as pd
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('research/fastapi_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API configuration
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"

# Test queries for search endpoints
TEST_QUERIES = [
    "artificial intelligence",
    "climate change",
    "cryptocurrency",
    "renewable energy",
    "global economy"
]

# Test prompts for LLM endpoints
TEST_PROMPTS = [
    "Explain the impact of artificial intelligence on healthcare",
    "Summarize the current state of climate change research",
    "What are the main challenges facing cryptocurrency adoption?",
    "Describe recent innovations in renewable energy technology",
    "Analyze the current global economic outlook"
]

def analyze_endpoint_design():
    """Analyze the endpoint design and RESTful practices."""
    logger.info("Analyzing endpoint design and RESTful practices...")
    
    # Get OpenAPI schema
    try:
        response = requests.get(f"{API_URL}/openapi.json")
        if response.status_code == 200:
            openapi_schema = response.json()
            
            # Extract endpoints
            paths = openapi_schema.get("paths", {})
            endpoints = []
            
            for path, methods in paths.items():
                for method, details in methods.items():
                    if method.lower() not in ["get", "post", "put", "delete", "patch"]:
                        continue
                    
                    endpoints.append({
                        "path": path,
                        "method": method.upper(),
                        "summary": details.get("summary", ""),
                        "description": details.get("description", ""),
                        "tags": details.get("tags", []),
                        "parameters": len(details.get("parameters", [])),
                        "request_body": "Yes" if "requestBody" in details else "No",
                        "responses": list(details.get("responses", {}).keys())
                    })
            
            # Save endpoints to CSV
            df = pd.DataFrame(endpoints)
            df.to_csv('research/results/endpoint_analysis.csv', index=False)
            
            # Analyze RESTful practices
            rest_analysis = {
                "resource_oriented": 0,
                "uses_http_methods_correctly": 0,
                "uses_status_codes_correctly": 0,
                "has_documentation": 0
            }
            
            for endpoint in endpoints:
                # Check if path is resource-oriented (e.g., /articles, /sources)
                if "/" in endpoint["path"][1:]:
                    rest_analysis["resource_oriented"] += 1
                
                # Check if HTTP methods are used correctly
                if (endpoint["method"] == "GET" and "create" not in endpoint["summary"].lower() and "update" not in endpoint["summary"].lower()) or \
                   (endpoint["method"] == "POST" and "create" in endpoint["summary"].lower()) or \
                   (endpoint["method"] == "PUT" and "update" in endpoint["summary"].lower()) or \
                   (endpoint["method"] == "DELETE" and "delete" in endpoint["summary"].lower()):
                    rest_analysis["uses_http_methods_correctly"] += 1
                
                # Check if status codes are used correctly
                if "200" in endpoint["responses"] or "201" in endpoint["responses"]:
                    if "400" in endpoint["responses"] or "404" in endpoint["responses"] or "422" in endpoint["responses"]:
                        rest_analysis["uses_status_codes_correctly"] += 1
                
                # Check if endpoint has documentation
                if endpoint["summary"] or endpoint["description"]:
                    rest_analysis["has_documentation"] += 1
            
            # Calculate percentages
            total_endpoints = len(endpoints)
            if total_endpoints > 0:
                for key in rest_analysis:
                    rest_analysis[key] = (rest_analysis[key] / total_endpoints) * 100
            
            # Save RESTful analysis to JSON
            with open('research/results/restful_analysis.json', 'w') as f:
                json.dump(rest_analysis, f, indent=2)
            
            logger.info(f"Endpoint analysis completed. Found {total_endpoints} endpoints.")
            return endpoints, rest_analysis
        else:
            logger.error(f"Failed to get OpenAPI schema: {response.status_code} - {response.text}")
            return [], {}
    
    except Exception as e:
        logger.error(f"Error analyzing endpoint design: {e}")
        return [], {}

def test_endpoint_performance(endpoint: str, method: str = "GET", data: Optional[Dict[str, Any]] = None, 
                             num_requests: int = 50, concurrency: int = 10):
    """Test the performance of an endpoint."""
    logger.info(f"Testing performance of {method} {endpoint} with {num_requests} requests and concurrency {concurrency}...")
    
    results = []
    
    def make_request():
        start_time = time.time()
        try:
            if method.upper() == "GET":
                response = requests.get(f"{API_URL}{endpoint}", params=data)
            else:
                response = requests.post(f"{API_URL}{endpoint}", json=data)
            
            end_time = time.time()
            response_time = end_time - start_time
            status_code = response.status_code
            response_size = len(response.content)
            
            return {
                "response_time": response_time,
                "status_code": status_code,
                "response_size": response_size,
                "success": 200 <= status_code < 300
            }
        
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            return {
                "response_time": response_time,
                "status_code": 0,
                "response_size": 0,
                "success": False,
                "error": str(e)
            }
    
    # Execute requests in parallel
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        for future in tqdm(futures, desc=f"Testing {endpoint}"):
            results.append(future.result())
    
    # Calculate statistics
    response_times = [r["response_time"] for r in results]
    success_rate = sum(1 for r in results if r["success"]) / len(results) if results else 0
    
    stats = {
        "endpoint": endpoint,
        "method": method,
        "num_requests": num_requests,
        "concurrency": concurrency,
        "min_response_time": min(response_times) if response_times else 0,
        "max_response_time": max(response_times) if response_times else 0,
        "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
        "p50_response_time": sorted(response_times)[len(response_times) // 2] if response_times else 0,
        "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0,
        "p99_response_time": sorted(response_times)[int(len(response_times) * 0.99)] if response_times else 0,
        "success_rate": success_rate * 100,
        "avg_response_size": sum(r["response_size"] for r in results) / len(results) if results else 0
    }
    
    logger.info(f"Performance test completed for {endpoint}. Avg response time: {stats['avg_response_time']:.4f}s, Success rate: {stats['success_rate']:.2f}%")
    return stats

def benchmark_search_endpoints():
    """Benchmark the performance of search endpoints."""
    logger.info("Benchmarking search endpoints...")
    
    endpoints = [
        {
            "name": "Text Search",
            "endpoint": "/search/text",
            "method": "GET",
            "data_template": {"q": "{query}"}
        },
        {
            "name": "Semantic Search",
            "endpoint": "/search/semantic",
            "method": "GET",
            "data_template": {"q": "{query}"}
        }
    ]
    
    results = []
    
    for endpoint in endpoints:
        for query in TEST_QUERIES:
            # Replace placeholders in data template
            data = {k: v.format(query=query) if isinstance(v, str) else v 
                   for k, v in endpoint["data_template"].items()}
            
            # Test endpoint performance
            stats = test_endpoint_performance(
                endpoint=endpoint["endpoint"],
                method=endpoint["method"],
                data=data,
                num_requests=20,
                concurrency=5
            )
            
            # Add endpoint name and query to stats
            stats["endpoint_name"] = endpoint["name"]
            stats["query"] = query
            
            results.append(stats)
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('research/results/search_endpoint_benchmark.csv', index=False)
    
    # Create summary by endpoint
    summary = df.groupby("endpoint_name").agg({
        "avg_response_time": "mean",
        "p95_response_time": "mean",
        "success_rate": "mean",
        "avg_response_size": "mean"
    }).reset_index()
    
    summary.to_csv('research/results/search_endpoint_summary.csv', index=False)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.bar(summary["endpoint_name"], summary["avg_response_time"])
    plt.title("Average Response Time by Search Endpoint")
    plt.xlabel("Endpoint")
    plt.ylabel("Response Time (s)")
    plt.savefig('research/results/search_endpoint_performance.png')
    
    logger.info("Search endpoint benchmarking completed")
    return results

def benchmark_llm_endpoints():
    """Benchmark the performance of LLM endpoints."""
    logger.info("Benchmarking LLM endpoints...")
    
    endpoints = [
        {
            "name": "Text Generation",
            "endpoint": "/llm/generate",
            "method": "POST",
            "data_template": {"prompt": "{prompt}", "max_tokens": 100}
        },
        {
            "name": "Summarization",
            "endpoint": "/llm/summarize",
            "method": "POST",
            "data_template": {"text": "{prompt}", "max_length": 50}
        }
    ]
    
    results = []
    
    for endpoint in endpoints:
        for prompt in TEST_PROMPTS:
            # Replace placeholders in data template
            data = {k: v.format(prompt=prompt) if isinstance(v, str) else v 
                   for k, v in endpoint["data_template"].items()}
            
            # Test endpoint performance
            stats = test_endpoint_performance(
                endpoint=endpoint["endpoint"],
                method=endpoint["method"],
                data=data,
                num_requests=5,  # Fewer requests for LLM endpoints due to rate limits
                concurrency=1    # Sequential requests for LLM endpoints
            )
            
            # Add endpoint name and prompt to stats
            stats["endpoint_name"] = endpoint["name"]
            stats["prompt"] = prompt
            
            results.append(stats)
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('research/results/llm_endpoint_benchmark.csv', index=False)
    
    # Create summary by endpoint
    summary = df.groupby("endpoint_name").agg({
        "avg_response_time": "mean",
        "p95_response_time": "mean",
        "success_rate": "mean",
        "avg_response_size": "mean"
    }).reset_index()
    
    summary.to_csv('research/results/llm_endpoint_summary.csv', index=False)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.bar(summary["endpoint_name"], summary["avg_response_time"])
    plt.title("Average Response Time by LLM Endpoint")
    plt.xlabel("Endpoint")
    plt.ylabel("Response Time (s)")
    plt.savefig('research/results/llm_endpoint_performance.png')
    
    logger.info("LLM endpoint benchmarking completed")
    return results

def analyze_documentation():
    """Analyze the API documentation quality."""
    logger.info("Analyzing API documentation quality...")
    
    try:
        # Get OpenAPI schema
        response = requests.get(f"{API_URL}/openapi.json")
        if response.status_code == 200:
            openapi_schema = response.json()
            
            # Extract paths and components
            paths = openapi_schema.get("paths", {})
            components = openapi_schema.get("components", {})
            schemas = components.get("schemas", {})
            
            # Calculate documentation metrics
            total_endpoints = 0
            endpoints_with_summary = 0
            endpoints_with_description = 0
            endpoints_with_response_description = 0
            
            for path, methods in paths.items():
                for method, details in methods.items():
                    if method.lower() not in ["get", "post", "put", "delete", "patch"]:
                        continue
                    
                    total_endpoints += 1
                    
                    if details.get("summary"):
                        endpoints_with_summary += 1
                    
                    if details.get("description"):
                        endpoints_with_description += 1
                    
                    # Check if responses have descriptions
                    responses = details.get("responses", {})
                    has_response_description = False
                    
                    for status_code, response_details in responses.items():
                        if response_details.get("description"):
                            has_response_description = True
                            break
                    
                    if has_response_description:
                        endpoints_with_response_description += 1
            
            # Calculate schema documentation metrics
            total_schemas = len(schemas)
            schemas_with_description = 0
            schemas_with_property_descriptions = 0
            
            for schema_name, schema in schemas.items():
                if schema.get("description"):
                    schemas_with_description += 1
                
                properties = schema.get("properties", {})
                property_descriptions = 0
                total_properties = len(properties)
                
                for prop_name, prop in properties.items():
                    if prop.get("description"):
                        property_descriptions += 1
                
                if total_properties > 0 and property_descriptions / total_properties >= 0.5:
                    schemas_with_property_descriptions += 1
            
            # Calculate percentages
            doc_metrics = {
                "total_endpoints": total_endpoints,
                "endpoints_with_summary_percent": (endpoints_with_summary / total_endpoints * 100) if total_endpoints > 0 else 0,
                "endpoints_with_description_percent": (endpoints_with_description / total_endpoints * 100) if total_endpoints > 0 else 0,
                "endpoints_with_response_description_percent": (endpoints_with_response_description / total_endpoints * 100) if total_endpoints > 0 else 0,
                "total_schemas": total_schemas,
                "schemas_with_description_percent": (schemas_with_description / total_schemas * 100) if total_schemas > 0 else 0,
                "schemas_with_property_descriptions_percent": (schemas_with_property_descriptions / total_schemas * 100) if total_schemas > 0 else 0
            }
            
            # Save documentation metrics to JSON
            with open('research/results/documentation_analysis.json', 'w') as f:
                json.dump(doc_metrics, f, indent=2)
            
            logger.info(f"Documentation analysis completed. Found {total_endpoints} endpoints and {total_schemas} schemas.")
            return doc_metrics
        else:
            logger.error(f"Failed to get OpenAPI schema: {response.status_code} - {response.text}")
            return {}
    
    except Exception as e:
        logger.error(f"Error analyzing documentation: {e}")
        return {}

def analyze_error_handling():
    """Analyze the API error handling."""
    logger.info("Analyzing API error handling...")
    
    # Test cases for error handling
    test_cases = [
        {
            "name": "Missing Required Parameter",
            "endpoint": "/search/text",
            "method": "GET",
            "data": {}  # Missing 'q' parameter
        },
        {
            "name": "Invalid Parameter Type",
            "endpoint": "/search/text",
            "method": "GET",
            "data": {"q": "test", "limit": "invalid"}  # 'limit' should be an integer
        },
        {
            "name": "Resource Not Found",
            "endpoint": "/articles/00000000-0000-0000-0000-000000000000",  # Non-existent ID
            "method": "GET",
            "data": {}
        },
        {
            "name": "Invalid JSON Body",
            "endpoint": "/llm/generate",
            "method": "POST",
            "data": "invalid json",
            "is_json_string": True
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        logger.info(f"Testing error handling for: {test_case['name']}")
        
        try:
            if test_case["method"].upper() == "GET":
                response = requests.get(f"{API_URL}{test_case['endpoint']}", params=test_case['data'])
            else:
                if test_case.get("is_json_string"):
                    response = requests.post(
                        f"{API_URL}{test_case['endpoint']}", 
                        data=test_case['data'],
                        headers={"Content-Type": "application/json"}
                    )
                else:
                    response = requests.post(f"{API_URL}{test_case['endpoint']}", json=test_case['data'])
            
            # Analyze response
            result = {
                "test_case": test_case["name"],
                "endpoint": test_case["endpoint"],
                "method": test_case["method"],
                "status_code": response.status_code,
                "has_error_message": False,
                "error_message": "",
                "has_structured_error": False
            }
            
            # Check if response is JSON
            try:
                response_json = response.json()
                
                # Check if response has error message
                if "detail" in response_json or "error" in response_json:
                    result["has_error_message"] = True
                    result["error_message"] = response_json.get("detail") or response_json.get("error")
                
                # Check if error is structured
                if isinstance(response_json, dict) and len(response_json) > 1:
                    result["has_structured_error"] = True
            except:
                pass
            
            results.append(result)
        
        except Exception as e:
            logger.error(f"Error testing {test_case['name']}: {e}")
            results.append({
                "test_case": test_case["name"],
                "endpoint": test_case["endpoint"],
                "method": test_case["method"],
                "status_code": 0,
                "has_error_message": False,
                "error_message": str(e),
                "has_structured_error": False
            })
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv('research/results/error_handling_analysis.csv', index=False)
    
    # Calculate metrics
    metrics = {
        "total_test_cases": len(results),
        "appropriate_status_codes": sum(1 for r in results if 400 <= r["status_code"] < 500),
        "has_error_message": sum(1 for r in results if r["has_error_message"]),
        "has_structured_error": sum(1 for r in results if r["has_structured_error"])
    }
    
    # Calculate percentages
    total_cases = metrics["total_test_cases"]
    if total_cases > 0:
        metrics["appropriate_status_codes_percent"] = (metrics["appropriate_status_codes"] / total_cases) * 100
        metrics["has_error_message_percent"] = (metrics["has_error_message"] / total_cases) * 100
        metrics["has_structured_error_percent"] = (metrics["has_structured_error"] / total_cases) * 100
    
    # Save metrics to JSON
    with open('research/results/error_handling_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Error handling analysis completed. Tested {total_cases} cases.")
    return results, metrics

def main():
    """Main function to run all analyses."""
    logger.info("Starting FastAPI implementation analysis...")
    
    # Create results directory if it doesn't exist
    os.makedirs('research/results', exist_ok=True)
    
    # Run analyses
    analyze_endpoint_design()
    benchmark_search_endpoints()
    benchmark_llm_endpoints()
    analyze_documentation()
    analyze_error_handling()
    
    logger.info("FastAPI implementation analysis completed")

if __name__ == "__main__":
    main() 