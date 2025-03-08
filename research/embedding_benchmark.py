"""
Embedding Generation Performance Benchmark Script

This script benchmarks the performance of different embedding models and configurations:
1. Tests different embedding models (HuggingFace, OpenAI)
2. Measures performance across different batch sizes
3. Evaluates embedding quality using similarity tests
4. Analyzes resource usage (CPU, memory, time)
"""

import os
import sys
import time
import logging
import json
import psutil
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('research/embedding_benchmark.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Sample texts for benchmarking
SAMPLE_TEXTS_SHORT = [
    "Artificial intelligence is transforming industries worldwide.",
    "Climate change poses significant challenges to global agriculture.",
    "Renewable energy adoption continues to accelerate globally.",
    "Cryptocurrency markets remain volatile amid regulatory uncertainty.",
    "Quantum computing promises to revolutionize computational capabilities."
]

# Longer sample texts
SAMPLE_TEXTS_LONG = [
    """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and 
    takes actions that maximize its chance of achieving its goals. The term "artificial intelligence" had previously been used to describe 
    machines that mimic and display human cognitive skills that are associated with the human mind, such as learning and problem-solving. 
    This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, 
    which does not limit how intelligence can be articulated.
    """,
    """
    Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, such as through variations 
    in the solar cycle. But since the 1800s, human activities have been the main driver of climate change, primarily due to burning fossil 
    fuels like coal, oil and gas. Burning fossil fuels generates greenhouse gas emissions that act like a blanket wrapped around the Earth, 
    trapping the sun's heat and raising temperatures. Examples of greenhouse gas emissions that are causing climate change include carbon dioxide 
    and methane. These come from using gasoline for driving a car or coal for heating a building, for example.
    """,
    """
    Renewable energy is energy derived from natural sources that are replenished at a higher rate than they are consumed. Sunlight and wind, 
    for example, are such sources that are constantly being replenished. Renewable energy sources are plentiful and all around us. Fossil fuels 
    - coal, oil and gas - on the other hand, are non-renewable resources that take hundreds of millions of years to form. Fossil fuels, when burned 
    to produce energy, cause harmful greenhouse gas emissions, such as carbon dioxide. Renewable energy provides reliable power supplies and fuel 
    diversification, which enhance energy security and lower risk of fuel spills while reducing the need for imported fuels.
    """,
    """
    Cryptocurrencies are digital or virtual currencies that use cryptography for security and operate on decentralized networks based on blockchain 
    technology—a distributed ledger enforced by a disparate network of computers. A defining feature of cryptocurrencies is that they are generally 
    not issued by any central authority, rendering them theoretically immune to government interference or manipulation. Bitcoin, the first blockchain 
    cryptocurrency, is still the most popular and most valuable. Today, there are thousands of alternate cryptocurrencies with various functions and 
    specifications. Some of these are clones or forks of Bitcoin, while others are new currencies that were built from scratch.
    """,
    """
    Quantum computing is an area of computing focused on developing computer technology based on the principles of quantum theory, which explains 
    the behavior of energy and material on the atomic and subatomic levels. Quantum computers perform calculations based on the probability of an 
    object's state before it is measured - instead of just 1s or 0s - which means they have the potential to process exponentially more data compared 
    to classical computers. Classical computers, which include smartphones and laptops, encode information in binary "bits" that can either be 0s or 1s. 
    In a quantum computer, the basic unit of memory is a quantum bit or qubit.
    """
]

# Define embedding models to test
EMBEDDING_MODELS = [
    {
        "name": "all-MiniLM-L6-v2",
        "type": "huggingface",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384
    },
    {
        "name": "all-mpnet-base-v2",
        "type": "huggingface",
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "dimensions": 768
    },
    {
        "name": "bge-small-en-v1.5",
        "type": "huggingface",
        "model_name": "BAAI/bge-small-en-v1.5",
        "dimensions": 384
    },
    {
        "name": "bge-base-en-v1.5",
        "type": "huggingface",
        "model_name": "BAAI/bge-base-en-v1.5",
        "dimensions": 768
    }
]

# Add OpenAI model if API key is available
if os.getenv("OPENAI_API_KEY"):
    EMBEDDING_MODELS.append({
        "name": "text-embedding-3-small",
        "type": "openai",
        "model_name": "text-embedding-3-small",
        "dimensions": 1536
    })

# Batch sizes to test
BATCH_SIZES = [1, 5, 10, 25, 50, 100]

def get_embedding_model(model_config: Dict[str, Any]):
    """Get embedding model based on configuration."""
    if model_config["type"] == "huggingface":
        return HuggingFaceEmbeddings(model_name=model_config["model_name"])
    elif model_config["type"] == "openai":
        return OpenAIEmbeddings(model=model_config["model_name"])
    else:
        raise ValueError(f"Unsupported model type: {model_config['type']}")

def benchmark_embedding_generation(model_config: Dict[str, Any], texts: List[str], batch_size: int) -> Dict[str, Any]:
    """Benchmark embedding generation for a specific model and batch size."""
    logger.info(f"Benchmarking {model_config['name']} with batch size {batch_size}...")
    
    # Initialize embedding model
    embedding_model = get_embedding_model(model_config)
    
    # Prepare batches
    num_texts = len(texts)
    num_batches = (num_texts + batch_size - 1) // batch_size  # Ceiling division
    batches = [texts[i:i+batch_size] for i in range(0, num_texts, batch_size)]
    
    # Track resource usage
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Generate embeddings and measure time
    start_time = time.time()
    embeddings = []
    
    for batch in batches:
        if model_config["type"] == "huggingface":
            # HuggingFace embeddings
            batch_embeddings = embedding_model.embed_documents(batch)
        else:
            # OpenAI embeddings
            batch_embeddings = embedding_model.embed_documents(batch)
        
        embeddings.extend(batch_embeddings)
    
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time_per_text = total_time / num_texts
    texts_per_second = num_texts / total_time if total_time > 0 else 0
    memory_usage = end_memory - start_memory
    
    # Calculate embedding dimensions
    avg_dimensions = np.mean([len(emb) for emb in embeddings])
    
    # Calculate similarity matrix for quality assessment
    embeddings_np = np.array(embeddings)
    similarity_matrix = cosine_similarity(embeddings_np)
    
    # Calculate average similarity (excluding self-similarity)
    np.fill_diagonal(similarity_matrix, 0)  # Zero out diagonal
    avg_similarity = np.mean(similarity_matrix)
    
    return {
        "model_name": model_config["name"],
        "model_type": model_config["type"],
        "batch_size": batch_size,
        "num_texts": num_texts,
        "total_time": total_time,
        "avg_time_per_text": avg_time_per_text,
        "texts_per_second": texts_per_second,
        "memory_usage_mb": memory_usage,
        "avg_dimensions": avg_dimensions,
        "avg_similarity": avg_similarity
    }

def run_benchmarks():
    """Run all benchmarks."""
    logger.info("Starting embedding generation benchmarks...")
    
    results = []
    
    # Test with short texts
    logger.info("Benchmarking with short texts...")
    for model_config in EMBEDDING_MODELS:
        for batch_size in BATCH_SIZES:
            try:
                result = benchmark_embedding_generation(model_config, SAMPLE_TEXTS_SHORT, batch_size)
                result["text_type"] = "short"
                results.append(result)
            except Exception as e:
                logger.error(f"Error benchmarking {model_config['name']} with batch size {batch_size}: {e}")
    
    # Test with long texts
    logger.info("Benchmarking with long texts...")
    for model_config in EMBEDDING_MODELS:
        for batch_size in BATCH_SIZES:
            try:
                result = benchmark_embedding_generation(model_config, SAMPLE_TEXTS_LONG, batch_size)
                result["text_type"] = "long"
                results.append(result)
            except Exception as e:
                logger.error(f"Error benchmarking {model_config['name']} with batch size {batch_size}: {e}")
    
    # Save detailed results to JSON
    with open('research/results/embedding_benchmark_detailed.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    df.to_csv('research/results/embedding_benchmark_summary.csv', index=False)
    
    # Create pivot tables for analysis
    model_performance = df.pivot_table(
        index=['model_name', 'text_type'],
        columns=['batch_size'],
        values=['texts_per_second', 'avg_time_per_text'],
        aggfunc='mean'
    )
    
    model_performance.to_csv('research/results/embedding_model_performance.csv')
    
    logger.info("Embedding benchmarks completed. Results saved to research/results/")

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('research/results', exist_ok=True)
    
    # Run benchmarks
    run_benchmarks() 