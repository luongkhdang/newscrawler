"""
Embedding Model Comparison Script

This script compares different embedding models for quality and performance.
It focuses on HuggingFace embeddings for local deployment and compatibility with GroqAI.
It evaluates models on:
1. Embedding quality (semantic similarity)
2. Generation speed
3. Dimensionality and storage requirements
4. Clustering quality
"""

import os
import sys
import time
import logging
import json
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from langchain_huggingface import HuggingFaceEmbeddings

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('research/embedding_comparison.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Test data
TEST_ARTICLES = [
    {
        "title": "The Impact of Artificial Intelligence on Modern Healthcare",
        "content": "Artificial intelligence is revolutionizing healthcare by improving diagnosis accuracy, treatment planning, and patient care. Machine learning algorithms can analyze medical images, predict disease progression, and personalize treatment plans based on patient data."
    },
    {
        "title": "Advances in Renewable Energy Technologies",
        "content": "Renewable energy technologies like solar, wind, and hydroelectric power are becoming more efficient and cost-effective. These sustainable energy sources are crucial for reducing carbon emissions and combating climate change."
    },
    {
        "title": "The Future of Autonomous Vehicles",
        "content": "Self-driving cars are poised to transform transportation by making roads safer and reducing traffic congestion. Companies are investing heavily in developing autonomous vehicle technology, with some experts predicting widespread adoption within the next decade."
    },
    {
        "title": "Understanding Machine Learning Algorithms",
        "content": "Machine learning algorithms enable computers to learn from data and make predictions without explicit programming. These algorithms power many AI applications, from recommendation systems to natural language processing."
    },
    {
        "title": "Climate Change Impacts on Global Agriculture",
        "content": "Climate change is affecting agricultural productivity worldwide through changing precipitation patterns, temperature increases, and extreme weather events. Farmers are adapting by implementing sustainable farming practices and developing drought-resistant crops."
    }
]

# Test queries for semantic search evaluation
TEST_QUERIES = [
    "How is AI being used in healthcare?",
    "What are the benefits of renewable energy?",
    "When will self-driving cars become common?",
    "Explain how machine learning works",
    "How does climate change affect farming?"
]

# Define embedding models to test
def get_embedding_models():
    """Get embedding models to test."""
    models = []
    
    # HuggingFace embeddings - focusing on models that work well with GroqAI and local deployment
    huggingface_models = [
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5"
    ]
    
    for model_name in huggingface_models:
        models.append((model_name, HuggingFaceEmbeddings(model_name=model_name)))
    
    return models

def generate_embeddings(model, texts: List[str]) -> Tuple[List[List[float]], float]:
    """Generate embeddings for a list of texts and measure performance."""
    start_time = time.time()
    
    embeddings = []
    for text in texts:
        embedding = model.embed_query(text)
        embeddings.append(embedding)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return embeddings, total_time

def evaluate_semantic_search(model_name: str, model, articles: List[Dict[str, str]], queries: List[str]) -> Dict[str, Any]:
    """Evaluate semantic search quality using the model."""
    logger.info(f"Evaluating semantic search for {model_name}...")
    
    # Generate embeddings for articles (combine title and content)
    article_texts = [f"{article['title']}\n\n{article['content']}" for article in articles]
    article_embeddings, article_embedding_time = generate_embeddings(model, article_texts)
    
    # Generate embeddings for queries
    query_embeddings, query_embedding_time = generate_embeddings(model, queries)
    
    # Calculate similarity scores
    similarity_scores = []
    for query_embedding in query_embeddings:
        scores = []
        for article_embedding in article_embeddings:
            # Convert to numpy arrays for cosine similarity calculation
            query_np = np.array(query_embedding).reshape(1, -1)
            article_np = np.array(article_embedding).reshape(1, -1)
            score = cosine_similarity(query_np, article_np)[0][0]
            scores.append(score)
        similarity_scores.append(scores)
    
    # Calculate average embedding dimension
    avg_dimension = len(article_embeddings[0]) if article_embeddings else 0
    
    # Calculate average embedding time per document
    avg_embedding_time = article_embedding_time / len(articles) if articles else 0
    
    # Perform clustering to evaluate embedding quality
    embeddings_array = np.array(article_embeddings)
    kmeans = KMeans(n_clusters=min(len(articles), 3), random_state=42)
    clusters = kmeans.fit_predict(embeddings_array)
    
    # Calculate inertia (sum of squared distances to closest centroid)
    inertia = kmeans.inertia_
    
    return {
        "model_name": model_name,
        "embedding_dimension": avg_dimension,
        "avg_embedding_time_per_doc": avg_embedding_time,
        "total_embedding_time": article_embedding_time + query_embedding_time,
        "similarity_scores": similarity_scores,
        "clustering_inertia": inertia,
        "clusters": clusters.tolist()
    }

def main():
    """Main function to run the embedding model comparison."""
    logger.info("Starting embedding model comparison...")
    
    # Get embedding models to test
    models = get_embedding_models()
    
    results = []
    for model_name, model in models:
        try:
            logger.info(f"Testing {model_name}...")
            result = evaluate_semantic_search(model_name, model, TEST_ARTICLES, TEST_QUERIES)
            results.append(result)
        except Exception as e:
            logger.error(f"Error testing {model_name}: {e}")
    
    # Save detailed results to JSON
    with open('research/results/embedding_comparison_detailed.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary DataFrame
    summary = []
    for result in results:
        summary.append({
            "model_name": result["model_name"],
            "embedding_dimension": result["embedding_dimension"],
            "avg_embedding_time_per_doc": result["avg_embedding_time_per_doc"],
            "total_embedding_time": result["total_embedding_time"],
            "clustering_inertia": result["clustering_inertia"]
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('research/results/embedding_comparison_summary.csv', index=False)
    
    logger.info("Embedding model comparison completed. Results saved to research/results/embedding_comparison_summary.csv")

if __name__ == "__main__":
    main() 