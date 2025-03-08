"""
Chunking Strategies Analysis Script

This script tests different text chunking strategies for optimal retrieval performance.
It evaluates:
1. Different chunk sizes
2. Different overlap sizes
3. Different chunking methods (character, token, sentence, paragraph)
4. Impact on retrieval quality
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
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('research/chunking_strategies.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Sample long article for chunking tests
SAMPLE_ARTICLE = """
# The Future of Artificial Intelligence: Opportunities and Challenges

Artificial intelligence (AI) has emerged as one of the most transformative technologies of the 21st century. From virtual assistants like Siri and Alexa to sophisticated recommendation systems on platforms like Netflix and Amazon, AI has become an integral part of our daily lives. This article explores the future of AI, examining both the opportunities it presents and the challenges it poses.

## Current State of AI

Modern AI systems can be broadly categorized into narrow AI and general AI. Narrow AI, also known as weak AI, is designed to perform specific tasks, such as facial recognition or language translation. These systems have achieved remarkable success in recent years, with some even surpassing human performance in specific domains.

General AI, or strong AI, refers to systems that possess human-like intelligence across a wide range of tasks. Despite significant progress in narrow AI, general AI remains largely theoretical, with many experts believing it is still decades away from realization.

## Opportunities

### Healthcare

AI has the potential to revolutionize healthcare in numerous ways. Machine learning algorithms can analyze medical images with greater accuracy than human radiologists, potentially leading to earlier and more accurate diagnoses of conditions like cancer. AI can also help in drug discovery by predicting how different compounds will interact with biological targets, significantly reducing the time and cost of developing new medications.

Personalized medicine is another promising application of AI in healthcare. By analyzing a patient's genetic information, medical history, and lifestyle factors, AI systems can help doctors develop tailored treatment plans that are more effective and have fewer side effects.

### Transportation

Self-driving vehicles represent one of the most visible applications of AI. Companies like Tesla, Waymo, and Uber are investing heavily in autonomous vehicle technology, which has the potential to reduce accidents, ease traffic congestion, and provide mobility for those unable to drive.

Beyond cars, AI is being applied to optimize traffic flow in cities, improve public transportation systems, and enhance the efficiency of logistics and supply chain operations.

### Education

AI has the potential to transform education by providing personalized learning experiences tailored to each student's needs, abilities, and learning style. Intelligent tutoring systems can adapt to a student's progress, providing additional support in areas where they struggle and advancing more quickly through material they grasp easily.

AI can also assist teachers by automating administrative tasks, providing insights into student performance, and identifying students who may need additional help.

## Challenges

### Ethics and Bias

As AI systems become more prevalent, ethical concerns about their development and deployment have come to the forefront. One major concern is bias in AI systems, which can arise when the data used to train these systems reflects existing societal biases. For example, facial recognition systems have been shown to perform less accurately on women and people with darker skin tones, potentially leading to discriminatory outcomes.

Ensuring that AI systems are developed and used ethically requires a multidisciplinary approach, involving not just technologists but also ethicists, policymakers, and representatives from diverse communities.

### Privacy and Security

AI systems often rely on vast amounts of data, raising concerns about privacy and security. As these systems become more integrated into our lives, they collect increasingly sensitive information about us, from our shopping habits to our health status.

Ensuring that this data is collected, stored, and used in ways that respect privacy and maintain security is a significant challenge. This includes protecting against both unauthorized access to data and the misuse of data by those who have legitimate access.

### Employment and Economic Disruption

While AI has the potential to create new jobs and increase productivity, it also poses challenges for employment. Automation powered by AI could displace workers in various sectors, from manufacturing to customer service. The pace of this displacement and the ability of workers to transition to new roles will significantly impact the economic and social consequences of AI adoption.

Addressing these challenges may require new approaches to education and training, as well as social safety nets to support those affected by technological unemployment.

## The Path Forward

Navigating the future of AI requires a balanced approach that maximizes its benefits while addressing its challenges. This includes:

1. **Ethical Guidelines and Regulation**: Developing clear ethical guidelines and appropriate regulatory frameworks for AI development and deployment.

2. **Inclusive Development**: Ensuring that AI development includes diverse perspectives to minimize bias and ensure that benefits are widely shared.

3. **Education and Training**: Preparing the workforce for an AI-driven economy through education and training programs.

4. **Research and Innovation**: Continuing to invest in AI research to advance the field while addressing its limitations and challenges.

5. **International Cooperation**: Collaborating across national boundaries to address global challenges posed by AI.

By taking a thoughtful and proactive approach to AI development and deployment, we can harness its potential to address some of humanity's most pressing challenges while mitigating its risks.
"""

# Test queries for retrieval evaluation
TEST_QUERIES = [
    "How is AI being used in healthcare?",
    "What are the ethical concerns about AI?",
    "How might AI affect employment?",
    "What is the difference between narrow AI and general AI?",
    "What steps should be taken to ensure responsible AI development?"
]

# Define chunking strategies to test
CHUNKING_STRATEGIES = [
    {
        "name": "RecursiveCharacter-500-0",
        "splitter": RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0),
        "description": "Recursive character splitting with 500 chars and no overlap"
    },
    {
        "name": "RecursiveCharacter-500-100",
        "splitter": RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100),
        "description": "Recursive character splitting with 500 chars and 100 char overlap"
    },
    {
        "name": "RecursiveCharacter-1000-200",
        "splitter": RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200),
        "description": "Recursive character splitting with 1000 chars and 200 char overlap"
    },
    {
        "name": "RecursiveCharacter-1500-300",
        "splitter": RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300),
        "description": "Recursive character splitting with 1500 chars and 300 char overlap"
    },
    {
        "name": "Character-1000-0",
        "splitter": CharacterTextSplitter(chunk_size=1000, chunk_overlap=0),
        "description": "Simple character splitting with 1000 chars and no overlap"
    },
    {
        "name": "Character-1000-200",
        "splitter": CharacterTextSplitter(chunk_size=1000, chunk_overlap=200),
        "description": "Simple character splitting with 1000 chars and 200 char overlap"
    },
    {
        "name": "Token-256-0",
        "splitter": TokenTextSplitter(chunk_size=256, chunk_overlap=0),
        "description": "Token-based splitting with 256 tokens and no overlap"
    },
    {
        "name": "Token-256-32",
        "splitter": TokenTextSplitter(chunk_size=256, chunk_overlap=32),
        "description": "Token-based splitting with 256 tokens and 32 token overlap"
    },
    {
        "name": "Token-512-64",
        "splitter": TokenTextSplitter(chunk_size=512, chunk_overlap=64),
        "description": "Token-based splitting with 512 tokens and 64 token overlap"
    },
    {
        "name": "SentenceTransformers-256-32",
        "splitter": SentenceTransformersTokenTextSplitter(chunk_size=256, chunk_overlap=32),
        "description": "SentenceTransformers token splitting with 256 tokens and 32 token overlap"
    }
]

def chunk_text(splitter, text: str, metadata: Dict[str, Any]) -> List[Document]:
    """Split text into chunks using the specified splitter."""
    return splitter.create_documents(texts=[text], metadatas=[metadata])

def evaluate_chunking_strategy(strategy: Dict[str, Any], article: str, queries: List[str]) -> Dict[str, Any]:
    """Evaluate a chunking strategy for retrieval performance."""
    logger.info(f"Evaluating chunking strategy: {strategy['name']}...")
    
    # Create chunks
    start_time = time.time()
    chunks = chunk_text(
        strategy['splitter'], 
        article, 
        {"source": "sample_article", "title": "The Future of Artificial Intelligence"}
    )
    chunking_time = time.time() - start_time
    
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Generate embeddings for chunks
    start_time = time.time()
    chunk_embeddings = []
    for chunk in chunks:
        embedding = embedding_model.embed_query(chunk.page_content)
        chunk_embeddings.append(embedding)
    embedding_time = time.time() - start_time
    
    # Generate embeddings for queries
    query_embeddings = []
    for query in queries:
        embedding = embedding_model.embed_query(query)
        query_embeddings.append(embedding)
    
    # Evaluate retrieval performance
    retrieval_results = []
    for i, query_embedding in enumerate(query_embeddings):
        # Calculate similarity with each chunk
        similarities = []
        for chunk_embedding in chunk_embeddings:
            query_np = np.array(query_embedding).reshape(1, -1)
            chunk_np = np.array(chunk_embedding).reshape(1, -1)
            similarity = cosine_similarity(query_np, chunk_np)[0][0]
            similarities.append(similarity)
        
        # Get top 3 chunks
        if similarities:
            top_indices = np.argsort(similarities)[-3:][::-1]
            top_similarities = [similarities[i] for i in top_indices]
            top_chunks = [chunks[i].page_content[:100] + "..." for i in top_indices]  # First 100 chars
        else:
            top_indices = []
            top_similarities = []
            top_chunks = []
        
        retrieval_results.append({
            "query": queries[i],
            "top_indices": top_indices.tolist() if isinstance(top_indices, np.ndarray) else top_indices,
            "top_similarities": top_similarities,
            "top_chunks": top_chunks
        })
    
    # Calculate average similarity of top result
    avg_top_similarity = np.mean([result["top_similarities"][0] if result["top_similarities"] else 0 
                                 for result in retrieval_results])
    
    return {
        "strategy_name": strategy["name"],
        "strategy_description": strategy["description"],
        "num_chunks": len(chunks),
        "avg_chunk_size": np.mean([len(chunk.page_content) for chunk in chunks]) if chunks else 0,
        "chunking_time": chunking_time,
        "embedding_time": embedding_time,
        "avg_top_similarity": avg_top_similarity,
        "retrieval_results": retrieval_results
    }

def main():
    """Main function to run the chunking strategies analysis."""
    logger.info("Starting chunking strategies analysis...")
    
    results = []
    for strategy in CHUNKING_STRATEGIES:
        try:
            result = evaluate_chunking_strategy(strategy, SAMPLE_ARTICLE, TEST_QUERIES)
            results.append(result)
        except Exception as e:
            logger.error(f"Error evaluating strategy {strategy['name']}: {e}")
    
    # Save detailed results to JSON
    with open('research/results/chunking_strategies_detailed.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary DataFrame
    summary = []
    for result in results:
        summary.append({
            "strategy_name": result["strategy_name"],
            "strategy_description": result["strategy_description"],
            "num_chunks": result["num_chunks"],
            "avg_chunk_size": result["avg_chunk_size"],
            "chunking_time": result["chunking_time"],
            "embedding_time": result["embedding_time"],
            "avg_top_similarity": result["avg_top_similarity"]
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('research/results/chunking_strategies_summary.csv', index=False)
    
    logger.info("Chunking strategies analysis completed. Results saved to research/results/chunking_strategies_summary.csv")

if __name__ == "__main__":
    main() 