# NewsCrawler Research Scripts

This directory contains research scripts for the NewsCrawler project, focusing on benchmarking, testing, and optimizing various components of the pipeline.

## Overview

These scripts are part of the research plan outlined in `AI_WORKSPACE/PLANS/2025-03-10_Newspaper4k_PostgreSQL_LangChain_Pipeline_Research.md`. They are designed to evaluate different aspects of the system, from embedding generation to similarity search and retrieval-augmented generation.

## Prerequisites

Before running these scripts, make sure you have:

1. Set up the PostgreSQL database with pgvector extension
2. Installed all required dependencies from `requirements.txt`
3. Created a `.env` file with necessary environment variables (see `.env.example`)
4. Populated the database with some articles for testing

## Scripts

### Embedding Benchmarks

#### `embedding_benchmark.py`

Benchmarks the performance of different embedding models and configurations:
- Tests different embedding models (HuggingFace, OpenAI)
- Measures performance across different batch sizes
- Evaluates embedding quality using similarity tests
- Analyzes resource usage (CPU, memory, time)

**Usage:**
```bash
python research/embedding_benchmark.py
```

**Output:**
- `research/results/embedding_benchmark_detailed.json`: Detailed benchmark results
- `research/results/embedding_benchmark_summary.csv`: Summary of benchmark results
- `research/results/embedding_model_performance.csv`: Pivot table of model performance

### Similarity Search

#### `similarity_search_implementation.py`

Implements and tests similarity search using pgvector with:
- Different distance metrics (cosine, euclidean, dot product)
- Different index types (IVFFlat, HNSW)
- Performance benchmarking with varying dataset sizes
- Hybrid search combining vector and keyword search

**Usage:**
```bash
python research/similarity_search_implementation.py
```

**Output:**
- `research/results/index_performance_benchmark.csv`: Performance comparison of different indexes
- `research/results/hybrid_search_benchmark.csv`: Performance comparison of search methods
- `research/results/search_quality_evaluation.json`: Detailed evaluation of search quality

### Retrieval-Augmented Generation (RAG)

#### `rag_implementation.py`

Implements and tests a complete RAG pipeline using:
- Vector search for relevant article retrieval
- Hybrid search combining vector and keyword search
- GroqAI for text generation
- Local LLM fallback options
- Performance and quality evaluation

**Usage:**
```bash
python research/rag_implementation.py
```

**Output:**
- `research/results/retrieval_methods_detailed.json`: Detailed evaluation of retrieval methods
- `research/results/retrieval_methods_summary.csv`: Summary of retrieval performance
- `research/results/rag_configurations_detailed.json`: Detailed evaluation of RAG configurations
- `research/results/rag_configurations_summary.csv`: Summary of RAG performance

## Results Directory

All benchmark and test results are saved to the `research/results/` directory. This directory is created automatically if it doesn't exist.

## Environment Variables

The scripts require the following environment variables:

```
# Database connection
DATABASE_URL=postgresql://newscrawler:newscrawler_password@localhost:5432/newscrawler

# GroqAI API
GROQ_API_KEY=your_groq_api_key

# Optional: OpenAI API for embedding comparison
OPENAI_API_KEY=your_openai_api_key
```

## Key Findings

Based on our research, we recommend:

1. **Embedding Model**: BAAI/bge-small-en-v1.5 provides the best balance of performance and quality
2. **Chunking Strategy**: RecursiveCharacterTextSplitter with 500 character chunks and 100 character overlap
3. **Vector Search**: HNSW indexes with m=16 and ef_construction=64
4. **Distance Metric**: Cosine similarity for semantic search
5. **Hybrid Search**: Combining vector similarity (70%) with keyword search (30%)
6. **RAG Pipeline**: GroqAI with enhanced prompting and hybrid search retrieval
7. **Batch Processing**: Optimal batch size of 25-50 documents for embedding generation

## Next Steps

After completing the LangChain integration research, the next phases will focus on:

1. FastAPI implementation research
2. Docker containerization research
3. End-to-end pipeline testing
4. Documentation and knowledge transfer 