"""
LLM-related API endpoints using Groq.
"""

from fastapi import APIRouter, Depends, HTTPException, Body, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from datetime import datetime

from src.utils.groq_client import get_groq_client, GroqClient, DEFAULT_MODEL, LARGER_MODEL, MIXTRAL_MODEL
from src.llm.rag import RAGSystem
from src.database.session import get_db
from src.vector.embedding_models import MODEL_DIMENSIONS

router = APIRouter(prefix="/llm", tags=["llm"])


class TextGenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="The prompt to generate text from")
    model: str = Field(DEFAULT_MODEL, description="The model to use for generation")
    max_tokens: int = Field(1024, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Temperature for generation (higher = more creative)")
    system_message: Optional[str] = Field(None, description="Optional system message for context")


class TextGenerationResponse(BaseModel):
    """Response model for text generation."""
    generated_text: str = Field(..., description="The generated text")
    model: str = Field(..., description="The model used for generation")


class SummarizeRequest(BaseModel):
    """Request model for text summarization."""
    text: str = Field(..., description="The text to summarize")
    max_length: int = Field(200, description="Approximate maximum length of summary in words")


class SummarizeResponse(BaseModel):
    """Response model for text summarization."""
    summary: str = Field(..., description="The summarized text")


class SentimentAnalysisRequest(BaseModel):
    """Request model for sentiment analysis."""
    text: str = Field(..., description="The text to analyze")


class SentimentAnalysisResponse(BaseModel):
    """Response model for sentiment analysis."""
    analysis: str = Field(..., description="The sentiment analysis results")


class EntityExtractionRequest(BaseModel):
    """Request model for entity extraction."""
    text: str = Field(..., description="The text to extract entities from")


class EntityExtractionResponse(BaseModel):
    """Response model for entity extraction."""
    entities: str = Field(..., description="The extracted entities")


class AvailableModelsResponse(BaseModel):
    """Response model for available models."""
    models: List[Dict[str, Any]] = Field(..., description="List of available models")


class QuestionAnsweringRequest(BaseModel):
    """Request model for question answering using RAG."""
    question: str = Field(..., description="The question to answer")
    model: str = Field(LARGER_MODEL, description="The model to use for generation")
    max_tokens: int = Field(1024, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Temperature for generation")
    context_strategy: Optional[str] = Field("semantic", description="Strategy for context assembly (simple, chunked, semantic, temporal)")
    embedding_model: Optional[str] = Field(None, description="Embedding model to use")
    source_domain: Optional[str] = Field(None, description="Filter by source domain")
    category: Optional[str] = Field(None, description="Filter by category")
    start_date: Optional[datetime] = Field(None, description="Filter by start date")
    end_date: Optional[datetime] = Field(None, description="Filter by end date")


class QuestionAnsweringResponse(BaseModel):
    """Response model for question answering using RAG."""
    answer: str = Field(..., description="The answer to the question")
    sources: List[Dict[str, Any]] = Field(..., description="Sources used to generate the answer")
    evaluation: Optional[Dict[str, Any]] = Field(None, description="Evaluation metrics for the response")


class QuestionWithContextRequest(BaseModel):
    """Request model for question answering with specific context."""
    question: str = Field(..., description="The question to answer")
    context_article_ids: List[str] = Field(..., description="IDs of articles to use as context")
    model: str = Field(LARGER_MODEL, description="The model to use for generation")
    max_tokens: int = Field(1024, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Temperature for generation")
    context_strategy: Optional[str] = Field("semantic", description="Strategy for context assembly (simple, chunked, semantic, temporal)")


class ArticleSummarizeRequest(BaseModel):
    """Request model for article summarization."""
    article_id: str = Field(..., description="ID of the article to summarize")
    model: str = Field(LARGER_MODEL, description="The model to use for generation")
    max_tokens: int = Field(512, description="Maximum number of tokens to generate")
    temperature: float = Field(0.3, description="Temperature for generation")


class ArticleSummarizeResponse(BaseModel):
    """Response model for article summarization."""
    summary: str = Field(..., description="The summary of the article")
    article_id: str = Field(..., description="ID of the article")
    title: str = Field(..., description="Title of the article")
    url: str = Field(..., description="URL of the article")
    published_date: Optional[str] = Field(None, description="Publication date of the article")
    source_domain: str = Field(..., description="Source domain of the article")


class EmbeddingModelResponse(BaseModel):
    """Response model for available embedding models."""
    models: List[Dict[str, Any]] = Field(..., description="List of available embedding models")


class RAGMetricsResponse(BaseModel):
    """Response model for RAG metrics."""
    metrics: Dict[str, Any] = Field(..., description="RAG evaluation metrics")


class RAGCacheStatsResponse(BaseModel):
    """Response model for RAG cache statistics."""
    stats: Dict[str, Any] = Field(..., description="RAG cache statistics")


@router.get("/models", response_model=AvailableModelsResponse)
async def get_available_models():
    """Get a list of available LLM models."""
    models = [
        {
            "id": DEFAULT_MODEL,
            "name": "Llama 3 8B",
            "description": "Smaller, faster model suitable for most tasks",
            "context_length": 8192,
            "is_default": True
        },
        {
            "id": LARGER_MODEL,
            "name": "Llama 3 70B",
            "description": "Larger, more capable model for complex tasks",
            "context_length": 8192,
            "is_default": False
        },
        {
            "id": MIXTRAL_MODEL,
            "name": "Mixtral 8x7B",
            "description": "Mixture of experts model with large context window",
            "context_length": 32768,
            "is_default": False
        }
    ]
    
    return {"models": models}


@router.get("/embedding-models", response_model=EmbeddingModelResponse)
async def get_embedding_models():
    """Get a list of available embedding models."""
    models = [
        {
            "id": model_name,
            "name": model_name.split("/")[-1],
            "dimension": dimension,
            "provider": model_name.split("/")[0] if "/" in model_name else "unknown"
        }
        for model_name, dimension in MODEL_DIMENSIONS.items()
    ]
    
    return {"models": models}


@router.get("/context-strategies")
async def get_context_strategies():
    """Get a list of available context assembly strategies."""
    strategies = [
        {
            "id": "simple",
            "name": "Simple",
            "description": "Basic concatenation of articles"
        },
        {
            "id": "chunked",
            "name": "Chunked",
            "description": "Chunks articles and selects the most relevant chunks"
        },
        {
            "id": "semantic",
            "name": "Semantic",
            "description": "Uses semantic similarity to rerank and filter content"
        },
        {
            "id": "temporal",
            "name": "Temporal",
            "description": "Prioritizes recent articles"
        }
    ]
    
    return {"strategies": strategies}


@router.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """Generate text using the Groq API."""
    try:
        client = get_groq_client()
        generated_text = client.generate_completion(
            prompt=request.prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_message=request.system_message
        )
        
        return {
            "generated_text": generated_text,
            "model": request.model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    """Summarize text using the Groq API."""
    try:
        client = get_groq_client()
        summary = client.summarize_text(
            text=request.text,
            max_length=request.max_length
        )
        
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing text: {str(e)}")


@router.post("/analyze-sentiment", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(request: SentimentAnalysisRequest):
    """Analyze sentiment of text using the Groq API."""
    try:
        client = get_groq_client()
        result = client.analyze_sentiment(text=request.text)
        
        return {"analysis": result["analysis"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")


@router.post("/extract-entities", response_model=EntityExtractionResponse)
async def extract_entities(request: EntityExtractionRequest):
    """Extract entities from text using the Groq API."""
    try:
        client = get_groq_client()
        result = client.extract_entities(text=request.text)
        
        return {"entities": result["entities"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting entities: {str(e)}")


@router.post("/ask", response_model=QuestionAnsweringResponse)
async def answer_question(
    request: QuestionAnsweringRequest,
    db: Session = Depends(get_db)
):
    """Answer a question using RAG."""
    try:
        # Create filter criteria
        filter_criteria = {}
        if request.source_domain:
            filter_criteria["source_domain"] = request.source_domain
        if request.category:
            filter_criteria["category"] = request.category
        if request.start_date:
            filter_criteria["start_date"] = request.start_date
        if request.end_date:
            filter_criteria["end_date"] = request.end_date
        
        # Initialize RAG system with specified parameters
        rag_system = RAGSystem(
            db_session=db,
            context_strategy=request.context_strategy,
            embedding_model=request.embedding_model
        )
        
        # Answer the question
        result = rag_system.answer_question(
            query=request.question,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            filter_criteria=filter_criteria
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")


@router.post("/ask-with-context", response_model=QuestionAnsweringResponse)
async def answer_with_context(
    request: QuestionWithContextRequest,
    db: Session = Depends(get_db)
):
    """Answer a question using specific articles as context."""
    try:
        # Initialize RAG system with specified parameters
        rag_system = RAGSystem(
            db_session=db,
            context_strategy=request.context_strategy
        )
        
        # Answer the question with specific context
        result = rag_system.answer_with_specific_context(
            query=request.question,
            article_ids=request.context_article_ids,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question with context: {str(e)}")


@router.post("/summarize-article", response_model=ArticleSummarizeResponse)
async def summarize_article(
    request: ArticleSummarizeRequest,
    db: Session = Depends(get_db)
):
    """Summarize an article."""
    try:
        # Initialize RAG system
        rag_system = RAGSystem(db_session=db)
        
        # Summarize the article
        result = rag_system.summarize_article(
            article_id=request.article_id,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing article: {str(e)}")


@router.get("/metrics", response_model=RAGMetricsResponse)
async def get_rag_metrics(
    last_n: Optional[int] = Query(None, description="Only consider the last N evaluations"),
    db: Session = Depends(get_db)
):
    """Get RAG evaluation metrics."""
    try:
        # Initialize RAG system with evaluation enabled
        rag_system = RAGSystem(db_session=db, enable_evaluation=True)
        
        # Get metrics
        metrics = rag_system.get_evaluation_metrics(last_n)
        
        if "error" in metrics:
            raise HTTPException(status_code=400, detail=metrics["error"])
        
        return {"metrics": metrics}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting RAG metrics: {str(e)}")


@router.get("/cache-stats", response_model=RAGCacheStatsResponse)
async def get_cache_stats(db: Session = Depends(get_db)):
    """Get RAG cache statistics."""
    try:
        # Initialize RAG system with caching enabled
        rag_system = RAGSystem(db_session=db, enable_cache=True)
        
        # Get cache stats
        stats = rag_system.get_cache_stats()
        
        if "error" in stats:
            raise HTTPException(status_code=400, detail=stats["error"])
        
        return {"stats": stats}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cache stats: {str(e)}")


@router.post("/clear-cache", status_code=204)
async def clear_cache(db: Session = Depends(get_db)):
    """Clear the RAG response cache."""
    try:
        # Initialize RAG system with caching enabled
        rag_system = RAGSystem(db_session=db, enable_cache=True)
        
        # Clear cache
        rag_system.clear_cache()
        
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}") 