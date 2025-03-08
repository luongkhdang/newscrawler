"""
LLM-related API endpoints using Groq.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from src.utils.groq_client import get_groq_client, GroqClient, DEFAULT_MODEL, LARGER_MODEL, MIXTRAL_MODEL

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