"""
Embedding models module for the vector search system.
This module provides support for different embedding models.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Callable
import numpy as np
from functools import lru_cache

logger = logging.getLogger(__name__)

# Default embedding dimensions for different models
MODEL_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/multi-qa-mpnet-base-dot-v1": 768,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
    "sentence-transformers/all-distilroberta-v1": 768,
    "openai/text-embedding-ada-002": 1536,
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-3-large": 3072,
}

# Default model to use
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """
    Base class for embedding models.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the embedding model
        """
        self.model_name = model_name
        self.dimension = MODEL_DIMENSIONS.get(model_name, 384)
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension
        """
        return self.dimension


class SentenceTransformersModel(EmbeddingModel):
    """
    Embedding model using sentence-transformers.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize the sentence-transformers model.
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        super().__init__(model_name)
        self.model = None
    
    def _load_model(self):
        """Load the model if not already loaded."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded sentence-transformers model: {self.model_name}")
            except ImportError:
                logger.error("sentence-transformers not installed. Please install with: pip install sentence-transformers")
                raise
            except Exception as e:
                logger.error(f"Error loading sentence-transformers model: {e}")
                raise
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        self._load_model()
        
        # Truncate text if too long (most models have a token limit)
        # A rough approximation: 1 token ≈ 4 characters in English
        max_chars = 8192 * 4  # Assuming a max of 8192 tokens
        if len(text) > max_chars:
            text = text[:max_chars]
        
        # Generate embedding
        embedding = self.model.encode(text)
        
        # Convert to list of floats
        return embedding.tolist()


class OpenAIModel(EmbeddingModel):
    """
    Embedding model using OpenAI API.
    """
    
    def __init__(self, model_name: str = "openai/text-embedding-3-small"):
        """
        Initialize the OpenAI model.
        
        Args:
            model_name: Name of the OpenAI model
        """
        super().__init__(model_name)
        self.client = None
        self.api_key = os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.warning("OPENAI_API_KEY environment variable not set. OpenAI embeddings will not work.")
    
    def _initialize_client(self):
        """Initialize the OpenAI client if not already initialized."""
        if self.client is None:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"Initialized OpenAI client for model: {self.model_name}")
            except ImportError:
                logger.error("openai package not installed. Please install with: pip install openai")
                raise
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {e}")
                raise
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Cannot generate OpenAI embeddings.")
        
        self._initialize_client()
        
        # Extract the model name without the "openai/" prefix
        model_name = self.model_name.replace("openai/", "")
        
        # Generate embedding
        response = self.client.embeddings.create(
            input=text,
            model=model_name
        )
        
        # Extract embedding
        embedding = response.data[0].embedding
        
        return embedding


class HuggingFaceModel(EmbeddingModel):
    """
    Embedding model using Hugging Face Transformers.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the Hugging Face model.
        
        Args:
            model_name: Name of the Hugging Face model
        """
        super().__init__(model_name)
        self.tokenizer = None
        self.model = None
    
    def _load_model(self):
        """Load the model if not already loaded."""
        if self.model is None or self.tokenizer is None:
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                
                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                
                # Set to evaluation mode
                self.model.eval()
                
                logger.info(f"Loaded Hugging Face model: {self.model_name}")
            except ImportError:
                logger.error("transformers not installed. Please install with: pip install transformers torch")
                raise
            except Exception as e:
                logger.error(f"Error loading Hugging Face model: {e}")
                raise
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        self._load_model()
        
        try:
            import torch
            
            # Tokenize and prepare input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use mean pooling to get a single vector
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.sum(input_mask_expanded, 1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            
            # Convert to list of floats
            return mean_embeddings[0].tolist()
        except Exception as e:
            logger.error(f"Error generating embedding with Hugging Face model: {e}")
            raise


# Factory function to get embedding model
def get_embedding_model(model_name: Optional[str] = None) -> EmbeddingModel:
    """
    Get an embedding model instance.
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        EmbeddingModel instance
    """
    if model_name is None:
        model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
    
    # Determine model type
    if model_name.startswith("openai/"):
        return OpenAIModel(model_name)
    elif model_name.startswith("sentence-transformers/"):
        return SentenceTransformersModel(model_name)
    else:
        # Default to Hugging Face
        return HuggingFaceModel(model_name)


# Cache for embedding models
_embedding_models: Dict[str, EmbeddingModel] = {}

def get_cached_embedding_model(model_name: Optional[str] = None) -> EmbeddingModel:
    """
    Get a cached embedding model instance.
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        EmbeddingModel instance
    """
    if model_name is None:
        model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
    
    if model_name not in _embedding_models:
        _embedding_models[model_name] = get_embedding_model(model_name)
    
    return _embedding_models[model_name]


@lru_cache(maxsize=1024)
def generate_embedding_with_model(text: str, model_name: Optional[str] = None) -> List[float]:
    """
    Generate an embedding for the given text using the specified model.
    
    Args:
        text: Input text
        model_name: Name of the embedding model
        
    Returns:
        Embedding vector
    """
    model = get_cached_embedding_model(model_name)
    return model.get_embedding(text) 