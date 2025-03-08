"""
Groq API client utility for the NewsCrawler application.
This module provides a wrapper around the Groq API for LLM capabilities.
"""

import os
from typing import List, Dict, Any, Optional
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

# Default models
DEFAULT_MODEL = "llama3-8b-8192"
LARGER_MODEL = "llama3-70b-8192"
MIXTRAL_MODEL = "mixtral-8x7b-32768"

class GroqClient:
    """
    A client for interacting with the Groq API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Groq client.
        
        Args:
            api_key: The Groq API key. If not provided, it will be read from the environment variable GROQ_API_KEY.
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is required. Provide it directly or set the GROQ_API_KEY environment variable.")
        
        self.client = Groq(api_key=self.api_key)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_completion(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate a completion using the Groq API.
        
        Args:
            prompt: The user prompt to generate a completion for.
            model: The model to use for generation.
            max_tokens: The maximum number of tokens to generate.
            temperature: The temperature for generation (higher = more creative, lower = more deterministic).
            system_message: Optional system message to provide context or instructions.
            
        Returns:
            The generated text.
        """
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        # Generate completion
        response = self.client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """
        Summarize a text using the Groq API.
        
        Args:
            text: The text to summarize.
            max_length: The approximate maximum length of the summary in words.
            
        Returns:
            The summarized text.
        """
        system_message = f"You are a helpful assistant that summarizes text. Create a concise summary in approximately {max_length} words."
        prompt = f"Please summarize the following text:\n\n{text}"
        
        return self.generate_completion(
            prompt=prompt,
            system_message=system_message,
            temperature=0.3  # Lower temperature for more factual summaries
        )
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of a text using the Groq API.
        
        Args:
            text: The text to analyze.
            
        Returns:
            A dictionary containing sentiment analysis results.
        """
        system_message = "You are a sentiment analysis expert. Analyze the sentiment of the provided text and return a JSON object with the following fields: sentiment (positive, negative, or neutral), confidence (0-1), and a brief explanation."
        prompt = f"Analyze the sentiment of the following text:\n\n{text}"
        
        response = self.generate_completion(
            prompt=prompt,
            system_message=system_message,
            temperature=0.1  # Very low temperature for consistent analysis
        )
        
        # Note: In a production environment, you would want to parse this response
        # into a proper JSON object and handle potential parsing errors
        return {"analysis": response}
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract named entities from a text using the Groq API.
        
        Args:
            text: The text to extract entities from.
            
        Returns:
            A dictionary containing the extracted entities.
        """
        system_message = "You are an expert at named entity recognition. Extract all named entities from the provided text and categorize them (e.g., person, organization, location, date, etc.)."
        prompt = f"Extract named entities from the following text:\n\n{text}"
        
        response = self.generate_completion(
            prompt=prompt,
            system_message=system_message,
            temperature=0.2
        )
        
        return {"entities": response}


# Singleton instance for easy import
_groq_client = None

def get_groq_client() -> GroqClient:
    """
    Get or create a singleton instance of the GroqClient.
    
    Returns:
        A GroqClient instance.
    """
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqClient()
    return _groq_client 