"""
Test script to verify that the Groq API integration works in the API.
"""

import os
import sys
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up the API URL
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"

def test_llm_endpoints():
    """Test the LLM endpoints."""
    print("\n=== Testing LLM Endpoints ===")
    
    # Test the models endpoint
    print("\nTesting GET /llm/models")
    try:
        response = requests.get(f"{API_URL}/llm/models")
        if response.status_code == 200:
            models = response.json()["models"]
            print(f"Success! Found {len(models)} models:")
            for model in models:
                print(f"  - {model['name']} ({model['id']}): {model['description']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test the generate endpoint
    print("\nTesting POST /llm/generate")
    try:
        data = {
            "prompt": "What are the top 3 benefits of using Groq for LLM inference?",
            "model": "llama3-8b-8192",
            "max_tokens": 500,
            "temperature": 0.7
        }
        response = requests.post(f"{API_URL}/llm/generate", json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Generated text using model: {result['model']}")
            print(f"Generated text: {result['generated_text'][:200]}...")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test the summarize endpoint
    print("\nTesting POST /llm/summarize")
    try:
        data = {
            "text": """
            Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence displayed by animals including humans. 
            AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and 
            takes actions that maximize its chance of achieving its goals. The term "artificial intelligence" had previously been used to describe 
            machines that mimic and display human cognitive skills that are associated with the human mind, such as learning and problem-solving. 
            This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, 
            which does not limit how intelligence can be articulated.
            """,
            "max_length": 100
        }
        response = requests.post(f"{API_URL}/llm/summarize", json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Generated summary:")
            print(f"Summary: {result['summary']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test the analyze-sentiment endpoint
    print("\nTesting POST /llm/analyze-sentiment")
    try:
        data = {
            "text": "I absolutely love this product! It has exceeded all my expectations and I would highly recommend it to anyone."
        }
        response = requests.post(f"{API_URL}/llm/analyze-sentiment", json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Analyzed sentiment:")
            print(f"Analysis: {result['analysis']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test the extract-entities endpoint
    print("\nTesting POST /llm/extract-entities")
    try:
        data = {
            "text": "Apple CEO Tim Cook announced the new iPhone 15 at an event in Cupertino, California on September 12, 2023."
        }
        response = requests.post(f"{API_URL}/llm/extract-entities", json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Extracted entities:")
            print(f"Entities: {result['entities']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("\n=== LLM Endpoints Test Complete ===")

if __name__ == "__main__":
    print("Testing Groq API integration...")
    test_llm_endpoints()
    print("Test complete!") 