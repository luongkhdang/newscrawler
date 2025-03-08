import os
import sys
from groq import Groq

def test_groq_api():
    """Test the Groq API with a simple prompt."""
    # Use the API key provided
    api_key = "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"
    
    # Initialize the Groq client
    client = Groq(api_key=api_key)
    
    # Test prompt
    prompt = "What are the top 3 benefits of using Groq for LLM inference?"
    
    try:
        # Make a completion request
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",  # Using Llama 3 8B model
        )
        
        # Print the response
        print("\n=== Groq API Test Results ===")
        print(f"Model used: {chat_completion.model}")
        print(f"Response: {chat_completion.choices[0].message.content}")
        print(f"Completion tokens: {chat_completion.usage.completion_tokens}")
        print(f"Prompt tokens: {chat_completion.usage.prompt_tokens}")
        print(f"Total tokens: {chat_completion.usage.total_tokens}")
        print("=== Test Successful ===\n")
        return True
    
    except Exception as e:
        print(f"\n=== Groq API Test Failed ===")
        print(f"Error: {str(e)}")
        print("=== Test Failed ===\n")
        return False

if __name__ == "__main__":
    print("Testing Groq API integration...")
    success = test_groq_api()
    sys.exit(0 if success else 1) 