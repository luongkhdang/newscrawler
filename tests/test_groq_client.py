"""
Tests for the Groq client implementation.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from src.utils.groq_client import GroqClient, DEFAULT_MODEL


@pytest.fixture
def mock_groq_response():
    """Create a mock Groq API response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This is a test response from Groq."
    mock_response.usage = MagicMock()
    mock_response.usage.completion_tokens = 10
    mock_response.usage.prompt_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response.model = DEFAULT_MODEL
    return mock_response


@pytest.fixture
def groq_client():
    """Create a Groq client with a test API key."""
    with patch('groq.Groq'):
        # Use the environment variable if available, otherwise use a test key
        api_key = os.environ.get("GROQ_API_KEY", "test_api_key")
        return GroqClient(api_key=api_key)


def test_groq_client_initialization():
    """Test that the Groq client initializes correctly."""
    with patch('groq.Groq'):
        client = GroqClient(api_key="test_api_key")
        assert client.api_key == "test_api_key"
        assert client.client is not None


def test_groq_client_initialization_from_env():
    """Test that the Groq client initializes from environment variables."""
    with patch('groq.Groq'):
        with patch.dict(os.environ, {"GROQ_API_KEY": "env_test_api_key"}):
            client = GroqClient()
            assert client.api_key == "env_test_api_key"


def test_groq_client_initialization_error():
    """Test that the Groq client raises an error when no API key is provided."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError):
            GroqClient()


@patch("groq.Groq.chat.completions.create")
def test_generate_completion(mock_create, groq_client, mock_groq_response):
    """Test the generate_completion method."""
    # Set up the mock
    mock_create.return_value = mock_groq_response
    
    # Patch the client's create method
    groq_client.client.chat.completions.create = mock_create
    
    # Call the method
    result = groq_client.generate_completion(
        prompt="Test prompt",
        model=DEFAULT_MODEL,
        max_tokens=100,
        temperature=0.7
    )
    
    # Check the result
    assert result == "This is a test response from Groq."
    
    # Check that the mock was called correctly
    mock_create.assert_called_once()
    args, kwargs = mock_create.call_args
    assert kwargs["messages"][0]["role"] == "user"
    assert kwargs["messages"][0]["content"] == "Test prompt"
    assert kwargs["model"] == DEFAULT_MODEL
    assert kwargs["max_tokens"] == 100
    assert kwargs["temperature"] == 0.7


@patch("groq.Groq.chat.completions.create")
def test_generate_completion_with_system_message(mock_create, groq_client, mock_groq_response):
    """Test the generate_completion method with a system message."""
    # Set up the mock
    mock_create.return_value = mock_groq_response
    
    # Patch the client's create method
    groq_client.client.chat.completions.create = mock_create
    
    # Call the method
    result = groq_client.generate_completion(
        prompt="Test prompt",
        model=DEFAULT_MODEL,
        max_tokens=100,
        temperature=0.7,
        system_message="You are a helpful assistant."
    )
    
    # Check the result
    assert result == "This is a test response from Groq."
    
    # Check that the mock was called correctly
    mock_create.assert_called_once()
    args, kwargs = mock_create.call_args
    assert len(kwargs["messages"]) == 2
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][0]["content"] == "You are a helpful assistant."
    assert kwargs["messages"][1]["role"] == "user"
    assert kwargs["messages"][1]["content"] == "Test prompt"


@patch("src.utils.groq_client.GroqClient.generate_completion")
def test_summarize_text(mock_generate_completion, groq_client):
    """Test the summarize_text method."""
    # Set up the mock
    mock_generate_completion.return_value = "This is a summary."
    
    # Call the method
    result = groq_client.summarize_text("This is a long text that needs to be summarized.")
    
    # Check the result
    assert result == "This is a summary."
    
    # Check that the mock was called correctly
    mock_generate_completion.assert_called_once()
    args, kwargs = mock_generate_completion.call_args
    assert "summarize" in kwargs.get("prompt", "").lower()
    assert kwargs.get("temperature") < 0.7  # Should use a lower temperature for summaries


@patch("src.utils.groq_client.GroqClient.generate_completion")
def test_analyze_sentiment(mock_generate_completion, groq_client):
    """Test the analyze_sentiment method."""
    # Set up the mock
    mock_generate_completion.return_value = "Sentiment analysis result"
    
    # Call the method
    result = groq_client.analyze_sentiment("This is a text to analyze sentiment.")
    
    # Check the result
    assert result["analysis"] == "Sentiment analysis result"
    
    # Check that the mock was called correctly
    mock_generate_completion.assert_called_once()
    args, kwargs = mock_generate_completion.call_args
    assert "sentiment" in kwargs.get("prompt", "").lower()
    assert kwargs.get("temperature") < 0.3  # Should use a very low temperature for sentiment analysis


@patch("src.utils.groq_client.GroqClient.generate_completion")
def test_extract_entities(mock_generate_completion, groq_client):
    """Test the extract_entities method."""
    # Set up the mock
    mock_generate_completion.return_value = "Entity extraction result"
    
    # Call the method
    result = groq_client.extract_entities("This is a text to extract entities from.")
    
    # Check the result
    assert result["entities"] == "Entity extraction result"
    
    # Check that the mock was called correctly
    mock_generate_completion.assert_called_once()
    args, kwargs = mock_generate_completion.call_args
    assert "entities" in kwargs.get("prompt", "").lower()
    assert kwargs.get("temperature") < 0.7  # Should use a moderate temperature for entity extraction 