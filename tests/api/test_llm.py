"""
Tests for the LLM router endpoints.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.main import app
from src.database.models import Base, Article, Source
from src.database.session import get_db

# Create an in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Test data
TEST_SOURCES = [
    {"id": 1, "name": "Test Source 1", "url": "https://test1.com", "is_active": True},
    {"id": 2, "name": "Test Source 2", "url": "https://test2.com", "is_active": True},
]

TEST_ARTICLES = [
    {
        "id": 1,
        "title": "Climate Change Impact",
        "content": "This article discusses the impact of climate change on global ecosystems.",
        "url": "https://test1.com/article1",
        "published_at": "2023-01-01T00:00:00",
        "source_id": 1,
        "embedding": None,
    },
    {
        "id": 2,
        "title": "Technology Advancements",
        "content": "Recent advancements in AI technology are transforming industries.",
        "url": "https://test1.com/article2",
        "published_at": "2023-01-02T00:00:00",
        "source_id": 1,
        "embedding": None,
    },
]


# Override the dependency to use our test database
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture
def client():
    """
    Create a test client with a fresh database for each test.
    """
    # Create the tables
    Base.metadata.create_all(bind=engine)
    
    # Add test data
    db = TestingSessionLocal()
    
    # Add sources
    for source_data in TEST_SOURCES:
        source = Source(**source_data)
        db.add(source)
    
    # Add articles
    for article_data in TEST_ARTICLES:
        article = Article(**article_data)
        db.add(article)
    
    db.commit()
    db.close()
    
    # Create client
    with TestClient(app) as client:
        yield client
    
    # Clean up
    Base.metadata.drop_all(bind=engine)


@patch("src.llm.client.get_llm_client")
def test_ask_question(mock_get_llm_client, client):
    """Test asking a question to the LLM."""
    # Mock the LLM client
    mock_client = MagicMock()
    mock_client.generate_response.return_value = {
        "response": "This is a test response about climate change.",
        "sources": [{"id": 1, "title": "Climate Change Impact", "relevance": 0.9}]
    }
    mock_get_llm_client.return_value = mock_client
    
    # Test the endpoint
    response = client.post(
        "/llm/ask",
        json={"question": "What is the impact of climate change?"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "sources" in data
    assert data["response"] == "This is a test response about climate change."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["id"] == 1


@patch("src.llm.client.get_llm_client")
def test_ask_question_with_context(mock_get_llm_client, client):
    """Test asking a question with specific context articles."""
    # Mock the LLM client
    mock_client = MagicMock()
    mock_client.generate_response_with_context.return_value = {
        "response": "This is a test response about technology advancements.",
        "sources": [{"id": 2, "title": "Technology Advancements", "relevance": 0.95}]
    }
    mock_get_llm_client.return_value = mock_client
    
    # Test the endpoint
    response = client.post(
        "/llm/ask-with-context",
        json={
            "question": "What are recent technology advancements?",
            "context_article_ids": [2]
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "sources" in data
    assert data["response"] == "This is a test response about technology advancements."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["id"] == 2


@patch("src.llm.client.get_llm_client")
def test_summarize_article(mock_get_llm_client, client):
    """Test summarizing an article."""
    # Mock the LLM client
    mock_client = MagicMock()
    mock_client.summarize_article.return_value = {
        "summary": "This is a summary of the climate change article.",
        "article_id": 1
    }
    mock_get_llm_client.return_value = mock_client
    
    # Test the endpoint
    response = client.post(
        "/llm/summarize",
        json={"article_id": 1}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "article_id" in data
    assert data["summary"] == "This is a summary of the climate change article."
    assert data["article_id"] == 1


@patch("src.llm.client.get_llm_client")
def test_summarize_nonexistent_article(mock_get_llm_client, client):
    """Test summarizing a non-existent article."""
    # Mock the LLM client
    mock_client = MagicMock()
    mock_get_llm_client.return_value = mock_client
    
    # Test the endpoint with a non-existent article ID
    response = client.post(
        "/llm/summarize",
        json={"article_id": 999}
    )
    
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


@patch("src.llm.client.get_llm_client")
def test_ask_question_error_handling(mock_get_llm_client, client):
    """Test error handling in the ask endpoint."""
    # Mock the LLM client to raise an exception
    mock_client = MagicMock()
    mock_client.generate_response.side_effect = Exception("LLM service error")
    mock_get_llm_client.return_value = mock_client
    
    # Test the endpoint
    response = client.post(
        "/llm/ask",
        json={"question": "What is the impact of climate change?"}
    )
    
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data 