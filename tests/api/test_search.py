"""
Tests for the search router endpoints.
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
from src.utils.cache import search_cache

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
    {
        "id": 3,
        "title": "Political Developments",
        "content": "Recent political developments have significant implications for international relations.",
        "url": "https://test2.com/article1",
        "published_at": "2023-01-03T00:00:00",
        "source_id": 2,
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


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the search cache before each test."""
    search_cache.clear()
    yield


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


def test_text_search(client):
    """Test text search functionality."""
    response = client.get("/search/text?query=climate")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) >= 1
    # The first article should be about climate change
    assert "climate" in data["items"][0]["title"].lower() or "climate" in data["items"][0]["content"].lower()


def test_text_search_pagination(client):
    """Test pagination in text search."""
    response = client.get("/search/text?query=article&page=1&size=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) <= 2
    assert data["page"] == 1
    assert data["size"] == 2


def test_text_search_no_results(client):
    """Test text search with no matching results."""
    response = client.get("/search/text?query=nonexistentterm")
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 0
    assert data["total"] == 0


@patch("src.vector.processor.generate_embedding")
def test_semantic_search(mock_generate_embedding, client):
    """Test semantic search functionality."""
    # Mock the embedding generation
    mock_embedding = [0.1] * 384  # Simplified embedding vector
    mock_generate_embedding.return_value = mock_embedding
    
    # Mock the vector search results
    with patch("src.api.routers.search.vector_search") as mock_vector_search:
        mock_vector_search.return_value = [
            (1, 0.9),  # Article ID 1 with similarity score 0.9
            (2, 0.7),  # Article ID 2 with similarity score 0.7
        ]
        
        response = client.get("/search/semantic?query=climate impact")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) == 2
        # Results should be ordered by similarity score
        assert data["items"][0]["id"] == 1
        assert data["items"][1]["id"] == 2


@patch("src.vector.processor.generate_embedding")
def test_hybrid_search(mock_generate_embedding, client):
    """Test hybrid search functionality."""
    # Mock the embedding generation
    mock_embedding = [0.1] * 384  # Simplified embedding vector
    mock_generate_embedding.return_value = mock_embedding
    
    # Mock the vector search results
    with patch("src.api.routers.search.vector_search") as mock_vector_search:
        mock_vector_search.return_value = [
            (1, 0.9),  # Article ID 1 with similarity score 0.9
            (3, 0.6),  # Article ID 3 with similarity score 0.6
        ]
        
        response = client.get("/search/hybrid?query=climate political&semantic_weight=0.7&keyword_weight=0.3")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        
        # Hybrid search should combine results from both semantic and keyword search
        # The exact results depend on the implementation, but we can check basic structure
        assert len(data["items"]) > 0
        assert "id" in data["items"][0]
        assert "title" in data["items"][0]
        assert "content" in data["items"][0]
        assert "score" in data["items"][0]


def test_search_with_filters(client):
    """Test search with source filters."""
    response = client.get("/search/text?query=article&source_id=1")
    assert response.status_code == 200
    data = response.json()
    
    # All results should be from source_id 1
    for item in data["items"]:
        assert item["source_id"] == 1


def test_search_with_date_filters(client):
    """Test search with date filters."""
    response = client.get("/search/text?query=article&start_date=2023-01-02&end_date=2023-01-03")
    assert response.status_code == 200
    data = response.json()
    
    # All results should be within the date range
    for item in data["items"]:
        assert item["published_at"] >= "2023-01-02T00:00:00"
        assert item["published_at"] <= "2023-01-03T23:59:59" 