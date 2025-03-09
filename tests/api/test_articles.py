"""
Tests for the articles router endpoints.
"""
import pytest
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
        "title": "Test Article 1",
        "content": "This is test content for article 1",
        "url": "https://test1.com/article1",
        "published_at": "2023-01-01T00:00:00",
        "source_id": 1,
        "embedding": None,
    },
    {
        "id": 2,
        "title": "Test Article 2",
        "content": "This is test content for article 2",
        "url": "https://test1.com/article2",
        "published_at": "2023-01-02T00:00:00",
        "source_id": 1,
        "embedding": None,
    },
    {
        "id": 3,
        "title": "Test Article 3",
        "content": "This is test content for article 3",
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


def test_get_articles(client):
    """Test getting a list of articles."""
    response = client.get("/articles/")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 3
    assert data["total"] == 3
    assert data["page"] == 1
    assert data["size"] == 10


def test_get_articles_pagination(client):
    """Test pagination of articles."""
    response = client.get("/articles/?page=1&size=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2
    assert data["total"] == 3
    assert data["page"] == 1
    assert data["size"] == 2
    
    # Get second page
    response = client.get("/articles/?page=2&size=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["total"] == 3
    assert data["page"] == 2
    assert data["size"] == 2


def test_get_articles_filtering(client):
    """Test filtering articles by source."""
    response = client.get("/articles/?source_id=1")
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2
    for article in data["items"]:
        assert article["source_id"] == 1


def test_get_article_by_id(client):
    """Test getting a single article by ID."""
    response = client.get("/articles/1")
    assert response.status_code == 200
    article = response.json()
    assert article["id"] == 1
    assert article["title"] == "Climate Change Impact"


def test_get_article_not_found(client):
    """Test getting a non-existent article."""
    response = client.get("/articles/999")
    assert response.status_code == 404
    assert "detail" in response.json()


def test_get_latest_articles(client):
    """Test getting latest articles."""
    response = client.get("/articles/latest")
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 3
    # Should be ordered by published_at descending
    assert data["items"][0]["id"] == 3
    assert data["items"][1]["id"] == 2
    assert data["items"][2]["id"] == 1


def test_get_articles_by_source(client):
    """Test getting articles by source."""
    response = client.get("/articles/by-source/1")
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2
    for article in data["items"]:
        assert article["source_id"] == 1 