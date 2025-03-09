"""
Shared test fixtures for the NewsCrawler tests.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.main import app
from src.database.models import Base, Article, Source
from src.database.session import get_db
from src.utils.cache import vector_cache, search_cache

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
    vector_cache.clear()
    search_cache.clear()
    yield


@pytest.fixture
def test_db():
    """
    Create a fresh test database session.
    """
    # Create the tables
    Base.metadata.create_all(bind=engine)
    
    # Create a session
    db = TestingSessionLocal()
    
    yield db
    
    # Clean up
    db.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def populated_db(test_db):
    """
    Create a test database populated with test data.
    """
    # Add sources
    for source_data in TEST_SOURCES:
        source = Source(**source_data)
        test_db.add(source)
    
    # Add articles
    for article_data in TEST_ARTICLES:
        article = Article(**article_data)
        test_db.add(article)
    
    test_db.commit()
    
    return test_db


@pytest.fixture
def client(populated_db):
    """
    Create a test client with a populated database.
    """
    with TestClient(app) as client:
        yield client 