"""
Tests for the health router endpoints.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.main import app
from src.utils.cache import vector_cache, search_cache


@pytest.fixture
def client():
    """
    Create a test client.
    """
    with TestClient(app) as client:
        yield client


def test_health_check(client):
    """Test the health check endpoint."""
    # Add some items to the caches for testing
    vector_cache.set("test_key1", "test_value1")
    vector_cache.set("test_key2", "test_value2")
    search_cache.set("test_search", "test_result")
    
    # Test the endpoint
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    
    # Check basic structure
    assert "status" in data
    assert "version" in data
    assert "uptime" in data
    assert "memory_usage" in data
    assert "environment" in data
    assert "cache_stats" in data
    
    # Check status
    assert data["status"] == "healthy"
    
    # Check cache stats
    assert "vector_cache" in data["cache_stats"]
    assert "search_cache" in data["cache_stats"]
    assert data["cache_stats"]["vector_cache"]["size"] == 2
    assert data["cache_stats"]["search_cache"]["size"] == 1


@patch("psutil.Process")
def test_health_check_memory_usage(mock_process, client):
    """Test the memory usage reporting in health check."""
    # Mock the process memory info
    mock_process_instance = MagicMock()
    mock_process_instance.memory_info.return_value = MagicMock(rss=104857600)  # 100 MB
    mock_process.return_value = mock_process_instance
    
    # Test the endpoint
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    
    # Check memory usage
    assert "memory_usage" in data
    assert "mb" in data["memory_usage"]
    assert float(data["memory_usage"].split()[0]) == 100.0  # 100 MB


@patch("os.environ")
def test_health_check_environment(mock_environ, client):
    """Test the environment reporting in health check."""
    # Mock environment variables
    mock_environ.get.side_effect = lambda key, default=None: {
        "ENVIRONMENT": "test",
        "APP_VERSION": "1.0.0"
    }.get(key, default)
    
    # Test the endpoint
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    
    # Check environment info
    assert data["environment"] == "test"
    assert data["version"] == "1.0.0" 