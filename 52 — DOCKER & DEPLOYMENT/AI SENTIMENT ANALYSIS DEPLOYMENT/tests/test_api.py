import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------
# TEST ROOT ENDPOINT
# --------------------------------------------------

def test_root_endpoint(client):

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"
    assert data["message"] == "AI Sentiment Analysis API"


# --------------------------------------------------
# TEST HEALTH ENDPOINT
# --------------------------------------------------

def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


# --------------------------------------------------
# TEST POSITIVE SENTIMENT
# --------------------------------------------------

def test_positive_sentiment(client):

    response = client.post(
        "/predict",
        json={
            "text": "This product is absolutely amazing. I love it!"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["label"] == "POSITIVE"
    assert 0.0 <= data["confidence"] <= 1.0


# --------------------------------------------------
# TEST NEGATIVE SENTIMENT
# --------------------------------------------------

def test_negative_sentiment(client):

    response = client.post(
        "/predict",
        json={
            "text": "This product is terrible. I hate it."
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["label"] == "NEGATIVE"
    assert 0.0 <= data["confidence"] <= 1.0


# --------------------------------------------------
# TEST EMPTY INPUT
# --------------------------------------------------

def test_empty_text(client):

    response = client.post(
        "/predict",
        json={
            "text": ""
        }
    )

    assert response.status_code == 422


# --------------------------------------------------
# TEST MISSING INPUT
# --------------------------------------------------

def test_missing_text(client):

    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422


# --------------------------------------------------
# TEST INVALID DATA TYPE
# --------------------------------------------------

def test_invalid_data_type(client):

    response = client.post(
        "/predict",
        json={
            "text": 12345
        }
    )

    assert response.status_code == 422