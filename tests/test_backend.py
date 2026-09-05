import pytest
from fastapi.testclient import TestClient
import sys
import os
import numpy as np

# Add root directory to sys.path to allow importing backend.app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app import app, feature_names

client = TestClient(app)

@pytest.fixture
def mock_ml(mocker):
    mocker.patch('backend.app.clf.predict_proba', return_value=np.array([[0.2, 0.8]]))
    num_features = len(feature_names)
    
    # Mock SHAP values returning list format [class0_shap, class1_shap] where class1_shap has 1 row
    mock_shap_list = [np.array([[0.0] * num_features]), np.array([[0.1] * num_features])]
    mocker.patch('backend.app.explainer.shap_values', return_value=mock_shap_list)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Credit Risk ML API is running"}

def test_predict_success(mock_ml):
    payload = {
        "features": {
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_2": 0.5,
            "EXT_SOURCE_3": 0.5,
            "DAYS_BIRTH": -15000,
            "AMT_CREDIT": 500000.0,
            "AMT_ANNUITY": 25000.0
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "probability_of_default" in data
    assert "top_3_positive_contributions" in data
    assert "top_3_negative_contributions" in data
    assert data["probability_of_default"] == 0.8

def test_predict_missing_features(mock_ml):
    # Model should handle missing features by filling them with 0.0, 
    # but the request should still be a valid ApplicantFeatures object
    payload = {
        "features": {}
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "probability_of_default" in data

def test_predict_invalid_payload(mock_ml):
    payload = {
        "invalid_key": "invalid_value"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # FastAPI Validation Error
