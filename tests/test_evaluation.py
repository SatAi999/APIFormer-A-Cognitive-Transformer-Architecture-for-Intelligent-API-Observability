import pytest
import os
import numpy as np
import torch
from fastapi.testclient import TestClient

from src.visualization.plots import plot_attention_heatmap, plot_latency_timeline
from src.api.app import app, SessionSequenceRequest, APIEventPayload

@pytest.fixture
def client():
    return TestClient(app)

def test_plotting_utilities():
    temp_heatmap = "data/processed/plots/temp_test_heatmap.png"
    temp_timeline = "data/processed/plots/temp_test_timeline.png"
    
    # Remove files if exist
    if os.path.exists(temp_heatmap):
        os.remove(temp_heatmap)
    if os.path.exists(temp_timeline):
        os.remove(temp_timeline)
        
    attn = np.eye(4)
    tokens = ["GET /auth", "GET /products", "POST /checkout", "GET /logout"]
    latencies = [100.0, 120.0, 500.0, 90.0]
    anoms = [0, 0, 1, 0]
    
    # Test heatmap
    path_h = plot_attention_heatmap(attn, tokens, temp_heatmap)
    assert os.path.exists(path_h)
    assert path_h == temp_heatmap
    
    # Test timeline
    path_t = plot_latency_timeline(latencies, anoms, tokens, temp_timeline)
    assert os.path.exists(path_t)
    assert path_t == temp_timeline
    
    # Cleanup
    os.remove(temp_heatmap)
    os.remove(temp_timeline)

def test_fastapi_endpoints(client):
    # Test health check
    response = client.get("/health")
    assert response.status_code == 200
    res_json = response.json()
    assert "status" in res_json
    
    # Test model prediction endpoint
    # Send a mock session payload
    payload = {
        "events": [
            {
                "timestamp": "2026-07-05T21:40:00Z",
                "endpoint": "/api/v1/auth/login",
                "http_method": "POST",
                "status_code": 200,
                "latency_ms": 150.0,
                "payload_size_bytes": 450,
                "time_since_previous_request": 0.0,
                "device": "desktop-chrome",
                "geo_location": "US-East",
                "environment": "production",
                "authentication": "None"
            },
            {
                "timestamp": "2026-07-05T21:40:05Z",
                "endpoint": "/api/v1/products",
                "http_method": "GET",
                "status_code": 200,
                "latency_ms": 110.0,
                "payload_size_bytes": 12000,
                "time_since_previous_request": 5.0,
                "device": "desktop-chrome",
                "geo_location": "US-East",
                "environment": "production",
                "authentication": "Bearer"
            }
        ]
    }
    
    # We trigger the app startup event manually to load model configurations and weights
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json=payload)
        
        # If model checkpoint exists and loaded, check success
        # Since scripts/train.py was run and created the checkpoint, it should succeed!
        assert response.status_code == 200
        res = response.json()
        
        assert "predictions" in res
        assert "explainability" in res
        
        # Check intents
        intent = res["predictions"]["intent"]
        assert "label" in intent
        assert "confidence" in intent
        
        # Check bot detection
        bot = res["predictions"]["bot_detection"]
        assert "is_bot" in bot
        
        # Check step level anomalies
        steps = res["predictions"]["step_anomalies"]
        assert len(steps) == 2
        assert steps[0]["endpoint"] == "/api/v1/auth/login"
        assert "anomaly_probability" in steps[0]
        
        # Check attention matrix
        attn_matrix = res["explainability"]["attention_matrix"]
        assert len(attn_matrix) == 2
        assert len(attn_matrix[0]) == 2
