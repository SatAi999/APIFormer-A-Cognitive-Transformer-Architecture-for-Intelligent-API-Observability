import pytest
import os
import yaml
import json
from scripts.kafka_consumer import APIFormerStreamingConsumer

def test_orchestration_file_existence():
    # Check Dockerfiles and Compose configs
    assert os.path.exists("Dockerfile")
    assert os.path.exists("docker-compose.yml")
    assert os.path.exists("config/prometheus.yml")
    
    # Check Kubernetes manifests
    assert os.path.exists("k8s/api-deployment.yaml")
    assert os.path.exists("k8s/dashboard-deployment.yaml")
    assert os.path.exists("k8s/redis-deployment.yaml")
    assert os.path.exists("k8s/ingress.yaml")
    
    # Check compose syntax loading
    with open("docker-compose.yml", "r") as f:
        compose_cfg = yaml.safe_load(f)
        assert "services" in compose_cfg
        assert "api-service" in compose_cfg["services"]
        assert "dashboard" in compose_cfg["services"]
        assert "redis" in compose_cfg["services"]
        assert "kafka" in compose_cfg["services"]
        
    # Check prometheus target maps
    with open("config/prometheus.yml", "r") as f:
        prom_cfg = yaml.safe_load(f)
        assert "scrape_configs" in prom_cfg
        assert prom_cfg["scrape_configs"][0]["job_name"] == "apiformer-service"

def test_kafka_streaming_consumer():
    consumer = APIFormerStreamingConsumer(predict_url="http://localhost:8000/predict")
    
    # Check initial empty caches
    assert consumer.get_session_sequence("test_sess_99") == []
    
    # Test local sequence caching updates
    mock_events = [
        {"session_id": "test_sess_99", "timestamp": "2026-07-05T22:00:00Z", "endpoint": "/api/v1/auth/login", "http_method": "POST", "status_code": 200, "latency_ms": 120.0, "payload_size_bytes": 350, "time_since_previous_request": 0.0},
        {"session_id": "test_sess_99", "timestamp": "2026-07-05T22:00:05Z", "endpoint": "/api/v1/products", "http_method": "GET", "status_code": 200, "latency_ms": 95.0, "payload_size_bytes": 1000, "time_since_previous_request": 5.0}
    ]
    
    # Feed event 0
    # Since model is not running locally during tests, process_message will catch request error and return None,
    # but the sequence should still be successfully cached locally!
    consumer.process_message(json.dumps(mock_events[0]))
    cached = consumer.get_session_sequence("test_sess_99")
    assert len(cached) == 1
    assert cached[0]["endpoint"] == "/api/v1/auth/login"
    
    # Feed event 1
    consumer.process_message(json.dumps(mock_events[1]))
    cached = consumer.get_session_sequence("test_sess_99")
    assert len(cached) == 2
    assert cached[1]["endpoint"] == "/api/v1/products"
