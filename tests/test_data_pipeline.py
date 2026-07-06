import pytest
import pandas as pd
from datetime import datetime
import torch

from src.data.synthetic_gen import SyntheticAPIGenerator, UnifiedAPIEvent
from src.data.inspector import DatasetInspector
from src.data.normalizer import SchemaNormalizer
from src.data.session_builder import SessionBuilder
from src.data.tokenizer import APITokenizer
from src.data.graph_builder import ServiceDependencyGraph
from src.data.dataset import APIFormerDataset

@pytest.fixture
def sample_config():
    return {
        "generator": {
            "num_sessions": 10,
            "seed": 42,
            "normal_ratio": 0.5,
            "latency": {"mu": 4.8, "sigma": 0.6, "timeout_threshold": 5000.0},
            "payload_size": {"min_bytes": 100, "max_bytes": 1000}
        },
        "session_builder": {
            "idle_timeout_seconds": 300,
            "max_sequence_length": 10
        },
        "tokenizer": {
            "mask_token": "[MASK]",
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]"
        }
    }

def test_synthetic_generator(sample_config):
    generator = SyntheticAPIGenerator(sample_config)
    events = generator.generate_dataset()
    
    assert len(events) > 0
    assert isinstance(events[0], UnifiedAPIEvent)
    
    # Check that timestamps are chronologically sorted
    ts_list = [datetime.fromisoformat(e.timestamp) for e in events]
    assert all(ts_list[i] <= ts_list[i+1] for i in range(len(ts_list) - 1))

def test_inspector_and_normalizer(sample_config):
    generator = SyntheticAPIGenerator(sample_config)
    events = generator.generate_dataset()
    
    # Convert to dataframe and rename columns to simulate custom schemas
    df = pd.DataFrame([e.__dict__ for e in events])
    col_rename = {
        "timestamp": "ts",
        "session_id": "session_key",
        "endpoint": "request_uri",
        "status_code": "resp_code",
        "latency_ms": "elapsed"
    }
    df_renamed = df.rename(columns=col_rename)
    
    # Temp file write
    import os
    os.makedirs("data/raw", exist_ok=True)
    temp_json = "data/raw/temp_test_logs.json"
    df_renamed.to_json(temp_json, orient="records", lines=True)
    
    inspector = DatasetInspector()
    results = inspector.inspect(temp_json)
    
    # Assert correct field mappings
    mappings = results["schema_mapping"]
    assert mappings.get("timestamp") == "ts"
    assert mappings.get("session_id") == "session_key"
    assert mappings.get("endpoint") == "request_uri"
    assert mappings.get("status_code") == "resp_code"
    
    # Normalize
    normalizer = SchemaNormalizer(results)
    normalized_events = normalizer.normalize_dataframe(df_renamed)
    
    assert len(normalized_events) == len(events)
    assert normalized_events[0].endpoint == events[0].endpoint
    assert normalized_events[0].status_code == events[0].status_code

def test_session_builder(sample_config):
    generator = SyntheticAPIGenerator(sample_config)
    events = generator.generate_normal_session(datetime.now())
    
    # Remove session IDs to test time-window reconstruction
    for e in events:
        e.session_id = "None"
        
    builder = SessionBuilder(sample_config)
    sessions = builder.reconstruct_sessions(events)
    
    assert len(sessions) > 0
    # Every event must belong to a session
    for s_id, s_events in sessions.items():
        assert s_id != "None"
        assert len(s_events) <= sample_config["session_builder"]["max_sequence_length"]
        # Delays should be computed
        assert all(e.time_since_previous_request >= 0.0 for e in s_events)

def test_tokenizer(sample_config):
    generator = SyntheticAPIGenerator(sample_config)
    events = generator.generate_normal_session(datetime.now())
    
    builder = SessionBuilder(sample_config)
    sessions = builder.reconstruct_sessions(events)
    
    tokenizer = APITokenizer(sample_config)
    tokenizer.fit(sessions)
    
    # Check that variables are normalized
    assert tokenizer.normalize_endpoint("/api/v1/products/123") == "/api/v1/products/{id}"
    assert tokenizer.normalize_endpoint("/api/v1/products/abc-123-xyz") == "/api/v1/products/abc-{id}-xyz"
    
    # Check encoding
    encoded_sessions = [tokenizer.encode_session(sess) for sess in sessions.values()]
    assert len(encoded_sessions) == len(sessions)
    assert "endpoint" in encoded_sessions[0]
    assert len(encoded_sessions[0]["endpoint"]) == len(list(sessions.values())[0])

def test_graph_builder(sample_config):
    generator = SyntheticAPIGenerator(sample_config)
    events = generator.generate_dataset()
    
    graph_builder = ServiceDependencyGraph()
    graph_builder.build_graphs(events)
    
    metrics = graph_builder.compute_graph_metrics()
    assert len(metrics) > 0
    
    # Fetch vocab
    builder = SessionBuilder(sample_config)
    sessions = builder.reconstruct_sessions(events)
    tokenizer = APITokenizer(sample_config)
    tokenizer.fit(sessions)
    
    adj, edge_list = graph_builder.get_gnn_structures(tokenizer.vocabs["endpoint"])
    assert adj.shape[0] == len(tokenizer.vocabs["endpoint"])
    assert len(edge_list) >= 0

def test_pytorch_dataset(sample_config):
    generator = SyntheticAPIGenerator(sample_config)
    events = generator.generate_dataset()
    
    builder = SessionBuilder(sample_config)
    sessions = builder.reconstruct_sessions(events)
    
    tokenizer = APITokenizer(sample_config)
    tokenizer.fit(sessions)
    
    encoded_sessions = [tokenizer.encode_session(sess) for sess in sessions.values()]
    
    dataset = APIFormerDataset(
        encoded_sessions=encoded_sessions,
        vocab=tokenizer.vocabs["endpoint"],
        max_len=sample_config["session_builder"]["max_sequence_length"]
    )
    
    assert len(dataset) == len(encoded_sessions)
    sample = dataset[0]
    
    # Validate tensor outputs
    assert isinstance(sample["endpoint"], torch.Tensor)
    assert sample["endpoint"].shape[0] == sample_config["session_builder"]["max_sequence_length"]
    assert sample["mam_labels"].shape[0] == sample_config["session_builder"]["max_sequence_length"]
    assert sample["padding_mask"].dtype == torch.bool
