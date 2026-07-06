import pytest
import torch
import torch.nn as nn
from src.models.heads import InfoNCELoss, APIFormerPlusModel
from src.data.dataset import APIFormerDataset

@pytest.fixture
def mock_vocabs():
    return {
        "endpoint": {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2, "GET /api/v1/products": 3, "POST /api/v1/checkout": 4},
        "method": {"[PAD]": 0, "[UNK]": 1, "GET": 2, "POST": 3},
        "status": {"[PAD]": 0, "[UNK]": 1, "200": 2, "500": 3},
        "service": {"[PAD]": 0, "[UNK]": 1, "gateway": 2, "auth-service": 3},
        "device": {"[PAD]": 0, "[UNK]": 1, "desktop": 2, "mobile": 3},
        "location": {"[PAD]": 0, "[UNK]": 1, "US": 2, "EU": 3},
        "env": {"[PAD]": 0, "[UNK]": 1, "production": 2, "staging": 3},
        "auth": {"[PAD]": 0, "[UNK]": 1, "None": 2, "Bearer": 3}
    }

def test_infonce_loss():
    batch_size = 4
    d_model = 16
    
    loss_fn = InfoNCELoss(temperature=0.07)
    
    # 1. Aligned representation (identical views) should yield very low loss
    z1 = torch.randn(batch_size, d_model)
    z2 = z1.clone() # Positive alignment
    
    loss_aligned = loss_fn(z1, z2)
    
    # 2. Random views should yield higher loss
    z3 = torch.randn(batch_size, d_model)
    loss_random = loss_fn(z1, z3)
    
    assert loss_aligned.item() < loss_random.item()

def test_apiformer_plus_model_forward(mock_vocabs):
    d_model = 32
    seq_len = 10
    batch_size = 2
    num_nodes = 5
    
    model = APIFormerPlusModel(
        vocabs=mock_vocabs,
        d_embed=8,
        d_model=d_model,
        num_layers=1,
        num_heads=2,
        d_ff=64,
        gcn_num_nodes=num_nodes,
        num_intents=3,
        max_len=seq_len
    )
    
    # Generate mock inputs
    batch = {
        "endpoint": torch.randint(0, 4, (batch_size, seq_len)),
        "method": torch.randint(0, 4, (batch_size, seq_len)),
        "status": torch.randint(0, 4, (batch_size, seq_len)),
        "service": torch.randint(0, 4, (batch_size, seq_len)),
        "device": torch.randint(0, 4, (batch_size, seq_len)),
        "location": torch.randint(0, 4, (batch_size, seq_len)),
        "env": torch.randint(0, 4, (batch_size, seq_len)),
        "auth": torch.randint(0, 4, (batch_size, seq_len)),
        "latency": torch.randn(batch_size, seq_len),
        "payload": torch.randn(batch_size, seq_len),
        "time_gap": torch.rand(batch_size, seq_len),
        "padding_mask": torch.zeros(batch_size, seq_len, dtype=torch.bool),
        "v1_endpoint": torch.randint(0, 4, (batch_size, seq_len)),
        "v2_endpoint": torch.randint(0, 4, (batch_size, seq_len)),
        "v1_latency": torch.randn(batch_size, seq_len),
        "v2_latency": torch.randn(batch_size, seq_len)
    }
    
    adj = torch.eye(num_nodes)
    
    # 1. Pretraining Forward
    out_pre = model(batch, adj, pretrain=True)
    assert "mam_logits" in out_pre
    assert "next_logits" in out_pre
    assert "latency_preds" in out_pre
    assert "status_logits" in out_pre
    assert "z1" in out_pre
    assert "z2" in out_pre
    
    assert out_pre["mam_logits"].shape == (batch_size, seq_len, len(mock_vocabs["endpoint"]))
    assert out_pre["next_logits"].shape == (batch_size, seq_len, len(mock_vocabs["endpoint"]))
    assert out_pre["latency_preds"].shape == (batch_size, seq_len)
    assert out_pre["status_logits"].shape == (batch_size, seq_len, len(mock_vocabs["status"]))
    assert out_pre["z1"].shape == (batch_size, d_model)
    
    # 2. Fine-tuning Forward
    out_fine = model(batch, adj, pretrain=False)
    assert "anomaly_logits" in out_fine
    assert "intent_logits" in out_fine
    assert "bot_logits" in out_fine
    
    assert out_fine["anomaly_logits"].shape == (batch_size, seq_len, 2)
    assert out_fine["intent_logits"].shape == (batch_size, 3)
    assert out_fine["bot_logits"].shape == (batch_size, 2)

def test_mini_training_step(mock_vocabs):
    d_model = 32
    seq_len = 8
    batch_size = 2
    num_nodes = 5
    
    model = APIFormerPlusModel(
        vocabs=mock_vocabs,
        d_embed=8,
        d_model=d_model,
        num_layers=1,
        num_heads=2,
        d_ff=64,
        gcn_num_nodes=num_nodes,
        num_intents=3,
        max_len=seq_len
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Capture initial weights of a projection layer
    init_weight = model.next_head.weight.clone()
    
    # Build inputs
    batch = {
        "endpoint": torch.randint(0, 4, (batch_size, seq_len)),
        "method": torch.randint(0, 4, (batch_size, seq_len)),
        "status": torch.randint(0, 4, (batch_size, seq_len)),
        "service": torch.randint(0, 4, (batch_size, seq_len)),
        "device": torch.randint(0, 4, (batch_size, seq_len)),
        "location": torch.randint(0, 4, (batch_size, seq_len)),
        "env": torch.randint(0, 4, (batch_size, seq_len)),
        "auth": torch.randint(0, 4, (batch_size, seq_len)),
        "latency": torch.randn(batch_size, seq_len),
        "payload": torch.randn(batch_size, seq_len),
        "time_gap": torch.rand(batch_size, seq_len),
        "padding_mask": torch.zeros(batch_size, seq_len, dtype=torch.bool),
        "v1_endpoint": torch.randint(0, 4, (batch_size, seq_len)),
        "v2_endpoint": torch.randint(0, 4, (batch_size, seq_len)),
        "v1_latency": torch.randn(batch_size, seq_len),
        "v2_latency": torch.randn(batch_size, seq_len),
        "mam_labels": torch.randint(-100, 4, (batch_size, seq_len)),
        "next_endpoint_targets": torch.randint(0, 4, (batch_size, seq_len)),
        "anomaly_labels": torch.randint(0, 2, (batch_size, seq_len)),
        "intent_label": torch.randint(0, 3, (batch_size,)),
        "bot_label": torch.randint(0, 2, (batch_size,))
    }
    
    adj = torch.eye(num_nodes)
    
    # Check that memory bank updates work
    assert model.cognitive_engine.memory_bank.size == 0
    
    # Forward & Backward Pass (SSL)
    optimizer.zero_grad()
    outputs = model(batch, adj, pretrain=True)
    
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
    loss = ce_loss(outputs["next_logits"].view(-1, len(mock_vocabs["endpoint"])), batch["next_endpoint_targets"].view(-1))
    
    loss.backward()
    optimizer.step()
    
    # Verify weights updated
    updated_weight = model.next_head.weight
    assert not torch.equal(init_weight, updated_weight)
    
    # Update memory bank manually
    model.cognitive_engine.memory_bank.update(outputs["z1"], outputs["z1"].unsqueeze(1).expand(-1, seq_len, -1))
    assert model.cognitive_engine.memory_bank.size == batch_size
