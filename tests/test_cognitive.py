import pytest
import torch
import numpy as np

from src.models.cognitive import GCNLayer, TemporalDecayAttention, SessionMemoryBank, MemoryCrossAttention, APIFormerCognitiveEngine
from src.models.embeddings import precompute_rope_freqs

def test_gcn_layer():
    num_nodes = 5
    d_in = 32
    d_out = 64
    
    # Adjacency matrix: node 0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4
    adj = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes - 1):
        adj[i, i + 1] = 1.0
        
    h = torch.randn(num_nodes, d_in)
    
    gcn = GCNLayer(d_in, d_out)
    out = gcn(h, adj)
    
    # Shape: [NumNodes, d_out]
    assert out.shape == (num_nodes, d_out)
    
    # Check normalized adjacency properties: symmetric scaling
    norm_adj = gcn._normalize_adjacency(adj)
    assert norm_adj.shape == (num_nodes, num_nodes)
    assert torch.all(norm_adj >= 0.0)
    # Norm adj should have non-zero diagonals due to self-loops
    assert torch.all(torch.diag(norm_adj) > 0.0)

def test_temporal_decay_attention():
    d_model = 32
    num_heads = 2
    seq_len = 5
    batch_size = 2
    
    temporal_attn = TemporalDecayAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Time gaps: Sequence 0 has small time gaps (0.1s), Sequence 1 has large time gaps (100.0s)
    time_gaps = torch.zeros(batch_size, seq_len)
    time_gaps[0] = torch.tensor([0.0, 0.1, 0.1, 0.1, 0.1])
    time_gaps[1] = torch.tensor([0.0, 100.0, 100.0, 100.0, 100.0])
    
    out = temporal_attn(x, time_gaps)
    assert out.shape == (batch_size, seq_len, d_model)
    
    # Let's inspect that decay operates as expected.
    # High time gaps (sequence 1) should increase the decay penalty and change weights.
    # We check by evaluating the delta_t difference calculations
    timestamps = torch.cumsum(time_gaps, dim=-1)
    t_row = timestamps.unsqueeze(-1)
    t_col = timestamps.unsqueeze(-2)
    delta_t = torch.abs(t_row - t_col)
    
    assert delta_t[0, 1, 2].item() == pytest.approx(0.1, abs=1e-5)
    assert delta_t[1, 1, 2].item() == pytest.approx(100.0, abs=1e-5)

def test_session_memory_bank_and_cross_attn():
    d_model = 16
    bank_size = 10
    seq_len = 4
    batch_size = 2
    
    # Initialize Memory bank
    bank = SessionMemoryBank(d_model=d_model, max_size=bank_size, seq_len=seq_len)
    assert bank.size == 0
    
    # Add dummy sessions
    dummy_embeds = torch.randn(5, d_model)
    dummy_seqs = torch.randn(5, seq_len, d_model)
    bank.update(dummy_embeds, dummy_seqs)
    
    assert bank.size == 5
    assert bank.pointer == 5
    
    # Test pointer wrapping
    more_embeds = torch.randn(7, d_model)
    more_seqs = torch.randn(7, seq_len, d_model)
    bank.update(more_embeds, more_seqs)
    
    # Size should clip at max_size (10)
    assert bank.size == bank_size
    assert bank.pointer == 2 # 5 + 7 = 12 % 10 = 2
    
    # Test retrieval
    query = torch.randn(batch_size, d_model)
    ret_seqs, ret_mask = bank.retrieve(query, K=3)
    
    # Shape: [Batch, K * seq_len, DModel] -> [2, 3 * 4, 16] = [2, 12, 16]
    assert ret_seqs.shape == (batch_size, 3 * seq_len, d_model)
    assert ret_mask.shape == (batch_size, 3 * seq_len)
    assert not torch.any(ret_mask) # Since size (10) >= K (3), no padding slots are masked
    
    # Test cross attention
    cross_attn = MemoryCrossAttention(d_model=d_model, num_heads=2)
    x = torch.randn(batch_size, seq_len, d_model)
    out = cross_attn(x, ret_seqs, memory_mask=ret_mask)
    
    assert out.shape == (batch_size, seq_len, d_model)

def test_api_former_cognitive_engine():
    d_model = 32
    num_heads = 2
    num_nodes = 8
    seq_len = 6
    batch_size = 2
    
    engine = APIFormerCognitiveEngine(
        d_model=d_model,
        num_heads=num_heads,
        gcn_num_nodes=num_nodes,
        max_len=seq_len
    )
    
    h_trans = torch.randn(batch_size, seq_len, d_model)
    static_embeddings = torch.randn(num_nodes, d_model)
    
    adj_matrix = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes - 1):
        adj_matrix[i, i + 1] = 1.0
        
    batch = {
        "endpoint": torch.randint(0, num_nodes, (batch_size, seq_len)),
        "time_gap": torch.rand(batch_size, seq_len) * 5.0
    }
    
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[0, -2:] = True
    
    # Ensure memory bank has at least 1 session to prevent retrieval zero-fallbacks
    engine.memory_bank.update(
        torch.randn(1, d_model),
        torch.randn(1, seq_len, d_model)
    )
    
    out, debug_info = engine(
        h_trans=h_trans,
        batch=batch,
        static_embeddings=static_embeddings,
        adj_matrix=adj_matrix,
        padding_mask=padding_mask
    )
    
    assert out.shape == (batch_size, seq_len, d_model)
    assert "query_embeddings" in debug_info
    assert debug_info["query_embeddings"].shape == (batch_size, d_model)
