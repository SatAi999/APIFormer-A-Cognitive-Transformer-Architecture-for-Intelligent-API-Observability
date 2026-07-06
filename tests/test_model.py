import pytest
import torch
import numpy as np

from src.models.embeddings import Time2Vec, PositionalEncoding, precompute_rope_freqs, rotate_half, apply_rope, APIFormerEmbedding
from src.models.attention import LayerNormalization, ScaledDotProductAttention, MultiHeadSelfAttention
from src.models.transformer import FeedForwardNetwork, TransformerEncoderLayer, TransformerEncoder

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

def test_layer_normalization():
    d_model = 64
    x = torch.randn(8, 16, d_model) * 10.0 + 5.0 # Unnormalized input
    
    ln = LayerNormalization(d_model)
    out = ln(x)
    
    assert out.shape == x.shape
    # Check that mean is approximately 0 and standard deviation is approximately 1
    # along the normalized dimension
    mean = out.mean(dim=-1)
    std = out.std(dim=-1, unbiased=False)
    
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-5)

def test_time2vec():
    d_embed = 16
    t2v = Time2Vec(d_embed)
    
    x = torch.randn(4, 10)
    out = t2v(x)
    
    # Shape: [Batch, SeqLen, DEmbed]
    assert out.shape == (4, 10, d_embed)
    # Check linear dimension is untouched and periodic is bound by [-1, 1]
    assert torch.all(out[..., 1:] >= -1.0) and torch.all(out[..., 1:] <= 1.0)

def test_rope_orthogonality():
    head_dim = 32
    seq_len = 20
    cos, sin = precompute_rope_freqs(head_dim, max_seq_len=seq_len)
    
    assert cos.shape == (seq_len, head_dim)
    assert sin.shape == (seq_len, head_dim)
    
    # Check that applying RoPE does not alter the vector magnitude (orthogonal transformation)
    x = torch.randn(2, 4, seq_len, head_dim)
    x_rot = apply_rope(x, cos, sin)
    
    norm_x = torch.norm(x, dim=-1)
    norm_x_rot = torch.norm(x_rot, dim=-1)
    
    assert torch.allclose(norm_x, norm_x_rot, atol=1e-5)

def test_api_former_embedding(mock_vocabs):
    d_embed = 16
    d_model = 64
    seq_len = 12
    batch_size = 4
    
    embedder = APIFormerEmbedding(vocabs=mock_vocabs, d_embed=d_embed, d_model=d_model, max_len=50)
    
    # Build a mock batch
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
        "payload": torch.randn(batch_size, seq_len) * 1000.0,
        "time_gap": torch.rand(batch_size, seq_len) * 5.0
    }
    
    out = embedder(batch)
    assert out.shape == (batch_size, seq_len, d_model)

def test_multi_head_self_attention():
    d_model = 64
    num_heads = 4
    seq_len = 15
    batch_size = 2
    
    mhsa = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, max_len=50)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Precompute RoPE elements
    head_dim = d_model // num_heads
    cos, sin = precompute_rope_freqs(head_dim, max_seq_len=50)
    
    # Padding mask: mask out last 3 positions in batch 0
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[0, -3:] = True
    # attention mask shape: [Batch, 1, 1, SeqLen]
    attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)
    
    out, weights = mhsa(x, mask=attn_mask, rope_cos=cos, rope_sin=sin)
    
    assert out.shape == (batch_size, seq_len, d_model)
    # Weights size: [Batch, NumHeads, SeqLen, SeqLen]
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)
    
    # Verify that padded weights are approximately 0 (due to masking)
    # The last 3 keys for batch 0 should be masked (soft-max outputs zero)
    masked_prob_sum = weights[0, :, :, -3:].sum()
    assert masked_prob_sum.item() < 1e-4

def test_transformer_encoder(mock_vocabs):
    d_model = 64
    d_ff = 128
    num_heads = 4
    num_layers = 2
    seq_len = 10
    batch_size = 3
    
    embedder = APIFormerEmbedding(vocabs=mock_vocabs, d_embed=16, d_model=d_model, max_len=50)
    encoder = TransformerEncoder(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        use_rope=True,
        max_len=50
    )
    
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
        "time_gap": torch.randn(batch_size, seq_len)
    }
    
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[0, -4:] = True # mask last 4 positions in first sequence
    
    embeddings = embedder(batch)
    out, all_weights = encoder(embeddings, padding_mask=padding_mask)
    
    assert out.shape == (batch_size, seq_len, d_model)
    assert len(all_weights) == num_layers
    
    # Test pooling strategies
    pooled_cls = encoder.pool(out, padding_mask, strategy="cls")
    assert pooled_cls.shape == (batch_size, d_model)
    # CLS should be identical to the first sequence token representation
    assert torch.allclose(pooled_cls, out[:, 0, :])
    
    pooled_mean = encoder.pool(out, padding_mask, strategy="mean")
    assert pooled_mean.shape == (batch_size, d_model)
    
    pooled_max = encoder.pool(out, padding_mask, strategy="max")
    assert pooled_max.shape == (batch_size, d_model)
