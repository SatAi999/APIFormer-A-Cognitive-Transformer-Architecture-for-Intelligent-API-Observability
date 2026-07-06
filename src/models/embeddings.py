import torch
import torch.nn as nn
import math
from typing import Dict, Any, Tuple

class Time2Vec(nn.Module):
    """Time2Vec representation layer for continuous variables.

    
    Computes periodic and linear representations of numerical values:
    T2V(tau)[0] = w0 * tau + b0
    T2V(tau)[i] = sin(wi * tau + bi) for 1 <= i < d_embed
    """
    def __init__(self, d_embed: int):
        super().__init__()
        self.d_embed = d_embed
        
        # Linear/periodic weights and biases
        self.w = nn.Parameter(torch.randn(d_embed, 1))
        self.b = nn.Parameter(torch.randn(d_embed))
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.w)
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, SeqLen]
        # Reshape to [Batch, SeqLen, 1] for matrix multiplication
        x_unsqueezed = x.unsqueeze(-1)
        
        # Project: [Batch, SeqLen, d_embed]
        # x_unsqueezed * w^T -> [Batch, SeqLen, d_embed]
        proj = torch.matmul(x_unsqueezed, self.w.transpose(0, 1)) + self.b
        
        # Split into linear part and periodic part
        linear_part = proj[..., :1]
        periodic_part = torch.sin(proj[..., 1:])
        
        # Concatenate back
        return torch.cat([linear_part, periodic_part], dim=-1)


class PositionalEncoding(nn.Module):
    """Sinusoidal absolute positional encoding for sequence structures."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Shape: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, SeqLen, d_model]
        return x + self.pe[:, :x.size(1)]


def precompute_rope_freqs(head_dim: int, max_seq_len: int = 512, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precomputes rotary embedding cosine and sine frequency caches."""
    # head_dim must be even
    assert head_dim % 2 == 0, "Head dimension must be even for RoPE"
    
    # Generate rotation angles
    # dim_idx goes 0, 2, 4... head_dim-2
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    
    # Generate positions: [max_seq_len]
    t = torch.arange(max_seq_len, dtype=torch.float)
    
    # Outer product: [max_seq_len, head_dim // 2]
    freqss = torch.outer(t, inv_freq)
    
    # Concatenate angles for split halves: [max_seq_len, head_dim]
    # e.g. [a, b] -> [a, b, a, b]
    freqs = torch.cat([freqss, freqss], dim=-1)
    
    # Cache cos & sin
    cos = torch.cos(freqs) # [max_seq_len, head_dim]
    sin = torch.sin(freqs) # [max_seq_len, head_dim]
    
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions for real-valued RoPE."""
    # Split the last dimension (head_dim) in half
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies Rotary Position Embeddings to query or key head tensors.

    
    xq/xk shape: [Batch, NumHeads, SeqLen, HeadDim]
    cos/sin shape: [SeqLen, HeadDim] -> reshaped for broadcasting
    """
    # Reshape cos/sin to [1, 1, SeqLen, HeadDim]
    seq_len = x.size(2)
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    
    # RoPE Rotation: x * cos + rotate_half(x) * sin
    return (x * cos) + (rotate_half(x) * sin)


class APIFormerEmbedding(nn.Module):
    """Combines categorical, numerical, and temporal embeddings into a single vector representation."""
    
    def __init__(self, 
                 vocabs: Dict[str, Dict[str, int]], 
                 d_embed: int = 32, 
                 d_model: int = 128, 
                 max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        # Categorical embedding tables
        self.embeddings = nn.ModuleDict({
            cat: nn.Embedding(len(vocab), d_embed, padding_idx=vocab.get("[PAD]", 0))
            for cat, vocab in vocabs.items()
        })
        
        # Continuous numerical embedding tables via Time2Vec
        self.latency_t2v = Time2Vec(d_embed)
        self.payload_t2v = Time2Vec(d_embed)
        self.time_gap_t2v = Time2Vec(d_embed)
        
        # Projection from concatenated feature maps (8 categoricals + 3 continuous = 11 features)
        num_features = 11
        self.project = nn.Linear(num_features * d_embed, d_model)
        
        # Sinusoidal absolute positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Extract features
        endpoints = batch["endpoint"]       # [Batch, SeqLen]
        methods = batch["method"]           # [Batch, SeqLen]
        statuses = batch["status"]         # [Batch, SeqLen]
        services = batch["service"]         # [Batch, SeqLen]
        devices = batch["device"]           # [Batch, SeqLen]
        locations = batch["location"]       # [Batch, SeqLen]
        envs = batch["env"]                 # [Batch, SeqLen]
        auths = batch["auth"]               # [Batch, SeqLen]
        
        latencies = batch["latency"]       # [Batch, SeqLen]
        payloads = batch["payload"]         # [Batch, SeqLen]
        time_gaps = batch["time_gap"]       # [Batch, SeqLen]
        
        # 1. Categorical Embeddings -> [Batch, SeqLen, d_embed]
        e_ep = self.embeddings["endpoint"](endpoints)
        e_meth = self.embeddings["method"](methods)
        e_stat = self.embeddings["status"](statuses)
        e_serv = self.embeddings["service"](services)
        e_dev = self.embeddings["device"](devices)
        e_loc = self.embeddings["location"](locations)
        e_env = self.embeddings["env"](envs)
        e_auth = self.embeddings["auth"](auths)
        
        # 2. Continuous Projections -> [Batch, SeqLen, d_embed]
        e_lat = self.latency_t2v(latencies)
        e_pay = self.payload_t2v(payloads)
        e_gap = self.time_gap_t2v(time_gaps)
        
        # 3. Concatenate all dimensions -> [Batch, SeqLen, 11 * d_embed]
        fused = torch.cat([
            e_ep, e_meth, e_stat, e_serv, e_dev, e_loc, e_env, e_auth,
            e_lat, e_pay, e_gap
        ], dim=-1)
        
        # 4. Joint Feature Projection -> [Batch, SeqLen, d_model]
        projected = self.project(fused)
        
        # 5. Add Absolute Positional Encoding
        out = self.pos_encoder(projected)
        
        return self.dropout(out)
