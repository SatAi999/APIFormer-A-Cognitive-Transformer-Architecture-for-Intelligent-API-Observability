import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from src.models.embeddings import apply_rope

class LayerNormalization(nn.Module):
    """Custom Layer Normalization layer implemented from scratch."""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scale (gamma) and shift (beta) parameters
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate mean along the last dimension: [Batch, SeqLen, 1]
        mean = x.mean(dim=-1, keepdim=True)
        # Calculate variance along the last dimension
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Standardize and scale
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class ScaledDotProductAttention(nn.Module):
    """Computes scaled dot-product attention scores with masks and relative positional biases."""
    
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                rel_bias: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, k, v shape: [Batch, NumHeads, SeqLen, HeadDim]
        head_dim = q.size(-1)
        
        # Compute dot products: [Batch, NumHeads, SeqLen, SeqLen]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Apply Relative Position Bias if available
        if rel_bias is not None:
            # rel_bias shape: [NumHeads, SeqLen, SeqLen] or [Batch, NumHeads, SeqLen, SeqLen]
            scores = scores + rel_bias
            
        # Apply Attention Padding Mask if available
        if mask is not None:
            # mask shape: [Batch, 1, 1, SeqLen] or similar
            # Replace True values in mask with a large negative value
            scores = scores.masked_fill(mask, -1e9)
            
        # Calculate softmax probability distribution
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Weighted sum: [Batch, NumHeads, SeqLen, HeadDim]
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention block supporting relative biases and RoPE rotary encodings."""
    
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 use_rope: bool = True,
                 use_rel_bias: bool = True,
                 max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope
        self.use_rel_bias = use_rel_bias
        self.max_len = max_len
        
        # Q, K, V Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Attention execution core
        self.attention = ScaledDotProductAttention(dropout=dropout)
        
        # Relative Position Bias lookup table (T5 style)
        if self.use_rel_bias:
            # We map relative distances from -max_len to +max_len (total 2 * max_len)
            self.rel_bias_table = nn.Embedding(2 * max_len, num_heads)
            
    def _compute_relative_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generates relative position bias query-key index offsets."""
        # Create a grid of query positions (rows) and key positions (cols)
        q_pos = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(1) # [L, 1]
        k_pos = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0) # [1, L]
        
        # Relative distance: q - k (ranges from -L+1 to L-1)
        rel_dist = q_pos - k_pos
        
        # Shift index to make it positive: ranges from max_len - L + 1 to max_len + L - 1
        rel_index = rel_dist + self.max_len
        # Ensure it fits within index bounds (clipping)
        rel_index = torch.clamp(rel_index, 0, 2 * self.max_len - 1)
        
        # Lookup bias: [SeqLen, SeqLen, NumHeads]
        bias = self.rel_bias_table(rel_index)
        
        # Reshape to [NumHeads, SeqLen, SeqLen]
        return bias.permute(2, 0, 1)

    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                rope_cos: Optional[torch.Tensor] = None,
                rope_sin: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # Project inputs to Q, K, V -> [Batch, SeqLen, DModel]
        q_val = self.q_proj(x)
        k_val = self.k_proj(x)
        v_val = self.v_proj(x)
        
        # Split into heads and transpose: [Batch, NumHeads, SeqLen, HeadDim]
        q = q_val.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_val.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_val.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply Rotary Position Embeddings (RoPE) if enabled
        if self.use_rope and rope_cos is not None and rope_sin is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)
            
        # Get relative position bias
        rel_bias = None
        if self.use_rel_bias:
            rel_bias = self._compute_relative_bias(seq_len, x.device)
            
        # Calculate scaled dot product attention
        attn_out, attn_weights = self.attention(q, k, v, mask=mask, rel_bias=rel_bias)
        
        # Transpose and concatenate head outputs -> [Batch, SeqLen, DModel]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output linear project
        output = self.out_proj(attn_out)
        
        return output, attn_weights
