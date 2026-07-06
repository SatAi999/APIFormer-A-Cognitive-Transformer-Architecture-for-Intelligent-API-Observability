import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from src.models.attention import LayerNormalization, MultiHeadSelfAttention
from src.models.embeddings import precompute_rope_freqs

class FeedForwardNetwork(nn.Module):
    """Feed-Forward Network (FFN) block with linear expansion, GeLU activation, and dropout."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, SeqLen, DModel] -> [Batch, SeqLen, DFF]
        h = self.activation(self.linear1(x))
        h = self.dropout(h)
        # [Batch, SeqLen, DFF] -> [Batch, SeqLen, DModel]
        return self.linear2(h)


class TransformerEncoderLayer(nn.Module):
    """A single Transformer Encoder Layer using Pre-LayerNorm architecture."""
    
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int, 
                 use_rope: bool = True,
                 use_rel_bias: bool = True,
                 max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        # Self Attention Block
        self.ln1 = LayerNormalization(d_model)
        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=use_rope,
            use_rel_bias=use_rel_bias,
            max_len=max_len,
            dropout=dropout
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed Forward Block
        self.ln2 = LayerNormalization(d_model)
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                rope_cos: Optional[torch.Tensor] = None,
                rope_sin: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-LN MHSA block
        norm_x = self.ln1(x)
        attn_out, attn_weights = self.attention(norm_x, mask=mask, rope_cos=rope_cos, rope_sin=rope_sin)
        x = x + self.dropout1(attn_out)
        
        # Pre-LN FFN block
        norm_x2 = self.ln2(x)
        ffn_out = self.ffn(norm_x2)
        x = x + self.dropout2(ffn_out)
        
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """A stack of multiple Pre-LayerNorm Transformer Encoder Layers managing RoPE caches."""
    
    def __init__(self, 
                 d_model: int, 
                 num_layers: int, 
                 num_heads: int, 
                 d_ff: int, 
                 use_rope: bool = True,
                 use_rel_bias: bool = True,
                 max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope
        self.max_len = max_len
        
        # Precompute RoPE frequency caches if enabled
        if self.use_rope:
            cos, sin = precompute_rope_freqs(self.head_dim, max_seq_len=max_len)
            self.register_buffer("rope_cos", cos)
            self.register_buffer("rope_sin", sin)
        else:
            self.rope_cos = None
            self.rope_sin = None
            
        # Encoder layer stack
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                use_rope=use_rope,
                use_rel_bias=use_rel_bias,
                max_len=max_len,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization layer (critical for Pre-LN stability)
        self.final_ln = LayerNormalization(d_model)

    def forward(self, 
                x: torch.Tensor, 
                padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        # x shape: [Batch, SeqLen, DModel]
        # padding_mask shape: [Batch, SeqLen] (True indicates padded elements to be masked)
        
        # Expand padding mask for attention: [Batch, NumHeads, SeqLen, SeqLen]
        attn_mask = None
        if padding_mask is not None:
            # We want to mask out column positions corresponding to padding tokens
            # Shape: [Batch, 1, 1, SeqLen]
            attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            
        # Store attention weights across layers for explainability
        all_attn_weights = {}
        
        # Send through encoder layers
        for idx, layer in enumerate(self.layers):
            x, weights = layer(x, mask=attn_mask, rope_cos=self.rope_cos, rope_sin=self.rope_sin)
            all_attn_weights[idx] = weights
            
        # Final normalization
        x = self.final_ln(x)
        
        return x, all_attn_weights

    def pool(self, 
             x: torch.Tensor, 
             padding_mask: Optional[torch.Tensor] = None, 
             strategy: str = "cls") -> torch.Tensor:
        """Pools sequence representations into a unified session embedding.

        
        Strategies:
        - 'cls': Extracts the representation of the first token (index 0).
        - 'mean': Averages representations, ignoring padded tokens.
        - 'max': Takes maximum feature activation, ignoring padded tokens.
        """
        # x shape: [Batch, SeqLen, DModel]
        if strategy == "cls":
            # Extract index 0 along sequence dimension -> [Batch, DModel]
            return x[:, 0, :]
            
        # Convert padding_mask to inverse float multiplier (1.0 for valid, 0.0 for padding)
        # padding_mask: [Batch, SeqLen] -> True means padded
        if padding_mask is not None:
            # [Batch, SeqLen, 1]
            valid_mask = (~padding_mask).unsqueeze(-1).float()
        else:
            valid_mask = torch.ones_like(x[..., :1])
            
        if strategy == "mean":
            # Sum up valid representations
            summed = torch.sum(x * valid_mask, dim=1) # [Batch, DModel]
            # Count valid tokens per batch item (clip at 1.0 to avoid divide-by-zero on empty sequences)
            counts = torch.sum(valid_mask, dim=1).clamp(min=1.0) # [Batch, 1]
            return summed / counts
            
        elif strategy == "max":
            # Set padded elements to a large negative number so they don't impact max
            masked_x = x.masked_fill(padding_mask.unsqueeze(-1), -1e9) if padding_mask is not None else x
            maxed, _ = torch.max(masked_x, dim=1) # [Batch, DModel]
            return maxed
            
        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")
