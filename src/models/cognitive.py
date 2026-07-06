import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Tuple, Optional, List
from src.models.attention import LayerNormalization, ScaledDotProductAttention

class GCNLayer(nn.Module):
    """Custom Graph Convolutional Network (GCN) layer implemented from scratch."""
    
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(d_out))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.weight)
        
    def _normalize_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        """Applies symmetric normalization to the adjacency matrix with self-loops."""
        device = adj.device
        N = adj.size(0)
        
        # Add self-loops (A_tilde = A + I)
        identity = torch.eye(N, device=device)
        adj_tilde = adj + identity
        
        # Calculate row sums (Degree matrix D_tilde)
        row_sums = adj_tilde.sum(dim=1)
        
        # Compute D_tilde^{-1/2}
        d_inv_sqrt = torch.pow(row_sums, -0.5)
        # Handle division by zero for isolated nodes
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        
        # Symmetric Normalization: D^{-1/2} * A * D^{-1/2}
        # Broadcasting: d_inv_sqrt[:, None] applies column-wise, d_inv_sqrt[None, :] applies row-wise
        norm_adj = d_inv_sqrt[:, None] * adj_tilde * d_inv_sqrt[None, :]
        return norm_adj

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h shape: [NumNodes, d_in]
        # adj shape: [NumNodes, NumNodes]
        
        # 1. Normalize adjacency matrix
        norm_adj = self._normalize_adjacency(adj)
        
        # 2. Graph convolution: A_norm * H * W
        support = torch.matmul(h, self.weight) # [NumNodes, d_out]
        output = torch.matmul(norm_adj, support) + self.bias # [NumNodes, d_out]
        
        return F.relu(output)


class TemporalDecayAttention(nn.Module):
    """Refines representations by applying a Hawkes-process exponential decay penalty to self-attention."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Learnable temporal decay parameters (one per head, initialized to 0 log-scale)
        self.log_gamma = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, time_gaps: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: [Batch, SeqLen, DModel]
        # time_gaps shape: [Batch, SeqLen] (Inter-request delays)
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # 1. Compute absolute timestamps by cumulative summation of time gaps
        # timestamps: [Batch, SeqLen]
        timestamps = torch.cumsum(time_gaps, dim=-1)
        
        # Compute pairwise time differences: [Batch, SeqLen, SeqLen]
        t_row = timestamps.unsqueeze(-1) # [Batch, SeqLen, 1]
        t_col = timestamps.unsqueeze(-2) # [Batch, 1, SeqLen]
        delta_t = torch.abs(t_row - t_col) # [Batch, SeqLen, SeqLen]
        
        # 2. Query, Key, Value projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Base similarity scores: [Batch, NumHeads, SeqLen, SeqLen]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 4. Apply Hawkes Decay Penalty: log(1.0 + gamma * delta_t)
        # gamma shape: [1, NumHeads, 1, 1]
        gamma = torch.exp(self.log_gamma).unsqueeze(0)
        decay = torch.log(1.0 + gamma * delta_t.unsqueeze(1)) # [Batch, NumHeads, SeqLen, SeqLen]
        
        # Subtract decay from attention scores
        scores = scores - decay
        
        # Apply padding mask if provided
        if mask is not None:
            # mask shape: [Batch, 1, 1, SeqLen]
            scores = scores.masked_fill(mask, -1e9)
            
        # 5. Softmax + Output projection
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v) # [Batch, NumHeads, SeqLen, HeadDim]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(context)


class SessionMemoryBank:
    """Retrieval-based memory bank storing historical session embeddings and trace sequences."""
    
    def __init__(self, d_model: int, max_size: int = 500, seq_len: int = 128):
        self.d_model = d_model
        self.max_size = max_size
        self.seq_len = seq_len
        
        # Pre-allocate memory bank arrays on CPU
        self.session_embeddings = torch.zeros(max_size, d_model)
        self.session_sequences = torch.zeros(max_size, seq_len, d_model)
        self.size = 0
        self.pointer = 0

    def update(self, embeddings: torch.Tensor, sequences: torch.Tensor) -> None:
        """Inserts a batch of new sessions into the memory bank (rolling buffer)."""
        # embeddings: [Batch, DModel], sequences: [Batch, SeqLen, DModel]
        batch_size = embeddings.size(0)
        
        # Move inputs to CPU for storage
        embeddings = embeddings.detach().cpu()
        sequences = sequences.detach().cpu()
        
        for i in range(batch_size):
            self.session_embeddings[self.pointer] = embeddings[i]
            self.session_sequences[self.pointer] = sequences[i]
            
            self.pointer = (self.pointer + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def retrieve(self, query_embeddings: torch.Tensor, K: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """Looks up the top-K similar sessions and returns their sequence contexts.

        
        query_embeddings shape: [Batch, DModel]
        Returns:
          - retrieved_sequences: [Batch, K * SeqLen, DModel]
          - retrieved_mask: [Batch, K * SeqLen] (True indicates dummy/padding slots if K > size)
        """
        batch_size = query_embeddings.size(0)
        device = query_embeddings.device
        
        # If memory bank is empty, return zeros
        if self.size == 0:
            ret_seqs = torch.zeros(batch_size, K * self.seq_len, self.d_model, device=device)
            ret_mask = torch.ones(batch_size, K * self.seq_len, dtype=torch.bool, device=device)
            return ret_seqs, ret_mask
            
        # Get active bank slice
        bank_embeds = self.session_embeddings[:self.size].to(device) # [Size, DModel]
        bank_seqs = self.session_sequences[:self.size].to(device)     # [Size, SeqLen, DModel]
        
        # Compute normalized cosine similarity: [Batch, Size]
        query_norm = F.normalize(query_embeddings, p=2, dim=-1)
        bank_norm = F.normalize(bank_embeds, p=2, dim=-1)
        similarities = torch.matmul(query_norm, bank_norm.transpose(0, 1))
        
        # Retrieve indices
        actual_K = min(K, self.size)
        topk_vals, topk_indices = torch.topk(similarities, k=actual_K, dim=-1)
        
        retrieved_seq_list = []
        
        for b in range(batch_size):
            # Extract top-K session sequences for this batch item -> [K, SeqLen, DModel]
            b_indices = topk_indices[b]
            b_seqs = bank_seqs[b_indices]
            
            # Flatten to [K * SeqLen, DModel]
            flat_b_seqs = b_seqs.view(actual_K * self.seq_len, self.d_model)
            
            # If bank has fewer sessions than K, pad with zeros
            if actual_K < K:
                padding = torch.zeros((K - actual_K) * self.seq_len, self.d_model, device=device)
                flat_b_seqs = torch.cat([flat_b_seqs, padding], dim=0)
                
            retrieved_seq_list.append(flat_b_seqs)
            
        # Stack to [Batch, K * SeqLen, DModel]
        retrieved_sequences = torch.stack(retrieved_seq_list, dim=0)
        
        # Create mask indicating where memory features exist vs padding
        retrieved_mask = torch.zeros(batch_size, K * self.seq_len, dtype=torch.bool, device=device)
        if actual_K < K:
            retrieved_mask[:, actual_K * self.seq_len:] = True
            
        return retrieved_sequences, retrieved_mask


class MemoryCrossAttention(nn.Module):
    """Cross-attention layer that fuses current sequence keys against retrieved memory values."""
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout=dropout)
        
    def forward(self, x: torch.Tensor, memory: torch.Tensor, memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x (Queries): [Batch, SeqLen, DModel]
        # memory (Keys/Values): [Batch, MemSeqLen, DModel]
        # memory_mask: [Batch, MemSeqLen] (True indicates padded slots)
        batch_size, seq_len, _ = x.shape
        mem_len = memory.size(1)
        
        # Project Query from sequence, Key and Value from retrieved memories
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(batch_size, mem_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(batch_size, mem_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Format padding mask: [Batch, 1, 1, MemSeqLen]
        mask = None
        if memory_mask is not None:
            mask = memory_mask.unsqueeze(1).unsqueeze(2)
            
        # Cross Attention
        context, _ = self.attention(q, k, v, mask=mask)
        
        # Concatenate and project back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context)


class APIFormerCognitiveEngine(nn.Module):
    """Unified Cognitive Engine coordinating GCN layers, temporal decays, and retrieval memory banks."""
    
    def __init__(self, 
                 d_model: int, 
                 num_heads: int,
                 gcn_num_nodes: int,
                 use_rope: bool = True,
                 max_len: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # 1. Structural GCN Module
        self.gcn = GCNLayer(d_in=d_model, d_out=d_model)
        
        # 2. Temporal Hawkes Decay Module
        self.temporal_refiner = TemporalDecayAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        
        # 3. Retrieval Memory Module
        self.memory_bank = SessionMemoryBank(d_model=d_model, max_size=1000, seq_len=max_len)
        self.memory_cross_attn = MemoryCrossAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        
        # Final Norm layer
        self.ln_fuse = LayerNormalization(d_model)
        
    def forward(self, 
                h_trans: torch.Tensor, 
                batch: Dict[str, torch.Tensor],
                static_embeddings: torch.Tensor,
                adj_matrix: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # h_trans: [Batch, SeqLen, DModel] (Raw outputs from Transformer encoder)
        # static_embeddings: [NumNodes, DModel] (Weights of endpoint embedding layer)
        # adj_matrix: [NumNodes, NumNodes] (Static endpoint transition graph)
        # padding_mask: [Batch, SeqLen] (True indicates padded tokens)
        
        device = h_trans.device
        batch_size, seq_len, _ = h_trans.shape
        
        # --- 1. Graph Module: Run GCN over microservice topology ---
        h_nodes_refined = self.gcn(static_embeddings, adj_matrix) # [NumNodes, DModel]
        
        # Lookup GCN-refined node vectors for sequence tokens
        endpoints = batch["endpoint"] # [Batch, SeqLen]
        h_graph_seq = h_nodes_refined[endpoints] # [Batch, SeqLen, DModel]
        
        # Add GCN structural representation to sequence
        h_fused = h_trans + h_graph_seq
        
        # --- 2. Temporal Module: Apply Hawkes-decay attention ---
        time_gaps = batch["time_gap"] # [Batch, SeqLen]
        attn_mask = padding_mask.unsqueeze(1).unsqueeze(2) if padding_mask is not None else None
        h_temp = self.temporal_refiner(h_fused, time_gaps, mask=attn_mask)
        h_fused = h_fused + h_temp
        
        # --- 3. Memory Module: Cosine Retrieval & Cross-Attention ---
        # Pool active sequence to create a query representation -> [Batch, DModel]
        # Using simple mean pooling of valid sequence tokens
        if padding_mask is not None:
            valid_mask = (~padding_mask).unsqueeze(-1).float()
            summed = torch.sum(h_fused * valid_mask, dim=1)
            counts = torch.sum(valid_mask, dim=1).clamp(min=1.0)
            query_embeddings = summed / counts
        else:
            query_embeddings = h_fused.mean(dim=1)
            
        # Retrieve nearest neighbor sessions
        retrieved_seqs, retrieved_mask = self.memory_bank.retrieve(query_embeddings, K=3)
        
        # Fuse retrieved session memories
        h_mem = self.memory_cross_attn(h_fused, retrieved_seqs, memory_mask=retrieved_mask)
        
        # Unified fusion
        h_cognitive = self.ln_fuse(h_fused + h_mem)
        
        # Return Unified Cognitive State and metadata for debugging
        debug_info = {
            "query_embeddings": query_embeddings,
            "retrieved_mask": retrieved_mask
        }
        return h_cognitive, debug_info
