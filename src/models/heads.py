import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from src.models.embeddings import APIFormerEmbedding
from src.models.transformer import TransformerEncoder
from src.models.cognitive import APIFormerCognitiveEngine

class InfoNCELoss(nn.Module):
    """Computes symmetric InfoNCE contrastive loss across dual-augmented session views."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.size(0)
        # Normalize representations to unit hypersphere
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)
        
        # Compute cosine similarity matrix between all positive and negative views: [Batch, Batch]
        sim_matrix = torch.matmul(z1, z2.transpose(0, 1)) / self.temperature
        
        # Labels are diagonal (index i of z1 maps to index i of z2)
        labels = torch.arange(batch_size, device=z1.device)
        
        # Symmetric InfoNCE loss calculation
        loss_1 = F.cross_entropy(sim_matrix, labels)
        loss_2 = F.cross_entropy(sim_matrix.transpose(0, 1), labels)
        
        return (loss_1 + loss_2) / 2.0


class APIFormerPlusModel(nn.Module):
    """Flagship APIFormer+ Foundation Model wrapper.

    
    Integrates Multi-Feature Embeddings, Custom Transformer Encoder, 
    Unified Cognitive Modules (GCN, Hawkes, Memory), and Multi-Task Predictors.
    """
    
    def __init__(self, 
                 vocabs: Dict[str, Dict[str, int]], 
                 d_embed: int = 32, 
                 d_model: int = 128, 
                 num_layers: int = 4, 
                 num_heads: int = 4, 
                 d_ff: int = 256, 
                 gcn_num_nodes: int = 100,
                 num_intents: int = 5,
                 max_len: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        
        self.vocabs = vocabs
        self.d_model = d_model
        
        # 1. Scaffolding Base Backbone Layers
        self.embedding = APIFormerEmbedding(
            vocabs=vocabs, 
            d_embed=d_embed, 
            d_model=d_model, 
            max_len=max_len, 
            dropout=dropout
        )
        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            use_rope=True,
            max_len=max_len,
            dropout=dropout
        )
        
        # 2. Unified Cognitive Engine
        self.cognitive_engine = APIFormerCognitiveEngine(
            d_model=d_model,
            num_heads=num_heads,
            gcn_num_nodes=gcn_num_nodes,
            use_rope=True,
            max_len=max_len,
            dropout=dropout
        )
        
        # 3. Pretraining Prediction Heads
        v_endpoints = len(vocabs["endpoint"])
        v_statuses = len(vocabs["status"])
        
        self.mam_head = nn.Linear(d_model, v_endpoints)
        self.next_head = nn.Linear(d_model, v_endpoints)
        self.latency_head = nn.Linear(d_model, 1)
        self.status_head = nn.Linear(d_model, v_statuses)
        
        # Contrastive Loss function
        self.contrastive_loss_fn = InfoNCELoss()
        
        # 4. Fine-Tuning Multi-Task Prediction Heads
        self.anomaly_head = nn.Linear(d_model, 2)  # Binary anomaly classification
        self.intent_head = nn.Linear(d_model, num_intents)  # Shopping, Admin, bot_scraping, etc.
        self.bot_head = nn.Linear(d_model, 2)  # Binary user vs bot classification

    def _pool_session(self, h: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Helper to mean pool sequence embeddings, ignoring padded tokens."""
        return self.encoder.pool(h, padding_mask, strategy="mean")

    def forward(self, 
                batch: Dict[str, torch.Tensor], 
                adj_matrix: torch.Tensor,
                pretrain: bool = True) -> Dict[str, torch.Tensor]:
        # batch features: endpoint, method, status, service, device, location, env, auth
        # latency, payload, time_gap, padding_mask
        # adj_matrix shape: [NumNodes, NumNodes]
        
        padding_mask = batch.get("padding_mask")
        
        # --- 1. Forward Pass on Input Sequence ---
        # Embed -> [Batch, SeqLen, DModel]
        h_emb = self.embedding(batch)
        # Encoder -> [Batch, SeqLen, DModel]
        h_enc, _ = self.encoder(h_emb, padding_mask=padding_mask)
        
        # Cognitive Engine -> [Batch, SeqLen, DModel]
        static_embed_weights = self.embedding.embeddings["endpoint"].weight # [NumNodes, DEmbed]
        # Project static embeddings to DModel to match GCN dimensions
        static_embed_projected = self.embedding.project.weight[:, :static_embed_weights.size(1)].transpose(0, 1)
        # Let's project properly: static_embed_weights is [NumNodes, DEmbed], project is [DEmbed * 11, DModel]
        # We project each using a linear projection matching the feature index (index 0 is endpoint)
        # The endpoint embeddings are mapped via index 0. Weight slice: self.embedding.project.weight[:, 0:d_embed]
        # Weight mapping shape: [DModel, DEmbed] -> transpose to [DEmbed, DModel]
        W_ep = self.embedding.project.weight[:, :static_embed_weights.size(1)] # [DModel, DEmbed]
        b_ep = self.embedding.project.bias / 11.0 # distributed bias
        static_embeddings = torch.matmul(static_embed_weights, W_ep.transpose(0, 1)) + b_ep # [NumNodes, DModel]
        
        h_cognitive, debug_info = self.cognitive_engine(
            h_trans=h_enc,
            batch=batch,
            static_embeddings=static_embeddings,
            adj_matrix=adj_matrix,
            padding_mask=padding_mask
        )
        
        outputs = {}
        
        if pretrain:
            # Pretraining logits
            outputs["mam_logits"] = self.mam_head(h_cognitive)        # [Batch, SeqLen, V_endpoint]
            outputs["next_logits"] = self.next_head(h_cognitive)      # [Batch, SeqLen, V_endpoint]
            outputs["latency_preds"] = self.latency_head(h_cognitive).squeeze(-1) # [Batch, SeqLen]
            outputs["status_logits"] = self.status_head(h_cognitive)  # [Batch, SeqLen, V_status]
            
            # Compute Contrastive representations of views
            # We build View 1 and View 2 batch dicts
            batch_v1 = {k: v for k, v in batch.items()}
            batch_v1["endpoint"] = batch["v1_endpoint"]
            batch_v1["latency"] = batch["v1_latency"]
            
            batch_v2 = {k: v for k, v in batch.items()}
            batch_v2["endpoint"] = batch["v2_endpoint"]
            batch_v2["latency"] = batch["v2_latency"]
            
            # Forward passes on views (ignoring memory loops inside contrastive to prevent recursive complexity)
            # Embed -> Encoder -> Pool -> z
            emb_v1 = self.embedding(batch_v1)
            enc_v1, _ = self.encoder(emb_v1, padding_mask=padding_mask)
            z1 = self._pool_session(enc_v1, padding_mask)
            
            emb_v2 = self.embedding(batch_v2)
            enc_v2, _ = self.encoder(emb_v2, padding_mask=padding_mask)
            z2 = self._pool_session(enc_v2, padding_mask)
            
            outputs["z1"] = z1
            outputs["z2"] = z2
            
        else:
            # Fine-tuning downstream logits
            outputs["anomaly_logits"] = self.anomaly_head(h_cognitive) # [Batch, SeqLen, 2]
            
            z_session = self._pool_session(h_cognitive, padding_mask)
            outputs["intent_logits"] = self.intent_head(z_session)   # [Batch, NumIntents]
            outputs["bot_logits"] = self.bot_head(z_session)         # [Batch, 2]
            
        return outputs
