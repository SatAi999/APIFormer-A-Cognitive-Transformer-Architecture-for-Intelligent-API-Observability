import os
import torch
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from src.utils.logger import logger
from src.data.tokenizer import APITokenizer
from src.models.heads import APIFormerPlusModel

app = FastAPI(
    title="APIFormer+ serving API",
    description="Cognitive Foundation Transformer serving engine for enterprise API traffic telemetry and security predictions.",
    version="1.0.0"
)

# Global variables for model state
model = None
tokenizer = None
vocabs = None
adj_matrix = None
device = None

class APIEventPayload(BaseModel):
    timestamp: str = Field(..., example="2026-07-05T21:40:00Z")
    endpoint: str = Field(..., example="/api/v1/products/1005")
    http_method: str = Field(..., example="GET")
    status_code: int = Field(..., example=200)
    latency_ms: float = Field(..., example=120.5)
    payload_size_bytes: int = Field(..., example=1500)
    time_since_previous_request: float = Field(..., example=1.5)
    device: str = Field(default="unknown", example="desktop-chrome")
    geo_location: str = Field(default="unknown", example="US-East")
    environment: str = Field(default="production", example="production")
    authentication: str = Field(default="None", example="Bearer")

class SessionSequenceRequest(BaseModel):
    events: List[APIEventPayload] = Field(..., max_items=128)

@app.on_event("startup")
def startup_event():
    global model, tokenizer, vocabs, adj_matrix, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"FastAPI starting model on: {device}")
    
    checkpoint_path = "data/checkpoints/apiformer_plus.pt"
    vocab_path = "data/processed/tokenizer_vocab.json"
    adj_path = "data/processed/gnn_adjacency.npy"
    config_path = "config/pipeline_config.yaml"
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Cannot start FastAPI. Checkpoint {checkpoint_path} not found.")
        return
        
    # Load vocabulary
    with open(vocab_path, "r") as f:
        vocabs = json.load(f)
        
    # Load tokenizer
    with open(config_path, "r") as f:
        import yaml
        config = yaml.safe_load(f)
    tokenizer = APITokenizer(config)
    tokenizer.vocabs = vocabs
    tokenizer._build_inverse_vocabs()
    
    # Load Adjacency
    adj_matrix_np = np.load(adj_path)
    adj_matrix = torch.tensor(adj_matrix_np, dtype=torch.float32, device=device)
    
    # Load Model
    model = APIFormerPlusModel(
        vocabs=vocabs,
        d_embed=32,
        d_model=128,
        num_layers=4,
        num_heads=4,
        d_ff=256,
        gcn_num_nodes=adj_matrix.size(0),
        num_intents=5,
        max_len=128,
        dropout=0.1
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("FastAPI: APIFormer+ Model loaded successfully and warm.")

@app.get("/health")
def health_check():
    if model is None:
        return {"status": "unhealthy", "message": "Model not loaded"}
    return {"status": "healthy", "device": str(device)}

@app.post("/predict")
def predict_session(request: SessionSequenceRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model server not initialized.")
        
    events = request.events
    seq_len = len(events)
    max_len = 128
    
    # 1. Preprocess and tokenize event list
    encoded = {cat: [] for cat in vocabs.keys()}
    encoded["latency"] = []
    encoded["payload"] = []
    encoded["time_gap"] = []
    
    intent_map = ["normal", "503_cascade", "credential_stuffing", "bot_scraping", "sequence_abuse"]
    
    services_map = {
        "gateway": ["/api/v1/auth/login", "/api/v1/products", "/api/v1/products/{id}", "/api/v1/cart", "/api/v1/checkout", "/api/v1/admin/dashboard", "/api/v1/admin/users", "/api/v1/admin/logs"],
        "auth-service": ["/internal/v1/auth/validate", "/internal/v1/auth/login"],
        "catalog-service": ["/internal/v1/products", "/internal/v1/products/{id}"],
        "cart-service": ["/internal/v1/cart/add", "/internal/v1/cart/view"],
        "payment-service": ["/internal/v1/payments/charge", "/internal/v1/payments/refund"],
        "invoice-service": ["/internal/v1/invoices/generate"],
    }
    
    for e in events:
        norm_ep = tokenizer.normalize_endpoint(e.endpoint)
        
        encoded["endpoint"].append(vocabs["endpoint"].get(norm_ep, vocabs["endpoint"].get("[UNK]", 1)))
        encoded["method"].append(vocabs["method"].get(e.http_method.upper(), vocabs["method"].get("[UNK]", 1)))
        encoded["status"].append(vocabs["status"].get(str(e.status_code), vocabs["status"].get("[UNK]", 1)))
        
        # services can be inferred from endpoint or mock catalog
        # Let's map dynamically
        service_val = "gateway"
        for s, endpoints in services_map.items():
            if norm_ep in endpoints:
                service_val = s
                break
        encoded["service"].append(vocabs["service"].get(service_val, vocabs["service"].get("[UNK]", 1)))
        encoded["device"].append(vocabs["device"].get(e.device, vocabs["device"].get("[UNK]", 1)))
        encoded["location"].append(vocabs["location"].get(e.geo_location, vocabs["location"].get("[UNK]", 1)))
        encoded["env"].append(vocabs["env"].get(e.environment, vocabs["env"].get("[UNK]", 1)))
        encoded["auth"].append(vocabs["auth"].get(e.authentication, vocabs["auth"].get("[UNK]", 1)))
        
        encoded["latency"].append(float(e.latency_ms))
        encoded["payload"].append(float(e.payload_size_bytes))
        encoded["time_gap"].append(float(e.time_since_previous_request))
        
    # 2. Add padding to match max_len
    pad_idx = vocabs["endpoint"].get("[PAD]", 0)
    
    padded_batch = {}
    for cat in vocabs.keys():
        vals = encoded[cat][:max_len]
        vals += [pad_idx] * (max_len - len(vals))
        padded_batch[cat] = torch.tensor([vals], dtype=torch.long, device=device) # [1, MaxLen]
        
    for cat in ["latency", "payload", "time_gap"]:
        vals = encoded[cat][:max_len]
        vals += [0.0] * (max_len - len(vals))
        padded_batch[cat] = torch.tensor([vals], dtype=torch.float32, device=device) # [1, MaxLen]
        
    padding_mask = torch.zeros(max_len, dtype=torch.bool, device=device)
    padding_mask[seq_len:] = True
    padded_batch["padding_mask"] = padding_mask.unsqueeze(0) # [1, MaxLen]
    
    # 3. Model Inference (downstream heads and attention extraction)
    with torch.no_grad():
        # Fine-tuning predictions
        outputs = model(padded_batch, adj_matrix, pretrain=False)
        
        # Anomaly probabilities per step
        anomaly_logits = outputs["anomaly_logits"][0, :seq_len] # [SeqLen, 2]
        anomaly_probs = torch.softmax(anomaly_logits, dim=-1)[:, 1].cpu().numpy().tolist()
        
        # Intent classification probabilities
        intent_logits = outputs["intent_logits"][0] # [NumIntents]
        intent_probs = torch.softmax(intent_logits, dim=-1).cpu().numpy().tolist()
        intent_pred_idx = int(torch.argmax(intent_logits).item())
        intent_pred_label = intent_map[intent_pred_idx]
        
        # Bot classification probabilities
        bot_logits = outputs["bot_logits"][0] # [2]
        bot_probs = torch.softmax(bot_logits, dim=-1).cpu().numpy().tolist()
        is_bot = bool(torch.argmax(bot_logits).item() == 1)
        
        # Extract attention weights for explainability
        h_emb = model.embedding(padded_batch)
        h_enc, all_attn_weights = model.encoder(h_emb, padding_mask=padded_batch["padding_mask"])
        
        # Compute average attention matrix across all layers & heads
        num_layers = len(all_attn_weights)
        num_heads = all_attn_weights[0].size(1)
        
        avg_attn = torch.zeros(max_len, max_len, device=device)
        for idx in range(num_layers):
            # Sum heads
            avg_attn += all_attn_weights[idx][0].mean(dim=0)
        avg_attn /= num_layers
        
        # Sliced sequence attention
        seq_attn = avg_attn[:seq_len, :seq_len].cpu().numpy().tolist()
        
    return {
        "predictions": {
            "intent": {
                "label": intent_pred_label,
                "confidence": float(intent_probs[intent_pred_idx]),
                "probabilities": {intent_map[i]: float(intent_probs[i]) for i in range(len(intent_map))}
            },
            "bot_detection": {
                "is_bot": is_bot,
                "confidence": float(bot_probs[1] if is_bot else bot_probs[0]),
                "probabilities": {"user": float(bot_probs[0]), "bot": float(bot_probs[1])}
            },
            "step_anomalies": [
                {
                    "step": i,
                    "endpoint": events[i].endpoint,
                    "anomaly_probability": float(anomaly_probs[i]),
                    "is_anomalous": bool(anomaly_probs[i] > 0.5)
                }
                for i in range(seq_len)
            ]
        },
        "explainability": {
            "attention_matrix": seq_attn
        }
    }
