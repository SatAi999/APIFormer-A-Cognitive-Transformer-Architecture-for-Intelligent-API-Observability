import os
import torch
import numpy as np
import json
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, confusion_matrix

from src.utils.logger import logger
from src.data.dataset import APIFormerDataset
from src.models.heads import APIFormerPlusModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running evaluation on device: {device}")
    
    # Paths
    sessions_path = "data/processed/encoded_sessions.pt"
    adj_path = "data/processed/gnn_adjacency.npy"
    vocab_path = "data/processed/tokenizer_vocab.json"
    checkpoint_path = "data/checkpoints/apiformer_plus.pt"
    
    if not os.path.exists(checkpoint_path):
        logger.error("Model checkpoint not found. Please run scripts/train.py first.")
        return
        
    # Load files
    encoded_sessions = torch.load(sessions_path, weights_only=False)
    adj_matrix_np = np.load(adj_path)
    
    with open(vocab_path, "r") as f:
        vocabs = json.load(f)
        
    # Validation split (using same seed as train.py to ensure correct split)
    total_samples = len(encoded_sessions)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    
    generator = torch.Generator().manual_seed(42)
    _, val_sessions = torch.utils.data.random_split(
        encoded_sessions, [train_size, val_size], generator=generator
    )
    
    val_dataset = APIFormerDataset(val_sessions, vocabs["endpoint"])
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    adj_matrix = torch.tensor(adj_matrix_np, dtype=torch.float32, device=device)
    num_nodes = adj_matrix.size(0)
    pad_idx = vocabs["endpoint"].get("[PAD]", 0)
    
    # Initialize model
    model = APIFormerPlusModel(
        vocabs=vocabs,
        d_embed=32,
        d_model=128,
        num_layers=4,
        num_heads=4,
        d_ff=256,
        gcn_num_nodes=num_nodes,
        num_intents=5,
        max_len=128,
        dropout=0.1
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Collections for metrics
    all_anomaly_preds = []
    all_anomaly_targets = []
    all_anomaly_probs = []
    
    all_intent_preds = []
    all_intent_targets = []
    
    all_bot_preds = []
    all_bot_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            outputs = model(batch, adj_matrix, pretrain=False)
            
            # 1. Anomaly metrics
            anom_logits = outputs["anomaly_logits"] # [B, L, 2]
            anom_labels = batch["anomaly_labels"]   # [B, L]
            padding_mask = batch["padding_mask"]   # [B, L]
            
            # Softmax to get probabilities for class 1 (anomaly)
            probs = torch.softmax(anom_logits, dim=-1)[..., 1] # [B, L]
            preds = torch.argmax(anom_logits, dim=-1) # [B, L]
            
            # Flatten, ignoring padded tokens
            valid_indices = ~padding_mask
            
            all_anomaly_preds.extend(preds[valid_indices].cpu().numpy().tolist())
            all_anomaly_targets.extend(anom_labels[valid_indices].cpu().numpy().tolist())
            all_anomaly_probs.extend(probs[valid_indices].cpu().numpy().tolist())
            
            # 2. Intent metrics (session-level)
            intent_logits = outputs["intent_logits"] # [B, NumIntents]
            intent_labels = batch["intent_label"]     # [B]
            intent_preds = torch.argmax(intent_logits, dim=-1)
            
            all_intent_preds.extend(intent_preds.cpu().numpy().tolist())
            all_intent_targets.extend(intent_labels.cpu().numpy().tolist())
            
            # 3. Bot metrics (session-level)
            bot_logits = outputs["bot_logits"] # [B, 2]
            bot_labels = batch["bot_label"]     # [B]
            bot_preds = torch.argmax(bot_logits, dim=-1)
            
            all_bot_preds.extend(bot_preds.cpu().numpy().tolist())
            all_bot_targets.extend(bot_labels.cpu().numpy().tolist())
            
    # Calculate Anomaly metrics
    anom_prec, anom_rec, anom_f1, _ = precision_recall_fscore_support(
        all_anomaly_targets, all_anomaly_preds, average='binary', zero_division=0
    )
    try:
        anom_auc = roc_auc_score(all_anomaly_targets, all_anomaly_probs)
    except Exception:
        anom_auc = 0.5
        
    # Calculate Intent metrics
    intent_acc = accuracy_score(all_intent_targets, all_intent_preds)
    intent_prec, intent_rec, intent_f1, _ = precision_recall_fscore_support(
        all_intent_targets, all_intent_preds, average='macro', zero_division=0
    )
    intent_cm = confusion_matrix(all_intent_targets, all_intent_preds).tolist()
    
    # Calculate Bot metrics
    bot_acc = accuracy_score(all_bot_targets, all_bot_preds)
    bot_prec, bot_rec, bot_f1, _ = precision_recall_fscore_support(
        all_bot_targets, all_bot_preds, average='binary', zero_division=0
    )
    
    metrics_results = {
        "anomaly": {
            "precision": float(anom_prec),
            "recall": float(anom_rec),
            "f1_score": float(anom_f1),
            "auc_roc": float(anom_auc)
        },
        "intent": {
            "accuracy": float(intent_acc),
            "precision": float(intent_prec),
            "recall": float(intent_rec),
            "f1_score": float(intent_f1),
            "confusion_matrix": intent_cm
        },
        "bot": {
            "accuracy": float(bot_acc),
            "precision": float(bot_prec),
            "recall": float(bot_rec),
            "f1_score": float(bot_f1)
        }
    }
    
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/evaluation_metrics.json", "w") as f:
        json.dump(metrics_results, f, indent=2)
        
    logger.info("Evaluation Complete. Metrics exported to data/processed/evaluation_metrics.json")
    print(json.dumps(metrics_results, indent=2))

if __name__ == "__main__":
    main()
