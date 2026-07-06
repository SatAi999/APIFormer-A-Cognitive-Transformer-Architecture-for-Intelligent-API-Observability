import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import json

from src.utils.logger import logger
from src.data.dataset import APIFormerDataset
from src.models.heads import APIFormerPlusModel

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def train_epoch_ssl(model, dataloader, optimizer, adj_matrix, device, pad_idx):
    model.train()
    total_loss = 0.0
    total_mam_loss = 0.0
    total_next_loss = 0.0
    total_lat_loss = 0.0
    total_stat_loss = 0.0
    total_cont_loss = 0.0
    
    # Loss functions
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    mse_loss_fn = nn.MSELoss(reduction='none') # Compute element-wise to ignore padded tokens
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch, adj_matrix, pretrain=True)
        
        # 1. MAM Loss (Ignore pad/non-masked positions via label=-100)
        mam_logits = outputs["mam_logits"] # [B, L, V_endpoint]
        mam_labels = batch["mam_labels"]   # [B, L]
        loss_mam = ce_loss_fn(mam_logits.view(-1, mam_logits.size(-1)), mam_labels.view(-1))
        
        # 2. Next API Prediction Loss
        next_logits = outputs["next_logits"]         # [B, L, V_endpoint]
        next_targets = batch["next_endpoint_targets"] # [B, L]
        # Ignore pad indices
        ce_loss_next_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
        loss_next = ce_loss_next_fn(next_logits.view(-1, next_logits.size(-1)), next_targets.view(-1))
        
        # 3. Latency Prediction Loss (Continuous MSE)
        latency_preds = outputs["latency_preds"] # [B, L]
        latency_targets = batch["latency"]       # [B, L]
        raw_mse = mse_loss_fn(latency_preds, latency_targets)
        # Apply padding mask: ignore padding elements
        padding_mask = batch["padding_mask"] # [B, L] (True for padding)
        valid_mse = raw_mse.masked_fill(padding_mask, 0.0)
        num_valid = (~padding_mask).sum().clamp(min=1.0)
        loss_latency = valid_mse.sum() / num_valid
        
        # 4. Status Code Prediction Loss
        status_logits = outputs["status_logits"] # [B, L, V_status]
        status_targets = batch["status"]         # [B, L]
        ce_loss_status_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
        loss_status = ce_loss_status_fn(status_logits.view(-1, status_logits.size(-1)), status_targets.view(-1))
        
        # 5. Contrastive Session Loss
        z1, z2 = outputs["z1"], outputs["z2"]
        loss_contrast = model.contrastive_loss_fn(z1, z2)
        
        # Combined Pretraining Loss
        loss = loss_mam + loss_next + 0.1 * loss_latency + loss_status + 0.5 * loss_contrast
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mam_loss += loss_mam.item()
        total_next_loss += loss_next.item()
        total_lat_loss += loss_latency.item()
        total_stat_loss += loss_status.item()
        total_cont_loss += loss_contrast.item()
        
        # --- Dynamically Update Memory Bank ---
        # Run evaluation pass to push fresh normal/anomaly sequences into cognitive engine's retrieval buffer
        with torch.no_grad():
            h_emb = model.embedding(batch)
            h_enc, _ = model.encoder(h_emb, padding_mask=batch["padding_mask"])
            z = model._pool_session(h_enc, batch["padding_mask"])
            model.cognitive_engine.memory_bank.update(z, h_enc)
            
    num_batches = len(dataloader)
    return {
        "loss": total_loss / num_batches,
        "loss_mam": total_mam_loss / num_batches,
        "loss_next": total_next_loss / num_batches,
        "loss_lat": total_lat_loss / num_batches,
        "loss_stat": total_stat_loss / num_batches,
        "loss_cont": total_cont_loss / num_batches
    }


def train_epoch_mtl(model, dataloader, optimizer, adj_matrix, device):
    model.train()
    total_loss = 0.0
    total_anom_loss = 0.0
    total_intent_loss = 0.0
    total_bot_loss = 0.0
    
    # Loss functions
    ce_loss_fn = nn.CrossEntropyLoss()
    
    for batch_idx, batch in enumerate(dataloader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # Forward pass in downstream fine-tuning mode
        outputs = model(batch, adj_matrix, pretrain=False)
        
        # 1. Step-level Anomaly Prediction Loss (Ignore padded slots)
        anom_logits = outputs["anomaly_logits"] # [B, L, 2]
        anom_labels = batch["anomaly_labels"]   # [B, L]
        # Ignore index for anomaly labels where padding occurs:
        # We can map padding positions to -100 in labels dynamically
        padding_mask = batch["padding_mask"]
        labels_masked = anom_labels.clone()
        labels_masked[padding_mask] = -100
        
        ce_loss_anom_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss_anom = ce_loss_anom_fn(anom_logits.view(-1, 2), labels_masked.view(-1))
        
        # 2. Session intent Classification Loss
        intent_logits = outputs["intent_logits"] # [B, NumIntents]
        intent_labels = batch["intent_label"]     # [B]
        loss_intent = ce_loss_fn(intent_logits, intent_labels)
        
        # 3. Session Bot Detection Loss
        bot_logits = outputs["bot_logits"] # [B, 2]
        bot_labels = batch["bot_label"]     # [B]
        loss_bot = ce_loss_fn(bot_logits, bot_labels)
        
        # Combined fine-tuning loss
        loss = loss_anom + loss_intent + loss_bot
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_anom_loss += loss_anom.item()
        total_intent_loss += loss_intent.item()
        total_bot_loss += loss_bot.item()
        
    num_batches = len(dataloader)
    return {
        "loss": total_loss / num_batches,
        "loss_anom": total_anom_loss / num_batches,
        "loss_intent": total_intent_loss / num_batches,
        "loss_bot": total_bot_loss / num_batches
    }


def evaluate(model, dataloader, adj_matrix, device, pad_idx, pretrain=True):
    model.eval()
    total_loss = 0.0
    
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    mse_loss_fn = nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            outputs = model(batch, adj_matrix, pretrain=pretrain)
            
            if pretrain:
                mam_logits = outputs["mam_logits"]
                mam_labels = batch["mam_labels"]
                loss_mam = ce_loss_fn(mam_logits.view(-1, mam_logits.size(-1)), mam_labels.view(-1))
                
                next_logits = outputs["next_logits"]
                next_targets = batch["next_endpoint_targets"]
                ce_loss_next_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
                loss_next = ce_loss_next_fn(next_logits.view(-1, next_logits.size(-1)), next_targets.view(-1))
                
                latency_preds = outputs["latency_preds"]
                latency_targets = batch["latency"]
                raw_mse = mse_loss_fn(latency_preds, latency_targets)
                padding_mask = batch["padding_mask"]
                valid_mse = raw_mse.masked_fill(padding_mask, 0.0)
                num_valid = (~padding_mask).sum().clamp(min=1.0)
                loss_latency = valid_mse.sum() / num_valid
                
                status_logits = outputs["status_logits"]
                status_targets = batch["status"]
                ce_loss_status_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
                loss_status = ce_loss_status_fn(status_logits.view(-1, status_logits.size(-1)), status_targets.view(-1))
                
                z1, z2 = outputs["z1"], outputs["z2"]
                loss_contrast = model.contrastive_loss_fn(z1, z2)
                
                loss = loss_mam + loss_next + 0.1 * loss_latency + loss_status + 0.5 * loss_contrast
            else:
                anom_logits = outputs["anomaly_logits"]
                anom_labels = batch["anomaly_labels"]
                padding_mask = batch["padding_mask"]
                labels_masked = anom_labels.clone()
                labels_masked[padding_mask] = -100
                ce_loss_anom_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss_anom = ce_loss_anom_fn(anom_logits.view(-1, 2), labels_masked.view(-1))
                
                intent_logits = outputs["intent_logits"]
                intent_labels = batch["intent_label"]
                loss_intent = ce_loss_fn(intent_logits, intent_labels)
                
                bot_logits = outputs["bot_logits"]
                bot_labels = batch["bot_label"]
                loss_bot = ce_loss_fn(bot_logits, bot_labels)
                
                loss = loss_anom + loss_intent + loss_bot
                
            total_loss += loss.item()
            
    return total_loss / len(dataloader)


def main():
    console = Console()
    console.print(Panel("[bold green]APIFormer+ Optimization & Training Engine[/bold green]", expand=False))
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: [bold yellow]{device}[/bold yellow]")
    
    # 1. Load Processed Dataset Tensors
    sessions_path = "data/processed/encoded_sessions.pt"
    adj_path = "data/processed/gnn_adjacency.npy"
    vocab_path = "data/processed/tokenizer_vocab.json"
    
    if not os.path.exists(sessions_path):
        logger.error("Processed sessions not found. Please run scripts/run_pipeline.py first.")
        return
        
    logger.info("Loading processed sessions and graph files...")
    encoded_sessions = torch.load(sessions_path, weights_only=False)
    adj_matrix_np = np.load(adj_path)
    
    with open(vocab_path, "r") as f:
        vocabs = json.load(f)
        
    adj_matrix = torch.tensor(adj_matrix_np, dtype=torch.float32, device=device)
    num_nodes = adj_matrix.size(0)
    pad_idx = vocabs["endpoint"].get("[PAD]", 0)
    
    # 2. Divide Dataset into Train/Val sets (80/20)
    total_samples = len(encoded_sessions)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    
    # Seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_sessions, val_sessions = random_split(
        encoded_sessions, [train_size, val_size], generator=generator
    )
    
    logger.info(f"Dataset split: [green]Train={len(train_sessions)}[/green] | [cyan]Val={len(val_sessions)}[/cyan]")
    
    train_dataset = APIFormerDataset(train_sessions, vocabs["endpoint"])
    val_dataset = APIFormerDataset(val_sessions, vocabs["endpoint"])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 3. Instantiate Foundation Model Architecture
    logger.info("Initializing APIFormerPlusModel...")
    model = APIFormerPlusModel(
        vocabs=vocabs,
        d_embed=32,
        d_model=128,
        num_layers=4,
        num_heads=4,
        d_ff=256,
        gcn_num_nodes=num_nodes,
        num_intents=5, # normal, 503_cascade, credential_stuffing, bot_scraping, sequence_abuse
        max_len=128,
        dropout=0.1
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    # 4. Phase 1: Self-Supervised Pretraining Epochs
    ssl_epochs = 5
    logger.info(f"Starting Phase 1: Self-Supervised Pretraining ({ssl_epochs} epochs)...")
    
    table_ssl = Table(title="Phase 1: Self-Supervised Pretraining Metrics")
    table_ssl.add_column("Epoch", style="cyan")
    table_ssl.add_column("Train Loss", style="green")
    table_ssl.add_column("Val Loss", style="green")
    table_ssl.add_column("MAM Loss", style="yellow")
    table_ssl.add_column("Next Loss", style="yellow")
    table_ssl.add_column("Latency Loss (MSE)", style="magenta")
    table_ssl.add_column("Contrastive Loss", style="magenta")
    
    for epoch in range(1, ssl_epochs + 1):
        metrics = train_epoch_ssl(model, train_loader, optimizer, adj_matrix, device, pad_idx)
        val_loss = evaluate(model, val_loader, adj_matrix, device, pad_idx, pretrain=True)
        
        table_ssl.add_row(
            str(epoch),
            f"{metrics['loss']:.4f}",
            f"{val_loss:.4f}",
            f"{metrics['loss_mam']:.4f}",
            f"{metrics['loss_next']:.4f}",
            f"{metrics['loss_lat']:.4f}",
            f"{metrics['loss_cont']:.4f}"
        )
    console.print(table_ssl)
    
    # 5. Phase 2: Downstream Multi-Task Fine-Tuning Epochs
    mtl_epochs = 3
    logger.info(f"Starting Phase 2: Downstream Multi-Task Fine-Tuning ({mtl_epochs} epochs)...")
    
    table_mtl = Table(title="Phase 2: Downstream Multi-Task Fine-Tuning Metrics")
    table_mtl.add_column("Epoch", style="cyan")
    table_mtl.add_column("Train Loss", style="green")
    table_mtl.add_column("Val Loss", style="green")
    table_mtl.add_column("Anomaly Loss", style="yellow")
    table_mtl.add_column("Intent Loss", style="yellow")
    table_mtl.add_column("Bot Loss", style="magenta")
    
    for epoch in range(1, mtl_epochs + 1):
        metrics = train_epoch_mtl(model, train_loader, optimizer, adj_matrix, device)
        val_loss = evaluate(model, val_loader, adj_matrix, device, pad_idx, pretrain=False)
        
        table_mtl.add_row(
            str(epoch),
            f"{metrics['loss']:.4f}",
            f"{val_loss:.4f}",
            f"{metrics['loss_anom']:.4f}",
            f"{metrics['loss_intent']:.4f}",
            f"{metrics['loss_bot']:.4f}"
        )
    console.print(table_mtl)
    
    # 6. Checkpoint Model Weights
    checkpoint_dir = "data/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "apiformer_plus.pt")
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocabs": vocabs
    }, checkpoint_path)
    
    console.print(Panel.fit(f"[bold green]APIFormer+ Optimization Complete. Checkpoint saved to {checkpoint_path}[/bold green]"))

if __name__ == "__main__":
    main()
