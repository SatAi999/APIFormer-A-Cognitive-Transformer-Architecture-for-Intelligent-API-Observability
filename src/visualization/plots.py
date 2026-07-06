import os
import matplotlib
matplotlib.use('Agg') # Non-interactive backend suitable for server scripts
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List

from src.utils.logger import logger

def plot_attention_heatmap(attn_matrix: np.ndarray, 
                           token_labels: List[str], 
                           save_path: str = "data/processed/plots/attention_heatmap.png") -> str:
    """Plots a self-attention weights matrix as a heatmap and saves it as a PNG file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    seq_len = len(token_labels)
    # Clip attention matrix to match sequence length
    matrix = attn_matrix[:seq_len, :seq_len]
    
    plt.figure(figsize=(10, 8))
    
    # Custom color palette (sleek deep blue/purple mix)
    cmap = sns.cubehelix_palette(as_cmap=True, dark=0.08, light=0.95, reverse=False)
    
    # Plot heatmap
    ax = sns.heatmap(
        matrix, 
        xticklabels=token_labels, 
        yticklabels=token_labels, 
        cmap=cmap, 
        annot=True, 
        fmt=".2f",
        annot_kws={"size": 8},
        square=True,
        cbar_kws={"label": "Attention Weight"}
    )
    
    plt.title("APIFormer+ Self-Attention Distribution Map", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Key Tokens", fontsize=11, labelpad=10)
    plt.ylabel("Query Tokens", fontsize=11, labelpad=10)
    
    # Rotate labels to avoid overlapping
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"Attention heatmap exported to: {save_path}")
    return save_path

def plot_latency_timeline(latencies: List[float], 
                          anomaly_labels: List[int],
                          token_labels: List[str],
                          save_path: str = "data/processed/plots/latency_timeline.png") -> str:
    """Plots request sequence latencies, highlighting anomalous steps."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    steps = np.arange(len(latencies))
    
    plt.figure(figsize=(10, 5))
    
    # Plot line
    plt.plot(steps, latencies, color='#3498db', marker='o', linestyle='-', linewidth=2, label="Latency (ms)")
    
    # Highlight anomalies
    anom_indices = [i for i, x in enumerate(anomaly_labels) if x == 1]
    anom_lats = [latencies[i] for i in anom_indices]
    
    if anom_indices:
        plt.scatter(anom_indices, anom_lats, color='#e74c3c', s=100, zorder=5, label="Anomalous Event")
        
    plt.title("Session API Latency Timeline", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Request Steps", fontsize=10)
    plt.ylabel("Latency (ms)", fontsize=10)
    
    # Set step labels
    plt.xticks(steps, [f"{i}: {labels[:15]}..." for i, labels in enumerate(token_labels)], rotation=30, ha='right', fontsize=8)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"Latency timeline exported to: {save_path}")
    return save_path
