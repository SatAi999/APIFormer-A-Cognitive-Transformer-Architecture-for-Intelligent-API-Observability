import os
import yaml
import pandas as pd
import json
import torch
import numpy as np
from typing import Dict, Any
from dataclasses import asdict

from src.utils.logger import logger
from src.data.synthetic_gen import SyntheticAPIGenerator
from src.data.inspector import DatasetInspector
from src.data.normalizer import SchemaNormalizer
from src.data.session_builder import SessionBuilder
from src.data.tokenizer import APITokenizer
from src.data.graph_builder import ServiceDependencyGraph
from src.data.dataset import APIFormerDataset

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    console = Console()
    console.print(Panel("[bold green]APIFormer+ Data Pipeline Orchestrator[/bold green]", expand=False))
    
    # 1. Load Configurations
    config_path = "config/pipeline_config.yaml"
    logger.info(f"Loading configuration from: {config_path}")
    config = load_yaml_config(config_path)
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # 2. Generate Synthetic Data
    logger.info("Initializing Synthetic Traffic Generator...")
    generator = SyntheticAPIGenerator(config)
    raw_events = generator.generate_dataset()
    logger.info(f"Generated {len(raw_events)} events.")
    
    # Save raw events to JSON Lines to simulate a production telemetry log file
    raw_file_path = "data/raw/telemetry_logs.json"
    raw_dicts = [e.__dict__ if hasattr(e, "__dict__") else e for e in raw_events]
    
    # Introduce some noise and rename columns to test the DatasetInspector
    df_raw = pd.DataFrame(raw_dicts)
    
    # Rename some columns to simulate real-world schema differences
    col_rename = {
        "timestamp": "ts",
        "session_id": "sessionID",
        "trace_id": "trace",
        "endpoint": "url_path",
        "http_method": "method",
        "status_code": "status",
        "latency_ms": "duration",
        "payload_size_bytes": "bytes_size"
    }
    df_raw = df_raw.rename(columns=col_rename)
    
    # Save raw data to file
    df_raw.to_json(raw_file_path, orient="records", lines=True)
    logger.info(f"Raw data with custom schemas exported to: {raw_file_path}")
    
    # 3. Automatically Inspect the Dataset
    inspector = DatasetInspector()
    inspection_results = inspector.inspect(raw_file_path)
    
    # Display mapped columns in a table
    table = Table(title="Column Mapping Discovery Results")
    table.add_column("Source Log Column", style="cyan")
    table.add_column("Inferred Standard Field", style="green")
    
    for src_col, std_field in inspection_results["schema_mapping"].items():
        table.add_row(src_col, std_field)
    console.print(table)
    
    # 4. Normalize the Dataset
    normalizer = SchemaNormalizer(inspection_results)
    normalized_events = normalizer.normalize_dataframe(df_raw)
    
    # 5. Reconstruct Sessions
    session_builder = SessionBuilder(config)
    sessions = session_builder.reconstruct_sessions(normalized_events)
    
    # 6. Tokenize API Dimensions
    tokenizer = APITokenizer(config)
    tokenizer.fit(sessions)
    
    vocab_path = "data/processed/tokenizer_vocab.json"
    tokenizer.save(vocab_path)
    
    # Encode all sessions
    encoded_sessions = [tokenizer.encode_session(sess) for sess in sessions.values()]
    logger.info(f"Successfully encoded {len(encoded_sessions)} session sequences.")
    
    # 7. Construct Service Dependency Graph
    graph_builder = ServiceDependencyGraph()
    graph_builder.build_graphs(normalized_events)
    metrics = graph_builder.compute_graph_metrics()
    
    # Top metrics display
    table_metrics = Table(title="Microservice Endpoint Topological Importance")
    table_metrics.add_column("Endpoint Node", style="magenta")
    table_metrics.add_column("PageRank Score", style="green")
    table_metrics.add_column("In-Degree / Fan-In", style="cyan")
    table_metrics.add_column("Out-Degree / Fan-Out", style="cyan")
    
    # Sort by PageRank and take top 5
    top_nodes = sorted(metrics.items(), key=lambda x: x[1]["pagerank"], reverse=True)[:5]
    for node, data in top_nodes:
        table_metrics.add_row(
            node, 
            f"{data['pagerank']:.4f}", 
            f"{data['in_degree']:.0f}", 
            f"{data['out_degree']:.0f}"
        )
    console.print(table_metrics)
    
    # Get GNN adjacency matrix
    adj_matrix, edge_list = graph_builder.get_gnn_structures(tokenizer.vocabs["endpoint"])
    logger.info(f"GNN Adjacency Matrix compiled. Shape: {adj_matrix.shape}. Non-zero edges: {len(edge_list)}")
    
    # Save Graph metrics & Adjacency
    np.save("data/processed/gnn_adjacency.npy", adj_matrix)
    with open("data/processed/graph_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Graph structures exported to data/processed/")
    
    # 8. Create PyTorch Dataset
    pytorch_dataset = APIFormerDataset(encoded_sessions, tokenizer.vocabs["endpoint"])
    logger.info(f"PyTorch Dataset instantiated. Size: {len(pytorch_dataset)} samples.")
    
    # Verify a single sample
    sample = pytorch_dataset[0]
    
    # Display sample structure
    table_sample = Table(title="PyTorch Batch Tensor Verification")
    table_sample.add_column("Tensor Feature", style="yellow")
    table_sample.add_column("Shape", style="green")
    table_sample.add_column("Data Type", style="cyan")
    
    for key, value in sample.items():
        table_sample.add_row(key, str(list(value.shape)), str(value.dtype))
    console.print(table_sample)
    
    # Save the PyTorch Dataset as a serialized file
    torch.save(encoded_sessions, "data/processed/encoded_sessions.pt")
    logger.info("Pytorch encoded sessions saved to data/processed/encoded_sessions.pt")
    
    # Export formats
    output_cfg = config.get("output", {})
    export_dir = output_cfg.get("export_dir", "./data/processed")
    formats = output_cfg.get("formats", ["json"])
    
    os.makedirs(export_dir, exist_ok=True)
    
    if "json" in formats:
        # Save as JSON
        json_sessions = []
        for s_id, s_events in sessions.items():
            json_sessions.append({
                "session_id": s_id,
                "events": [asdict(e) for e in s_events]
            })
        with open(os.path.join(export_dir, "sessions.json"), "w") as f:
            json.dump(json_sessions, f, indent=2)
            
    if "csv" in formats:
        # Flatten sessions to single CSV
        flat_events = []
        for s_events in sessions.values():
            for e in s_events:
                flat_events.append(asdict(e))
        pd.DataFrame(flat_events).to_csv(os.path.join(export_dir, "sessions.csv"), index=False)
        
    if "parquet" in formats:
        flat_events = []
        for s_events in sessions.values():
            for e in s_events:
                flat_events.append(asdict(e))
        pd.DataFrame(flat_events).to_parquet(os.path.join(export_dir, "sessions.parquet"), index=False)
        
    console.print(Panel.fit("[bold green]APIFormer+ Unified Data Pipeline Executed Successfully![/bold green]"))

if __name__ == "__main__":
    main()
