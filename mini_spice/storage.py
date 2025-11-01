"""Storage utilities for logging."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


def append_jsonl(filepath: str, data: Dict[str, Any]) -> None:
    """Append JSON object to JSONL file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    
    with open(filepath, "a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")


def log_episode(
    log_file: str,
    step: int,
    role: str,
    doc_id: Optional[str] = None,
    valid: Optional[bool] = None,
    ans_type: Optional[str] = None,
    p_pass: Optional[float] = None,
    rC: Optional[float] = None,
    rR_mean: Optional[float] = None,
    loss: Optional[float] = None,
    q: Optional[str] = None,
    a_star: Optional[str] = None,
    preds: Optional[list[str]] = None,
    seed: Optional[int] = None,
    run_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    **kwargs
) -> None:
    """Log training episode to JSONL file."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    episode = {
        "step": step,
        "timestamp": timestamp,
        "role": role,
        "doc_id": doc_id,
        "valid": valid,
        "type": ans_type,
        "p_pass": p_pass,
        "rC": rC,
        "rR": rR_mean,
        "rR_mean": rR_mean,
        "loss": loss,
        "q": q,
        "a_star": a_star,
        "pred": preds[0] if preds and len(preds) > 0 else None,
        "preds": preds,
        "seed": seed,
        "run_id": run_id,
        **kwargs
    }
    
    append_jsonl(log_file, episode)


def export_to_parquet(jsonl_file: str, parquet_file: str) -> None:
    """Convert JSONL to Parquet format."""
    # Read JSONL
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save as Parquet
    os.makedirs(os.path.dirname(parquet_file) if os.path.dirname(parquet_file) else ".", exist_ok=True)
    df.to_parquet(parquet_file, index=False)


def create_run_log(runs_dir: str, run_id: Optional[str] = None) -> str:
    """Create new run log file path."""
    if run_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}"
    
    os.makedirs(runs_dir, exist_ok=True)
    log_file = os.path.join(runs_dir, f"{run_id}.jsonl")
    return log_file


def save_checkpoint_manifest(
    checkpoint_dir: str,
    step: int,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    corpus_hash: Optional[str] = None,
    git_commit: Optional[str] = None
) -> None:
    """Save checkpoint manifest with metadata."""
    manifest = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": metrics,
        "corpus_hash": corpus_hash,
        "git_commit": git_commit
    }
    
    manifest_path = os.path.join(checkpoint_dir, "manifest.json")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

