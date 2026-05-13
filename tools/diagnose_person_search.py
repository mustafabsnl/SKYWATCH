import os
import sys
import argparse
import numpy as np
from pathlib import Path
import json

# Fix import paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config import AppConfig
from src.database.db import Database
from src.core.face_analyzer import FaceAnalyzer
from src.utils.logger import EventLogger


def check_database(db: Database, target_id: int):
    print(f"\n--- Checking Target ID: {target_id} ---")
    
    # 1. Check raw database row
    with db._get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT c.id, c.name, c.status, e.embedding FROM criminals c LEFT JOIN embeddings e ON e.criminal_id = c.id WHERE c.id = ?", (target_id,))
        row = cursor.fetchone()
        
    if not row:
        print(f"FAILED: Target ID {target_id} not found in 'criminals' table.")
        return None
        
    print(f"OK: Target found: {row['name']} (Status: {row['status']})")
    
    if not row['embedding']:
        print("FAILED: Target has no embedding record in database.")
        return None
        
    raw_emb = row['embedding']
    print(f"OK: Raw embedding found in DB. Type: {type(raw_emb)}")
    
    # 2. Check coercion logic
    try:
        from src.database.db import _coerce_embedding
        emb = _coerce_embedding(raw_emb, person_id=target_id)
        if emb is None:
            print("FAILED: Embedding coercion returned None.")
            return None
            
        print(f"OK: Coercion successful. Shape: {emb.shape}, Dtype: {emb.dtype}")
        norm = np.linalg.norm(emb)
        print(f"OK: Embedding Norm: {norm:.4f}")
        
        if not np.isclose(norm, 1.0, atol=1e-4):
            print(f"WARN: Warning: Embedding is not L2-normalized! (Norm: {norm})")
            
        return emb
        
    except Exception as e:
        print(f"FAILED: Error during embedding coercion: {e}")
        return None


def run_diagnostics(target_id: int):
    print(f"=== SKYWATCH Person Search Diagnostics ===")
    print(f"Project Root: {PROJECT_ROOT}")
    
    # Load config
    cfg = AppConfig()
    print("OK: Configuration loaded.")
    
    ps_cfg = cfg.get("face", {}).get("person_search", {})
    threshold = float(ps_cfg.get("cosine_threshold", 0.50))
    print(f"INFO: Configured Person Search Threshold: {threshold}")
    
    logger = EventLogger(cfg)
    db = Database(cfg, logger)
    analyzer = FaceAnalyzer(cfg)
    
    # Check Database and Embedding
    target_emb = check_database(db, target_id)
    
    if target_emb is None:
        print("\nFAILED: Diagnostics failed at database/embedding validation step.")
        return
        
    # Analyze similarities
    print("\n--- Running Similarity Tests ---")
    
    with db._get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT c.id, c.name FROM criminals c JOIN embeddings e ON e.criminal_id = c.id")
        others = cursor.fetchall()
        
    print(f"Found {len(others)} total people with embeddings in DB.")
    
    matches = []
    for other in others:
        other_id = other['id']
        other_name = other['name']
        if other_id == target_id:
            continue
            
        info = db.get_person_embedding_for_search(other_id)
        if not info or info.get("embedding") is None:
            continue
            
        other_emb = info["embedding"]
        score = analyzer.compare(target_emb, other_emb)
        
        matches.append((other_id, other_name, float(score)))
        
    # Sort matches by score descending
    matches.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop 5 Similar Identities in DB:")
    for m in matches[:5]:
        score_marker = "MATCH" if m[2] >= threshold else "CLEAN"
        print(f" - {m[1]} (ID: {m[0]}): Score = {m[2]:.4f} [{score_marker}]")
        
    print("\n--- Pipeline Emulation ---")
    info = db.get_person_embedding_for_search(target_id)
    if info and info.get("embedding") is not None:
        print("OK: pipeline.set_mode('PERSON_SEARCH') target load works.")
    else:
        print("FAILED: db.get_person_embedding_for_search() failed during emulation.")

    print("\n=== Diagnostics Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose Person Search Mode")
    parser.add_argument("target_id", type=int, help="Target Person ID to diagnose")
    args = parser.parse_args()
    
    run_diagnostics(args.target_id)
