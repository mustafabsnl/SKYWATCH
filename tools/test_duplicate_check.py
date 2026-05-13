import sys
import os
from pathlib import Path
import numpy as np

# Proje ana dizinini path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import AppConfig
from src.utils.logger import EventLogger
from src.database.db import Database

def main():
    print("Test: Duplicate Check")
    cfg = AppConfig()
    logger = EventLogger(cfg)
    db = Database(cfg, logger)
    
    # 1. En son eklenen kisinin embedding'ini alalim
    conn = db._get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT criminal_id, embedding FROM embeddings ORDER BY rowid DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        print("[TEST_DUP] Veritabaninda hic embedding yok!")
        return
        
    last_id = row[0]
    db_emb_raw = row[1]
    
    # Raw embedding'i numpy array'e donustur
    from src.database.db import _coerce_embedding
    last_emb = _coerce_embedding(db_emb_raw, logger=logger, person_id=last_id)
    
    if last_emb is None:
        print(f"[TEST_DUP] Veritabanindan alinan embedding parse edilemedi (ID: {last_id})")
        return
        
    print(f"[TEST_DUP] loaded person_id={last_id} shape={last_emb.shape}")
    
    # 2. Ayni embedding'i find_duplicate_person ile aratalim
    threshold = 0.62
    uncertain_low = 0.52
    
    result = db.find_duplicate_person(last_emb, threshold, uncertain_low)
    
    print(f"[TEST_DUP] result={result}")
    if result is not None and result["level"] == "duplicate" and result["similarity"] > 0.98:
        print("[TEST_DUP] PASS")
    else:
        print("[TEST_DUP] FAIL - Aynisi olmasina ragmen kesin duplicate dönmedi!")
        
    # 3. Rastgele (farkli) bir embedding olusturalim ve test edelim
    random_emb = np.random.randn(512).astype(np.float32)
    random_emb = random_emb / np.linalg.norm(random_emb)
    
    rand_result = db.find_duplicate_person(random_emb, threshold, uncertain_low)
    
    print(f"[TEST_DUP_RANDOM] result={rand_result}")
    if rand_result is None or rand_result["level"] == "no_match":
         print("[TEST_DUP_RANDOM] PASS")
    else:
         print("[TEST_DUP_RANDOM] FAIL - Rastgele vektor gercek biriyle eslesti (cok dusuk ihtimal, esik deger hatali olabilir!)")

if __name__ == "__main__":
    main()
