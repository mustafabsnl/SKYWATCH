"""
SKYWATCH — DB Embedding Duplicate Diagnostic

Bu script veritabanındaki tüm kişilerin embedding'lerini analiz eder:
- Her embedding için hash hesaplar
- Aynı hash'e sahip farklı person_id'leri bulur
- Duplicate grupları raporlar

Kullanım:
    python tools/diagnose_db_embedding_duplicates.py
"""
import sys
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config import AppConfig
from src.database.db import Database, _coerce_embedding
from src.utils.logger import EventLogger


def main():
    config = AppConfig()
    logger = EventLogger(config)
    db = Database(config, logger)

    print("=" * 70)
    print("SKYWATCH — DB Embedding Duplicate Diagnostic")
    print("=" * 70)

    # DB'den tüm embedding satırlarını çek
    with db._get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT c.id AS person_id, c.name, c.status, e.embedding
            FROM embeddings e
            JOIN criminals c ON e.criminal_id = c.id
            ORDER BY c.id
            """
        )
        rows = cursor.fetchall()

    print(f"\nToplam embedding satırı: {len(rows)}")
    print("-" * 70)

    hash_to_records: dict[str, list[dict]] = defaultdict(list)
    all_records: list[dict] = []

    for row in rows:
        pid = int(row["person_id"])
        name = row["name"] or ""
        status = row["status"] or ""
        raw_emb = row["embedding"]

        emb = _coerce_embedding(raw_emb, logger=None, person_id=pid)
        if emb is None:
            print(f"[DB_EMB_SKIP] id={pid} name={name} reason=invalid_embedding")
            continue

        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb))
        emb_hash = hashlib.sha1(emb.tobytes()).hexdigest()[:12]

        rec = {
            "person_id": pid,
            "name": name,
            "status": status,
            "shape": emb.shape,
            "norm": norm,
            "hash": emb_hash,
        }
        all_records.append(rec)
        hash_to_records[emb_hash].append(rec)

        print(
            f"[DB_EMB] id={pid:3d} name={name:20s} status={status:10s} "
            f"shape={emb.shape} norm={norm:.4f} hash={emb_hash}"
        )

    print("-" * 70)
    print(f"\nGeçerli embedding sayısı: {len(all_records)}")
    print(f"Unique hash sayısı: {len(hash_to_records)}")

    # Duplicate grupları bul
    dup_groups = {h: recs for h, recs in hash_to_records.items() if len(recs) > 1}

    if dup_groups:
        print("\n" + "=" * 70)
        print("!!! DUPLICATE EMBEDDING GRUPLARI BULUNDU !!!")
        print("=" * 70)

        for h, recs in dup_groups.items():
            pids = [r["person_id"] for r in recs]
            names = [r["name"] for r in recs]
            statuses = [r["status"] for r in recs]
            print(
                f"\n[DB_DUPLICATE_EMBEDDING_GROUP] hash={h}\n"
                f"    person_ids = {pids}\n"
                f"    names      = {names}\n"
                f"    statuses   = {statuses}"
            )

        print("\n" + "-" * 70)
        print(
            f"[DB_DUPLICATE_EMBEDDING_FOUND] groups={len(dup_groups)} "
            f"total_affected_persons={sum(len(recs) for recs in dup_groups.values())}"
        )
        print(
            "\nUYARI: Bu kişiler aynı embedding'e sahip olduğu için\n"
            "GENERAL modda matematiksel olarak birbirinden ayırt edilemez.\n"
            "Sistem status_priority (WANTED > CRIMINAL > CLEAN) ve\n"
            "en küçük person_id ile deterministik seçim yapacaktır."
        )
    else:
        print("\n" + "=" * 70)
        print("Duplicate embedding bulunamadı. Tüm kişiler benzersiz embedding'e sahip.")
        print("=" * 70)

    # Özet
    print("\n" + "=" * 70)
    print("ÖZET")
    print("=" * 70)
    print(f"  Toplam satır        : {len(rows)}")
    print(f"  Geçerli embedding   : {len(all_records)}")
    print(f"  Unique hash         : {len(hash_to_records)}")
    print(f"  Duplicate grup      : {len(dup_groups)}")
    if dup_groups:
        print(f"  Etkilenen kişi      : {sum(len(recs) for recs in dup_groups.values())}")


if __name__ == "__main__":
    main()
