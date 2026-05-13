"""
SKYWATCH — Verify GENERAL multi-person matching (no single-ID lock).

Loads all DB embedding rows as candidates (same shape as Pipeline),
aggregates scores per person_id, ensures two different people can be
matched independently when their own embeddings are used as live probes.

Also detects duplicate embeddings (same hash across different person_ids).
"""
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config import AppConfig
from src.database.db import Database
from src.core.face_analyzer import FaceAnalyzer
from src.utils.logger import EventLogger


def _best_per_person_scores(live: np.ndarray, candidates: list[dict], face: FaceAnalyzer):
    best_by_pid: dict[int, float] = {}
    for c in candidates:
        pid = int(c["person_id"])
        emb = c["embedding"]
        score = float(face.compare(live, emb))
        if score > best_by_pid.get(pid, -1.0):
            best_by_pid[pid] = score
    ranked = sorted(best_by_pid.items(), key=lambda x: x[1], reverse=True)
    return best_by_pid, ranked


def main():
    config = AppConfig()
    logger = EventLogger(config)
    db = Database(config, logger)
    face = FaceAnalyzer(config)

    cands = db.get_all_person_embeddings_for_general()
    print("=" * 70)
    print("SKYWATCH — GENERAL multi-match diagnostic (hash-aware)")
    print("=" * 70)
    print(f"\nCandidate rows: {len(cands)}")
    pids = sorted({int(c["person_id"]) for c in cands})
    print(f"Unique person_id values: {len(pids)} -> {pids}")

    # Build hash map
    hash_to_pids: dict[str, list[int]] = defaultdict(list)
    pid_to_hash: dict[int, str] = {}
    for c in cands:
        h = c.get("embedding_hash", "")
        pid = int(c["person_id"])
        if h:
            if pid not in hash_to_pids[h]:
                hash_to_pids[h].append(pid)
            pid_to_hash[pid] = h
            print(f"  person_id={pid:3d} hash={h} name={c.get('name', '')[:20]:20s} status={c.get('status', '')}")

    dup_groups = {h: pids for h, pids in hash_to_pids.items() if len(pids) > 1}
    print(f"\nUnique hashes: {len(hash_to_pids)}")
    if dup_groups:
        print(f"\n!!! DUPLICATE EMBEDDING GROUPS FOUND !!!")
        for h, dpids in dup_groups.items():
            print(f"  hash={h} person_ids={dpids}")
        print(
            "\nNOTE: These persons share identical embeddings.\n"
            "GENERAL will use status_priority resolver (WANTED > CRIMINAL > CLEAN, then smallest ID)."
        )

    if len(pids) < 2:
        print("\nSKIP: Need at least 2 distinct person_id rows in DB to prove multi-match.")
        return

    face_cfg = config.get("face", {}) if hasattr(config, "get") else {}
    g = face_cfg.get("general") if isinstance(face_cfg.get("general"), dict) else {}
    thr = float(g.get("cosine_threshold", face_cfg.get("general_match_threshold", 0.55)))

    # Pick two persons with DIFFERENT hashes if possible
    unique_hash_pids = [pid for pid in pids if len(hash_to_pids.get(pid_to_hash.get(pid, ""), [])) == 1]
    if len(unique_hash_pids) >= 2:
        a, b = unique_hash_pids[0], unique_hash_pids[1]
        print(f"\nUsing two persons with UNIQUE hashes: {a}, {b}")
    else:
        a, b = pids[0], pids[1]
        print(f"\nUsing first two persons (may share hash): {a}, {b}")

    emb_a = next(c["embedding"] for c in cands if int(c["person_id"]) == a)
    emb_b = next(c["embedding"] for c in cands if int(c["person_id"]) == b)

    _, r1 = _best_per_person_scores(emb_a, cands, face)
    _, r2 = _best_per_person_scores(emb_b, cands, face)

    top1 = r1[0][0] if r1 else None
    top2 = r2[0][0] if r2 else None
    s1 = r1[0][1] if r1 else 0.0
    s2 = r2[0][1] if r2 else 0.0

    print(f"\nProbe embedding of person_id={a} -> best person_id={top1} score={s1:.4f} (threshold={thr})")
    print(f"Probe embedding of person_id={b} -> best person_id={top2} score={s2:.4f} (threshold={thr})")

    # Check if they share hash
    hash_a = pid_to_hash.get(a, "")
    hash_b = pid_to_hash.get(b, "")
    same_hash = bool(hash_a and hash_a == hash_b)

    if same_hash:
        print(f"\nNOTE: person_id={a} and {b} share the SAME hash ({hash_a}).")
        print("They cannot be distinguished mathematically; GENERAL will use resolver.")
        print("This is expected behavior for duplicate embeddings.")
    else:
        ok_a = top1 == a and s1 >= thr
        ok_b = top2 == b and s2 >= thr
        if ok_a and ok_b:
            print("\nPASS: Two different persons each resolve to their own ID (no global single-target lock).")
        else:
            print("\nWARN: At least one probe did not self-match at threshold; check thresholds or embeddings.")


if __name__ == "__main__":
    main()
