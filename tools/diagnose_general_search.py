"""
SKYWATCH — GENERAL Mode Diagnostic Tool
Tests the full GENERAL mode pipeline path:
  DB cache → embedding comparison → MatchResult → criminal_info → DecisionEngine
"""
import os
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config import AppConfig
from src.database.db import Database
from src.utils.logger import EventLogger
from src.engine.decision import DecisionEngine
from src.core.models import Track, MatchResult


def main():
    config = AppConfig()
    logger = EventLogger(config)
    db = Database(config, logger)

    print("=" * 60)
    print("SKYWATCH — GENERAL Mode Diagnostic")
    print("=" * 60)

    # 1. DB Cache check
    all_embs = db.get_all_embeddings()
    print(f"\n[1] DB Cache: {len(all_embs)} valid embeddings loaded")
    if not all_embs:
        print("    FAIL: No embeddings in database. Add persons with face photos first.")
        return

    for pid, emb in all_embs:
        info = db.get_criminal_info(pid)
        name = info.get("name", "?") if info else "?"
        status = info.get("status", "?") if info else "?"
        print(f"    person_id={pid} name={name} status={status} "
              f"emb_shape={emb.shape} norm={np.linalg.norm(emb):.3f}")

    # 2. Pick first person as test subject
    test_pid, test_emb = all_embs[0]
    test_info = db.get_criminal_info(test_pid)
    print(f"\n[2] Test subject: person_id={test_pid} name={test_info.get('name', '?')}")

    # 3. Simulate GENERAL search — compare test_emb against all DB embeddings
    face_cfg = config.get("face", {}) if hasattr(config, "get") else {}
    threshold = float(face_cfg.get("general_match_threshold", 0.55))
    margin = float(face_cfg.get("general_match_margin", 0.05))
    print(f"\n[3] GENERAL Search Config: threshold={threshold}, margin={margin}")

    best_cid = None
    best_score = 0.0
    second_score = 0.0
    for cid, db_emb in all_embs:
        score = float(np.dot(test_emb, db_emb))
        if score > best_score:
            second_score = best_score
            best_score = score
            best_cid = cid
        elif score > second_score:
            second_score = score

    print(f"    best_id={best_cid} best_score={best_score:.4f} "
          f"second_score={second_score:.4f}")

    if best_cid is None or best_score < threshold:
        print(f"    FAIL: No match above threshold ({threshold})")
    elif len(all_embs) > 1 and (best_score - second_score) < margin:
        print(f"    FAIL: Ambiguous margin ({best_score - second_score:.4f} < {margin})")
    else:
        print(f"    PASS: Match found — person_id={best_cid} score={best_score:.4f}")

    # 4. Criminal info retrieval
    criminal_info = db.get_criminal_info(best_cid) if best_cid else None
    print(f"\n[4] criminal_info: {criminal_info is not None}")
    if criminal_info:
        print(f"    name={criminal_info.get('name')} status={criminal_info.get('status')}")
    else:
        print("    FAIL: get_criminal_info returned None")
        return

    # 5. DecisionEngine evaluation
    decision_engine = DecisionEngine(logger=logger)
    decision_engine.set_mode("GENERAL", {})

    mock_track = Track(
        track_id=999,
        bbox=[100, 100, 200, 200],
        is_confirmed=True,
        age=10,
        source="deepsort",
    )
    mock_track.velocity_ok = True
    mock_track.face_embedding = test_emb
    mock_track.criminal_match = MatchResult(criminal_id=best_cid, confidence=best_score)
    mock_track.global_id = "TEST-T999"

    result = decision_engine.evaluate(mock_track, criminal_info)
    print(f"\n[5] DecisionEngine result:")
    print(f"    status={result.status} criminal_id={result.criminal_id} "
          f"confidence={result.confidence:.4f} danger_level={result.danger_level}")

    db_status = criminal_info.get("status", "").upper()
    expected_map = {"WANTED": "WANTED", "CRIMINAL": "CRIMINAL", "CLEARED": "CLEAN", "CLEAN": "CLEAN"}
    expected = expected_map.get(db_status, "UNKNOWN")
    if result.status == expected:
        print(f"    PASS: Expected '{expected}', got '{result.status}'")
    else:
        print(f"    FAIL: Expected '{expected}', got '{result.status}'")

    # 6. Alert eligibility
    alert_statuses = ("WANTED", "CRIMINAL", "HEDEF BULUNDU", "TARGET_FOUND")
    if result.status in alert_statuses:
        print(f"\n[6] Alert: YES — {result.status} would trigger dashboard alert")
    else:
        print(f"\n[6] Alert: NO — {result.status} does not trigger dashboard alert")

    print("\n" + "=" * 60)
    print("Diagnostic complete.")


if __name__ == "__main__":
    main()
