"""
Runtime mode diagnostics: Pipeline.set_mode for GENERAL and PERSON_SEARCH.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# best.pt references ultralytics.nn.modules.skywatch_modules — load file directly (avoid package __init__ deps)
import importlib.util

_swm_path = PROJECT_ROOT / "src" / "ultralytics_patch" / "nn" / "modules" / "skywatch_modules.py"
_swm_spec = importlib.util.spec_from_file_location("ultralytics.nn.modules.skywatch_modules", _swm_path)
_swm_mod = importlib.util.module_from_spec(_swm_spec)
sys.modules["ultralytics.nn.modules.skywatch_modules"] = _swm_mod
assert _swm_spec.loader is not None
_swm_spec.loader.exec_module(_swm_mod)

from utils.config import AppConfig
from utils.logger import EventLogger
from engine.pipeline import Pipeline
from database.db import Database


def _embedding_count(cfg: AppConfig, logger: EventLogger) -> int:
    db = Database(cfg, logger)
    embs = db.get_all_embeddings()
    return len(embs)


def _first_person_id_with_embedding(cfg: AppConfig, logger: EventLogger) -> int | None:
    db = Database(cfg, logger)
    embs = db.get_all_embeddings()
    if not embs:
        return None
    return int(embs[0][0])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("GENERAL", "PERSON_SEARCH"), required=True)
    p.add_argument("--person-id", type=int, default=None, help="Required for PERSON_SEARCH")
    args = p.parse_args()

    cfg = AppConfig()
    logger = EventLogger(cfg)
    print(f"[DIAG] project_root={cfg.project_root}")

    if args.mode == "GENERAL":
        pipe = Pipeline(cfg, logger)
        ok = pipe.set_mode("GENERAL", {})
        assert pipe.current_mode == "GENERAL", f"expected GENERAL, got {pipe.current_mode}"
        assert pipe.target_embedding is None, "GENERAL must have target_embedding=None"
        n = _embedding_count(cfg, logger)
        print(f"[DIAG] set_mode ok={ok} current_mode={pipe.current_mode} target_embedding is None={pipe.target_embedding is None}")
        print(f"[DIAG] DB embedding count (get_all_embeddings): {n}")
        return 0 if ok else 1

    pid = args.person_id
    if pid is None:
        pid = _first_person_id_with_embedding(cfg, logger)
        if pid is None:
            print("[DIAG] PERSON_SEARCH requires a --person-id or at least one row in embeddings.")
            print(
                "  Example: python tools/diagnose_runtime_modes.py "
                "--mode PERSON_SEARCH --person-id <valid_id_from_criminals>"
            )
            return 2

    pipe = Pipeline(cfg, logger)
    ok = pipe.set_mode("PERSON_SEARCH", {"target_person_id": int(pid), "target_person_name": ""})
    assert pipe.current_mode == "PERSON_SEARCH", f"expected PERSON_SEARCH, got {pipe.current_mode}"
    assert pipe.target_embedding is not None, "target_embedding must be loaded"
    te = pipe.target_embedding
    nrm = float(np.linalg.norm(te))
    print(
        f"[DIAG] set_mode ok={ok} current_mode={pipe.current_mode} "
        f"emb_shape={tuple(te.shape)} norm={nrm:.4f}"
    )
    if abs(nrm - 1.0) > 0.05:
        print(f"[DIAG] WARNING: expected norm ~1.0, got {nrm:.4f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
