"""The prompt_pool partitions used for development: eval12 (4 chunks), eval2 (6 chunks) and 'tune' (16 chunks: few-shot sources + aligner tuning).
dev_eval and the test split are never reachable from here."""
import json
from pathlib import Path

from .data import chunk_key, rows_for

SPL = Path(__file__).resolve().parents[1] / "splits"


def _spec():
    e12 = json.loads((SPL / "pool_eval12.json").read_text(encoding="utf-8"))
    e2 = json.loads((SPL / "pool_eval2.json").read_text(encoding="utf-8"))
    return {"eval12": set(e12["chunks"]), "eval2": set(e2["chunks"]), "tune": set(e2["fewshot_chunks"])}


def load_set(name: str) -> list[dict]:
    chunks = _spec()[name]
    return sorted((r for r in rows_for("prompt_pool") if chunk_key(r) in chunks), key=lambda r: r["id"])


def fewshot_chunks() -> set[str]:
    return _spec()["tune"]
