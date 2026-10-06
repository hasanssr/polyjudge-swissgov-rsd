"""Scoring of prediction dicts against gold rows (dev sets only; the test labels are never handled by this module's callers before Task 5).
Point estimates: the organizers' compute_correlations (via official_eval). Chunk-bootstrap confidence intervals: scipy Spearman on resampled chunks."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .official_eval import official_spearman

LANGS = ("de", "fr", "it")


def _lang(row_id: str) -> str:
    return row_id.split("_")[1]


def official_scores(preds: dict, gold_rows: list[dict]) -> dict:
    """{'de','fr','it','macro'}: pooled over both sides and all documents of a language (organizers' function)."""
    out = {}
    for lang in LANGS:
        P, G = [], []
        for r in gold_rows:
            if _lang(r["id"]) == lang:
                P += list(preds[r["id"]][0]) + list(preds[r["id"]][1])
                G += list(r["labels_a"]) + list(r["labels_b"])
        out[lang] = official_spearman(P, G)
    out["macro"] = sum(out[l] for l in LANGS) / 3
    return out


def _chunk_arrays(preds: dict, gold_rows: list[dict], chunk_of: dict):
    """{lang: {chunk: (pred, gold)}} with gold == -1 removed."""
    acc: dict = {l: {} for l in LANGS}
    for r in gold_rows:
        p = np.array(list(preds[r["id"]][0]) + list(preds[r["id"]][1]), dtype=float)
        g = np.array(list(r["labels_a"]) + list(r["labels_b"]), dtype=float)
        m = g != -1
        acc[_lang(r["id"])].setdefault(chunk_of[r["id"]], []).append((p[m], g[m]))
    return {l: {c: (np.concatenate([x[0] for x in v]), np.concatenate([x[1] for x in v])) for c, v in d.items()} for l, d in acc.items()}


def paired_chunk_bootstrap(preds_by_method: dict, gold_rows: list[dict], chunk_of: dict, ref: str, n: int = 1000, seed: int = 0) -> dict:
    """Resample CHUNKS with replacement (same resamples for every method). Returns per method:
    {'ci': (lo, hi) of its macro Spearman, 'diff': (mean, lo, hi, P(diff>0)) of macro(method) - macro(ref)}."""
    chunks = sorted({chunk_of[r["id"]] for r in gold_rows})
    arrs = {m: _chunk_arrays(p, gold_rows, chunk_of) for m, p in preds_by_method.items()}
    rng = np.random.default_rng(seed)
    macro = {m: [] for m in preds_by_method}
    for _ in range(n):
        pick = [chunks[i] for i in rng.integers(0, len(chunks), len(chunks))]
        for m in preds_by_method:
            vals = []
            for l in LANGS:
                parts = [arrs[m][l][c] for c in pick if c in arrs[m][l]]
                p = np.concatenate([x[0] for x in parts])
                g = np.concatenate([x[1] for x in parts])
                vals.append(0.0 if p.size == 0 or p.min() == p.max() else float(spearmanr(p, g)[0]))
            macro[m].append(sum(vals) / 3)
    out = {}
    for m in preds_by_method:
        d = np.array(macro[m]) - np.array(macro[ref])
        out[m] = {"ci": (float(np.percentile(macro[m], 2.5)), float(np.percentile(macro[m], 97.5))),
                  "diff": (float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.mean(d > 0)))}
    return out
