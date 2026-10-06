"""Leave-one-chunk-out cross-validation harness for the model-free variants (and later: base + accepted LLM flags).
Uses ALL 26 prompt_pool chunks (78 docs); hyper-parameters for a held-out chunk are chosen on the other 25 chunks only.
Selection uses scipy's Spearman (fast); the reported point estimates use the organizers' official function; chunk-bootstrap CIs use scipy."""
from __future__ import annotations

import pickle
import itertools
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .aligner import BEST_V2, HeuristicAlignerV2, HeuristicAlignerV3
from .data import chunk_key, rows_for
from .detlayer import DetConfig, det_scores
from .heuristic import heuristic_labels
from .official_eval import official_spearman
from .segmentation import apply_punct_mask

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "output" / "cv_cache.pkl"
LANGS = ("de", "fr", "it")

V2_GRID = [dict(complex_penalty=c, tau=t, trim=False, rescue=False) for c, t in itertools.product((1.5, 0.0, 3.0), (1.0, 2.0))]   # simplest first
V3_GRID = [dict(w_stem=w, p_short=s, p11=p, **BEST_V2) for w, s, p in sorted(itertools.product((0.0, 1.0), (0.0, 1.5), (0.0, 1.5)), key=lambda x: (sum(v > 0 for v in x), x))]
SEV_GRID = [(0.4, 0.4), (0.6, 0.4), (0.4, 0.2), (0.6, 0.2)]   # (severity of number flags, severity of parenthesis flags)
W_LAYER = 0.2


def key_of(kind: str, cfg: dict):
    return (kind,) + tuple(sorted(cfg.items()))


def make_aligner(kind: str, cfg: dict):
    return HeuristicAlignerV2(**cfg) if kind == "v2" else HeuristicAlignerV3(**cfg)


class Store:
    """Per (aligner config, doc) cached arrays: heuristic labels and the two deterministic flag arrays."""

    def __init__(self):
        self.rows = sorted(rows_for("prompt_pool"), key=lambda r: r["id"])
        self.chunk = {r["id"]: chunk_key(r) for r in self.rows}
        self.chunks = sorted(set(self.chunk.values()))
        self.byid = {r["id"]: r for r in self.rows}
        self.h, self.num, self.par = {}, {}, {}
        if CACHE.exists():
            self.h, self.num, self.par = pickle.loads(CACHE.read_bytes())

    def save(self):
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_bytes(pickle.dumps((self.h, self.num, self.par)))

    def heur(self, kind, cfg, r):
        k = (key_of(kind, cfg), r["id"])
        if k not in self.h:
            self.h[k] = heuristic_labels(r, r["subset"].split("_")[1], make_aligner(kind, cfg))
        return self.h[k]

    def flags(self, kind, cfg, r, which):
        store = self.num if which == "num" else self.par
        k = (key_of(kind, cfg), r["id"])
        if k not in store:
            dc = DetConfig(sev_num=1.0, use_par=False) if which == "num" else DetConfig(sev_par=1.0, use_num=False)
            la, lb, _ = det_scores(r, dc, make_aligner(kind, cfg))
            store[k] = (la, lb)
        return store[k]


def _arr(pred_a, pred_b, r):
    p = np.array(list(pred_a) + list(pred_b), dtype=float)
    g = np.array(list(r["labels_a"]) + list(r["labels_b"]), dtype=float)
    m = g != -1
    return p[m], g[m]


def macro_scipy(preds: dict, docs, byid):
    """preds[id] = (pa, pb); macro of per-language pooled scipy Spearman over `docs` (ids). Constant -> 0."""
    out = []
    for lang in LANGS:
        P, G = [], []
        for d in docs:
            r = byid[d]
            if r["subset"].endswith(lang):
                p, g = _arr(*preds[d], r)
                P.append(p); G.append(g)
        p, g = np.concatenate(P), np.concatenate(G)
        out.append(0.0 if p.min() == p.max() else float(spearmanr(p, g)[0]))
    return sum(out) / 3


def add_layers(r, h, num, par, sev=(0.4, 0.4), use_num=True, use_par=True, w=W_LAYER):
    """h + w * max(sev_num * num_flag, sev_par * par_flag), punctuation = -1."""
    out = []
    for side, toks in (("a", r["text_a"].split()), ("b", r["text_b"].split())):
        i = 0 if side == "a" else 1
        hv = [max(x, 0.0) for x in h[i]]
        n = num[i] if use_num else [0.0] * len(hv)
        p = par[i] if use_par else [0.0] * len(hv)
        out.append(apply_punct_mask(toks, [a + w * max(sev[0] * max(b, 0), sev[1] * max(c, 0)) for a, b, c in zip(hv, n, p)]))
    return tuple(out)


def official_macro(preds: dict, docs, byid):
    res = {}
    for lang in LANGS:
        P, G = [], []
        for d in docs:
            r = byid[d]
            if r["subset"].endswith(lang):
                P += list(preds[d][0]) + list(preds[d][1])
                G += list(r["labels_a"]) + list(r["labels_b"])
        res[lang] = official_spearman(P, G)
    res["macro"] = sum(res[l] for l in LANGS) / 3
    return res


def bootstrap(variants: dict, st: Store, n=1000, seed=0, docs=None):
    """Chunk bootstrap. variants[name] = preds dict. Returns {name: {lang/macro: (lo, hi)}} and paired draws for differences."""
    rng = np.random.default_rng(seed)
    rows = [r for r in st.rows if docs is None or r["id"] in docs]
    chunks = sorted({st.chunk[r["id"]] for r in rows})
    arrs = {}   # name -> lang -> chunk -> (pred, gold)
    for name, preds in variants.items():
        arrs[name] = {l: {c: [] for c in chunks} for l in LANGS}
        for r in rows:
            lang = r["subset"][-2:]
            p, g = _arr(*preds[r["id"]], r)
            arrs[name][lang][st.chunk[r["id"]]].append((p, g))
        for l in LANGS:
            for c in chunks:
                lst = arrs[name][l][c]
                arrs[name][l][c] = (np.concatenate([x[0] for x in lst]), np.concatenate([x[1] for x in lst])) if lst else (np.array([]), np.array([]))
    draws = {name: {k: [] for k in (*LANGS, "macro")} for name in variants}
    for _ in range(n):
        pick = rng.choice(len(chunks), len(chunks), replace=True)
        for name in variants:
            vals = []
            for l in LANGS:
                p = np.concatenate([arrs[name][l][chunks[i]][0] for i in pick])
                g = np.concatenate([arrs[name][l][chunks[i]][1] for i in pick])
                v = 0.0 if p.size == 0 or p.min() == p.max() else float(spearmanr(p, g)[0])
                draws[name][l].append(v)
                vals.append(v)
            draws[name]["macro"].append(sum(vals) / 3)
    ci = {name: {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in d.items()} for name, d in draws.items()}
    return ci, draws
