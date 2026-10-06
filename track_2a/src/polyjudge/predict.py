"""Predict: INPUT texts only -> prediction files in the organizers' format (docs/prediction_format.md).

Methods
  B0  model-free sentence alignment (aligner v2): unaligned sentence = 1.0, everything else 0      [additional baseline, no Apertus]
  M0  simple Apertus prompting baseline: one B1 call per sentence group (70B), spans -> token scores   [reference for 'improvement over prompting']
  M1  B0 + Apertus verifier on deterministic flags (number / parenthesis / untranslated); a flag is accepted only if BOTH independent samples say
      "real" at level 1.0                                                                            [+ 0.2 * level]
  M2  PolyJudge multi-agent on aligned pairs: blind extractors (EN / target) -> Compare -> verifier   [B0 + 0.2 * accepted level]
  M3  Apertus-judged alignment: 8B answers "which candidate carries the same content" in both directions, 70B decides the disagreements with the aligner;
      presence score (both unaligned 1.0, judged verdict, both aligned 0)
Modes
  offline  model-free only (B0)            replay  every LLM request must be in the cache, a miss raises CacheMiss (nothing is sent)
  live     the API is called (call budgets and cache apply)
Determinism: all choices are fixed; with the replay cache the output is bit-for-bit reproducible.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .aligner import BEST_V2, HeuristicAlignerV2
from .candidates import build_candidates
from .config import load_llm_config
from .debate import run_debate
from .detlayer import DetConfig, det_scores
from .evalsets import fewshot_chunks
from .groups import make_groups
from .heuristic import heuristic_labels
from .inputs import FORBIDDEN_KEYS, INPUT_KEYS, Input
from .judge import JUDGE_BATCH, run_judge
from .lla import BATCH as ALIGN_BATCH, build_items, presence_labels, run_items
from .llm import CACHE_DIR, LLMClient
from .pipeline import llm_scores, map_run, run_groups
from .prompts import frozen_shots
from .prompts_v2 import select_shots2
from .segmentation import apply_punct_mask
from .verifier import _batches, run_verifier

METHODS = ("B0", "M0", "M1", "M2", "M3")
MODES = ("offline", "replay", "live")
M8, M70 = "swiss-ai/Apertus-v1.5-8B", "swiss-ai/Apertus-v1.5-70B"
W = 0.2                                  # weight of accepted flags (rank-invariant for a binary base; fixed a priori)
LANGS = ("de", "fr", "it")
AL = HeuristicAlignerV2(**BEST_V2)


class ModeError(RuntimeError):
    pass


@dataclass
class Clients:
    c8: LLMClient | None = None
    c70: LLMClient | None = None

    async def aclose(self):
        for c in (self.c8, self.c70):
            if c is not None:
                await c.aclose()

    def summary(self) -> str:
        return " | ".join(f"{n}: {c.summary()}" for n, c in (("8B", self.c8), ("70B", self.c70)) if c is not None)


def make_clients(mode: str, cache_dir: Path = CACHE_DIR, concurrency: int = 4, sdk_client=None, budgets=None, max_concurrency: int | None = None,
                 raise_after: int = 100, log_path=None) -> Clients:
    """offline: none; replay: cache only (a miss raises CacheMiss); live: real API (host must be api.inference.cscs.ch).
    budgets = {"8": budget, "70": budget} (any object with reserve(); default: the persistent per-model budgets)."""
    if mode == "offline":
        return Clients()
    if mode == "replay":
        return Clients(LLMClient(None, model=M8, replay=True, cache_dir=cache_dir, log_path=None), LLMClient(None, model=M70, replay=True, cache_dir=cache_dir, log_path=None))
    if mode == "live":
        b = budgets or {}
        kw = dict(concurrency=concurrency, cache_dir=cache_dir, max_concurrency=max_concurrency, raise_after=raise_after)
        if sdk_client is not None:            # tests: fake SDK, no network
            return Clients(LLMClient(None, model=M8, sdk_client=sdk_client, budget=b.get("8"), log_path=log_path, **kw),
                           LLMClient(None, model=M70, sdk_client=sdk_client, budget=b.get("70"), log_path=log_path, **kw))
        cfg8, cfg70 = load_llm_config(name=M8), load_llm_config(name=M70)
        for c in (cfg8, cfg70):
            if urlparse(c.base_url).netloc != "api.inference.cscs.ch":
                raise ModeError("unexpected API host (expected api.inference.cscs.ch)")
        return Clients(LLMClient(cfg8, model=M8, budget=b.get("8"), log_path=log_path, **kw), LLMClient(cfg70, model=M70, budget=b.get("70"), log_path=log_path, **kw))
    raise ModeError(f"unknown mode {mode!r}")


# ------------------------------------------------------------------------------------------------ helpers
def _rows(inputs: list[Input]) -> list[dict]:
    rows = [i.as_row() for i in inputs]
    assert all(set(r) == set(INPUT_KEYS) and not set(r) & set(FORBIDDEN_KEYS) for r in rows)
    return rows


def _tokens(row):
    return row["text_a"].split(), row["text_b"].split()


def _flag_array(n_tokens: int, spans: list[tuple[int, int, float]]) -> list[float]:
    arr = [0.0] * n_tokens
    for a, b, lvl in spans:
        for p in range(a, b):
            arr[p] = max(arr[p], lvl)
    return arr


def _add_flags(row: dict, base, accepted: list[tuple[str, int, int, float]]):
    """base + W * level on the flagged tokens (punctuation stays -1)."""
    ta, tb = _tokens(row)
    arrs = {"a": _flag_array(len(ta), [(a, b, l) for s, a, b, l in accepted if s == "a"]),
            "b": _flag_array(len(tb), [(a, b, l) for s, a, b, l in accepted if s == "b"])}
    return tuple(apply_punct_mask(toks, [max(h, 0.0) + W * x for h, x in zip(bs, arrs[sd])])
                 for sd, toks, bs in (("a", ta, base[0]), ("b", tb, base[1])))


# ------------------------------------------------------------------------------------------------ methods
async def run_b0(inputs, cl: Clients):
    return {i.id: heuristic_labels(i.as_row(), i.lang, AL) for i in inputs}, {}


async def run_m0(inputs, cl: Clients):
    rows = _rows(inputs)
    groups = [g for r in rows for g in make_groups(r, "all")]
    shots = {l: frozen_shots(l, "b1") for l in LANGS}
    runs = await run_groups(cl.c70, "b1", groups, shots, M70)
    for run in runs:
        map_run(run)
    preds = {}
    for r in rows:
        la, lb = llm_scores(r, runs)
        ta, tb = _tokens(r)
        preds[r["id"]] = (apply_punct_mask(ta, la), apply_punct_mask(tb, lb))
    return preds, {"groups": len(groups), "failed_groups": sum(x.failed for x in runs)}


def _det_candidates(rows):
    cands = []
    for r in rows:
        num = det_scores(r, DetConfig(sev_num=1.0, use_par=False), AL)[:2]
        par = det_scores(r, DetConfig(sev_par=1.0, use_num=False), AL)[:2]
        cands += build_candidates(r, AL, num, par, None)
    return cands


async def run_m1(inputs, cl: Clients):
    rows = _rows(inputs)
    base = {r["id"]: heuristic_labels(r, r["subset"].split("_")[1], AL) for r in rows}
    cands = _det_candidates(rows)
    d_a, s_a = await run_verifier(cl.c70, M70, cands, temperature=0.0)                   # independent sample 1
    d_b, s_b = await run_verifier(cl.c70, M70, cands, temperature=0.7, seed=101)         # independent sample 2
    acc = {i for i in range(len(cands)) if i in d_a and i in d_b and d_a[i][0] and d_b[i][0] and d_a[i][1] == 1.0 and d_b[i][1] == 1.0}
    per_doc: dict[str, list] = {}
    for i in sorted(acc):
        c = cands[i]
        per_doc.setdefault(c.doc, []).append((c.side, c.start, c.end, 1.0))
    preds = {r["id"]: _add_flags(r, base[r["id"]], per_doc.get(r["id"], [])) for r in rows}
    return preds, {"candidates": len(cands), "accepted": len(acc), "verifier_failed": [s_a["failed"], s_b["failed"]]}


async def run_m2(inputs, cl: Clients):
    rows = _rows(inputs)
    base = {r["id"]: heuristic_labels(r, r["subset"].split("_")[1], AL) for r in rows}
    groups = [g for r in rows for g in make_groups(r, "aligned")]
    shots = {l: select_shots2(l, fewshot_chunks()) for l in LANGS}                      # few-shot sources: prompt_pool tuning chunks only
    runs, st = await run_debate(cl.c70, M70, groups, shots)
    spans: dict[str, list] = {}
    for run in runs:
        spans.setdefault(run.group.doc_id, []).extend(s for s in run.spans if s["mapped"])
    cands = []
    for r in rows:
        zero = ([0.0] * len(_tokens(r)[0]), [0.0] * len(_tokens(r)[1]))
        cands += build_candidates(r, AL, zero, zero, spans.get(r["id"]))
    dec, sv = await run_verifier(cl.c70, M70, cands, temperature=0.0)
    per_doc: dict[str, list] = {}
    for i, (real, level, _) in sorted(dec.items()):
        if real:
            c = cands[i]
            per_doc.setdefault(c.doc, []).append((c.side, c.start, c.end, level))
    preds = {r["id"]: _add_flags(r, base[r["id"]], per_doc.get(r["id"], [])) for r in rows}
    return preds, {"groups": len(groups), "candidates": len(cands), "accepted": sum(len(v) for v in per_doc.values()), **st, "verifier_failed": sv["failed"]}


async def run_m3(inputs, cl: Clients):
    rows = _rows(inputs)
    data = []
    for r in rows:
        lang = r["subset"].split("_")[1]
        items, en, tg, _ = build_items(r, lang, AL)
        data.append((r, lang, items, en, tg))
    await asyncio.gather(*(run_items(cl.c8, M8, lang, items, en, tg) for _, lang, items, en, tg in data))
    cases = [(r["id"], lang, it, en, tg) for r, lang, items, en, tg in data for it in items if it.llm is not None and it.heur_unaligned != it.llm_unaligned]
    verdicts, js = await run_judge(cl.c70, M70, cases) if cases else ({}, {"calls": 0, "failed_batches": 0})
    preds = {}
    for r, lang, items, en, tg in data:
        judged = {(sd, ix): v for (d, sd, ix), v in verdicts.items() if d == r["id"]}
        preds[r["id"]] = presence_labels(r, items, en, tg, judged)
    return preds, {"disagreements": len(cases), "judge_failed_batches": js["failed_batches"]}


RUNNERS = {"B0": run_b0, "M0": run_m0, "M1": run_m1, "M2": run_m2, "M3": run_m3}


async def predict(method: str, inputs: list[Input], mode: str, clients: Clients | None = None):
    """-> ({doc id: (labels_a, labels_b)}, stats). Inputs must be `Input` objects (no labels exist on them)."""
    if method not in RUNNERS:
        raise ValueError(f"unknown method {method!r}")
    if not all(isinstance(i, Input) for i in inputs):
        raise TypeError("predict() accepts Input objects only")
    if mode == "offline" and method != "B0":
        raise ModeError(f"{method} needs Apertus: use --mode replay or --mode live (offline = model-free only)")
    own = clients is None
    cl = clients or make_clients(mode)
    try:
        preds, stats = await RUNNERS[method](inputs, cl)
    finally:
        if own:
            await cl.aclose()
    for i in inputs:      # format contract
        la, lb = preds[i.id]
        ta, tb = i.text_a.split(), i.text_b.split()
        assert len(la) == len(ta) and len(lb) == len(tb), f"label length mismatch in {i.id}"
        assert all(isinstance(x, (int, float)) and math.isfinite(x) for x in list(la) + list(lb))
        assert [x == -1 for x in la] == [all(c in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" for c in t) for t in ta], "punctuation must be -1"
    return preds, stats


# ------------------------------------------------------------------------------------------------ files
def prediction_path(out_dir: Path, method: str, lang: str) -> Path:
    """`<prefix><lang>.jsonl.jsonl` with prefix `<out_dir>/<method>_admin_` (the organizers' evaluator appends `<lang>.jsonl.jsonl`)."""
    return Path(out_dir) / f"{method}_admin_{lang}.jsonl.jsonl"


def write_predictions(preds: dict, inputs: list[Input], out_dir: Path, method: str) -> dict[str, str]:
    """One file per language, one JSON object per line in input order: {"id","text_a","text_b","labels_a","labels_b"}. Returns {path: sha256}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for lang in LANGS:
        rows = [i for i in inputs if i.lang == lang]
        if not rows:
            continue
        p = prediction_path(out_dir, method, lang)
        with p.open("w", encoding="utf-8", newline="\n") as f:
            for i in rows:
                la, lb = preds[i.id]
                f.write(json.dumps({"id": i.id, "text_a": i.text_a, "text_b": i.text_b, "labels_a": [float(x) for x in la], "labels_b": [float(x) for x in lb]},
                                   ensure_ascii=False, separators=(",", ":")) + "\n")
        hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def read_predictions(out_dir: Path, method: str) -> dict:
    preds = {}
    for lang in LANGS:
        p = prediction_path(out_dir, method, lang)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                d = json.loads(line)
                preds[d["id"]] = (d["labels_a"], d["labels_b"])
    return preds


# ------------------------------------------------------------------------------------------------ cost estimates (no calls)
RETRY = {"M0": 1.03, "M1": 1.08, "M2": 1.04, "M3": 1.03}       # measured first-attempt failure overheads of earlier runs
DISAGREE_RATE = 0.09                                             # aligner vs 8B status disagreements per sentence (93 / 1 028 on dev_select; 40 / 1 444 on the pool sets)
M2_CANDS_PER_GROUP, CANDS_PER_VERIFY_CALL = 1.0, 7.7             # measured in the debate ablation (69 candidates / 66 groups; 809 / 83 calls)


def estimate_calls(method: str, inputs: list[Input]) -> dict:
    """Expected number of API calls (70B, 8B) from the INPUT texts only; assumptions are the constants above."""
    rows = _rows(inputs)
    if method == "B0":
        return {"70B": 0, "8B": 0}
    if method == "M0":
        g = sum(len(make_groups(r, "all")) for r in rows)
        return {"70B": math.ceil(g * RETRY["M0"]), "8B": 0, "groups": g}
    if method == "M1":
        cands = _det_candidates(rows)
        one = sum(len(_batches([c for c in cands if c.lang == l])) for l in LANGS)
        return {"70B": math.ceil(2 * one * RETRY["M1"]), "8B": 0, "candidates": len(cands), "verifier_batches_per_sample": one}
    if method == "M2":
        g = sum(len(make_groups(r, "aligned")) for r in rows)
        ver = math.ceil(g * M2_CANDS_PER_GROUP / CANDS_PER_VERIFY_CALL * 1.08)
        return {"70B": math.ceil((3 * g) * RETRY["M2"]) + ver, "8B": 0, "aligned_groups": g, "verifier_est": ver}
    if method == "M3":
        n8, nsent = 0, 0
        for r in rows:
            items, _, _, _ = build_items(r, r["subset"].split("_")[1], AL)
            for s in ("a", "b"):
                k = sum(1 for it in items if it.side == s)
                n8 += -(-k // ALIGN_BATCH)
                nsent += k
        judge = max(3, math.ceil(nsent * DISAGREE_RATE / JUDGE_BATCH * 1.3))     # +30 %: batches are formed per language
        return {"70B": judge, "8B": math.ceil(n8 * RETRY["M3"]), "sentences": nsent}
    raise ValueError(method)


# ------------------------------------------------------------------------------------------------ CLI
def _resolve_inputs(a) -> list[Input]:
    from .inputs import load_inputs, load_inputs_by_ids
    if a.ids_file:
        return load_inputs_by_ids(set(json.loads(Path(a.ids_file).read_text(encoding="utf-8"))["doc_ids"]))
    return load_inputs(a.input)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m polyjudge.predict")
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--mode", choices=MODES, default="offline")
    ap.add_argument("--input", default="prompt_pool", help="prompt_pool | dev_select | dev_final | test")
    ap.add_argument("--ids-file", default=None, help="json file with a 'doc_ids' list of prompt_pool documents (e.g. splits/pool_eval2.json)")
    ap.add_argument("--out", default="output/predictions")
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--estimate", action="store_true", help="print the expected call counts and exit (no calls)")
    a = ap.parse_args(argv)
    inputs = _resolve_inputs(a)
    if a.limit_docs:
        inputs = inputs[: a.limit_docs]
    if a.estimate:
        print(json.dumps({"method": a.method, "docs": len(inputs), **estimate_calls(a.method, inputs)}))
        return 0
    preds, stats = asyncio.run(predict(a.method, inputs, a.mode))
    hashes = write_predictions(preds, inputs, Path(a.out), a.method)
    print(json.dumps({"method": a.method, "mode": a.mode, "docs": len(inputs), "stats": stats, "files": hashes}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
