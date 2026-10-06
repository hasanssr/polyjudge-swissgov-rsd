"""First LLM layer: run groups through B1 / Compare, map spans to tokens (verbatim only), combine with the heuristic, score.

No self-consistency, no extractors, no debate: one call per sentence group (plus at most one retry after a parse failure / abort / truncation).
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from .diffs import ParseResult, parse_b1, parse_compare
from .groups import Group, make_groups
from .heuristic import heuristic_labels
from .official_eval import official_spearman
from .prompts import build_messages
from .prompts_v2 import build_messages_v2, parse_compare2
from .spans import ScoreMapper

PARSERS = {"b1": parse_b1, "compare": parse_compare, "compare2": parse_compare2}
MAX_TOKENS = {"b1": 600, "compare": 700, "compare2": 600}
MAPPER_EXACT = ScoreMapper(stages=("exact",))              # primary policy: not verbatim -> dropped, never guessed
MAPPER_NORM = ScoreMapper(stages=("exact", "normalized"))  # sensitivity only (no extra calls)


@dataclass
class GroupRun:
    group: Group
    kind: str
    calls: int = 0
    first_failed: bool = False
    failed: bool = False                 # no parseable answer after the retry
    aborted: int = 0                     # repetition-guard aborts
    truncated: int = 0                   # finish_reason == length
    parse: ParseResult | None = None
    raw: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    spans: list = field(default_factory=list)   # per record: dict(side, quote, severity, type, reason, mapped, mapped_norm, pos)


async def _run_one(client, kind, g: Group, shots, model) -> GroupRun:
    run = GroupRun(g, kind)
    msgs = build_messages_v2(g.lang, g, shots) if kind == "compare2" else build_messages(kind, g.lang, g, shots)
    for attempt in range(2):
        r = await client.chat(msgs, model=model, temperature=0.0 if attempt == 0 else 0.3, seed=None if attempt == 0 else 7,
                              max_tokens=MAX_TOKENS[kind], json_mode=True, rep_guard=True, tag=f"{kind}-{g.lang}")
        run.calls += 1
        run.prompt_tokens += r.prompt_tokens or 0
        run.completion_tokens += r.completion_tokens or 0
        run.latency += r.latency_s
        run.aborted += int(r.aborted)
        run.truncated += int(r.finish_reason == "length")
        res = PARSERS[kind](r.text) if not (r.aborted or r.finish_reason == "length") else ParseResult(False, error="aborted/truncated")
        run.parse, run.raw = res, r.text
        if res.ok:
            return run
        if attempt == 0:
            run.first_failed = True
    run.failed = True
    return run


async def run_groups(client, kind: str, groups: list[Group], shots_by_lang: dict, model: str) -> list[GroupRun]:
    return list(await asyncio.gather(*(_run_one(client, kind, g, shots_by_lang[g.lang], model) for g in groups)))


def map_run(run: GroupRun) -> None:
    """Map each parsed record separately (verbatim policy) and keep per-record diagnostics with doc token positions."""
    g = run.group
    run.spans = []
    if not run.parse or not run.parse.ok:
        return
    for rec in run.parse.records:
        toks, pos = (g.en_tokens, g.en_pos) if rec["side"] == "a" else (g.tg_tokens, g.tg_pos)
        ex = MAPPER_EXACT.map(toks, [rec])
        nm = MAPPER_NORM.map(toks, [rec])
        hit = [pos[i] for i, v in enumerate(ex.labels) if v > 0]
        run.spans.append({"side": rec["side"], "quote": rec["quote"], "severity": rec["severity"], "type": rec.get("type", ""),
                          "reason": rec.get("reason", ""), "mapped": bool(hit), "mapped_norm": any(v > 0 for v in nm.labels), "pos": hit,
                          "pos_norm": [pos[i] for i, v in enumerate(nm.labels) if v > 0]})


def llm_scores(row: dict, runs: list[GroupRun], norm: bool = False):
    """Doc-level LLM score arrays (la, lb) (>= 0; max over spans). Punctuation handled by `finalize`."""
    la, lb = [0.0] * len(row["text_a"].split()), [0.0] * len(row["text_b"].split())
    for run in runs:
        if run.group.doc_id != row["id"]:
            continue
        for sp in run.spans:
            arr = la if sp["side"] == "a" else lb
            for i in (sp["pos_norm"] if norm else sp["pos"]):
                arr[i] = max(arr[i], sp["severity"])
    return la, lb


def finalize(row: dict, la, lb, h=None):
    """final = max(heuristic, LLM) with -1 kept on punctuation (taken from the heuristic/gold punctuation rule)."""
    ha, hb = h if h is not None else (None, None)
    from .segmentation import apply_punct_mask
    ta, tb = row["text_a"].split(), row["text_b"].split()
    fa = [max(x, ha[i]) if ha else x for i, x in enumerate(la)]
    fb = [max(x, hb[i]) if hb else x for i, x in enumerate(lb)]
    return apply_punct_mask(ta, fa), apply_punct_mask(tb, fb)


def score(rows: list[dict], preds: dict) -> dict:
    """Official pooled Spearman per language over `rows`; preds[id] = (la, lb). Returns {lang: rho, 'macro': mean}."""
    out = {}
    for lang in ("de", "fr", "it"):
        P, G = [], []
        for r in rows:
            if r["subset"] != f"admin_{lang}":
                continue
            a, b = preds[r["id"]]
            P += list(a) + list(b)
            G += list(r["labels_a"]) + list(r["labels_b"])
        out[lang] = official_spearman(P, G)
    out["macro"] = sum(out[l] for l in ("de", "fr", "it")) / 3
    return out


def heuristic_preds(rows, aligner):
    return {r["id"]: heuristic_labels(r, r["subset"].split("_")[1], aligner) for r in rows}
