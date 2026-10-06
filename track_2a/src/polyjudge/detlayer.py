"""Deterministic mismatch layer (no LLM): inside aligned beads, flag
  * numbers / dates that occur on one side of the bead but not on the other (digit tokens), and
  * parenthetical segments "( ... )" that have no counterpart on the other side (their content tokens).
Severities are constants chosen on the tuning set (scripts/30_task_f.py det)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .aligner import acronyms, numbers_v2, stems
from .groups import doc_beads
from .segmentation import apply_punct_mask


@dataclass
class DetConfig:
    sev_num: float = 0.4
    sev_par: float = 0.4
    use_num: bool = True
    use_par: bool = True


def _token_value(tok: str) -> set[str]:
    return numbers_v2(tok)


def _parentheticals(tokens: list[str], start: int, end: int):
    """[(first_content_token, last_content_token_exclusive)] of '(' ... ')' groups inside tokens[start:end] (no nesting)."""
    out, open_at = [], None
    for i in range(start, end):
        if tokens[i] == "(":
            open_at = i
        elif tokens[i] == ")" and open_at is not None:
            if i - open_at > 1:
                out.append((open_at + 1, i))
            open_at = None
    return out


def _signature(tokens: list[str], a: int, b: int) -> set[str]:
    text = " ".join(tokens[a:b])
    return {"n" + x for x in numbers_v2(text)} | {"a" + x for x in acronyms(text)} | {"s" + x for x in stems(text)}


def det_scores(row: dict, cfg: DetConfig = DetConfig(), aligner=None):
    """-> (la, lb, info): token arrays (>= 0, -1 on punctuation) and counters of flagged items (number tokens, parentheticals)."""
    from .groups import _ALIGNER
    lang, en, tg, links = doc_beads(row, aligner or _ALIGNER)
    ta, tb = row["text_a"].split(), row["text_b"].split()
    la, lb = [0.0] * len(ta), [0.0] * len(tb)
    info = {"num_tokens": 0, "par_groups": 0}
    for l in links:
        if l.kind.startswith("unaligned"):
            continue
        ea, eb = [(en[i].start, en[i].end) for i in l.en], [(tg[j].start, tg[j].end) for j in l.tgt]
        if cfg.use_num:
            vals_a = set().union(*[_token_value(" ".join(ta[s:e])) for s, e in ea])
            vals_b = set().union(*[_token_value(" ".join(tb[s:e])) for s, e in eb])
            for toks, ranges, arr, other in ((ta, ea, la, vals_b), (tb, eb, lb, vals_a)):
                for s, e in ranges:
                    for i in range(s, e):
                        v = _token_value(toks[i])
                        if v and not (v & other):          # a number/date token with no counterpart on the other side
                            arr[i] = max(arr[i], cfg.sev_num)
                            info["num_tokens"] += 1
        if cfg.use_par:
            pa = [(s, e, _signature(ta, s, e)) for rs, re_ in ea for (s, e) in _parentheticals(ta, rs, re_)]
            pb = [(s, e, _signature(tb, s, e)) for rs, re_ in eb for (s, e) in _parentheticals(tb, rs, re_)]
            for mine, theirs, arr in ((pa, pb, la), (pb, pa, lb)):
                for s, e, sig in mine:
                    if not any(sig & sg for _, _, sg in theirs) and not (not sig and theirs):
                        for i in range(s, e):
                            arr[i] = max(arr[i], cfg.sev_par)
                        info["par_groups"] += 1
    return apply_punct_mask(ta, la), apply_punct_mask(tb, lb), info
