"""Span+severity records <-> token labels.

* `labels_to_records`  gold token labels -> records (few-shot examples, oracle round trip)
* `ScoreMapper.map`    records -> token labels (model-free; used on LLM output)

A record is {"quote": str, "severity": 0.2|0.4|0.6|0.8|1.0, "before": optional str, "occurrence": optional int}.
`quote` is copied verbatim from the text the model saw (the untokenized segment). All matching is done on the
whitespace-free character stream, so tokenization differences ("Switzerland's" vs "Switzerland ' s") do not matter.
"""
import math
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .segmentation import is_punct_token

LEVELS = (0.2, 0.4, 0.6, 0.8, 1.0)


def snap_severity(x: float) -> float | None:
    """Nearest gold level; values < 0.1 mean 'no difference' and are dropped."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if x != x or x < 0.1:  # NaN or "no difference"
        return None
    return min(1.0, max(0.2, math.floor(min(x, 1.0) * 5 + 0.5) / 5))  # nearest level, ties round up


def _norm(s: str) -> str:
    """Case-folded alphanumerics only (drops punctuation, whitespace, accents kept)."""
    return "".join(c for c in unicodedata.normalize("NFKC", s).casefold() if c.isalnum())


def _squash_map(text: str) -> tuple[str, list[int]]:
    chars, idx = [], []
    for i, ch in enumerate(text):
        if not ch.isspace():
            chars.append(ch)
            idx.append(i)
    return "".join(chars), idx


# ---------------------------------------------------------------- gold -> records
def runs_from_labels(labels: list[float]) -> list[tuple[int, int, float]]:
    """Maximal runs (first_tok, last_tok, level) of equal positive label; ASCII punctuation (-1) never breaks a run."""
    out, cur = [], None
    for i, l in enumerate(labels):
        if l == -1:
            continue
        if l > 0 and cur is not None and cur[2] == round(l, 2):
            cur[1] = i
        elif l > 0:
            cur = [i, i, round(l, 2)]
            out.append(cur)
        else:
            cur = None
    return [tuple(r) for r in out]


def labels_to_records(tokens: list[str], untokenized: str, labels: list[float], n_before: int = 4) -> list[dict]:
    """Convert gold token labels into verbatim-quote records (quote taken from the untokenized text)."""
    return runs_to_records(tokens, untokenized, runs_from_labels(labels), n_before)


def runs_to_records(tokens: list[str], untokenized: str, runs: list[tuple[int, int, float]], n_before: int = 4) -> list[dict]:
    """One verbatim-quote record per run (first_tok, last_tok, level); output order = run order."""
    squashed, idx = _squash_map(untokenized)
    assert len(squashed) == sum(len(t) for t in tokens), "tokenized / untokenized streams differ"
    starts, c = [], 0
    for t in tokens:
        starts.append(c)
        c += len(t)
    records = []
    for a, b, lvl in runs:
        s_char, e_char = starts[a], starts[b] + len(tokens[b])  # [s_char, e_char) in squashed space
        quote = untokenized[idx[s_char]: idx[e_char - 1] + 1]
        # up to n_before whitespace words immediately before the quote, verbatim
        before = " ".join(untokenized[: idx[s_char]].split()[-n_before:])
        records.append({"quote": quote, "severity": lvl, "before": before})  # "" = quote starts the text
    return records


# ---------------------------------------------------------------- records -> tokens
@dataclass
class MapResult:
    labels: list[float]
    diagnostics: list[dict] = field(default_factory=list)


class ScoreMapper:
    """Deterministic span -> token score mapping with a three-stage matching cascade:
    1. exact      squashed (whitespace-free) string, aligned to token boundaries
    2. normalized case-folded, punctuation-free, aligned to token boundaries
    3. fuzzy      best-matching token window by character similarity >= fuzzy_threshold
    Unmatched records are dropped (and reported in diagnostics)."""

    def __init__(self, fuzzy_threshold: float = 0.82, max_fuzzy_candidates: int = 40, stages=("exact", "normalized", "fuzzy")):
        self.stages = tuple(stages)  # e.g. ("exact",) = verbatim only: anything else is dropped, not guessed
        self.fuzzy_threshold = fuzzy_threshold
        self.max_fuzzy_candidates = max_fuzzy_candidates

    # -- document index
    def _index(self, tokens):
        S = "".join(tokens)
        starts, c = [], 0
        for t in tokens:
            starts.append(c)
            c += len(t)
        ends = [s + len(t) for s, t in zip(starts, tokens)]
        ntok = [_norm(t) for t in tokens]
        nstarts, c = [], 0
        for t in ntok:
            nstarts.append(c)
            c += len(t)
        return {
            "tokens": tokens, "S": S, "N": "".join(ntok), "ntok": ntok,
            "start_tok": {s: i for i, s in enumerate(starts)},
            "end_tok": {e: i for i, e in enumerate(ends)},
            "nstart_tok": {s: i for i, (s, t) in enumerate(zip(nstarts, ntok)) if t},
            "nend_tok": {s + len(t): i for i, (s, t) in enumerate(zip(nstarts, ntok)) if t},
        }

    # -- stages: each returns a list of (first_tok, last_tok) candidates, in document order
    @staticmethod
    def _find_all(hay: str, needle: str):
        p = hay.find(needle)
        while p != -1:
            yield p
            p = hay.find(needle, p + 1)

    def _exact(self, ix, quote):
        q = "".join(quote.split())
        if not q:
            return []
        return [(ix["start_tok"][p], ix["end_tok"][p + len(q)])
                for p in self._find_all(ix["S"], q)
                if p in ix["start_tok"] and p + len(q) in ix["end_tok"]]

    def _normalized(self, ix, quote):
        q = _norm(quote)
        if len(q) < 2:
            return []
        return [(ix["nstart_tok"][p], ix["nend_tok"][p + len(q)])
                for p in self._find_all(ix["N"], q)
                if p in ix["nstart_tok"] and p + len(q) in ix["nend_tok"]]

    def _fuzzy(self, ix, quote):
        qn = _norm(quote)
        qtoks = [t for t in (_norm(w) for w in quote.split()) if t]
        if len(qn) < 4 or not qtoks:
            return []
        ntok, n = ix["ntok"], len(ix["ntok"])
        lo, hi = 0.7 * len(qn), 1.3 * len(qn)
        anchors = qtoks[:2]

        def close(t, a):  # cheap pre-filter, then the real ratio
            if not t or abs(len(t) - len(a)) > 3 or t[0] != a[0]:
                return False
            return SequenceMatcher(None, t, a, autojunk=False).ratio() >= 0.75

        # candidate starts: tokens similar to one of the first two quote tokens
        starts = [i for i in range(n) if any(close(ntok[i], a) for a in anchors)]
        best, sm = (self.fuzzy_threshold - 1e-9, None), SequenceMatcher(autojunk=False)
        sm.set_seq2(qn)
        for i in starts[: self.max_fuzzy_candidates]:
            acc = ""
            for j in range(i, n):
                acc += ntok[j]
                if len(acc) > hi:
                    break
                if len(acc) >= lo and ntok[j]:
                    sm.set_seq1(acc)
                    if sm.real_quick_ratio() >= best[0] and sm.quick_ratio() >= best[0]:
                        r = sm.ratio()
                        if r > best[0]:
                            best = (r, (i, j))
        if best[1] and best[0] >= self.fuzzy_threshold:
            i, j = best[1]
            while i < j and not ix["ntok"][i]:
                i += 1
            while j > i and not ix["ntok"][j]:
                j -= 1
            return [(i, j)]
        return []

    # -- choosing among repeated occurrences
    def _choose(self, ix, cands, rec, claimed, quote):
        if len(cands) == 1:
            return cands
        if rec.get("occurrence") is not None:
            return [cands[min(int(rec["occurrence"]), len(cands) - 1)]]
        if rec.get("before") is not None:  # even "" is a hint: the quote starts the text
            before = _norm(rec["before"])

            def score(c):
                prev = "".join(ix["ntok"][max(0, c[0] - 8): c[0]])
                if not before:
                    return 1.0 if not "".join(ix["ntok"][: c[0]]) else 0.0
                return SequenceMatcher(None, prev[-len(before):], before, autojunk=False).ratio()

            top = max(score(c) for c in cands)
            tied = [c for c in cands if score(c) >= top - 1e-9]
            free = [c for c in tied if c[0] not in claimed]
            return [(free or tied)[0]]
        n_words = len([w for w in quote.split() if _norm(w)])
        if n_words >= 3 or re.search(r"\d", quote):  # distinctive: the same difference applies to every copy
            return cands
        free = [c for c in cands if c[0] not in claimed]  # generic short string: consume occurrences in order
        return [(free or cands)[0]]

    def map(self, tokens: list[str], records: list[dict]) -> MapResult:
        ix = self._index(tokens)
        labels = [0.0] * len(tokens)
        claimed: set[int] = set()
        diags = []
        for rec in records:
            quote, sev = rec.get("quote", ""), snap_severity(rec.get("severity"))
            if not isinstance(quote, str) or not quote.strip() or sev is None:
                diags.append({"quote": quote, "stage": "invalid", "hits": 0})
                continue
            stage, cands = "unmatched", []
            for name, fn in (("exact", self._exact), ("normalized", self._normalized), ("fuzzy", self._fuzzy)):
                if name not in self.stages:
                    continue
                cands = fn(ix, quote)
                if cands:
                    stage = name
                    break
            chosen = self._choose(ix, cands, rec, claimed, quote) if cands else []
            for a, b in chosen:
                claimed.add(a)
                for i in range(a, b + 1):
                    labels[i] = max(labels[i], sev)  # overlapping records: highest severity wins
            diags.append({"quote": quote, "stage": stage, "candidates": len(cands), "hits": len(chosen)})
        # keep -1 on ASCII punctuation, whatever the records covered
        labels = [-1.0 if is_punct_token(t) else l for t, l in zip(tokens, labels)]
        return MapResult(labels, diags)
