"""Candidate flags inside ALIGNED beads: deterministic layer (numbers, unmatched parentheticals, untranslated text) + cached Compare-v2 spans.
A candidate is a token span of one sentence; the LLM verifier decides whether it reflects a real meaning difference."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .groups import doc_beads
from .spans import runs_to_records

STOP = {
    "en": set("the of and to in is are was were be been for with on at by from as that this these those it its or an a not which has have had will can may must also their they there than into about other more such only".split()),
    "de": set("der die das und ist sind war waren wird werden für mit von zu im den dem des ein eine einer eines nicht auch auf bei als oder sich nach über aus durch wie noch nur zum zur am wenn dass sowie bis".split()),
    "fr": set("le la les des du de et est sont était été pour avec dans sur par au aux un une ne pas ou qui que ce ces se son sa ses leur leurs plus aussi comme entre cette ainsi mais".split()),
    "it": set("il lo la i gli le dei del della delle di e è sono era stato per con in su da al alla un una non o che si suo sua loro più anche come tra fra questo questa questi ma sul nel".split()),
}


@dataclass
class Cand:
    doc: str
    lang: str
    side: str            # "a" English / "b" target
    start: int           # doc token positions [start, end)
    end: int
    types: str           # e.g. "number", "paren", "foreign", "compare2", joined with "+"
    quote: str           # untokenized text of the span
    bead: int            # index into the doc's link list
    en_text: str         # the bead's English sentences (one per line)
    tg_text: str
    reason: str = ""


def _foreign(text: str, own: str, other: str) -> bool:
    words = [w.lower() for w in re.findall(r"[^\W\d_]+", text)]
    if len(words) < 4:
        return False
    sh = lambda lang: sum(w in STOP[lang] for w in words) / len(words)
    return sh(other) >= 0.25 and sh(other) > sh(own)


def _spans_from_flags(flags, lo, hi):
    """Maximal runs of flagged tokens inside [lo, hi); ASCII punctuation (-1) does not break a run."""
    out, cur = [], None
    for i in range(lo, hi):
        v = flags[i]
        if v > 0:
            cur = [cur[0], i + 1] if cur else [i, i + 1]
        elif v == -1:
            continue
        else:
            if cur:
                out.append(tuple(cur))
            cur = None
    if cur:
        out.append(tuple(cur))
    return out


def build_candidates(row: dict, aligner, num_flags, par_flags, compare2_spans: list[dict] | None = None) -> list[Cand]:
    lang, en, tg, links = doc_beads(row, aligner)
    ta, tb = row["text_a"].split(), row["text_b"].split()
    sent = {"a": en, "b": tg}
    toks = {"a": ta, "b": tb}
    raw: list[tuple[str, int, int, str, str]] = []   # (side, start, end, type, reason)
    bead_of = {("a", i): bi for bi, l in enumerate(links) for i in l.en}
    bead_of.update({("b", j): bi for bi, l in enumerate(links) for j in l.tgt})
    for bi, l in enumerate(links):
        if l.kind.startswith("unaligned"):
            continue
        for side, idxs in (("a", l.en), ("b", l.tgt)):
            for i in idxs:
                seg = sent[side][i]
                for flags, typ in ((num_flags[0 if side == "a" else 1], "number"), (par_flags[0 if side == "a" else 1], "paren")):
                    for a, b in _spans_from_flags(flags, seg.start, seg.end):
                        raw.append((side, a, b, typ, ""))
                own, other = ("en", lang) if side == "a" else (lang, "en")
                if _foreign(seg.text, own, other):
                    raw.append((side, seg.start, seg.end, "foreign", ""))
    for sp in compare2_spans or []:
        pos = sorted(sp["pos"])
        if not pos:
            continue
        side = sp["side"]
        for i, seg in enumerate(sent[side]):
            inside = [p for p in pos if seg.start <= p < seg.end]
            if inside and not links[bead_of[(side, i)]].kind.startswith("unaligned"):
                raw.append((side, min(inside), max(inside) + 1, "compare2", sp.get("reason", "")))
    # merge overlapping spans of the same side
    raw.sort(key=lambda x: (x[0], x[1], x[2]))
    merged: list[list] = []
    for side, a, b, typ, reason in raw:
        if merged and merged[-1][0] == side and a < merged[-1][2]:
            m = merged[-1]
            m[2] = max(m[2], b)
            if typ not in m[3].split("+"):
                m[3] += "+" + typ
            m[4] = m[4] or reason
        else:
            merged.append([side, a, b, typ, reason])
    out = []
    for side, a, b, typ, reason in merged:
        for i, seg in enumerate(sent[side]):          # the sentence containing the span start
            if seg.start <= a < seg.end:
                b = min(b, seg.end)
                rel = runs_to_records(toks[side][seg.start:seg.end], seg.text, [(a - seg.start, b - 1 - seg.start, 0.2)])[0]
                bi = bead_of[(side, i)]
                l = links[bi]
                out.append(Cand(row["id"], lang, side, a, b, typ, rel["quote"], bi, "\n".join(en[k].text for k in l.en), "\n".join(tg[k].text for k in l.tgt), reason))
                break
    return out
