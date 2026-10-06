"""Sentence groups: the unit sent to the LLM (never a whole document).

A group = consecutive beads of the (model-free) sentence alignment. Mode "all" keeps every bead (used by the B1 baseline);
mode "aligned" drops unaligned (1-0 / 0-1) beads, because the hybrid scores those by the heuristic and never sends them to the LLM.
Each group remembers the doc token positions of its sentences, so mapped spans can be written back to the document exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .aligner import BEST_V2, HeuristicAlignerV2
from .heuristic import doc_segments

LANG_NAME = {"de": "German", "fr": "French", "it": "Italian"}
_ALIGNER = HeuristicAlignerV2(**BEST_V2)


@dataclass
class Group:
    doc_id: str
    lang: str
    bead_ids: list[int]
    kinds: list[str]
    en_text: str                      # sentences joined by "\n" (untokenized text, as shown to the model)
    tg_text: str
    en_tokens: list[str]              # tokenized tokens of those sentences, in order
    tg_tokens: list[str]
    en_pos: list[int]                 # position of each of those tokens in the document token array
    tg_pos: list[int]
    en_bead: list[int] = field(default_factory=list)   # bead index (position inside bead_ids) per en token
    tg_bead: list[int] = field(default_factory=list)

    @property
    def n_chars(self) -> int:
        return len(self.en_text) + len(self.tg_text)


def doc_beads(row: dict, aligner=_ALIGNER):
    lang = row["subset"].split("_")[1]
    en, tg = doc_segments(row["text_a"], row["text_a_untokenized"], row["text_b"], row["text_b_untokenized"], lang)
    return lang, en, tg, aligner.align(en, tg, lang)


def make_groups(row: dict, mode: str = "all", max_beads: int = 6, max_chars: int = 1400, aligner=_ALIGNER) -> list[Group]:
    assert mode in ("all", "aligned")
    lang, en, tg, links = doc_beads(row, aligner)
    ta, tb = row["text_a"].split(), row["text_b"].split()
    groups: list[Group] = []
    cur: list[int] = []

    def flush():
        if not cur:
            return
        g = Group(row["id"], lang, list(cur), [links[b].kind for b in cur], "", "", [], [], [], [])
        en_txt, tg_txt = [], []
        for k, b in enumerate(cur):
            for i in links[b].en:
                en_txt.append(en[i].text)
                g.en_tokens += ta[en[i].start:en[i].end]
                g.en_pos += range(en[i].start, en[i].end)
                g.en_bead += [k] * (en[i].end - en[i].start)
            for j in links[b].tgt:
                tg_txt.append(tg[j].text)
                g.tg_tokens += tb[tg[j].start:tg[j].end]
                g.tg_pos += range(tg[j].start, tg[j].end)
                g.tg_bead += [k] * (tg[j].end - tg[j].start)
        g.en_text, g.tg_text = "\n".join(en_txt), "\n".join(tg_txt)
        groups.append(g)
        cur.clear()

    def bead_chars(b):
        return sum(len(en[i].text) for i in links[b].en) + sum(len(tg[j].text) for j in links[b].tgt) + 2

    chars = 0
    for b, l in enumerate(links):
        if mode == "aligned" and l.kind.startswith("unaligned"):
            continue
        bc = bead_chars(b)
        if cur and (len(cur) >= max_beads or chars + bc > max_chars):
            flush()
            chars = 0
        cur.append(b)
        chars += bc
    flush()
    return groups


def group_labels(row: dict, g: Group):
    """Gold labels of the group's tokens (en side, target side)."""
    return [row["labels_a"][i] for i in g.en_pos], [row["labels_b"][i] for i in g.tg_pos]
