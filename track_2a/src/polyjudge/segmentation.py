"""Sentence segmentation on the untokenized text + mapping back to gold token indices.

Gold labels are per whitespace token of the *tokenized* text. Tokenization only inserts/removes
whitespace, so the whitespace-free character streams of both versions are identical; sentence
boundaries found in the untokenized text are mapped to token boundaries via cumulative char counts.
"""
import re
import string
from bisect import bisect_left
from dataclasses import dataclass

PUNCT = set(string.punctuation)  # gold uses -1 for ASCII punctuation tokens only
ABBREV = {
    "en": {"no", "nos", "dr", "mr", "mrs", "ms", "prof", "st", "vs", "etc", "approx", "fig", "art", "para", "incl", "e.g", "i.e", "cf", "ca", "co"},
    "de": {"nr", "art", "abs", "bzw", "usw", "ca", "mio", "mrd", "dr", "prof", "vgl", "inkl", "ggf", "z.b", "d.h", "u.a", "z.t", "evtl", "str", "tel", "bsp", "lit", "kap", "ziff", "sog"},
    "fr": {"no", "nos", "art", "al", "dr", "mme", "mm", "env", "p.ex", "c.-à-d", "cf", "etc", "tél", "ch", "fr", "ss", "let", "ex", "inc"},
    "it": {"n", "art", "cpv", "dr", "sig", "es", "ecc", "ca", "tel", "cfr", "pag", "seg", "lett", "cap", "fr"},
}
_END = re.compile(r"([.!?…]+[)\]»\"'”’]*)(\s+)(?=[A-ZÀ-ÖØ-Þ0-9«„\"“(‹\[•–-])")
_LAST_WORD = re.compile(r"([^\s(«\"“„]+)\.[)\]»\"”’]*$")


@dataclass(frozen=True)
class Segment:
    idx: int
    text: str
    start: int  # first token index (inclusive)
    end: int    # last token index (exclusive)


def is_punct_token(tok: str) -> bool:
    return bool(tok) and all(ch in PUNCT for ch in tok)


def _is_abbrev(prefix: str, lang: str) -> bool:
    m = _LAST_WORD.search(prefix)
    if not m:
        return False
    w = m.group(1)
    wl = w.lower()
    if wl in ABBREV.get(lang, ()) or wl in ABBREV["en"]:
        return True
    if len(w) == 1 and w.isalpha() and w.isupper():  # initials: "J. Smith"
        return True
    if lang == "de" and re.fullmatch(r"\d{1,2}", w):  # ordinals: "am 5. Mai"
        return True
    return False


def split_sentences(untok: str, lang: str = "en") -> list[str]:
    """Split on line breaks, then on sentence-final punctuation followed by an uppercase/digit/opener."""
    out: list[str] = []
    for line in untok.split("\n"):
        line = line.strip()
        if not line:
            continue
        start = 0
        for m in _END.finditer(line):
            if _is_abbrev(line[start:m.end(1)], lang):
                continue
            out.append(line[start:m.end(1)].strip())
            start = m.end()
        if line[start:].strip():
            out.append(line[start:].strip())
    return out


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def segment_document(tokenized: str, untokenized: str, lang: str = "en") -> list[Segment]:
    """Sentences with their half-open token ranges; ranges tile [0, n_tokens) exactly once."""
    tokens = tokenized.split()
    n = len(tokens)
    if n == 0:
        return []
    sents = split_sentences(untokenized, lang)
    ends, acc = [], 0
    for t in tokens:
        acc += len(t)
        ends.append(acc)  # ends[i] = chars consumed after token i
    bounds, acc = [], 0
    for s in sents:
        acc += len(_squash(s))
        # token boundary at-or-after this char offset (exact when the char streams agree)
        bounds.append(min(bisect_left(ends, acc) + 1, n))
    bounds[-1] = n  # the last sentence always closes the document
    segs, prev = [], 0
    for s, b in zip(sents, bounds):
        if b <= prev:  # degenerate sentence with no tokens: skip
            continue
        segs.append(Segment(len(segs), s, prev, b))
        prev = b
    if prev < n:  # safety net: leftover tokens go to the last segment
        last = segs[-1]
        segs[-1] = Segment(last.idx, last.text, last.start, n)
    return segs


def to_token_labels(tokens: list[str], segments: list[Segment], seg_scores: list[float]) -> list[float]:
    """Broadcast segment scores to tokens; punctuation positions are fixed to -1."""
    assert len(segments) == len(seg_scores)
    labels = [0.0] * len(tokens)
    for seg, sc in zip(segments, seg_scores):
        for i in range(seg.start, seg.end):
            labels[i] = float(sc)
    return apply_punct_mask(tokens, labels)


def apply_punct_mask(tokens: list[str], token_scores: list[float]) -> list[float]:
    """Token-level scores (e.g. from evidence spans) -> final labels with -1 on punctuation."""
    assert len(tokens) == len(token_scores)
    return [-1.0 if is_punct_token(t) else float(s) for t, s in zip(tokens, token_scores)]
