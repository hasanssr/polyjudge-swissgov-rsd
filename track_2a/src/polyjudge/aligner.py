"""EN -> target sentence alignment behind a swappable interface.

`SentenceAligner.align` takes the two segment lists and returns monotone, non-overlapping links
(beads). `HeuristicAligner` is model-free; an LLM-based aligner can implement the same interface.
"""
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .segmentation import Segment


@dataclass(frozen=True)
class AlignLink:
    en: tuple[int, ...]   # indices into the EN segment list (may be empty = target-only)
    tgt: tuple[int, ...]  # indices into the target segment list (may be empty = EN-only / omission)

    @property
    def kind(self) -> str:
        a, b = len(self.en), len(self.tgt)
        if a and not b:
            return "unaligned_en"
        if b and not a:
            return "unaligned_tgt"
        if a == 1 and b == 1:
            return "1-to-1"
        if a == 1:
            return "1-to-many"
        if b == 1:
            return "many-to-1"
        return "many-to-many"


class SentenceAligner(ABC):
    @abstractmethod
    def align(self, en: list[Segment], tgt: list[Segment], lang: str) -> list[AlignLink]:
        """Monotone alignment covering every EN and target segment exactly once."""


_NUM = re.compile(r"\d+(?:[.,'’]\d+)*")
_CAPS = re.compile(r"(?<!^)(?<![.!?]\s)\b([A-ZÀ-ÖØ-Þ][\wÀ-ÿ-]{3,}|[A-Z]{2,})\b")
# expected target/EN character-length ratio per language (rough, from typical DE/FR/IT vs EN text)
EXPECTED_RATIO = {"de": 1.10, "fr": 1.18, "it": 1.12}
BEADS = [(1, 1, 0.0), (1, 2, -0.7), (2, 1, -0.7), (2, 2, -1.4), (1, 0, -1.6), (0, 1, -1.6), (1, 3, -1.8), (3, 1, -1.8)]


def _numbers(s: str) -> set[str]:
    return {re.sub(r"[.,'’]", "", m) for m in _NUM.findall(s)}


def _entities(s: str) -> set[str]:
    return {m.lower()[:5] for m in _CAPS.findall(s)}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


class HeuristicAligner(SentenceAligner):
    """Gale-Church-style monotone DP using length ratio, shared numbers, shared name-like strings
    and relative position. Beads: 1-1, 1-2, 2-1, 2-2, 1-3, 3-1, 1-0, 0-1."""

    def __init__(self, w_len=1.0, w_num=2.0, w_ent=1.5, w_pos=4.0, sigma=0.45):
        self.w_len, self.w_num, self.w_ent, self.w_pos, self.sigma = w_len, w_num, w_ent, w_pos, sigma

    def _bead_score(self, en_txt, tgt_txt, en_pos, tgt_pos, lang) -> float:
        ratio = (len(tgt_txt) + 1) / (len(en_txt) + 1)
        z = math.log(ratio / EXPECTED_RATIO.get(lang, 1.1)) / self.sigma
        s = -self.w_len * 0.5 * z * z
        ne, nt = _numbers(en_txt), _numbers(tgt_txt)
        if ne or nt:
            s += self.w_num * (_jaccard(ne, nt) - 0.25)  # shared numbers help; mismatching ones hurt a bit
        s += self.w_ent * _jaccard(_entities(en_txt), _entities(tgt_txt))
        s -= self.w_pos * abs(en_pos - tgt_pos)
        return s

    def align(self, en, tgt, lang):
        n, m = len(en), len(tgt)
        if n == 0 or m == 0:
            return [AlignLink(tuple(range(n)), tuple(range(m)))] if n or m else []
        # relative position of a span's midpoint, measured in characters so long sentences weigh more
        def cum(segs):
            c = [0]
            for g in segs:
                c.append(c[-1] + len(g.text) + 1)
            return c
        ce, ct = cum(en), cum(tgt)
        txt = lambda segs, i, j: " ".join(g.text for g in segs[i:j])
        NEG = -1e18
        best = [[NEG] * (m + 1) for _ in range(n + 1)]
        back = [[None] * (m + 1) for _ in range(n + 1)]
        best[0][0] = 0.0
        for i in range(n + 1):
            for j in range(m + 1):
                if best[i][j] == NEG:
                    continue
                for a, b, prior in BEADS:
                    ni, nj = i + a, j + b
                    if ni > n or nj > m:
                        continue
                    if a and b:
                        pe = (ce[i] + ce[ni]) / 2 / ce[n]
                        pt = (ct[j] + ct[nj]) / 2 / ct[m]
                        sc = prior + self._bead_score(txt(en, i, ni), txt(tgt, j, nj), pe, pt, lang)
                    else:
                        sc = prior
                    if best[i][j] + sc > best[ni][nj]:
                        best[ni][nj] = best[i][j] + sc
                        back[ni][nj] = (i, j)
        links, i, j = [], n, m
        while (i, j) != (0, 0):
            pi, pj = back[i][j]
            links.append(AlignLink(tuple(range(pi, i)), tuple(range(pj, j))))
            i, j = pi, pj
        return links[::-1]


# ====================================================================================================
# V2: richer evidence + penalty for weakly supported complex beads + length-budget trimming + rescue of
# out-of-order sentences. Tuned on prompt_pool only (see scripts/11_eval_aligners_pool.py).
# ====================================================================================================
MONTHS = {
    "01": "january januar janvier gennaio jan janv gen", "02": "february februar février fevrier febbraio feb févr",
    "03": "march märz mars marzo mär", "04": "april avril aprile apr avr", "05": "may mai maggio mai",
    "06": "june juni juin giugno jun giu", "07": "july juli juillet luglio jul juil lug",
    "08": "august août aout agosto aug", "09": "september septembre settembre sep sept set",
    "10": "october oktober octobre ottobre oct okt ott", "11": "november novembre nov", "12": "december dezember décembre dicembre dec dez",
}
MONTH_LOOKUP = {w: k for k, ws in MONTHS.items() for w in ws.split()}
_THOUSANDS = re.compile(r"(?<=\d)['’‘ ](?=\d{3}\b)")
_ACRO = re.compile(r"\b[A-ZÀ-Ö]{2,}[a-z]?\b")
_WORD = re.compile(r"[A-Za-zÀ-ÿ]+")


def numbers_v2(s: str) -> set[str]:
    s = _THOUSANDS.sub("", s)
    out = {m.lstrip("0") or "0" for m in re.findall(r"\d+", s)}
    out |= {MONTH_LOOKUP[w.lower()].lstrip("0") for w in _WORD.findall(s) if len(w) >= 3 and w.lower() in MONTH_LOOKUP}
    return {x for x in out if len(x) >= 2 or x in {str(i) for i in range(1, 10)}} - {"0"}


def acronyms(s: str) -> set[str]:
    return {a.lower() for a in _ACRO.findall(s) if a.lower() not in {"pdf", "mb", "kb", "de", "fr", "it", "en"}}


def evidence(en_txt: str, tgt_txt: str) -> float:
    """Model-free cross-lingual support: shared numbers/dates, shared name-like strings, shared acronyms."""
    ne, nt = numbers_v2(en_txt), numbers_v2(tgt_txt)
    return 2.0 * _jaccard(ne, nt) + 1.5 * _jaccard(_entities(en_txt), _entities(tgt_txt)) + 2.0 * _jaccard(acronyms(en_txt), acronyms(tgt_txt))


class HeuristicAlignerV2(HeuristicAligner):
    _feat_cache: dict = {}

    def __init__(self, complex_penalty=2.0, tau=1.0, trim=True, trim_ratio=0.8, rescue=True, rescue_min=0.7, p11=0.0, p11_minw=6, **kw):
        super().__init__(**kw)
        self.p11, self.p11_minw = p11, p11_minw
        self.complex_penalty, self.tau, self.trim, self.trim_ratio = complex_penalty, tau, trim, trim_ratio
        self.rescue, self.rescue_min = rescue, rescue_min

    @staticmethod
    def _sent_feats(segs):
        return [(numbers_v2(g.text), _entities(g.text), acronyms(g.text)) for g in segs]

    def align(self, en, tgt, lang):
        n, m = len(en), len(tgt)
        if n == 0 or m == 0:
            return [AlignLink(tuple(range(n)), tuple(range(m)))] if n or m else []
        key = (id(en), id(tgt))  # segment lists are cached per document by `heuristic.doc_segments`
        if key not in self._feat_cache:
            self._feat_cache[key] = (self._sent_feats(en), self._sent_feats(tgt), {})
        fe, ft, memo = self._feat_cache[key]

        def ev(i, ni, j, nj):
            k = (i, ni, j, nj)
            if k not in memo:
                u = lambda feats, a, b, t: set().union(*(f[t] for f in feats[a:b]))
                memo[k] = (2.0 * _jaccard(u(fe, i, ni, 0), u(ft, j, nj, 0)) + 1.5 * _jaccard(u(fe, i, ni, 1), u(ft, j, nj, 1))
                           + 2.0 * _jaccard(u(fe, i, ni, 2), u(ft, j, nj, 2)))
            return memo[k]

        def cum(segs):
            c = [0]
            for g in segs:
                c.append(c[-1] + len(g.text) + 1)
            return c
        ce, ct = cum(en), cum(tgt)
        wen, wtg = [len(g.text.split()) for g in en], [len(g.text.split()) for g in tgt]
        exp = EXPECTED_RATIO.get(lang, 1.1)
        NEG = -1e18
        best = [[NEG] * (m + 1) for _ in range(n + 1)]
        back = [[None] * (m + 1) for _ in range(n + 1)]
        best[0][0] = 0.0
        for i in range(n + 1):
            for j in range(m + 1):
                if best[i][j] == NEG:
                    continue
                for a, b, prior in BEADS:
                    ni, nj = i + a, j + b
                    if ni > n or nj > m:
                        continue
                    if a and b:
                        pe = (ce[i] + ce[ni]) / 2 / ce[n]
                        pt = (ct[j] + ct[nj]) / 2 / ct[m]
                        ratio = (ct[nj] - ct[j]) / max(1, ce[ni] - ce[i])
                        z = math.log(ratio / exp) / self.sigma
                        e = ev(i, ni, j, nj)
                        sc = prior - self.w_len * 0.5 * z * z + e - self.w_pos * abs(pe - pt)
                        if a + b > 2:
                            sc -= self.complex_penalty * (1.0 - min(1.0, e / self.tau))
                        elif self.p11 and e == 0 and wen[i] >= self.p11_minw and wtg[j] >= self.p11_minw:
                            sc -= self.p11  # long 1-1 pair without any cross-lingual evidence is suspicious
                    else:
                        sc = prior
                    if best[i][j] + sc > best[ni][nj]:
                        best[ni][nj] = best[i][j] + sc
                        back[ni][nj] = (i, j)
        links, i, j = [], n, m
        while (i, j) != (0, 0):
            pi, pj = back[i][j]
            links.append(AlignLink(tuple(range(pi, i)), tuple(range(pj, j))))
            i, j = pi, pj
        links = links[::-1]
        if self.trim:
            links = self._trim(links, en, tgt, lang)
        if self.rescue:
            links = self._rescue(links, en, tgt, lang)
        return links

    # 1-to-many / many-to-1: keep only the sentences needed to explain the single sentence's length, best evidence first
    def _trim(self, links, en, tgt, lang):
        out = []
        for l in links:
            a, b = len(l.en), len(l.tgt)
            if (a == 1 and b >= 2) or (b == 1 and a >= 2):
                single_is_en = a == 1
                one = en[l.en[0]] if single_is_en else tgt[l.tgt[0]]
                many_idx = l.tgt if single_is_en else l.en
                many = tgt if single_is_en else en
                exp = EXPECTED_RATIO.get(lang, 1.1)
                need = len(one.text) * (exp if single_is_en else 1 / exp) * self.trim_ratio
                order = sorted(many_idx, key=lambda k: -evidence(one.text, many[k].text) if single_is_en else -evidence(many[k].text, one.text))
                kept, total = [], 0
                for k in order:
                    if total >= need and kept:
                        break
                    kept.append(k)
                    total += len(many[k].text)
                kept = tuple(sorted(kept))
                dropped = [k for k in many_idx if k not in kept]
                if dropped:
                    # extras become single unaligned sentences, in document order around the kept bead
                    before = [k for k in dropped if k < kept[0]]
                    after = [k for k in dropped if k > kept[0]]
                    mk = (lambda ks: [AlignLink((), (k,)) if single_is_en else AlignLink((k,), ()) for k in ks])
                    core = AlignLink(l.en, kept) if single_is_en else AlignLink(kept, l.tgt)
                    out += mk(before) + [core] + mk(after)
                    continue
            out.append(l)
        return out

    # out-of-order rescue: pair unaligned EN sentences with unaligned target sentences that have real evidence
    def _rescue(self, links, en, tgt, lang):
        ue = [l.en[0] for l in links if l.kind == "unaligned_en"]
        ut = [l.tgt[0] for l in links if l.kind == "unaligned_tgt"]
        cands = sorted(((evidence(en[i].text, tgt[j].text), i, j) for i in ue for j in ut), reverse=True)
        used_e, used_t, pairs = set(), set(), []
        for ev, i, j in cands:
            if ev < self.rescue_min:
                break
            if i in used_e or j in used_t:
                continue
            ratio = (len(tgt[j].text) + 1) / (len(en[i].text) + 1)
            if not 0.4 <= ratio / EXPECTED_RATIO.get(lang, 1.1) <= 2.5:
                continue
            used_e.add(i)
            used_t.add(j)
            pairs.append(AlignLink((i,), (j,)))
        if not pairs:
            return links
        keep = [l for l in links if not ((l.kind == "unaligned_en" and l.en[0] in used_e) or (l.kind == "unaligned_tgt" and l.tgt[0] in used_t))]
        return sorted(keep + pairs, key=lambda l: min(l.en) if l.en else (min(l.tgt) / max(1, len(tgt)) * len(en)))


# Config chosen on prompt_pool only (scripts/11_eval_aligners_pool.py): plateau for complex_penalty in 1.5..3.0;
# trimming and rescue gave no gain (kept implemented, off by default).
BEST_V2 = dict(complex_penalty=1.5, tau=1.0, trim=False, rescue=False)


# ====================================================================================================
# V3 (Task D): V2 + stem-overlap evidence (romance/English cognates), penalty for short zero-evidence pairs (navigation / boilerplate lines)
# and for long zero-evidence 1-1 pairs. Parameters are tuned on the 'tune' prompt_pool chunks only (never on eval12 / eval2).
# ====================================================================================================
import unicodedata as _ud

_WORD3 = re.compile(r"[^\W\d_]{6,}", re.U)


def stems(s: str) -> set[str]:
    """5-letter, accent-stripped, lower-cased prefixes of words with >= 6 letters (cheap cognate signal across EN/FR/IT, partly DE)."""
    out = set()
    for w in _WORD3.findall(s):
        w = "".join(c for c in _ud.normalize("NFD", w.lower()) if _ud.category(c) != "Mn")
        out.add(w[:5])
    return out


class HeuristicAlignerV3(HeuristicAlignerV2):
    def __init__(self, w_stem=1.0, p_short=0.0, short_words=6, p11=0.0, p11_minw=8, **kw):
        super().__init__(**kw)
        self.w_stem, self.p_short, self.short_words, self.p11_, self.p11_minw_ = w_stem, p_short, short_words, p11, p11_minw

    @staticmethod
    def _sent_feats(segs):
        return [(numbers_v2(g.text), _entities(g.text), acronyms(g.text), stems(g.text)) for g in segs]

    def align(self, en, tgt, lang):
        n, m = len(en), len(tgt)
        if n == 0 or m == 0:
            return [AlignLink(tuple(range(n)), tuple(range(m)))] if n or m else []
        key = ("v3", id(en), id(tgt))
        if key not in self._feat_cache:
            self._feat_cache[key] = (self._sent_feats(en), self._sent_feats(tgt), {})
        fe, ft, memo = self._feat_cache[key]

        def ev(i, ni, j, nj):
            k = (i, ni, j, nj)
            if k not in memo:
                u = lambda feats, a, b, t: set().union(*(f[t] for f in feats[a:b]))
                memo[k] = (2.0 * _jaccard(u(fe, i, ni, 0), u(ft, j, nj, 0)) + 1.5 * _jaccard(u(fe, i, ni, 1), u(ft, j, nj, 1))
                           + 2.0 * _jaccard(u(fe, i, ni, 2), u(ft, j, nj, 2)))
                memo[("s",) + k] = _jaccard(u(fe, i, ni, 3), u(ft, j, nj, 3))
            return memo[k] + self.w_stem * memo[("s",) + k], memo[k]

        def cum(segs):
            c = [0]
            for g in segs:
                c.append(c[-1] + len(g.text) + 1)
            return c
        ce, ct = cum(en), cum(tgt)
        wen, wtg = [len(g.text.split()) for g in en], [len(g.text.split()) for g in tgt]
        cw_e, cw_t = [0], [0]
        for w in wen: cw_e.append(cw_e[-1] + w)
        for w in wtg: cw_t.append(cw_t[-1] + w)
        exp = EXPECTED_RATIO.get(lang, 1.1)
        NEG = -1e18
        best = [[NEG] * (m + 1) for _ in range(n + 1)]
        back = [[None] * (m + 1) for _ in range(n + 1)]
        best[0][0] = 0.0
        for i in range(n + 1):
            for j in range(m + 1):
                if best[i][j] == NEG:
                    continue
                for a, b, prior in BEADS:
                    ni, nj = i + a, j + b
                    if ni > n or nj > m:
                        continue
                    if a and b:
                        pe = (ce[i] + ce[ni]) / 2 / ce[n]
                        pt = (ct[j] + ct[nj]) / 2 / ct[m]
                        z = math.log(((ct[nj] - ct[j]) / max(1, ce[ni] - ce[i])) / exp) / self.sigma
                        e, e_base = ev(i, ni, j, nj)
                        sc = prior - self.w_len * 0.5 * z * z + e - self.w_pos * abs(pe - pt)
                        if a + b > 2:
                            sc -= self.complex_penalty * (1.0 - min(1.0, e / self.tau))
                        elif e == 0:
                            words = max(cw_e[ni] - cw_e[i], cw_t[nj] - cw_t[j])
                            if words <= self.short_words:
                                sc -= self.p_short                    # short line without any cross-lingual evidence: probably navigation/boilerplate
                            elif min(cw_e[ni] - cw_e[i], cw_t[nj] - cw_t[j]) >= self.p11_minw_:
                                sc -= self.p11_
                    else:
                        sc = prior
                    if best[i][j] + sc > best[ni][nj]:
                        best[ni][nj] = best[i][j] + sc
                        back[ni][nj] = (i, j)
        links, i, j = [], n, m
        while (i, j) != (0, 0):
            pi, pj = back[i][j]
            links.append(AlignLink(tuple(range(pi, i)), tuple(range(pj, j))))
            i, j = pi, pj
        return links[::-1]

# Task D result: best on the tuning set (see scripts/24_tune_v3.py): stem evidence off, short-line penalty off, long zero-evidence 1-1 penalty 1.5.
BEST_V3 = dict(w_stem=0.0, p_short=0.0, p11=1.5, **BEST_V2)
