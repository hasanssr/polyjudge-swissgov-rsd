"""LLM-assisted sentence alignment (Task E).

For every English sentence the model sees candidate target sentences (own aligner partners + neighbours + top-scoring others) and answers which
candidates carry the same content, or none; the same is done in the reverse direction for target sentences. Strict JSON, batched (BATCH items per call).
A sentence is "aligned" for the LLM iff its answer is non-empty. Presence score combines this with the aligner's verdict.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from .aligner import _jaccard, evidence, stems
from .diffs import extract_json
from .groups import LANG_NAME
from .heuristic import doc_segments

BATCH = 8
MAX_CANDS = 6

SYSTEM = """You help align the sentences of two versions of the same Swiss government web page: an {SRC} version and a {DST} version.
Every {SRC} sentence is an item with a number [1], [2], ...; its candidate {DST} sentences are labelled with letters A, B, C, .... Decide which candidates carry the same content as the {SRC} sentence.
Rules:
- A candidate counts if it expresses the same statement, title, label or menu entry (fully or in large part), even with different words, a different word order, or one added/missing detail.
- A candidate that merely shares the topic or a few words does NOT count.
- Most sentences DO have a counterpart among their candidates. Use an empty list only if you are sure that no candidate expresses the same content.
- Decide every item on its own. Several candidates may count if the sentence was split into several sentences.
Reply with JSON only: {{"results": [{{"id": <item number>, "match": [<letters of the matching candidates>]}}, ...]}} with exactly one entry per item; "match" is an empty list when no candidate matches. Example: {{"id": 3, "match": ["B"]}} or {{"id": 4, "match": []}}."""

# invented demonstration (not taken from the data): item 1 -> candidate 2, item 2 -> candidate 1, item 3 -> none
_DEMO = {
    "de": dict(office="Das Amt ist an Feiertagen geschlossen.", other="Ein ganz anderer Satz über Bankkonten.", contact="Kontakt", rules="Die neuen Regeln treten 2031 in Kraft."),
    "fr": dict(office="Le bureau est fermé les jours fériés.", other="Une toute autre phrase sur les comptes bancaires.", contact="Contact", rules="Les nouvelles règles entrent en vigueur en 2031."),
    "it": dict(office="L'ufficio è chiuso nei giorni festivi.", other="Una frase completamente diversa sui conti bancari.", contact="Contatto", rules="Le nuove regole entrano in vigore nel 2031."),
}
_EN = dict(office="The office is closed on public holidays.", other="A completely different sentence about bank accounts.", contact="Contact", rules="The new rules enter into force in 2031.")
DEMO_ANSWER = '{"results": [{"id": 1, "match": ["B"]}, {"id": 2, "match": ["A"]}, {"id": 3, "match": []}]}'


def _demo_user(lang: str, side: str) -> str:
    t, e = _DEMO[lang], _EN
    if side == "a":    # English sentences, target-language candidates
        src, items = "English", [(e["office"], [t["other"], t["office"]]), (e["contact"], [t["contact"], t["office"]]), (e["rules"], [t["contact"], t["other"]])]
    else:              # target-language sentences, English candidates
        src, items = LANG_NAME[lang], [(t["office"], [e["other"], e["office"]]), (t["contact"], [e["contact"], e["office"]]), (t["rules"], [e["contact"], e["other"]])]
    lines = []
    for k, (txt, cs) in enumerate(items, 1):
        lines.append(f"[{k}] {src}: {txt}" + chr(10) + "    CANDIDATES:")
        lines += [f"    {chr(64 + c)}. {x}" for c, x in enumerate(cs, 1)]
    return chr(10).join(lines)



PROMPT_VERSION = "lla-v4"
ANCHORS = 2
# two invented anchor items prepended to every batch (known answers: both have a counterpart); answers are discarded but checked
_ANCH = {
    "de": [("Applications must be submitted in writing.", "Gesuche sind schriftlich einzureichen."), ("Download", "Herunterladen")],
    "fr": [("Applications must be submitted in writing.", "Les demandes doivent être déposées par écrit."), ("Download", "Télécharger")],
    "it": [("Applications must be submitted in writing.", "Le domande devono essere presentate per iscritto."), ("Download", "Scaricare")],
}
_ANCH_OTHER = {"de": "Die Preise werden jährlich angepasst.", "fr": "Les prix sont adaptés chaque année.", "it": "I prezzi sono adeguati ogni anno."}
_ANCH_OTHER_EN = "Prices are adjusted every year."


def anchor_lines(lang: str, side: str) -> list[str]:
    out = []
    for k, (en_t, dst_t) in enumerate(_ANCH[lang], 1):
        if side == "a":
            src, text, cands = "English", en_t, [_ANCH_OTHER[lang], dst_t] if k == 1 else [dst_t, _ANCH_OTHER[lang]]
        else:
            src, text, cands = LANG_NAME[lang], dst_t, [_ANCH_OTHER_EN, en_t] if k == 1 else [en_t, _ANCH_OTHER_EN]
        out.append(f"[{k}] {src}: {text}" + chr(10) + "    CANDIDATES:")
        out += [f"    {chr(64 + c)}. {x}" for c, x in enumerate(cands, 1)]
    return out


def anchor_expected(side: str) -> dict:
    return {1: [2], 2: [1]}


@dataclass
class Item:
    side: str                 # "a" (English sentence, candidates = target) or "b" (target sentence, candidates = English)
    idx: int                  # sentence index in its own side
    text: str
    cands: list[int]          # candidate sentence indices on the other side (document order)
    heur: list[int]           # the aligner's partners (other side); [] = aligner says unaligned
    llm: list[int] | None = None   # LLM answer (None = no valid answer)
    call_failed: bool = False

    @property
    def heur_unaligned(self) -> bool:
        return not self.heur

    @property
    def llm_unaligned(self) -> bool | None:
        return None if self.llm is None else (len(self.llm) == 0)


def _score(a: str, b: str) -> float:
    return evidence(a, b) + _jaccard(stems(a), stems(b))


def build_items(row: dict, lang: str, aligner) -> tuple[list[Item], list, list, list]:
    en, tg = doc_segments(row["text_a"], row["text_a_untokenized"], row["text_b"], row["text_b_untokenized"], lang)
    links = aligner.align(en, tg, lang)
    partners_a = {i: list(l.tgt) for l in links for i in l.en}
    partners_b = {j: list(l.en) for l in links for j in l.tgt}
    cum = lambda segs: [sum(len(s.text) + 1 for s in segs[:k]) for k in range(len(segs) + 1)]
    ce, ct = cum(en), cum(tg)
    items: list[Item] = []
    for side, own, other, part, co, cx in (("a", en, tg, partners_a, ce, ct), ("b", tg, en, partners_b, ct, ce)):
        for i, s in enumerate(own):
            heur = part.get(i, [])
            cands = set(heur)
            if heur:
                cands |= {j for j in (min(heur) - 1, max(heur) + 1) if 0 <= j < len(other)}
            else:   # unaligned: neighbours around the expected relative position
                rel = (co[i] + len(s.text) / 2) / max(1, co[-1])
                c = min(range(len(other)), key=lambda j: abs((cx[j] + len(other[j].text) / 2) / max(1, cx[-1]) - rel))
                cands |= {j for j in (c - 1, c, c + 1) if 0 <= j < len(other)}
            ranked = sorted((j for j in range(len(other)) if j not in cands),
                            key=lambda j: -(_score(s.text, other[j].text) if side == "a" else _score(other[j].text, s.text)))
            cands |= set(ranked[:2])
            cands = sorted(cands)
            if len(cands) > MAX_CANDS:   # keep own partners first, then the closest to the expected position
                keep = list(heur) + [j for j in cands if j not in heur]
                cands = sorted(keep[:MAX_CANDS])
            items.append(Item(side, i, s.text, cands, heur))
    return items, en, tg, links


def clip(s: str, n: int = 380) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def make_messages(lang: str, batch: list[Item], en, tg) -> list[dict]:
    side = batch[0].side
    src, dst = ("English", LANG_NAME[lang]) if side == "a" else (LANG_NAME[lang], "English")
    other = tg if side == "a" else en
    lines = anchor_lines(lang, side)
    for k, it in enumerate(batch, ANCHORS + 1):
        lines.append(f"[{k}] {src}: {clip(it.text)}\n    CANDIDATES:")
        lines += [f"    {chr(64 + c)}. {clip(other[j].text)}" for c, j in enumerate(it.cands, 1)]
    return [{"role": "system", "content": SYSTEM.format(SRC=src, DST=dst)},
            {"role": "user", "content": _demo_user(lang, side)}, {"role": "assistant", "content": DEMO_ANSWER},
            {"role": "user", "content": "\n".join(lines)}]


def _to_num(x):
    """'B' / 'b' -> 2; ints pass through; anything else -> 0 (invalid)."""
    if isinstance(x, str) and len(x.strip()) == 1 and x.strip().isalpha():
        return ord(x.strip().upper()) - 64
    return x if isinstance(x, int) else 0


def parse_matches(text: str, batch: list[Item], anchors: dict | None = None) -> bool:
    """Fill item.llm from the answer; True iff every item received a valid answer."""
    obj = extract_json(text)
    res = obj.get("results") if isinstance(obj, dict) else None
    if not isinstance(res, list):
        return False
    got = {}
    for r in res:
        if isinstance(r, dict) and isinstance(r.get("id"), int) and isinstance(r.get("match"), list):
            got[r["id"]] = [_to_num(x) for x in r["match"]]
        elif isinstance(r, dict) and isinstance(r.get("id"), int) and isinstance(r.get("match"), str) and r["match"].strip().upper() == "NONE":
            got[r["id"]] = []
    ok = True
    if anchors is not None:
        anchors["seen"] = anchors.get("seen", 0) + ANCHORS
        anchors["ok"] = anchors.get("ok", 0) + sum(1 for k, v in anchor_expected(batch[0].side).items() if got.get(k) == v)
    for k, it in enumerate(batch, ANCHORS + 1):
        m = got.get(k)
        if m is None or not all(isinstance(x, int) and 1 <= x <= len(it.cands) for x in m):
            ok = False
            continue
        it.llm = sorted({it.cands[x - 1] for x in m})
    return ok


async def run_items(client, model: str, lang: str, items: list[Item], en, tg, batch_size: int = BATCH) -> dict:
    """One call per batch of BATCH items of the same side; one retry (temperature 0.3) for batches with an unusable answer."""
    batches = [[it for it in items if it.side == s][k:k + batch_size] for s in ("a", "b") for k in range(0, sum(1 for it in items if it.side == s), batch_size)]
    stats = {"calls": 0, "first_failed": 0, "final_failed": 0, "batches": len(batches), "anchor_seen": 0, "anchor_ok": 0}

    async def one(batch):
        msgs = make_messages(lang, batch, en, tg)
        for attempt in range(2):
            r = await client.chat(msgs, model=model, temperature=0.0 if attempt == 0 else 0.3, seed=None if attempt == 0 else 11, max_tokens=300,
                                  json_mode=True, rep_guard=True, tag=f"align-{lang}")
            stats["calls"] += 1
            anc = {}
            good = not (r.aborted or r.finish_reason == "length") and parse_matches(r.text, batch, anc)
            stats["anchor_seen"] += anc.get("seen", 0)
            stats["anchor_ok"] += anc.get("ok", 0)
            if good:
                return
            if attempt == 0:
                stats["first_failed"] += 1
        stats["final_failed"] += 1
        for it in batch:
            it.call_failed = it.llm is None

    await asyncio.gather(*(one(b) for b in batches))
    return stats


def presence_labels(row: dict, items: list[Item], en, tg, judged: dict | None = None):
    """Token scores: both unaligned -> 1.0; disagree -> 0.5 (or the judge's verdict 1.0 / 0.0); both aligned -> 0. Missing LLM answer -> aligner verdict.
    judged: {(side, idx): True if judge says 'no counterpart'}. Returns (labels_a, labels_b) with -1 on punctuation."""
    from .segmentation import apply_punct_mask
    la, lb = [0.0] * len(row["text_a"].split()), [0.0] * len(row["text_b"].split())
    for it in items:
        if it.llm is None:
            v = 1.0 if it.heur_unaligned else 0.0
        elif judged is not None and (it.side, it.idx) in judged:
            v = 1.0 if judged[(it.side, it.idx)] else 0.0
        elif it.heur_unaligned and it.llm_unaligned:
            v = 1.0
        elif it.heur_unaligned != it.llm_unaligned:
            v = 0.5
        else:
            v = 0.0
        seg = (en if it.side == "a" else tg)[it.idx]
        arr = la if it.side == "a" else lb
        arr[seg.start:seg.end] = [v] * (seg.end - seg.start)
    return apply_punct_mask(row["text_a"].split(), la), apply_punct_mask(row["text_b"].split(), lb)
