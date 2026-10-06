"""70B judge for the sentence-alignment disagreements: sees the sentence and two proposals (aligner vs 8B) and decides which is right."""
from __future__ import annotations

import asyncio
import hashlib

from .diffs import extract_json
from .groups import LANG_NAME
from .lla import Item, clip

JUDGE_BATCH = 8
SYSTEM = """You verify sentence alignments between an English and a {LANG} version of the same Swiss government web page.
For every item you see one sentence and two proposals, A and B. A proposal either names the counterpart sentence in the other version or says that there is NO counterpart.
A counterpart must express the same content as the sentence (fully or in large part); the same topic, similar layout or a few shared words are not enough.
Decide which proposal is right. If the proposed counterpart does not carry the same content, the "NO counterpart" proposal is the right one.
Reply with JSON only: {{"results": [{{"id": 1, "verdict": "A"}}, {{"id": 2, "verdict": "B"}}]}} with one entry per item."""


def _flip(it: Item, doc_id: str) -> bool:
    """Deterministic random order of the two proposals (avoids a fixed A/B position bias)."""
    return hashlib.md5(f"{doc_id}-{it.side}-{it.idx}".encode()).digest()[0] % 2 == 1


def judge_messages(lang: str, batch: list[tuple[str, Item, list, list]]) -> list[dict]:
    lines = []
    for k, (doc_id, it, en, tg) in enumerate(batch, 1):
        src = "English" if it.side == "a" else LANG_NAME[lang]
        other = tg if it.side == "a" else en
        prop = lambda idxs: ("counterpart: " + " || ".join(f'"{clip(other[j].text, 300)}"' for j in idxs)) if idxs else "NO counterpart"
        pa, pb = (prop(it.heur), prop(it.llm)) if not _flip(it, doc_id) else (prop(it.llm), prop(it.heur))
        lines.append(f"[{k}] {src} sentence: {clip(it.text, 300)}\n    A: {pa}\n    B: {pb}")
    return [{"role": "system", "content": SYSTEM.format(LANG=LANG_NAME[lang])}, {"role": "user", "content": "\n".join(lines)}]


async def run_judge(client, model: str, cases: list[tuple[str, str, Item, list, list]]) -> tuple[dict, dict]:
    """cases: (doc_id, lang, item, en, tg) of status disagreements. Returns ({(doc_id, side, idx): True if 'no counterpart'}, stats)."""
    by_lang: dict[str, list] = {}
    for c in cases:
        by_lang.setdefault(c[1], []).append(c)
    batches = [(lang, lst[k:k + JUDGE_BATCH]) for lang, lst in by_lang.items() for k in range(0, len(lst), JUDGE_BATCH)]
    verdicts, stats = {}, {"calls": 0, "failed_batches": 0, "batches": len(batches)}

    async def one(lang, chunk):
        batch = [(d, it, en, tg) for d, _, it, en, tg in chunk]
        msgs = judge_messages(lang, batch)
        for attempt in range(2):
            r = await client.chat(msgs, model=model, temperature=0.0 if attempt == 0 else 0.3, seed=None if attempt == 0 else 5, max_tokens=250,
                                  json_mode=True, rep_guard=True, tag="judge")
            stats["calls"] += 1
            obj = extract_json(r.text)
            res = obj.get("results") if isinstance(obj, dict) else None
            if isinstance(res, list):
                got = {x["id"]: str(x.get("verdict", "")).strip().upper() for x in res if isinstance(x, dict) and isinstance(x.get("id"), int)}
                if all(got.get(k) in ("A", "B") for k in range(1, len(batch) + 1)):
                    for k, (d, it, en, tg) in enumerate(batch, 1):
                        flip = _flip(it, d)
                        chosen_is_heur = (got[k] == "A") != flip      # A is the aligner's proposal iff not flipped
                        chosen = it.heur if chosen_is_heur else it.llm
                        verdicts[(d, it.side, it.idx)] = (len(chosen) == 0)
                    return
        stats["failed_batches"] += 1

    await asyncio.gather(*(one(l, c) for l, c in batches))
    return verdicts, stats
