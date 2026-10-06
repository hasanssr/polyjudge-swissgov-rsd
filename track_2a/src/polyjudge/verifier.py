"""LLM verifier for candidate flags: for each candidate, 'does this reflect a real meaning difference?' -> reason first, then yes/no and a level 0.2..1.0.
Candidates are batched by aligned pair (bead): several pairs and their candidates per call. Strict JSON; candidate ids are K1, K2, ... (letters+digits, never
plain numbers, to avoid the item/candidate numbering confusion seen in the alignment experiment)."""
from __future__ import annotations

import asyncio

from .candidates import Cand
from .diffs import extract_json
from .groups import LANG_NAME
from .prompts import RUBRIC
from .spans import snap_severity

MAX_PAIRS, MAX_CHARS, MAX_FLAGS = 6, 3200, 12

SYSTEM = """You check flags on aligned sentences of an English and a {LANG} version of the same Swiss government web page.
Each PAIR shows the English and the {LANG} sentences that correspond to each other, followed by FLAGS: short spans (marked K1, K2, ...) that an automatic detector found suspicious.
For every flag decide whether it reflects a REAL difference in MEANING between the two versions.
REAL: the flagged span carries content that is missing, added or different in the other version (a different number, date or amount; an extra clause or remark; a remark in brackets that exists in only one version; different names or negation).
NOT REAL: wording choices, a freely translated title or organisation name, the same number or date in another notation (3.043 vs 3,043; 05.12.2014 vs 5 December 2014), abbreviations, text left in another language on both sides, formatting.
Write the "reason" FIRST: one short sentence that names the concrete difference, or says why the two versions agree. Then "real": "yes" or "no". If yes, give a "level" by this scale:
{RUBRIC}
Reply with JSON only: {{"results": [{{"id": "K1", "reason": "...", "real": "no"}}, {{"id": "K2", "reason": "...", "real": "yes", "level": 0.4}}]}} with one entry per flag.
Examples (invented):
PAIR: ENGLISH: The fee is CHF 50. | {LANG}: Die Gebühr beträgt 80 Franken.  FLAG K1 [English] "CHF 50" -> {{"id": "K1", "reason": "English gives CHF 50, the other version 80 francs.", "real": "yes", "level": 0.8}}
PAIR: ENGLISH: The office opens on 5 March. | {LANG}: Das Amt öffnet am 05.03.  FLAG K2 [English] "5" -> {{"id": "K2", "reason": "Same date in a different notation.", "real": "no"}}
PAIR: ENGLISH: Apply online (in German). | {LANG}: Online anmelden.  FLAG K3 [English] "in German" -> {{"id": "K3", "reason": "The remark about the language exists only in the English version.", "real": "yes", "level": 0.4}}"""


def _batches(cands: list[Cand]):
    """Group candidate indices by pair; fill calls with up to MAX_PAIRS pairs / MAX_CHARS characters / MAX_FLAGS flags."""
    pairs: dict[tuple[str, int], list[int]] = {}
    for i, c in enumerate(cands):
        pairs.setdefault((c.doc, c.bead), []).append(i)
    out, cur, chars, flags = [], [], 0, 0
    for key, idxs in pairs.items():
        c0 = cands[idxs[0]]
        size = len(c0.en_text) + len(c0.tg_text)
        if cur and (len(cur) >= MAX_PAIRS or chars + size > MAX_CHARS or flags + len(idxs) > MAX_FLAGS):
            out.append(cur)
            cur, chars, flags = [], 0, 0
        cur.append(idxs)
        chars += size
        flags += len(idxs)
    if cur:
        out.append(cur)
    return out


def _clip(s: str, n: int = 600) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def messages(cands: list[Cand], batch: list[list[int]]) -> tuple[list[dict], dict]:
    lang = cands[batch[0][0]].lang
    lines, ids, k = [], {}, 0
    for p, idxs in enumerate(batch, 1):
        c0 = cands[idxs[0]]
        lines.append(f"PAIR {p}:\nENGLISH: {_clip(c0.en_text.replace(chr(10), ' | '))}\n{LANG_NAME[lang].upper()}: {_clip(c0.tg_text.replace(chr(10), ' | '))}\nFLAGS:")
        for i in idxs:
            k += 1
            c = cands[i]
            ids[f"K{k}"] = i
            lines.append(f'  K{k} [{"English" if c.side == "a" else LANG_NAME[lang]}] "{_clip(c.quote, 160)}"')
    return [{"role": "system", "content": SYSTEM.format(LANG=LANG_NAME[lang], RUBRIC=RUBRIC)}, {"role": "user", "content": "\n".join(lines)}], ids


def parse(text: str, ids: dict) -> dict | None:
    """-> {cand_index: (real: bool, level: float, reason)} or None if some flag has no valid answer."""
    obj = extract_json(text)
    res = obj.get("results") if isinstance(obj, dict) else None
    if not isinstance(res, list):
        return None
    out = {}
    for r in res:
        if not (isinstance(r, dict) and str(r.get("id", "")).strip().upper() in ids):
            continue
        real = str(r.get("real", "")).strip().lower()
        if real not in ("yes", "no"):
            continue
        lvl = snap_severity(r.get("level")) if real == "yes" else None
        out[ids[str(r["id"]).strip().upper()]] = (real == "yes", (lvl or 0.4) if real == "yes" else 0.0, str(r.get("reason", ""))[:200])
    return out if all(i in out for i in ids.values()) else None


async def run_verifier(client, model: str, cands: list[Cand], subset: list[int] | None = None, temperature: float = 0.0, seed: int | None = None):
    """Verify cands (optionally only the given indices). Returns ({idx: (real, level, reason)}, stats)."""
    sel = subset if subset is not None else list(range(len(cands)))
    sub = [cands[i] for i in sel]
    batches = {}
    for lang in sorted({c.lang for c in sub}):
        loc = [j for j, c in enumerate(sub) if c.lang == lang]
        for b in _batches([sub[j] for j in loc]):
            batches.setdefault(lang, []).append([[loc[x] for x in idxs] for idxs in b])
    stats = {"calls": 0, "batches": sum(len(v) for v in batches.values()), "first_failed": 0, "failed": 0}
    results: dict = {}

    async def one(batch):
        msgs, ids = messages(sub, batch)
        for attempt in range(2):
            # first attempt: the requested sample (temperature, seed); retry: a different seed (and at least temperature 0.3)
            r = await client.chat(msgs, model=model, temperature=temperature if attempt == 0 else max(temperature, 0.3),
                                  seed=seed if attempt == 0 else (13 if seed is None else seed + 1), max_tokens=900,
                                  json_mode=True, rep_guard=True, tag="verify")
            stats["calls"] += 1
            got = None if (r.aborted or r.finish_reason == "length" or r.dry_run) else parse(r.text, ids)
            if got is not None:
                for j, v in got.items():
                    results[sel[j]] = v
                return
            if attempt == 0:
                stats["first_failed"] += 1
        stats["failed"] += 1

    await asyncio.gather(*(one(b) for v in batches.values() for b in v))
    return results, stats
