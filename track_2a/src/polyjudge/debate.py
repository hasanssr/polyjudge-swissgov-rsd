"""Task D: debate-style agents. Two blind EXTRACTOR agents (English text only / target text only) -> COMPARE (texts + both component lists) -> verifier.
Reuses the Compare-v2 prompt/parser/filters and the candidate verifier."""
from __future__ import annotations

import asyncio

from .diffs import ParseResult, extract_json
from .groups import LANG_NAME, Group
from .pipeline import GroupRun, map_run
from .prompts import format_input
from .prompts_v2 import COMPARE2_SYSTEM, RUBRIC, parse_compare2

EXTRACTOR_SYSTEM = """You extract the checkable meaning components of ONE text. You will not see any translation.
List only what the text explicitly states. For every component copy the exact words from the text.
Component types: claim (a factual statement), number (any quantity, percentage, amount, date, deadline, duration, year), condition (if/unless/only when, restriction, exception),
obligation (must / shall / may / is entitled to / is responsible for), entity (organisations, laws, places, products, people, document or programme titles),
other (anything else that carries meaning, e.g. a remark in brackets, a link label, a footer phrase).
Rules: "quote" is copied EXACTLY from the text and is as short as possible while still complete (at most 8 words). Do not infer, translate or summarise. "gist" is a neutral paraphrase in English of at most 10 words.
Answer with JSON only: {"components": [{"type": "claim", "quote": "...", "gist": "..."}]}"""

COMPARE_WITH_COMPONENTS_NOTE = """
You are also given a list of components that two independent analysts extracted from each text (each analyst saw only one of the texts).
Use the lists to find candidate differences (a component of one text without counterpart in the other, or components that disagree), then verify every candidate against the texts."""


def parse_components(text: str) -> list[dict] | None:
    obj = extract_json(text)
    comps = obj.get("components") if isinstance(obj, dict) else None
    if not isinstance(comps, list):
        return None
    out = []
    for c in comps:
        if isinstance(c, dict) and isinstance(c.get("quote"), str) and c["quote"].strip():
            out.append({"type": str(c.get("type", "other"))[:12], "quote": c["quote"].strip(), "gist": str(c.get("gist", ""))[:80]})
    return out[:30]


def _fmt(comps: list[dict]) -> str:
    return "\n".join(f'- {c["type"]}: "{c["quote"]}" ({c["gist"]})' for c in comps) or "(none)"


def compare_messages(lang: str, g: Group, shots, en_comps: list[dict], tg_comps: list[dict]) -> list[dict]:
    """Compare-v2 system prompt (+ note about components); demonstrations as in Compare-v2 (texts only); final user turn with texts and components."""
    msgs = [{"role": "system", "content": COMPARE2_SYSTEM.format(LANG=LANG_NAME[lang], RUBRIC=RUBRIC) + COMPARE_WITH_COMPONENTS_NOTE}]
    import json
    for row, sg, out in shots:
        msgs.append({"role": "user", "content": format_input("compare", lang, sg.en_text, sg.tg_text)})
        msgs.append({"role": "assistant", "content": json.dumps(out, ensure_ascii=False)})
    L = LANG_NAME[lang]
    msgs.append({"role": "user", "content": f"{format_input('compare', lang, g.en_text, g.tg_text)}\n\nENGLISH COMPONENTS:\n{_fmt(en_comps)}\n\n{L.upper()} COMPONENTS:\n{_fmt(tg_comps)}"})
    return msgs


async def run_debate(client, model: str, groups: list[Group], shots_by_lang: dict, sem: asyncio.Semaphore | None = None):
    """-> list[GroupRun] (kind 'compare2') whose spans come from extractors -> compare. Stats dict returned as second value."""
    stats = {"extractor_calls": 0, "extractor_failed": 0, "compare_calls": 0, "compare_failed": 0}

    async def extract(g: Group, side: str):
        text = g.en_text if side == "a" else g.tg_text
        msgs = [{"role": "system", "content": EXTRACTOR_SYSTEM}, {"role": "user", "content": f"TEXT ({'English' if side == 'a' else LANG_NAME[g.lang]}):\n{text}"}]
        for attempt in range(2):
            r = await client.chat(msgs, model=model, temperature=0.0 if attempt == 0 else 0.3, seed=None if attempt == 0 else 17, max_tokens=700,
                                  json_mode=True, rep_guard=True, tag="extract")
            stats["extractor_calls"] += 1
            comps = None if (r.aborted or r.finish_reason == "length" or r.dry_run) else parse_components(r.text)
            if comps is not None:
                return comps
        stats["extractor_failed"] += 1
        return []

    async def one(g: Group) -> GroupRun:
        en_c, tg_c = await asyncio.gather(extract(g, "a"), extract(g, "b"))      # blind to each other
        run = GroupRun(g, "compare2")
        msgs = compare_messages(g.lang, g, shots_by_lang[g.lang], en_c, tg_c)
        for attempt in range(2):
            r = await client.chat(msgs, model=model, temperature=0.0 if attempt == 0 else 0.3, seed=None if attempt == 0 else 19, max_tokens=700,
                                  json_mode=True, rep_guard=True, tag="debate-compare")
            stats["compare_calls"] += 1
            run.calls += 1
            res = ParseResult(False, error="aborted") if (r.aborted or r.finish_reason == "length" or r.dry_run) else parse_compare2(r.text)
            run.parse, run.raw = res, r.text
            if res.ok:
                break
            run.first_failed = attempt == 0
        else:
            run.failed = True
            stats["compare_failed"] += 1
        map_run(run)
        return run

    return list(await asyncio.gather(*(one(g) for g in groups))), stats
