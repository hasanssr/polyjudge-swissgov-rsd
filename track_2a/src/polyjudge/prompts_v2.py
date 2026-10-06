"""Compare v2 (Task F): minimal spans (<= 8 words), reason BEFORE the quote with rejection of 'same information' reports,
few-shot demonstrations with concrete gold-derived reasons (and two empty examples), parser-side filters.
Few-shot sources: prompt_pool chunks outside every evaluation set (asserted)."""
from __future__ import annotations

import json
import random
import re

from .data import chunk_key, rows_for
from .diffs import ParseResult, parse_compare
from .groups import LANG_NAME, Group, group_labels, make_groups
from .prompts import RUBRIC, eval_chunks, format_input, _stats
from .spans import runs_from_labels, runs_to_records

MAX_WORDS = 8

COMPARE2_SYSTEM = """You are a bilingual reviewer comparing an English text and its {LANG} counterpart.
Both texts are a few consecutive, aligned sentences, one per line. Find the places where the MEANING differs.

Difference types:
- omission: content of the English text has no counterpart in the {LANG} text (quote it on the English side)
- addition: content of the {LANG} text has no counterpart in the English text (quote it on the {LANG} side)
- meaning_shift: both texts contain the passage but it says something different (quote BOTH sides)

Rules:
1. Write the "reason" FIRST. It must say concretely what differs and show both wordings (for example: English "CHF 3 million" vs {LANG} "3 million euros").
   If you cannot name a concrete difference in content, report nothing. If your reason would say that the texts convey the same information or differ only in wording, do NOT report the item.
2. Quote the SHORTEST span that carries the difference: at most 8 words per side. Never quote a whole sentence (unless it has at most 8 words).
3. Every quote must be copied EXACTLY from the text of its side (not translated, not paraphrased).
4. "before" = up to 4 words immediately preceding the quote in that text, copied exactly ("" at the start).
5. severity in {{0.2,0.4,0.6,0.8,1.0}}:
{RUBRIC}
6. Word order, grammar, articles, freely translated titles and organisation names, and standard translation choices are NOT differences.
7. Numbers, dates, names and negations must match exactly: a mismatch is important information (at least 0.4).
8. Most aligned sentence pairs are fully equivalent: the correct answer is usually an empty list.
Answer with JSON only, in this format (reason first):
{{"differences": [{{"reason": "concrete difference", "type": "omission" or "addition" or "meaning_shift", "en": {{"quote": "...", "before": "..."}} or null, "tgt": {{"quote": "...", "before": "..."}} or null, "severity": 0.4}}]}}"""

_EQUIV = re.compile(r"same (information|meaning|content|message|statement|thing|idea)|convey(s)? the same|(are|is) (essentially )?(equivalent|identical)|no (real |actual |significant )?difference|"
                    r"(only|just|merely) (a )?(slightly |slight )?(different |difference in )?(wording|phrasing|word choice|translation|style)|different wording|different phrasing|paraphras|similar (meaning|information|content)", re.I)


def parse_compare2(text: str) -> ParseResult:
    """parse_compare + (a) quotes of more than MAX_WORDS words are dropped, (b) records whose reason says the texts are the same are dropped."""
    res = parse_compare(text)
    if not res.ok:
        return res
    keep, dropped, why = [], 0, {"long_quote": 0, "same_info_reason": 0}
    for r in res.records:
        if len(r["quote"].split()) > MAX_WORDS:
            dropped += 1
            why["long_quote"] += 1
        elif _EQUIV.search(r.get("reason", "")):
            dropped += 1
            why["same_info_reason"] += 1
        else:
            keep.append(r)
    out = ParseResult(True, keep, res.n_raw, res.n_dropped + dropped)
    out.drop_reasons = why   # type: ignore[attr-defined]
    return out


# ---------------------------------------------------------------- few-shot with concrete, gold-derived reasons
def gold_compare2_output(row: dict, g: Group) -> dict | None:
    """Gold spans paired per bead; every record gets a CONCRETE reason built from the gold quotes. None if any quote exceeds MAX_WORDS."""
    la, lb = group_labels(row, g)
    ra, rb = runs_from_labels(la), runs_from_labels(lb)
    qa, qb = runs_to_records(g.en_tokens, g.en_text, ra), runs_to_records(g.tg_tokens, g.tg_text, rb)
    A, B = {}, {}
    for run, rec in zip(ra, qa):
        A.setdefault(g.en_bead[run[0]], []).append((run[2], rec))
    for run, rec in zip(rb, qb):
        B.setdefault(g.tg_bead[run[0]], []).append((run[2], rec))
    lang, out = LANG_NAME[g.lang], []
    side = lambda x: {"quote": x[1]["quote"], "before": x[1]["before"]}
    for k in range(len(g.bead_ids)):
        a_l, b_l = A.get(k, []), B.get(k, [])
        for i in range(max(len(a_l), len(b_l))):
            a = a_l[i] if i < len(a_l) else None
            b = b_l[i] if i < len(b_l) else None
            for x in (a, b):
                if x and len(x[1]["quote"].split()) > MAX_WORDS:
                    return None
            if a and b:
                out.append({"reason": f'English says "{a[1]["quote"]}" where the {lang} text says "{b[1]["quote"]}".', "type": "meaning_shift",
                            "en": side(a), "tgt": side(b), "severity": max(a[0], b[0])})
            elif a:
                out.append({"reason": f'"{a[1]["quote"]}" appears only in the English text.', "type": "omission", "en": side(a), "tgt": None, "severity": a[0]})
            else:
                out.append({"reason": f'"{b[1]["quote"]}" appears only in the {lang} text.', "type": "addition", "en": None, "tgt": side(b), "severity": b[0]})
    return {"differences": out}


def _frozen_shots2(lang: str):
    """packaging patch: the exact M2 few-shot examples chosen by select_shots2 on the prompt pool, frozen in data/few_shot/ (or None)."""
    from .prompts import FEW_SHOT_DIR, few_shot_rows
    spec_path, rows = FEW_SHOT_DIR / "m2_shots.json", few_shot_rows()
    if rows is None or not spec_path.is_file():
        return None
    out = []
    for item in json.loads(spec_path.read_text(encoding="utf-8"))[lang]:
        row = rows[item["doc"]]
        assert chunk_key(row) not in eval_chunks(), "frozen few-shot example lies in an evaluation chunk"
        g = next(g for g in make_groups(row, "aligned", max_beads=3, max_chars=800) if g.bead_ids == item["beads"])
        out.append((row, g, gold_compare2_output(row, g)))
    return out


def select_shots2(lang: str, allowed_chunks: set[str], seed: int = 0):
    """4 shots from allowed chunks only: mid (short gold spans), empty, high (short spans, level >= 0.8), a second empty. -> [(row, group, expected_output)]"""
    assert not (set(allowed_chunks) & eval_chunks()), "few-shot chunks overlap an evaluation set"
    frozen = _frozen_shots2(lang)
    if frozen is not None:
        return frozen
    cands = {"empty": [], "mid": [], "high": []}
    for row in sorted(rows_for("prompt_pool"), key=lambda r: r["id"]):
        if row["subset"] != f"admin_{lang}" or chunk_key(row) not in allowed_chunks:
            continue
        for g in make_groups(row, "aligned", max_beads=3, max_chars=800):
            if g.n_chars < 120:
                continue
            st = _stats(row, g)
            out = gold_compare2_output(row, g)
            if out is None:
                continue
            if st["nz"] == 0 and st["beads"] >= 2:
                cands["empty"].append((g.n_chars, row, g, out))
            elif 1 <= st["runs"] <= 3 and st["mx"] >= 0.4 and st["mx"] < 0.8:
                cands["mid"].append((g.n_chars, row, g, out))
            elif st["runs"] <= 4 and st["mx"] >= 0.8:
                cands["high"].append((g.n_chars, row, g, out))
    rng = random.Random(f"v2-{seed}-{lang}")
    shots = []
    for name in ("mid", "empty", "high", "empty"):
        pool = [c for c in sorted(cands[name], key=lambda x: (x[0], x[1]["id"]))[:8] if all(c[1]["id"] != s[0]["id"] or c[2].bead_ids != s[1].bead_ids for s in shots)]
        assert pool, f"no few-shot candidate '{name}' for {lang}"
        _, row, g, out = rng.choice(pool)
        assert chunk_key(row) in allowed_chunks and chunk_key(row) not in eval_chunks()
        shots.append((row, g, out))
    return shots


def build_messages_v2(lang: str, g: Group, shots) -> list[dict]:
    msgs = [{"role": "system", "content": COMPARE2_SYSTEM.format(LANG=LANG_NAME[lang], RUBRIC=RUBRIC)}]
    for row, sg, out in shots:
        msgs.append({"role": "user", "content": format_input("compare", lang, sg.en_text, sg.tg_text)})
        msgs.append({"role": "assistant", "content": json.dumps(out, ensure_ascii=False)})
    msgs.append({"role": "user", "content": format_input("compare", lang, g.en_text, g.tg_text)})
    return msgs
