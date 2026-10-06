"""Prompts (B1 baseline, Compare-only) and few-shot construction from gold labels of prompt_pool chunks.

Few-shot examples may only come from chunks listed in `allowed_chunks` (the prompt_pool chunks that are NOT evaluation chunks);
`select_shots` asserts this. Rubric = organizers' annotation guideline (paper Appendix A.2), levels 0.2 ... 1.0.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .data import ROOT, chunk_key, rows_for
from .groups import LANG_NAME, Group, group_labels, make_groups
from .spans import runs_from_labels, runs_to_records

EVAL12_PATH = Path(__file__).resolve().parents[1] / "splits" / "pool_eval12.json"

RUBRIC = """   0.2 mostly equivalent, only unimportant details differ
   0.4 roughly equivalent, but some important information differs or is missing
   0.6 not equivalent, but the two share some details
   0.8 not equivalent, but on the same topic
   1.0 completely dissimilar; or the passage exists only in this text (no counterpart)"""

B1_SYSTEM = """You compare two versions of the same Swiss government web content: an English version and a {LANG} version.
Both texts are a few consecutive sentences, one per line; they follow the same order, but a sentence may exist in only one version.
Find where their MEANING differs. Ignore differences that are only grammatical, word order, or normal translation choices.

Report every difference as a record. Differences can be:
- an addition: text that exists only in one version (report it on the side where it exists),
- an omission: the same thing seen from the other side (report only the side that has the text),
- a meaning shift: both sides have the passage but it says something different (report BOTH passages).

Rules:
1. "quote" must be copied EXACTLY (character for character) from that side's text. Never translate or paraphrase quotes.
2. Quote the smallest span that carries the difference, but a whole sentence if the whole sentence differs.
3. "before" = the up to 4 words immediately preceding the quote in that text, copied exactly ("" at the very start).
4. "severity" must be one of 0.2, 0.4, 0.6, 0.8, 1.0:
{RUBRIC}
5. If the two texts say the same thing, return {{"differences": []}}. Most sentences are equivalent: be conservative.
Answer with JSON only, in this format:
{{"differences": [{{"side": "en" or "tgt", "quote": "...", "severity": 0.6, "before": "..."}}]}}"""

COMPARE_SYSTEM = """You are a bilingual reviewer comparing an English text and its {LANG} counterpart.
Both texts are a few consecutive, aligned sentences, one per line.
Decide where the MEANING differs.

Difference types:
- omission: content of the English text has no counterpart in the {LANG} text (report the quote on the English side)
- addition: content of the {LANG} text has no counterpart in the English text (report the quote on the {LANG} side)
- meaning_shift: both texts contain the passage but it says something different (report a quote on BOTH sides)
- if the texts are equivalent in meaning, return an empty list

Rules:
1. Every quote must be copied EXACTLY from the text of its side (not translated, not paraphrased).
2. Quote the smallest span that carries the difference; a whole sentence if the whole sentence differs.
3. "before" = up to 4 words immediately preceding the quote in that text, copied exactly ("" at the start).
4. severity in {{0.2,0.4,0.6,0.8,1.0}} using the scale:
{RUBRIC}
5. Differences in word order, grammar, articles, or standard translation choices are NOT differences.
6. Numbers, dates, names and negations must match exactly: a mismatch is important information (at least 0.4).
7. Be conservative: if unsure, do not report.
Answer with JSON only, in this format:
{{"differences": [{{"type": "omission" or "addition" or "meaning_shift", "en": {{"quote": "...", "before": "..."}} or null, "tgt": {{"quote": "...", "before": "..."}} or null, "severity": 0.6, "reason": "one short sentence"}}]}}"""


def system_prompt(kind: str, lang: str) -> str:
    tpl = {"b1": B1_SYSTEM, "compare": COMPARE_SYSTEM}[kind]
    return tpl.format(LANG=LANG_NAME[lang], RUBRIC=RUBRIC)


def format_input(kind: str, lang: str, en_text: str, tg_text: str) -> str:
    if kind == "b1":
        return f"ENGLISH:\n{en_text}\n\n{LANG_NAME[lang].upper()}:\n{tg_text}"
    return f"ENGLISH TEXT:\n{en_text}\n\n{LANG_NAME[lang].upper()} TEXT:\n{tg_text}"


# --------------------------------------------------------------------------- gold -> expected model output
def gold_b1_output(row: dict, g: Group) -> dict:
    la, lb = group_labels(row, g)
    recs = [{"side": "en", **{k: r[k] for k in ("quote", "severity", "before")}} for r in runs_to_records(g.en_tokens, g.en_text, runs_from_labels(la))]
    recs += [{"side": "tgt", **{k: r[k] for k in ("quote", "severity", "before")}} for r in runs_to_records(g.tg_tokens, g.tg_text, runs_from_labels(lb))]
    return {"differences": recs}


def gold_compare_output(row: dict, g: Group) -> dict:
    """Gold spans paired per bead (in order); a pair -> meaning_shift (severity = max of the two), leftovers -> omission (EN) / addition (target)."""
    la, lb = group_labels(row, g)
    ra, rb = runs_from_labels(la), runs_from_labels(lb)
    qa, qb = runs_to_records(g.en_tokens, g.en_text, ra), runs_to_records(g.tg_tokens, g.tg_text, rb)
    by_bead_a, by_bead_b = {}, {}
    for run, rec in zip(ra, qa):
        by_bead_a.setdefault(g.en_bead[run[0]], []).append((run[2], rec))
    for run, rec in zip(rb, qb):
        by_bead_b.setdefault(g.tg_bead[run[0]], []).append((run[2], rec))
    out = []
    lang = LANG_NAME[g.lang]
    for k in range(len(g.bead_ids)):
        A, B = by_bead_a.get(k, []), by_bead_b.get(k, [])
        for i in range(max(len(A), len(B))):
            a = A[i] if i < len(A) else None
            b = B[i] if i < len(B) else None
            side = lambda x: {"quote": x[1]["quote"], "before": x[1]["before"]} if x else None
            if a and b:
                out.append({"type": "meaning_shift", "en": side(a), "tgt": side(b), "severity": max(a[0], b[0]),
                            "reason": "The two passages differ in meaning or detail."})
            elif a:
                out.append({"type": "omission", "en": side(a), "tgt": None, "severity": a[0], "reason": "This content appears only in the English text."})
            else:
                out.append({"type": "addition", "en": None, "tgt": side(b), "severity": b[0], "reason": f"This content appears only in the {lang} text."})
    return {"differences": out}


# --------------------------------------------------------------------------- few-shot selection
def _stats(row, g):
    la, lb = group_labels(row, g)
    vals = [x for x in la + lb if x > 0]
    runs = len(runs_from_labels(la)) + len(runs_from_labels(lb))
    unal_hi = False
    for side, lab, bead in (("a", la, g.en_bead), ("b", lb, g.tg_bead)):
        for k, kind in enumerate(g.kinds):
            if kind.startswith("unaligned"):
                v = [x for x, bk in zip(lab, bead) if bk == k and x != -1]
                if v and sum(v) / len(v) >= 0.8:
                    unal_hi = True
    return dict(nz=len(vals), mx=max(vals, default=0.0), runs=runs, unal_hi=unal_hi, beads=len(g.bead_ids))


def select_shots(lang: str, kind: str, allowed_chunks: set[str], seed: int = 0, k: int = 3):
    """Deterministic k-shot selection for one language from `allowed_chunks` ONLY. Returns [(row, group)] ordered mid, empty, high."""
    ev = eval_chunks()  # hard leakage guard: NO evaluation chunk (pool_eval12, pool_eval2, ...) can be a few-shot source
    assert not (set(allowed_chunks) & ev), "few-shot chunks overlap an evaluation set"
    mode = "all" if kind == "b1" else "aligned"
    cands = {"empty": [], "mid": [], "high": []}
    for row in sorted(rows_for("prompt_pool"), key=lambda r: r["id"]):
        if row["subset"] != f"admin_{lang}":
            continue
        if chunk_key(row) not in allowed_chunks:
            continue
        for g in make_groups(row, mode, max_beads=3, max_chars=800):
            if g.n_chars < 120:
                continue
            st = _stats(row, g)
            if st["nz"] == 0 and st["beads"] >= 2:
                cands["empty"].append((g.n_chars, row, g))
            elif 1 <= st["runs"] <= 3 and st["mx"] in (0.4, 0.6):
                cands["mid"].append((g.n_chars, row, g))
            elif st["runs"] <= 4 and ((kind == "b1" and st["unal_hi"]) or (kind != "b1" and st["mx"] >= 0.8)):
                cands["high"].append((g.n_chars, row, g))
    rng = random.Random(f"{seed}-{lang}-{kind}")
    shots = []
    for name in ("mid", "empty", "high")[:k]:
        pool = sorted(cands[name], key=lambda x: (x[0], x[1]["id"]))[:6]
        assert pool, f"no few-shot candidate '{name}' for {lang}/{kind}"
        _, row, g = rng.choice(pool)
        assert chunk_key(row) in allowed_chunks  # leakage guard: never from an evaluation chunk
        shots.append((row, g))
    return shots



def eval_chunks() -> set[str]:
    """Union of the chunk ids of every splits/pool_eval*.json evaluation set."""
    out: set[str] = set()
    for p in sorted(EVAL12_PATH.parent.glob("pool_eval*.json")):
        out |= set(json.loads(p.read_text(encoding="utf-8"))["chunks"])
    return out


FEW_SHOT_DIR = ROOT / "data" / "few_shot"


def few_shot_rows() -> dict | None:
    """packaging patch: the frozen few-shot documents (data/few_shot/few_shot_rows.jsonl, prompt_pool records only) by id, or None."""
    p = FEW_SHOT_DIR / "few_shot_rows.jsonl"
    if not p.is_file():
        return None
    return {r["id"]: r for r in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())}


def frozen_shots(lang: str, kind: str):
    """The exact round-1 few-shot examples (splits/shots_round1.json), rebuilt as [(row, group)]; asserted not to touch any evaluation chunk."""
    spec = json.loads((EVAL12_PATH.parent / "shots_round1.json").read_text(encoding="utf-8"))[f"{lang}/{kind}"]
    rows = few_shot_rows() or {r["id"]: r for r in rows_for("prompt_pool")}   # packaging patch: frozen file first
    mode = "all" if kind == "b1" else "aligned"
    ev, out = eval_chunks(), []
    for item in spec:
        row = rows[item["doc"]]
        assert chunk_key(row) not in ev, "frozen few-shot example lies in an evaluation chunk"
        g = next(g for g in make_groups(row, mode, max_beads=3, max_chars=800) if g.bead_ids == item["beads"])
        out.append((row, g))
    return out

def build_messages(kind: str, lang: str, g: Group, shots) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt(kind, lang)}]
    for row, sg in shots:
        out = gold_b1_output(row, sg) if kind == "b1" else gold_compare_output(row, sg)
        msgs.append({"role": "user", "content": format_input(kind, lang, sg.en_text, sg.tg_text)})
        msgs.append({"role": "assistant", "content": json.dumps(out, ensure_ascii=False)})
    msgs.append({"role": "user", "content": format_input(kind, lang, g.en_text, g.tg_text)})
    return msgs
