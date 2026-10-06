"""Parsing and normalising LLM answers (B1 and Compare schemas) into mapper records.

Side convention: "a" = English side, "b" = target-language side. Any of en/english/a (EN) or tgt/target/b/de/fr/it/german/french/italian/...
(target) is accepted: the 70B model answered `side:"fr"` for a French pair instead of the prompt's "tgt".
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .spans import snap_severity

SIDE_A = {"en", "eng", "english", "englisch", "anglais", "inglese", "a", "source", "src"}
SIDE_B = {"tgt", "target", "b", "de", "fr", "it", "german", "french", "italian", "deutsch", "allemand", "français", "francais",
          "italiano", "tedesco", "translation", "other"}


def normalize_side(x) -> str | None:
    """-> 'a' (English) | 'b' (target) | None (unknown)."""
    if not isinstance(x, str):
        return None
    k = x.strip().strip("\"'.:;<>[]() ").lower()
    if k in SIDE_A:
        return "a"
    if k in SIDE_B:
        return "b"
    return None


def extract_json(text: str):
    """json.loads, else code-fence stripping, else the first balanced {...} (string-aware). None if nothing parses."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    for cand in (t, re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)):
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            pass
    start = t.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = t.find("{", start + 1)
    return None


@dataclass
class ParseResult:
    ok: bool                       # a JSON object with a "differences" list was obtained
    records: list[dict] = field(default_factory=list)   # {"side","quote","severity","before", ["type","reason"]}
    n_raw: int = 0                 # difference items in the answer
    n_dropped: int = 0             # items discarded (bad side / empty quote / bad severity / no side left)
    error: str | None = None


def _differences(text: str):
    obj = extract_json(text)
    if isinstance(obj, list):
        obj = {"differences": obj}
    if not isinstance(obj, dict) or not isinstance(obj.get("differences"), list):
        return None
    return obj["differences"]


def _clean(side, quote, severity, before):
    sev = snap_severity(severity)
    if side not in ("a", "b") or not isinstance(quote, str) or not quote.strip() or sev is None:
        return None
    return {"side": side, "quote": quote, "severity": sev, "before": before if isinstance(before, str) else ""}


def parse_b1(text: str) -> ParseResult:
    """B1 schema: {"differences": [{"side","quote","severity","before"}]}."""
    items = _differences(text)
    if items is None:
        return ParseResult(False, error="no 'differences' list")
    recs, dropped = [], 0
    for it in items:
        r = _clean(normalize_side(it.get("side")), it.get("quote"), it.get("severity"), it.get("before")) if isinstance(it, dict) else None
        if r is None:
            dropped += 1
        else:
            recs.append(r)
    return ParseResult(True, recs, len(items), dropped)


def parse_compare(text: str) -> ParseResult:
    """Compare schema: {"differences": [{"type","en":{quote,before}|null,"tgt":{...}|null,"severity","reason"}]}.
    Side keys en/tgt/de/fr/it/... are normalised; omission keeps only the EN side, addition only the target side
    (violations are repaired, not rejected); meaning_shift keeps both."""
    items = _differences(text)
    if items is None:
        return ParseResult(False, error="no 'differences' list")
    recs, dropped = [], 0
    for it in items:
        if not isinstance(it, dict):
            dropped += 1
            continue
        typ = str(it.get("type", "")).strip().lower()
        sides = {}
        if "quote" in it and normalize_side(it.get("side")):          # tolerate B1-style items
            sides[normalize_side(it["side"])] = {"quote": it["quote"], "before": it.get("before", "")}
        for k, v in it.items():
            sd = normalize_side(k)
            if sd and k != "side" and isinstance(v, dict) and sd not in sides:
                sides[sd] = v
        if typ == "omission":
            sides.pop("b", None)
        elif typ == "addition":
            sides.pop("a", None)
        got = 0
        for sd, v in sides.items():
            r = _clean(sd, v.get("quote"), it.get("severity"), v.get("before"))
            if r:
                r["type"], r["reason"] = typ, (it.get("reason") if isinstance(it.get("reason"), str) else "")[:300]
                recs.append(r)
                got += 1
        if not got:
            dropped += 1
    return ParseResult(True, recs, len(items), dropped)
