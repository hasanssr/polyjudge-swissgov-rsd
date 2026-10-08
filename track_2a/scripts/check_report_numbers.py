#!/usr/bin/env python3
"""Check the numbers in technical_report.md and docs/PolyJudge_2page_summary.html against docs/test_results.md.

Standard library only; reads text files in this repository, nothing else. Exit code 1 on any mismatch.

1. Table rows (hard check). Every row of a report table that starts with a method id (B0, M0, M1, M2, M3) and
   whose other cells are only numbers / intervals / dashes must equal, number for number (sign included),
   the leading columns of some row of the same method in a table of docs/test_results.md (trailing columns
   may be left out of the report table).
2. Other numbers (hard check). Every decimal number and every number >= 1000 in the report text must occur in
   docs/test_results.md or in one of the other documents in SOURCES (sign-insensitive; spaces/commas in thousands
   are ignored). Numbers that are not in test_results.md are listed with the document that supplies them.
3. Derived claims (hard check), computed from the parsed test_results.md tables.
"""
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
RESULTS = DOCS / "test_results.md"
TARGETS = [ROOT / "technical_report.md", DOCS / "PolyJudge_2page_summary.html"]
SOURCES = ["dev_select_report.md", "apertus_role.md", "round1_findings.md", "round2_findings.md",
           "baselines_from_paper.md", "prereg.md", "export_list.md", "code_changes_after_freeze.md",
           "technical_report_notes.md", "reproducibility_measurements.md", "owner_confirmations.md"]
METHODS = ("B0", "M0", "M1", "M2", "M3")

NUM = re.compile(r"[+\-−]?\d+(?:\.\d+)?")
THOUSANDS = re.compile(r"(?<![\w.])\d{1,3}(?:[  ,]\d{3})+(?![\d.])")
HEX = re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{6,}\b")


def norm_minus(s):
    return s.replace("−", "-")


def numbers_in(cell):
    return [norm_minus(m) for m in NUM.findall(norm_minus(cell))]


def clean_cell(c):
    return html.unescape(re.sub(r"<[^>]+>", "", c)).replace("*", "").replace("`", "").strip()


def md_rows(text):
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|") and not re.fullmatch(r"\|[\s:|\-]+\|", s):
            yield [clean_cell(c) for c in s.strip("|").split("|")]


def html_rows(text):
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", text, flags=re.S | re.I):
        yield [clean_cell(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S | re.I)]


def method_of(first):
    m = re.match(r"(B0|M[0-3])(?![A-Za-z0-9])", first)
    return m.group(1) if m else None


def numeric_only(cells):
    """True if the cell has no words, only numbers, signs, brackets, commas, dashes, %."""
    return all(re.fullmatch(r"[\s\d.,+\-−\[\]%–]*", c) or c.startswith("(reference)") or c in ("reference", "–", "-")
               for c in cells)


def row_signature(cells):
    out = []
    for c in cells[1:]:
        out.extend(numbers_in(c))
    return out


def load_reference():
    ref = {m: [] for m in METHODS}
    for cells in md_rows(RESULTS.read_text(encoding="utf-8")):
        m = method_of(cells[0]) if cells else None
        if m and len(cells) > 1 and numeric_only(cells[1:]):
            ref[m].append(row_signature(cells))
    return ref


def text_numbers(text):
    """Numbers to check generically: decimals and numbers >= 1000. Returns set of normalised absolute strings."""
    t = html.unescape(re.sub(r"<style.*?</style>|<[^>]+>", " ", text, flags=re.S | re.I))
    t = re.sub(r"https?://\S+|arXiv:\s?\d+\.\d+", " ", t)
    t = re.sub(r"`[^`\n]*`", lambda m: " " if HEX.search(m.group(0)) or "/" in m.group(0) or "_" in m.group(0) else m.group(0), t)
    t = HEX.sub(" ", t)
    t = re.sub(r"\bv\d+\.\d+\b|\bApertus-v\d+\.\d+|python:?\s?3\.11|Python 3\.11", " ", t)
    t = norm_minus(t)
    found = set()
    for m in THOUSANDS.finditer(t):
        found.add(re.sub(r"[  ,]", "", m.group(0)))
    t = THOUSANDS.sub(" ", t)
    for m in re.finditer(r"\d+\.\d+", t):
        found.add(m.group(0))
    for m in re.finditer(r"(?<![\d.])\d{4,}(?![\d.])", t):
        found.add(m.group(0))
    return found


def canon(s):
    """Sign-insensitive form for matching prose numbers (tables are checked with signs)."""
    return s.lstrip("+-")


def main():
    errors, info = [], []
    ref = load_reference()
    results_text = RESULTS.read_text(encoding="utf-8")
    ref_numbers = {canon(n) for n in text_numbers(results_text)}
    other = {}
    for name in SOURCES:
        p = DOCS / name
        if p.exists():
            other[name] = {canon(n) for n in text_numbers(p.read_text(encoding="utf-8"))}

    checked_rows = 0
    for target in TARGETS:
        text = target.read_text(encoding="utf-8")
        rows = html_rows(text) if target.suffix == ".html" else md_rows(text)
        for cells in rows:
            m = method_of(cells[0]) if cells else None
            if not m or len(cells) < 3 or not numeric_only(cells[1:]):
                continue
            sig = row_signature(cells)
            if not sig:
                continue
            checked_rows += 1
            # a report table may leave out trailing columns of the source table, never change a value
            if not any(r[:len(sig)] == sig for r in ref[m]):
                errors.append(f"{target.name}: row '{cells[0]}' numbers {sig} do not match the leading columns of any {m} row of test_results.md")

        for n in sorted(text_numbers(text)):
            c = canon(n)
            if c in ref_numbers:
                continue
            src = [k for k, v in other.items() if c in v]
            if src:
                info.append(f"{target.name}: {n} not in test_results.md; found in {', '.join(src)}")
            else:
                errors.append(f"{target.name}: number {n} found in no source document")

    # derived claims
    rtext = results_text
    def row(table_title_re, method):
        sec = re.split(r"\n## ", rtext)
        for s in sec:
            if re.match(table_title_re, s):
                for cells in md_rows(s):
                    if cells and method_of(cells[0]) == method:
                        return cells
        raise SystemExit(f"cannot find {method} in section {table_title_re}")
    macro = {m: float(row_signature(row(r"Scores", m))[3]) for m in METHODS}
    flagged = {m: int(row_signature(row(r"Tokens flagged", m))[0]) for m in METHODS}
    gold_pos = int(row_signature(row(r"Tokens flagged", "M0"))[1])
    derived = [
        ("M1 - B0 macro = 0.003", round(macro["M1"] - macro["B0"], 3) == 0.003),
        ("M1 flags 74 tokens more than B0 (5086 - 5012)", flagged["M1"] - flagged["B0"] == 74),
        ("M0 flags more than twice the gold nonzero tokens", flagged["M0"] > 2 * gold_pos),
        ("M1 macro 0.276, M0 macro 0.213 (difference 0.063)", round(macro["M1"] - macro["M0"], 3) == 0.063),
        ("M2 and M3 macro both below M1", macro["M2"] < macro["M1"] and macro["M3"] < macro["M1"]),
    ]
    for label, ok in derived:
        print(("PASS  " if ok else "FAIL  ") + "derived: " + label)
        if not ok:
            errors.append("derived claim failed: " + label)

    print(f"checked {checked_rows} table row(s) against test_results.md")
    print(f"{len(info)} number(s) outside test_results.md, each found in another source document:")
    for i in info:
        print("  " + i)
    if errors:
        print(f"\nFAILED: {len(errors)} problem(s)")
        for e in errors:
            print("  " + e)
        return 1
    print("\nOK: no mismatch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
