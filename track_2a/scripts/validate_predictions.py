#!/usr/bin/env python3
"""Validate prediction files against their input files (stdlib only).

Format (from the organizers' evaluation script, see docs/prediction_format.md):
  * one jsonl file per language, named <prefix>{de,fr,it}.jsonl.jsonl
  * rows in the same order as the matching input file
  * fields exactly {id, text_a, text_b, labels_a, labels_b}
  * id, text_a, text_b equal the input row at the same position
  * len(labels_x) == len(text_x.split())
  * label = float in [0, 1] (no step size enforced), or -1 on ASCII punctuation
    tokens only (a token made solely of ASCII punctuation characters)

An input file is matched to its prediction by language: the input name must end
in {de,fr,it}.jsonl (optionally .jsonl.jsonl), the prediction name in
{de,fr,it}.jsonl.jsonl. --prefix additionally pins the prediction prefix.

Exit code 0 = valid, 1 = invalid.
"""
import argparse
import json
import math
import re
import string
import sys
from pathlib import Path

FIELDS = ("id", "text_a", "text_b", "labels_a", "labels_b")
PUNCT = frozenset(string.punctuation)
IN_RE = re.compile(r"(?P<lang>de|fr|it)\.jsonl(?:\.jsonl)?$")
PRED_RE = re.compile(r"(?P<lang>de|fr|it)\.jsonl\.jsonl$")


def is_ascii_punct(token: str) -> bool:
    return bool(token) and all(c in PUNCT for c in token)


def read_jsonl(path: Path, errors: list):
    rows = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                errors.append(f"{path.name}:{n}: empty line")
                rows.append(None)
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{n}: invalid JSON ({exc.msg})")
                rows.append(None)
    return rows


def check_labels(where, side, labels, text, allow_punct_values):
    errs = []
    if not isinstance(labels, list):
        return [f"{where}: labels_{side} is not a list"]
    if not isinstance(text, str):
        return [f"{where}: text_{side} is not a string"]
    tokens = text.split()
    if len(labels) != len(tokens):
        errs.append(f"{where}: labels_{side} has {len(labels)} entries, text_{side} has {len(tokens)} tokens")
        return errs
    for k, (tok, lab) in enumerate(zip(tokens, labels)):
        if isinstance(lab, bool) or not isinstance(lab, (int, float)) or not math.isfinite(lab):
            errs.append(f"{where}: labels_{side}[{k}] is not a finite number: {lab!r}")
        elif is_ascii_punct(tok):
            if lab != -1 and not (allow_punct_values and 0 <= lab <= 1):
                errs.append(f"{where}: labels_{side}[{k}]={lab} on ASCII punctuation token {tok!r}, expected -1")
        elif not 0 <= lab <= 1:
            errs.append(f"{where}: labels_{side}[{k}]={lab} on token {tok!r}, expected float in [0, 1] (-1 only on ASCII punctuation)")
        if len(errs) >= 5:
            break
    return errs


def validate_pair(inp: Path, pred: Path, allow_punct_values: bool) -> list:
    errors = []
    in_rows = read_jsonl(inp, errors)
    pr_rows = read_jsonl(pred, errors)
    if len(in_rows) != len(pr_rows):
        errors.append(f"{pred.name}: {len(pr_rows)} rows, input has {len(in_rows)}")
    for i, (src, row) in enumerate(zip(in_rows, pr_rows), 1):
        where = f"{pred.name}:{i}"
        if row is None or src is None:
            continue
        if not isinstance(row, dict):
            errors.append(f"{where}: row is not an object")
            continue
        if set(row) != set(FIELDS):
            missing, extra = set(FIELDS) - set(row), set(row) - set(FIELDS)
            errors.append(f"{where}: fields differ (missing={sorted(missing)}, extra={sorted(extra)})")
            continue
        for f in ("id", "text_a", "text_b"):
            if row[f] != src.get(f):
                errors.append(f"{where}: '{f}' differs from input (row order or content changed)")
        for side in ("a", "b"):
            errors += check_labels(where, side, row[f"labels_{side}"], row[f"text_{side}"], allow_punct_values)
    return errors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--prefix", default=None, help="require prediction names to be <prefix>{lang}.jsonl.jsonl")
    ap.add_argument("--allow-punct-values", action="store_true",
                    help="accept values in [0,1] (not only -1) on ASCII punctuation tokens")
    ap.add_argument("--max-errors", type=int, default=20)
    args = ap.parse_args(argv)

    errors = []
    inputs = {}
    for p in sorted(args.input_dir.glob("*.jsonl*")):
        m = IN_RE.search(p.name)
        if m:
            inputs[m["lang"]] = p
    if not inputs:
        errors.append(f"no <prefix>{{de,fr,it}}.jsonl input files in {args.input_dir}")
    if not args.pred_dir.is_dir():
        errors.append(f"prediction dir not found: {args.pred_dir}")
    else:
        preds = {}
        for p in sorted(args.pred_dir.iterdir()):
            m = PRED_RE.search(p.name)
            if not m:
                errors.append(f"unexpected file (name must end in {{de,fr,it}}.jsonl.jsonl): {p.name}")
            elif args.prefix is not None and p.name != f"{args.prefix}{m['lang']}.jsonl.jsonl":
                errors.append(f"wrong prefix: {p.name} (expected {args.prefix}{m['lang']}.jsonl.jsonl)")
            else:
                preds[m["lang"]] = p
        for lang, inp in inputs.items():
            if lang not in preds:
                errors.append(f"missing prediction file for language '{lang}' (input {inp.name})")
            else:
                errors += validate_pair(inp, preds[lang], args.allow_punct_values)
        for lang in preds.keys() - inputs.keys():
            errors.append(f"prediction for language '{lang}' has no input file")

    for e in errors[: args.max_errors]:
        print("ERROR", e, file=sys.stderr)
    if len(errors) > args.max_errors:
        print(f"... and {len(errors) - args.max_errors} more", file=sys.stderr)
    if errors:
        print(f"INVALID: {len(errors)} problem(s)", file=sys.stderr)
        return 1
    print(f"OK: {len(inputs)} file(s) valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
