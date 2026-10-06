#!/usr/bin/env python3
"""Minimal secrets/label scan over a directory tree (stdlib only). Prints file:line and the PATTERN NAME, never the matched text.
Exit 1 if anything is found.  usage: python scripts/scan_secrets.py [root]"""
import re
import sys
from pathlib import Path

PATTERNS = {
    "openai-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    "hf token": re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    "github token": re.compile(r"\b(?:ghp|gho|ghs|github_pat)_[A-Za-z0-9_]{20,}"),
    "aws key id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "bearer token": re.compile(r"[Bb]earer\s+[A-Za-z0-9._-]{20,}"),
    "key assignment with value": re.compile(r"""(?i)\b(?:api[_-]?key|secret|token|password)\b["']?\s*[:=]\s*["'][^"'\s$<{]{12,}["']"""),
    "LLM_API_KEY with value": re.compile(r"LLM_API_KEY\s*=\s*[^\s$]{6,}"),
    "gold label field": re.compile(r'"(?:gold_)?labels_[ab]"\s*:\s*\[(?![^\]]*\])'),
}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache"}
# prediction files legitimately contain labels_a/labels_b (our own predictions); only flag the label pattern elsewhere
LABEL_OK = ("predictions",)
ALLOW = ("DO-NOT-LEAK",)   # the fake secret used by tests/test_llm.py to prove that secrets are never logged


def main(root: Path) -> int:
    hits = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file() or SKIP_DIRS & set(p.parts):
            continue
        rel = p.relative_to(root)
        if p.name == ".env" or (p.name.startswith(".env.") and p.name != ".env.example"):
            print(f"{rel}: forbidden file name"); hits += 1; continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if any(a in line for a in ALLOW):
                continue
            for name, rx in PATTERNS.items():
                if name == "gold label field" and (rel.parts[0] in LABEL_OK or p.name == "scan_secrets.py"):
                    continue
                if p.name == "scan_secrets.py":
                    continue
                if rx.search(line):
                    print(f"{rel}:{n}: {name}"); hits += 1
    print(f"scan finished: {hits} finding(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else ".")))
