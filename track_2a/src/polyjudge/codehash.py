"""SHA-256 over the method code (polyjudge/*.py except this file) and the split files: pins what a preregistration refers to."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILES = ("dev_split.json", "dev_eval_split.json", "pool_eval2.json", "pool_eval12.json", "shots_round1.json")


def code_hash() -> str:
    h = hashlib.sha256()
    files = [p for p in sorted((ROOT / "polyjudge").glob("*.py")) if p.name != "codehash.py"] + [ROOT / "splits" / n for n in SPLIT_FILES]
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))      # line-ending independent
    return h.hexdigest()


if __name__ == "__main__":
    print(code_hash())
