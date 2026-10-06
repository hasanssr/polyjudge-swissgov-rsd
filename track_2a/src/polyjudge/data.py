"""Dataset access with LOCKS. Only the DEV files (= HF "validation") can be loaded here; the test files are never named in this module (see inputs.py).

Reads the three `data/{lang}/dev_{lang}.jsonl` files of the pinned HF dataset revision with `huggingface_hub`
(no `datasets`/pandas: pandas' DLLs are blocked by the application-control policy on this machine).

Locks
* `dev_final` rows are removed from every loader unless the environment variable POLYJUDGE_UNLOCK contains "dev_final".
* `dev_select` can only be opened after docs/prereg.md exists (preregistration first).
* `dev_eval` (the union of both) is deprecated and refuses to open.
* POLYJUDGE_FORBID_SETS (comma separated) forbids named sets for the current session.
"""
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "splits" / "dev_split.json"
SPLIT2_PATH = ROOT / "splits" / "dev_eval_split.json"
PREREG_PATH = ROOT / "docs" / "prereg.md"
LANGS = ("de", "fr", "it")
REPO_ID = "ZurichNLP/SwissGov-RSD"
REVISION = "f16aa0536811b8c0cf975fede21eec5996e4b3c1"  # pinned (same revision as the earlier `datasets` cache)


def _dev_file(lang: str) -> Path:
    name = f"data/{lang}/dev_{lang}.jsonl"
    try:
        return Path(hf_hub_download(REPO_ID, name, repo_type="dataset", revision=REVISION, local_files_only=True))
    except Exception:  # not cached yet -> download this one file
        return Path(hf_hub_download(REPO_ID, name, repo_type="dataset", revision=REVISION))


def chunk_key(row: dict) -> str:
    """Stable chunk id: the same English chunk appears once per language, so key on its text.
    (`chunk_id` is None for fr/it rows and therefore unusable.)"""
    return "c" + hashlib.sha1(row["text_a"].encode("utf-8")).hexdigest()[:10]


def _unlocked(name: str) -> bool:
    return name in {s.strip() for s in os.environ.get("POLYJUDGE_UNLOCK", "").split(",") if s.strip()}


def _read_all_dev() -> list[dict]:
    rows = []
    for lang in LANGS:
        with open(_dev_file(lang), encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def load_split() -> dict:
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))


def load_split2() -> dict:
    """dev_eval split into dev_select / dev_final (chunk ids only)."""
    return json.loads(SPLIT2_PATH.read_text(encoding="utf-8"))


def final_chunks() -> set[str]:
    return set(load_split2()["dev_final"]) if SPLIT2_PATH.exists() else set()


def all_dev_chunk_keys() -> dict[str, str]:
    """{row id: chunk key} for ALL dev rows, locked or not. Returns keys only (no text, no labels)."""
    return {r["id"]: chunk_key(r) for r in _read_all_dev()}


def load_dev_rows() -> list[dict]:
    """Dev rows minus the locked chunks: dev_final (unless POLYJUDGE_UNLOCK=dev_final) and, before docs/prereg.md exists, dev_select."""
    rows = _read_all_dev()
    if _unlocked("dev_final"):
        return rows
    locked = final_chunks()
    if SPLIT2_PATH.exists() and not PREREG_PATH.exists():       # dev_select stays closed until the preregistration exists
        locked = locked | set(load_split2()["dev_select"])
    return [r for r in rows if chunk_key(r) not in locked]


def rows_for(set_name: str) -> list[dict]:
    """Rows of a named chunk set: 'prompt_pool', 'dev_select', 'dev_final'. See the module docstring for the locks."""
    forbidden = {s.strip() for s in os.environ.get("POLYJUDGE_FORBID_SETS", "").split(",") if s.strip()}
    if set_name in forbidden:
        raise PermissionError(f"set '{set_name}' is forbidden in this session (POLYJUDGE_FORBID_SETS)")
    if set_name == "dev_eval":
        raise PermissionError("'dev_eval' is deprecated: it is split into 'dev_select' and 'dev_final' (splits/dev_eval_split.json)")
    if set_name == "dev_select":
        if not PREREG_PATH.exists():
            raise PermissionError("dev_select is closed until the preregistration docs/prereg.md exists")
        chunks = set(load_split2()["dev_select"])
    elif set_name == "dev_final":
        if not _unlocked("dev_final"):
            raise PermissionError("dev_final is locked (set POLYJUDGE_UNLOCK=dev_final only after the selection report has been approved)")
        chunks = set(load_split2()["dev_final"])
    else:
        chunks = set(load_split()[set_name])
    return [r for r in load_dev_rows() if chunk_key(r) in chunks]
