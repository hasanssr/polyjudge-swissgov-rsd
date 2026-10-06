"""INPUT-only loader: what a prediction method is allowed to see. No labels, no annotator data, no chunk metadata.

* `Input` is a frozen dataclass with exactly the text fields below; there is no attribute that could hold a label.
* every loader builds `Input` objects from a WHITELIST of keys and drops everything else at read time (also for the test files, which do contain labels:
  they are parsed line by line, converted immediately, never stored, never logged).
* the test files are opened ONLY here (`load_inputs("test")`); polyjudge/data.py never names them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from .data import LANGS, REPO_ID, REVISION, ROOT, rows_for

INPUT_KEYS = ("id", "subset", "text_a", "text_a_untokenized", "text_b", "text_b_untokenized")
FORBIDDEN_KEYS = ("labels_a", "labels_b", "annotator_tag", "chunk_id")   # must never reach a method


@dataclass(frozen=True)
class Input:
    id: str
    subset: str
    text_a: str                 # English, tokenized (whitespace-separated tokens: the unit of prediction)
    text_a_untokenized: str
    text_b: str                 # target language, tokenized
    text_b_untokenized: str

    @property
    def lang(self) -> str:
        return self.subset.split("_")[1]

    def as_row(self) -> dict:
        """Dict view with the whitelisted keys only (what the model-free helpers expect)."""
        return {k: getattr(self, k) for k in INPUT_KEYS}


def _from_raw(d: dict) -> Input:
    return Input(**{k: d[k] for k in INPUT_KEYS})


def _test_file(lang: str) -> Path:
    # packaging patch: an inputs-only file shipped with the repo is preferred, so `make run` needs no network and no HF cache
    local = ROOT / "data" / "test_input" / f"test_{lang}.jsonl"
    if local.is_file():
        return local
    name = f"data/{lang}/test_{lang}.jsonl"
    try:
        return Path(hf_hub_download(REPO_ID, name, repo_type="dataset", revision=REVISION, local_files_only=True))
    except Exception:
        return Path(hf_hub_download(REPO_ID, name, repo_type="dataset", revision=REVISION))


def _load_test() -> list[Input]:
    out = []
    for lang in LANGS:
        with open(_test_file(lang), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(_from_raw(json.loads(line)))     # labels, if present in the line, are discarded here
    return out


def load_inputs(source: str) -> list[Input]:
    """source: 'test' | 'prompt_pool' | 'dev_select' | 'dev_final' (dev sets obey the locks in data.py)."""
    if source == "test":
        return _load_test()
    return [_from_raw(r) for r in rows_for(source)]


def load_inputs_by_ids(ids: set[str]) -> list[Input]:
    """Pool documents by id (e.g. the pool_eval2 documents)."""
    return [_from_raw(r) for r in rows_for("prompt_pool") if r["id"] in ids]
