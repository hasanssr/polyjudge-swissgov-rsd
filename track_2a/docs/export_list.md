# Export list for the packaging repo (Task 6) — a list only, nothing was moved or copied

State: code hash `8391d36ad46ca14bd1b5da25dbdd1eb6ee8ca043fff38f5e71b595d329e0bb62` (frozen, Addendum 3). Main system by Addendum 3: **M1** (see `docs/test_results.md`). Python 3.11.9 on Windows 11.
Mapping and `make run` design: `docs/packaging_plan.md` (`polyjudge/` → `src/polyjudge/`, replay cache → `data/llm_cache/`, predictions → `data/predictions/`).

## 1. Modules of `polyjudge/` (25 files, 470 685 bytes)

All are needed at run time **except** `cv.py` (leave-one-chunk-out analysis, dev only) and the empty `__init__.py` (keep it as the package marker).

| group | modules |
|---|---|
| prediction entry | `predict.py` (methods B0, M0–M3, `RUNNERS`, `predict`, `make_clients`, `write_predictions`, `read_predictions`, CLI `python -m polyjudge.predict`), `inputs.py` (inputs-only loader) |
| LLM access | `llm.py` (cache, replay mode, single-flight, budgets), `config.py` (`.env` / env vars `LLM_NAME`, `LLM_BASE_URL`, `LLM_API_KEY`) |
| model-free base (B0) | `aligner.py`, `heuristic.py`, `segmentation.py`, `spans.py`, `diffs.py`, `detlayer.py` |
| M0 | `pipeline.py`, `prompts.py`, `groups.py` |
| M1 / M2 / M3 | `candidates.py`, `verifier.py`, `debate.py`, `judge.py`, `lla.py`, `prompts_v2.py` |
| data / protocol | `data.py` (dev loaders and locks), `evalsets.py`, `codehash.py` |
| scoring (not needed for `make run`) | `score.py`, `official_eval.py` — **`official_eval.py` imports the organizers' `compute_correlations` out of `organizers_repo/scripts/evaluate_predictions_admin.py` (AST extraction; that repo has no licence and is not vendored).** For the packaging repo either drop the scoring modules from `make run`, or point the evaluation step at the organizers' script path. |
| dev only | `cv.py` |

Scripts worth shipping (`scripts/`, 50 files, 202 300 bytes): `50_run_test.py` (resumable test runner), `52_replay_test.py` (replay check), `60_score_test.py` (label fetch + single scoring), `43_dev_select.py`, `44_score_select.py`, `45_replay_check.py`, `_fakesdk.py`; the rest is dev tooling. Tests: `tests/` (15 files, 60 193 bytes; last full run 94 + 4 passed).

## 2. Third-party imports (installed versions)

| package | version | used by |
|---|---|---|
| numpy | 2.4.6 | alignment, scoring |
| scipy | 1.17.1 | `score.py` (bootstrap Spearman) |
| nlpstats | 0.0.1 | `official_eval.py` (organizers' correlation helper) |
| huggingface_hub | 1.33.0 | `data.py`, `inputs.py` (pinned dataset revision `f16aa0536811b8c0cf975fede21eec5996e4b3c1`) |
| openai | 3.24.0 | `llm.py` (OpenAI-compatible client for the Apertus endpoint; live mode only) |
| python-dotenv | 1.2.4 | `config.py` (imported as `dotenv`) |
| pytest | (dev) | `tests/` |

Standard library otherwise. Not imported by any module: pandas, sklearn, torch, transformers (their absence is intentional; the organizers' CLI needs them, we extract the one function).

## 3. Data sizes

| item | files | size |
|---|---|---|
| `splits/*.json` (dev_eval_split, dev_split, pool_eval12, pool_eval2, shots_round1) | 5 | 9 384 bytes |
| test inputs, labels removed (`id, subset, text_a/b` tokenized + untokenized), 168 docs | – | ≈ 1.86 MB as JSONL (`load_inputs("test")` reads them from the HF cache; for an offline `make run` ship this inputs-only file, **never the label-bearing test files**) |
| test predictions, 5 methods × 3 languages (`output/test_run/predictions/`) | 15 | 7 815 058 bytes |
| replay cache of the test run, all five methods | 5 174 | **5 066 793 bytes (5.07 MB raw)** — limit of `data/`: 100 MB |
| `docs/` | 14 | 98 397 bytes |

## 4. Replay-cache files (the test run)

Layout `cache/<key[:2]>/<key>.json` (key = sha256 of model, messages, temperature, seed, max_tokens, response_format); the complete list with the sha256 of every file is in `docs/export_replay_cache_files.txt` (5 174 lines). No key is shared between methods.

| method | cache entries | bytes | API calls sent in the run |
|---|---|---|---|
| B0 | 0 | 0 | 0 (model-free) |
| M0 | 956 | 720 944 | 956 (70B) |
| **M1 (main system)** | **332** | **622 701** | 332 (70B) |
| M2 | 2 495 | 3 058 895 | 2 494 (70B) |
| M3 | 1 391 | 664 253 | 1 314 (8B) + 77 (70B) |
| all | 5 174 | 5 066 793 | 70B 3 859 / 8B 1 314 |

`scripts/52_replay_test.py` re-generated all five methods from these files with 0 requests: all prediction hashes identical, 0 missing. If only the main system is shipped, `data/llm_cache/` needs just the 332 M1 entries (0.62 MB) plus B0 (no cache) for the model-free fallback; the other methods are documented in the report and reproducible from the full cache (5.07 MB), which fits comfortably. The older dev_select cache entries (M2 of the dev run is not replayable, see Addendum 2) are not part of the export.

## 5. Split files and protocol evidence to ship

* `splits/dev_split.json` (prompt pool and tuning split), `splits/pool_eval12.json`, `splits/pool_eval2.json`, `splits/shots_round1.json`, `splits/dev_eval_split.json` (dev_select / dev_final chunk ids; chunk ids only, no text).
* `docs/prereg.md` (original + Addenda 1–3), `docs/test_prediction_hashes.md`, `docs/test_results.md`, `docs/addendum2_replay_hashes.json`, `docs/export_replay_cache_files.txt`, `docs/prediction_format.md`, `docs/apertus_role.md`, `docs/dev_select_report.md`, `docs/technical_report_notes.md`.
* Backup of the whole test run (outside the project, outside OneDrive): `C:\Projects\backup_test_run_2026-10-05` (5 225 files, 22 281 234 bytes, all sha256 verified; `hashes.txt`).

## 6. Never ship

`.env`, `organizers_repo/`, `__pycache__`, `.pytest_cache`, the HF cache, `output/test_run/runner.lock`, `output/test_run/calls.jsonl` (raw call log), any file with test gold labels (the label-bearing JSONL files are only read by `scripts/60_score_test.py`, which does not store them).
