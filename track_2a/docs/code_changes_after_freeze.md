# Code changes after the freeze

The prediction code was frozen before the test run (Addendum 3). Packaging for `make run` required a few changes. None of them
touches an algorithm, a prompt, a parameter or the cache key; they only change where inputs, few-shot examples and the
scorer come from.

| | SHA-256 (`polyjudge/codehash.py`: package `*.py` except codehash.py + the 5 split files) |
|---|---|
| frozen (development project, used for the test run) | `8391d36ad46ca14bd1b5da25dbdd1eb6ee8ca043fff38f5e71b595d329e0bb62` |
| this repo (`/app` layout inside the image) | `d5a8bb46865c3fe1a0c067aaa3e76eb50b870db89df63d7829ece2a67929b735` |

The split files are byte-identical to the frozen ones; the hash differs only because of the file changes below.

## What changed
| file | change |
|---|---|
| `inputs.py` | `_test_file` first looks for the inputs-only file `data/test_input/test_<lang>.jsonl` (text fields only, no labels) and only otherwise falls back to the Hugging Face cache. Needed so `make run` works with no network. The whitelist loader itself is unchanged. |
| `official_eval.py` | **Replaced by a shim.** The original extracted the organizers' `compute_correlations` from their (unlicensed) repo at import time; `pipeline.py` imports the module at the top, so it could not simply be dropped. The shim imports nothing at load time and loads the function only if `ORGANIZERS_REPO` is set (`make eval` clones the organizers' repo, pinned, inside the image). No predictions depend on it. |
| `prompts.py`, `prompts_v2.py` | `frozen_shots` (M0) and `select_shots2` (M2) read the few-shot documents from the frozen file `data/few_shot/` when it exists (16 prompt-pool documents, CC-BY-4.0, see `data/few_shot/ATTRIBUTION.md`) instead of loading the whole dev split. The M2 selection was run once with the original code and frozen in `data/few_shot/m2_shots.json` (`scripts/export_few_shot.py`); the groups and the expected outputs are rebuilt from the documents with the original functions. |
| `cv.py` | moved to `src/dev_only/` (dev-only analysis, not in the image), content unchanged. |

All other modules are byte-identical to the frozen versions (compared after line-ending normalisation).

## Evidence that the predictions did not change
Fresh checkout, image built from scratch, then `make run` and `make run METHOD={M0,M2,M3}`; every `docker run` used `--network none`,
replay mode (a cache miss is an error). SHA-256 of every output file compared with `predictions/SHA256SUMS`
(which equals `docs/test_prediction_hashes.md`, recorded before any test label was read):

| method | output | files byte-identical |
|---|---|---|
| M1 (main) | `outputs/predictions/` | 3 / 3 |
| B0 | `outputs/baseline_b0/` | 3 / 3 |
| M0 | `outputs/extra/M0/` | 3 / 3 |
| M2 | `outputs/extra/M2/` | 3 / 3 |
| M3 | `outputs/extra/M3/` | 3 / 3 |

Because replay keys hash the full prompt, identical outputs with zero cache misses also show that the frozen few-shot
examples produce exactly the original M0/M2 prompts.
