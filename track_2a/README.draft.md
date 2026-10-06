# PolyJudge — Track 2A (UZH: cross-lingual semantic differences in Swiss government websites)

(Draft. The organizers' challenge text stays in `README.md`; merge or replace as you prefer.)

> **AI-assistance disclosure.** Implementation was done with Claude Code (Anthropic) as a coding assistant. The research design, preregistration protocol, experiment decisions and all interpretation were directed and reviewed by the author.

Two systems are presented, both reproduced by `make run` **offline** (no API key, no network):

| System | Output | How it works offline |
|---|---|---|
| **Main system M1** — model-free sentence alignment + Apertus-70B-v1.5 verifier (flags accepted only at level 1.0 and in 2 of 2 independent samples) | `outputs/predictions/` | Apertus responses are replayed from the committed cache `data/replay_cache/` (332 entries); a cache miss is an error, never an API call |
| **B0** — model-free alignment baseline (additional, documented, reproducible) | `outputs/baseline_b0/` | No model calls at all |

Comparison methods M0 (simple Apertus prompting baseline), M2 (multi-agent), M3 (Apertus-judged alignment) are kept in the repo and
documented in the report; their caches are in `data/replay_cache/extras/`.

## Requirements
Docker and `make`. Nothing else (no GPU, no API key, no model weights). Python is only needed for `make validate`.

## Run (judges)
```bash
git clone https://github.com/hasanssr/polyjudge-swissgov-rsd.git
cd polyjudge-swissgov-rsd/track_2a
make run
```
`make run` builds the image, then (with `docker run --network none`) writes M1 predictions to `outputs/predictions/` and B0
predictions to `outputs/baseline_b0/`, and validates both against the input files. It does **not** score anything.

Output: `<method>_admin_{de,fr,it}.jsonl.jsonl`, one row per input document in input order, fields
`id, text_a, text_b, labels_a, labels_b`; one float in [0, 1] per whitespace token, `-1` on ASCII punctuation tokens only
(`docs/prediction_format.md`). The predictions submitted with this repo are in `predictions/` (M1, B0) and `predictions/extra/`
(M0, M2, M3); SHA-256 values are in `predictions/SHA256SUMS` (identical to the hashes recorded before any test label was read) and `make run` reproduces them bit for bit.

## Other targets
| Command | What it does |
|---|---|
| `make run METHOD=M2` | An extra method (`M0`, `M2`, `M3`, or `B0` alone) from the replay cache → `outputs/extra/<METHOD>/`, also offline (`--network none`). M0 and M2 take their few-shot examples from the frozen file `data/few_shot/` (see its `ATTRIBUTION.md`). All five methods reproduce `predictions/` byte for byte. |
| `make live` | Re-runs M1 against Apertus. Needs `LLM_NAME`, `LLM_BASE_URL`, `LLM_API_KEY` in the environment (values are never printed). New responses are written to the cache inside the container only. |
| `make eval PRED_PREFIX=outputs/<dir>/<method>_admin_ [SPLIT=dev]` | Scores predictions with the organizers' own script: the image build clones github.com/ZurichNLP/SwissGov-RSD at commit `1807a42100e742ed03d337c54c4b9ea86995f565` (their repo has no licence, so nothing of it is vendored). Needs network at build time and is heavy (their requirements include torch). **`make eval` was not executed in this repo's verification** (only `make -n eval`); everything else listed here was run. |
| `make test` | Unit tests that need no organizers data (network disabled). |
| `make validate` | Runs the format validator on the host (python3 only). |

```bash
export LLM_NAME=swiss-ai/Apertus-v1.5-70B  LLM_BASE_URL=...  LLM_API_KEY=...
make live
```

## Layout
- `src/polyjudge/` code (runtime modules); `src/dev_only/cv.py` leave-one-chunk-out analysis, dev only (not in the image)
- `scripts/validate_predictions.py` format validator, `scripts/cache_manifest.py` which cache keys each method reads, `scripts/export_few_shot.py` how `data/few_shot/` was produced, `scripts/scan_secrets.py` secrets/label scan
- `requirements.lock` (pip freeze of the runtime image; the Dockerfile installs it), `requirements.txt` (top-level pins)
- `data/test_input/` test inputs, **text only** (CC-BY-4.0, ZurichNLP/SwissGov-RSD; no labels); `data/splits/` split ids; `data/replay_cache/` (M1) and `data/replay_cache/extras/` (M0, M2, M3); `data/few_shot/` the 16 frozen prompt-pool documents used as in-prompt examples by M0 and M2 (CC-BY-4.0, with attribution) — total well under 100 MB
- `predictions/` submitted predictions; `docs/` preregistration (with addenda), test results, hashes, prediction format, `code_changes_after_freeze.md`; `technical_report.md`

See `technical_report.md` for method, evaluation and limitations.
