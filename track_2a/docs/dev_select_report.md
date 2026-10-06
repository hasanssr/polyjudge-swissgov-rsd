# dev_select report (preregistered run, 2026-10-05)

Protocol: `docs/prereg.md` (CODE_HASH `5db749fb…`, unchanged during and after the run; addendum of 2026-10-05 appended afterwards without changing any rule). dev_final locked, test labels untouched.
Evaluation subset by the preregistered budget rule: **first 5 chunks of the seeded permutation of dev_select = 15 documents (5 per language)**. Predictions: `output/dev_select/<M>_admin_<lang>.jsonl.jsonl`
(scored by `scripts/44_score_select.py`; `results.json` there).

## Result (official macro Spearman; CI = bootstrap over the 5 chunks, 1 000 resamples)

| candidate | DE | FR | IT | macro | macro 95 % CI | diff to M0 [95 % CI] | P(diff > 0) |
|---|---|---|---|---|---|---|---|
| B0 model-free (additional baseline) | 0.596 | −0.049 | 0.600 | 0.382 | [0.004, 0.461] | +0.052 [−0.223, +0.179] | 0.66 |
| **M0** simple Apertus prompting (70B) | 0.333 | 0.098 | 0.324 | 0.251 | [0.085, 0.310] | reference | – |
| M1 B0 + verifier, level 1.0 in 2 of 2 samples | 0.596 | −0.040 | 0.602 | 0.386 | [0.021, 0.462] | +0.062 [−0.212, +0.182] | 0.70 |
| **M2** extractors → Compare → verifier | 0.547 | 0.028 | 0.600 | **0.392** | [0.039, 0.468] | +0.087 [−0.179, +0.196] | 0.77 |
| M3 8B alignment + 70B judge | 0.622 | −0.047 | 0.504 | 0.359 | [−0.008, 0.434] | +0.032 [−0.211, +0.151] | 0.66 |

Preregistered ranking: M2 (0.392) > M1 (0.386) > B0 (0.382) > M3 (0.359) > M0 (0.251). Only **M3** is more than 0.02 below the best and *may* be dropped from the test run (B0 and M0 are always kept).
Every difference to M0 is positive in the point estimate, but **every paired CI contains 0**: with 15 documents / 5 chunks nothing can be distinguished statistically (declared in advance).

Run facts: M0 98 groups, 0 failed; M1 142 candidates, **5 flags accepted** (verifier failed batches 2 / 1); M2 86 groups, 95 candidates, 48 accepted flags, 0 failed extractor/compare calls;
M3 93 sentence-level disagreements (9.0 % of 1 028 sentences; the dry-run assumed 2.8 %), 0 failed judge batches.

## Calls (this round)
70B 434 / 500 used (66 left): M0 98, M1 39, M2 280, M3 17. 8B 149 / 300 used (151 left): M3 alignment questions. Estimates vs actual: 70B 424 → 434, 8B 149 → 149.

## Reproducibility finding (not fixed, to be decided)
Replay of the live run (cache only, 0 calls, `scripts/45_replay_check.py`): **B0, M0, M1, M3 reproduce bit-for-bit; M2 does not** — 20 of 86 Compare requests are not in the cache.
Cause: in M2 the English extractor request is identical for the three language versions of a chunk (and for repeated groups). Of the 172 extractor requests only **140 are distinct** (25 keys occur 2–3 times);
the client sends identical requests that are *in flight at the same time* separately, the 70B answers differ slightly between calls, the cache keeps only the last answer, but each group's Compare prompt had been built
from its own answer. Consequences: (1) the live M2 predictions and scores above are valid as produced; (2) M2 is not replayable from the cache; (3) 32 extractor calls were wasted (19 %).
Proposed fix (needs approval because it changes `polyjudge/llm.py`, i.e. the pinned CODE_HASH, and must be recorded in a second addendum with a new pin): single-flight coalescing of identical in-flight requests
in `LLMClient.chat`. It does not change any prompt, rule or candidate definition.

Other observation: the runner process stopped responding after writing all predictions (before its scoring step); it was ended and the same frozen scoring functions were run from the saved files.
