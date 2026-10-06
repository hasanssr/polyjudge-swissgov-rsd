# Test results (public test split, Task 5b)

Scored once on 168 documents (56 per language), 56 English chunks; official `compute_correlations` (pooled per language over both sides, gold −1 ignored); macro = mean of de/fr/it.
Code hash `8391d36ad46ca14bd1b5da25dbdd1eb6ee8ca043fff38f5e71b595d329e0bb62` (= `FREEZE_CODE_HASH` of Addendum 3); prediction files and their sha256: `docs/test_prediction_hashes.md`; script: `scripts/60_score_test.py`; raw numbers: `output/test_results/results.json`.

## Scores

| method | DE | FR | IT | **macro** | macro 95 % CI (chunk bootstrap) | diff to M0 [95 % CI] | P(diff>0) |
|---|---|---|---|---|---|---|---|
| B0 (additional baseline) | 0.390 | 0.108 | 0.320 | **0.273** | [0.201, 0.330] | +0.060 [-0.003, +0.116] | 0.97 |
| M0 | 0.296 | 0.122 | 0.221 | **0.213** | [0.139, 0.278] | (reference) | – |
| M1 | 0.396 | 0.108 | 0.324 | **0.276** | [0.204, 0.334] | +0.063 [+0.001, +0.118] | 0.97 |
| M2 | 0.354 | 0.131 | 0.284 | **0.256** | [0.187, 0.314] | +0.043 [-0.021, +0.101] | 0.92 |
| M3 | 0.373 | 0.079 | 0.316 | **0.256** | [0.195, 0.301] | +0.043 [-0.007, +0.098] | 0.95 |

## Differences to the top candidate (M1), paired over chunks (same 1 000 resamples, seed 0)

| method | macro | diff to M1 [95 % CI] | P(diff>0) |
|---|---|---|---|
| B0 | 0.273 | -0.003 [-0.005, -0.001] | 0.00 |
| M0 | 0.213 | -0.063 [-0.118, -0.001] | 0.03 |
| M2 | 0.256 | -0.020 [-0.041, -0.000] | 0.03 |
| M3 | 0.256 | -0.019 [-0.044, +0.005] | 0.07 |

## Selection (Addendum 3, applied mechanically; eligible: M0–M3, B0 never)

* Ranking among eligible: M1 0.276 > M2 0.256 > M3 0.256 > M0 0.213
* Rule: top two (M1 0.2762, M2 0.2564) differ by 0.0198 >= 0.005 -> highest macro Spearman: M1
* **Main system: M1**
* Top two paired difference (M2 − M1): -0.020 [-0.041, -0.000]

**Caveat.** Choosing the best of four candidates on the same split is optimistic (selection bias): the main system's test score is not an unbiased estimate of its performance on new data; the CIs above are per-candidate and do not account for the selection.

## Stability: dev_select vs test (macro Spearman)

dev_select is the budget-scaled evaluation subset (15 documents, 5 chunks) — a very small sample; its ranking is only a rough check.

| method | dev_select macro | test macro | dev_select rank | test rank |
|---|---|---|---|---|
| B0 | 0.382 | 0.273 | 3 | 2 |
| M0 | 0.251 | 0.213 | 5 | 5 |
| M1 | 0.386 | 0.276 | 2 | 1 |
| M2 | 0.392 | 0.256 | 1 | 3 |
| M3 | 0.359 | 0.256 | 4 | 4 |

## Tokens flagged nonzero (test, gold −1 tokens excluded)

| method | predicted > 0 | gold > 0 | predicted = 1.0 | gold = 1.0 | scored tokens |
|---|---|---|---|---|---|
| B0 | 5012 | 15840 | 5012 | 6254 | 132080 |
| M0 | 33616 | 15840 | 6384 | 6254 | 132080 |
| M1 | 5086 | 15840 | 5012 | 6254 | 132080 |
| M2 | 8482 | 15840 | 5012 | 6254 | 132080 |
| M3 | 4121 | 15840 | 4121 | 6254 | 132080 |

## API calls sent in the test run (cost for the tie rule)

| method | calls sent (70B + 8B) | pipeline stages |
|---|---|---|
| M0 | 956 | 1 |
| M1 | 332 | 2 |
| M2 | 2494 | 3 |
| M3 | 1391 | 2 |
