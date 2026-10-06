# Preregistration — candidate selection on `dev_select` (written 2026-10-05, BEFORE any run on dev_select)

CODE_HASH: `5db749fb4c56b4d213ef278506bff04ffbb13a12d58302e9fa91121834bd05d7`

`scripts/43_dev_select.py` recomputes this hash (all `polyjudge/*.py` except `codehash.py`, plus the split files) and refuses to run if it differs. Nothing in `polyjudge/` or `splits/`
may change between this document and the dev_select report; the scripts in `scripts/` only orchestrate.

## Context
The challenge owner confirmed that the **main submitted system must build on Apertus**; the model-free method is submitted as an additional, documented baseline. Apertus-based
candidates will each be run once on the test split; the best one becomes the main system. This preregistration fixes how candidates are compared on `dev_select` before anyone has seen their scores.

## Data and locks
* `dev_select` = 40 chunks (120 documents) of the former dev_eval, fixed seed 2026 (`splits/dev_eval_split.json`). `dev_final` = the other 102 chunks: **locked**, not opened in this task.
* Prompts/rules may be improved on `prompt_pool` only. **No prompt, rule, split or hyper-parameter was changed after the last prompt_pool result** (the prompt_pool/eval sets
  were used for all earlier experiments; no further improvement was made this round because the call budget is needed for dev_select).
* Test labels are not readable by any code in `polyjudge/` before Task 5 (`inputs.py` keeps a whitelist of text fields); only test *inputs* are used (cost estimates).

## Candidates (exactly these five; none will be added after results are seen)
| id | system | Apertus use |
|---|---|---|
| **B0** | model-free: segmentation → aligner v2 (`BEST_V2`) → tokens of sentences without counterpart = 1.0, rest 0, ASCII punctuation −1 | none (additional baseline) |
| **M0** | simple prompting baseline: one B1 call per sentence group (≤ 6 aligned beads / ≤ 1 400 chars; all sentences), 3 frozen few-shot examples per language (`splits/shots_round1.json`), 70B, T = 0, JSON mode, repetition guard, one retry (T 0.3); spans mapped verbatim (exact only, unmapped quotes dropped); no heuristic | 70B |
| **M1** | B0 + verifier on deterministic flags (numbers without counterpart, unmatched parentheticals, untranslated sentence) inside aligned beads; 70B, two independent samples (T 0; T 0.7 seed 101); a flag is accepted only if **both** samples say "real" at **level 1.0**; final = B0 + 0.2·1.0 on accepted tokens | 70B ×2 |
| **M2** | PolyJudge multi-agent on aligned groups: blind extractors (EN / target, 70B) → Compare-v2 (≤ 8-word spans, reason first, same-info filter, 4 gold-derived few-shot demos from tuning chunks) → verifier (70B, T 0); accepted = "real" with its level; final = B0 + 0.2·level | 70B (extractors, compare, verifier) |
| **M3** | Apertus-judged alignment: 8B answers "which candidate carries the same content / none" for every sentence in both directions (letter-labelled candidates, 2 anchor items, batch 8); the 70B judge decides the sentence-level disagreements with the aligner (batch 8); presence score: both unaligned 1.0, judged verdict 1.0/0, both aligned 0, unjudged disagreement 0.5 | 8B + 70B |

Fixed constants: weight of accepted flags 0.2 (rank-invariant for a binary base), aligner v2 = {complex_penalty 1.5, tau 1.0, no trim, no rescue}, groups ≤ 6 beads/≤ 1 400 chars,
verifier batches ≤ 6 pairs/≤ 3 200 chars/≤ 12 flags.

## Primary metric and uncertainty
* **Primary metric: official macro Spearman** = mean over DE/FR/IT of the organizers' `compute_correlations` pooled per language over both sides and all evaluated documents.
* **Uncertainty: paired chunk bootstrap**, 1 000 resamples, seed 0, resampling chunks with replacement (identical resamples for all candidates); reported: 95 % CI of each candidate's macro score and
  of its **difference to M0** (mean, 95 % CI, P(diff > 0)). Bootstrap draws use scipy's Spearman; point estimates use the official function.

## Cost-dependent evaluation subset (decided by this rule, not by results)
Estimated cost per document, from the prompt_pool dry run (78 docs, inputs only): 70B ≈ M0 5.4 + M1 2.0 + M2 16.2 + M3 0.2 = **23.8**, 8B ≈ M3 7.8. Budgets this round: 70B 500, 8B 300 calls.
The full dev_select (120 documents) would need ≈ 3 000 70B and ≈ 940 8B calls — far over budget. Therefore:
* the evaluation subset = the **first K chunks of a seeded permutation (seed 7) of dev_select**, K = the largest value for which the summed dry-run estimate of the five candidates stays ≤ 85 % of
  each budget (70B ≤ 425, 8B ≤ 255);
* all five candidates are run on the same documents; no document is added, removed or replaced after seeing results.
**Consequence (stated in advance):** with K ≈ 5 chunks the confidence intervals will be wide and rankings between close candidates are largely noise; the report will say so.

## Decision rule
1. Candidates are ranked by their dev_select macro Spearman (primary metric).
2. Any candidate more than **0.02 below the best** may be dropped from the test run to save budget. B0 (required additional baseline) and M0 (reference for "improvement over simple prompting") are always kept.
3. All remaining candidates are run once on the test inputs (Task 5); predictions are frozen and hashed before the test labels are fetched; the best Apertus-based candidate (M0–M3) on the test score becomes the main system.
4. Secondary diagnostics (not used for selection): call counts, parse-failure rates, accepted-flag counts.

## Known limitations declared in advance
* dev_select score differences below ≈ 0.05 are not distinguishable with this subset; the 0.02 drop rule can therefore remove a candidate by chance.
* All prompts were developed on prompt_pool with ≤ 18 documents per experiment; no ablation of the 0.2 weight is possible (rank-invariant).
* M2 and M3 depend on several independent LLM calls per document; failed batches degrade gracefully (zero flags / aligner verdict) and are counted.


---

## Addendum (added on 2026-10-05, after the dev_select run had started)

This section was appended after the dev_select run (`scripts/43_dev_select.py run --mode live`, started 2026-10-05 12:22 local time) had begun; B0, M0 and M1 had already been written
and M2 was running. **Nothing above this line was edited.** SHA-256 of the preregistration text above this addendum (identical to the hash the runner recorded when it started):
`6c329cb475ed041ba33c186e5f87b6e6206a378587070094a253670caab55445`.

### Two confirmations from the challenge owner
UZH (**Michelle Wastl**, challenge owner) confirmed by e-mail on **5 October 2026** (as relayed to the project; the e-mail itself is not stored in this repository):

* **(a) What counts as Apertus-based.** A hybrid counts as Apertus-based when the **sentence pairing (alignment) is completely model-free**. Consequently **M1 and M2 qualify** as Apertus-based
  candidates, and **the aligner used inside M1 and M2 must not use any model** (no embedding model, no LLM, no API call path). In this project that aligner is the dynamic-programming
  aligner v2 (`polyjudge.aligner.HeuristicAlignerV2`); `tests/test_model_free_aligner.py` asserts that its import closure and the pairing stack around it contain no model or LLM code.
* **(b) Use of the public test split.** The public test split **may be used for the final ranking, after all methods are frozen**. This matches Task 5 (predictions frozen and hashed before the test labels are fetched).

### What this addendum does NOT change
* It changes **neither the candidate list** (B0, M0, M1, M2, M3 — none added, removed or modified) **nor any selection rule** (primary metric, paired chunk-bootstrap CI against M0, the evaluation-subset rule with its
  85 % budget cap, the "more than 0.02 below the best may be dropped" rule, the freeze-then-test procedure) **nor any threshold, constant or prompt**. The CODE_HASH above remains the pinned code.
* The confirmations only record how the two statements in the "Context" section are to be read: M1 and M2 qualify as Apertus-based because their pairing is model-free; the public test split is the final-ranking set.


---

## Addendum 2 (added on 2026-10-05, after the dev_select report): single-flight in `LLMClient.chat`

Approved by the project owner on 2026-10-05 after the dev_select report. **Nothing above this line was edited.**

* **Change.** `polyjudge/llm.py`: identical requests that are in flight at the same time are merged into **one API call** (same cache key, same result — or the same error — for every waiter);
  requests with `use_cache=False` are never merged. Supporting additions in the same file, none of which changes what is sent: an adaptive concurrency limiter (4 → 8 only after 100 requests without
  a 429, pinned to the base value after any 429), a chained budget (a request must fit every budget it is charged to) and a record of the cache keys each client used. `polyjudge/predict.py`:
  `make_clients` passes the budgets / concurrency options to the real clients, and the dry-run *estimate* of the judge calls was corrected (assumed disagreement rate 2.8 % → 9.0 %, as observed on dev_select);
  estimates never influence predictions. New tests: `tests/test_singleflight.py` (concurrent identical requests → 1 call, errors reach every waiter, adaptive concurrency, chained budgets) and `tests/test_runner.py`.
* **Old and new CODE_HASH.** old (Addendum-free preregistration): `5db749fb4c56b4d213ef278506bff04ffbb13a12d58302e9fa91121834bd05d7`;
  **new: `8391d36ad46ca14bd1b5da25dbdd1eb6ee8ca043fff38f5e71b595d329e0bb62`** (same definition: `polyjudge/*.py` except `codehash.py`, plus the split files).
* **Unchanged.** The candidate list (B0, M0, M1, M2, M3), all prompts, all hyper-parameters, the primary metric, the bootstrap, the evaluation-subset rule and the "more than 0.02 below the best" rule.
* **dev_select results.** B0, M0, M1 and M3: **unaffected** — regenerated in replay mode (cache only, 0 calls) with the new code, their prediction files have the same sha256 as the live run
  (12 of 12 files identical; full hashes in `docs/addendum2_replay_hashes.json`). **M2:** its dev_select predictions and scores **stand as produced live**; they are not recomputed. Their replay fails
  (20 of 86 Compare requests are not in the cache) because, before this change, identical extractor requests (the English text of a chunk is identical in its de/fr/it versions) were sent as separate calls whose answers differ slightly; the cache kept only the last answer. With single-flight this cannot happen for new runs.

---

## Addendum 3 (added on 2026-10-05, BEFORE any test prediction is generated): rules for the final test ranking

FREEZE_CODE_HASH: `8391d36ad46ca14bd1b5da25dbdd1eb6ee8ca043fff38f5e71b595d329e0bb62`

`scripts/50_run_test.py` refuses to generate test predictions unless the current code hash equals this value. Nothing in `polyjudge/` or `splits/` may change after this line. The runner and replay scripts
(`scripts/50_run_test.py`, `scripts/52_replay_test.py`) only orchestrate; their sha256 values are recorded in `output/test_run/state.json` (manifest) when the run starts.

1. **Eligible main systems: M0, M1, M2, M3.** B0 is **never** the main system; it is the additional, documented, reproducible model-free baseline (and is reported like the others).
2. **Main system = the eligible candidate with the highest official macro Spearman on the test split.** If the top two differ by **less than 0.005**, the **cheaper** one is taken, where cost = the total
   number of API calls actually sent in the test run (70B + 8B, from the runner's counters); if still equal, the one with fewer pipeline stages (M0 1, M1 2, M3 2, M2 3).
3. **Reporting.** All five candidates (B0, M0, M1, M2, M3) are reported with their official per-language and macro Spearman, a 95 % CI of the macro score, and the **paired bootstrap 95 % CI of the difference
   to M0** (resampling unit: the **test chunks**, 1 000 resamples, seed 0, identical resamples for all candidates). The report states explicitly that **choosing the best of four candidates on the same split is optimistic**
   (selection bias): the main system's test score is not an unbiased estimate of its performance on new data.
4. **Freeze, then fetch.** (a) Test predictions are generated from the test **inputs only** and saved with sha256 hashes; (b) every method is re-run in replay mode (0 calls) and the hashes compared; (c) only after the project owner
   states "scores approved to open" are the test labels fetched, by one committed script, with a row-alignment check against the inputs; (d) every candidate is scored **exactly once**; (e) nothing is changed afterwards.
   A method that is incomplete when the labels are opened (budget exhausted, failure) is reported as incomplete and excluded from the ranking; re-runs are allowed only **before** the labels are opened (from the cache).
5. **Budgets and run order.** Global limits **70B 4 300 / 8B 1 500 calls**; per-method sub-limits = dry-run estimate + 15 % (capped at the global limit): M1 381, M3 158 (70B) and 1 500 (8B), M0 1 130, M2 3 278 (70B);
   a request that would exceed any limit is never sent. Order: **B0, M1, M3, M0, M2**. Concurrency 4, raised to 8 only if no 429 has occurred. The M2 sub-limit is conservative: the dry-run estimate counts 2 850 requests, of which about 2 474 are distinct (≈ 13 % fewer calls with single-flight).
6. **What this addendum does not change.** Candidates, prompts, hyper-parameters, the primary metric and the selection rules of the preregistration and of Addendum 1 and 2. The dev_select ranking (M2 > M1 > B0 > M3 > M0) is not used to drop any candidate:
   the project owner decided that M3 stays in the test run.
