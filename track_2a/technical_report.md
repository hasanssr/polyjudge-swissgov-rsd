# Technical report — `PolyJudge`

A deeper write-up than the README: what we built, how it works, and what the
numbers say. Also covers the specific submission requirements of the UZH challenge.

- **Track:** `Track 2A — UZH: Detecting Cross-Lingual Semantic Differences in Swiss Government Websites`
- **Event:** Online
- **Team:** Solo — Hasan Sarı
- **Repository:** https://github.com/hasanssr/polyjudge-swissgov-rsd
- **Demo:** https://youtu.be/Yq5JUif_B9I (unlisted video); optional offline viewer in `demo/`

## 1. Summary

We predict, for each English government web page and its German, French or Italian version, a score in [0, 1] for every token on both sides (how strongly the token is part of a semantic difference). The **main system, M1**, splits both texts into sentences, pairs them with a model-free dynamic-programming aligner, scores every token of a sentence that has no counterpart as 1.0, and uses Apertus-70B as a verifier that may add a small extra flag on a few deterministic candidates (numbers without counterpart, unmatched parentheticals, untranslated sentences). **B0** is the same model-free base without the verifier; it is submitted as an additional, reproducible system. **M0** (simple Apertus prompting), **M2** (multi-agent) and **M3** (Apertus-judged alignment) are comparison methods.

On the public test split (168 documents, 56 per language, scored once) M1 reaches macro Spearman **0.276** (95 % CI [0.204, 0.334]), M0 reaches **0.213** ([0.139, 0.278]); the paired difference M1 − M0 is **+0.063 [+0.001, +0.118]**. The lower bound is barely above zero. The model-free B0 reaches 0.273; M1 is only 0.003 higher, and the verifier adds only 74 flagged tokens (5 086 vs 5 012). So **almost all of M1's score comes from the model-free base; the Apertus contribution on top of it is small.** M2 and M3 did not beat M1, and their CIs against M0 include zero. M1 was chosen as the main system by a rule written before the test run, from four candidates scored on the same split, so its score is optimistic.

The project name comes from the original multi-agent idea (M2, with a "judge" role); the submitted main system M1 is alignment-first.

### Problem

- **Task:** token-level recognition of semantic differences between an English document and its translated counterpart on Swiss government websites (SwissGov-RSD; Wastl, Vamvas, Sennrich).
- **Input / output:** one jsonl file per language (de, fr, it); fields `id, text_a, text_b, labels_a, labels_b`. `text_a` is the English side, `text_b` the target-language side. For each side we write one float in [0, 1] per whitespace token and `-1` on ASCII punctuation tokens (`docs/prediction_format.md`).
- **Metric:** the organizers' `compute_correlations`, token-level Spearman pooled per language over both sides, gold `-1` ignored; the headline number is the macro average over de/fr/it.

## 2. Architecture

Per the challenge owner's ruling, the **main system must build on Apertus**; a hybrid counts as Apertus-based when the sentence pairing is completely model-free (owner's e-mail of 5 October 2026; excerpts in `docs/owner_confirmations.md`, also recorded in `docs/prereg.md`, Addendum 1). `make run` reproduces M1 and B0 offline from a replay cache.

| id | role | what it is | Apertus use |
|---|---|---|---|
| **M1** | **main system** | B0 + verifier on deterministic flags; a flag counts only if both of two independent samples say "real" at level 1.0; final score = B0 + 0.2 on accepted tokens | 70B, 2 samples |
| **B0** | additional baseline, submitted and reproducible | model-free: segmentation → aligner v2 → tokens of sentences without counterpart = 1.0, rest 0, ASCII punctuation −1 | none |
| M0 | comparison | simple prompting: one 70B call per sentence group (≤ 6 aligned beads / ≤ 1 400 chars), 3 frozen few-shot examples per language, no heuristic | 70B |
| M2 | comparison | multi-agent on aligned groups: blind extractors (English / target) → Compare → verifier; final = B0 + 0.2·level of accepted flags | 70B in all three stages |
| M3 | comparison | Apertus-judged alignment: 8B answers "which candidate carries the same content / none" for every sentence in both directions; 70B judge decides disagreements with the aligner | 8B + 70B |

**Model-free base (B0, also the base of M1 and M2).** Sentence segmentation (rule-based, maps sentences to token ranges, keeps `-1` on ASCII punctuation); monotone sentence alignment by dynamic programming over 1-1, 1-2, 2-1, 2-2, 1-0 and 0-1 beads, scored with length ratio, relative position, shared numbers/dates (months normalised across EN/DE/FR/IT), names and acronyms, plus a penalty for weakly supported complex beads; fixed parameters {complex_penalty 1.5, tau 1.0, no trim, no rescue}. These were chosen on `prompt_pool` only. `tests/test_model_free_aligner.py` checks, by import closure and at runtime, that the aligner and the flag generation load no LLM client, no embedding model and no network code, and that M1 and M2 never call the LLM alignment code that belongs to M3.

**Verifier in M1.** Deterministic candidates inside aligned beads (numbers without counterpart, unmatched parentheticals, untranslated sentence) are shown to Apertus-70B (T = 0 and T = 0.7 with seed 101); a candidate is accepted only if both samples answer "real" at level 1.0. Batches hold at most 6 pairs, 3 200 characters and 12 flags. The weight 0.2 is irrelevant for the ranking: the base is binary, so any weight in (0, 1) gives the same order.

Data flow of the main system M1:

```text
test inputs (id, text_a, text_b; no labels)
   -> sentence segmentation (rule-based, maps sentences to token ranges)
   -> model-free aligner v2 (DP over 1-1 / 1-2 / 2-1 / 2-2 / 1-0 / 0-1 beads)
   -> sentences without counterpart: all their tokens = 1.0, everything else 0, ASCII punctuation -1
   -> deterministic flag candidates inside aligned beads
        (numbers without counterpart, unmatched parentheticals, untranslated sentence)
   -> Apertus-70B verifier, 2 independent samples (T 0 and T 0.7, seed 101);
        a flag is kept only if both say "real" at level 1.0
   -> final scores: base + 0.2 on the tokens of kept flags -> <method>_admin_{de,fr,it}.jsonl.jsonl
```

Code is in `src/polyjudge/`; the module list is in `docs/export_list.md`.

## 3. Use of Apertus

- **Model:** Apertus-v1.5, model ids `swiss-ai/Apertus-v1.5-70B` (set by `LLM_NAME`; used for M0, M1, M2 and the M3 judge) and `swiss-ai/Apertus-v1.5-8B` (M3 alignment questions). The service exposes no model revision information, so none is recorded.
- **Where it runs:** to the best of our records, all Apertus inference ran remotely on the CSCS Inference API (`https://api.inference.cscs.ch/v1`); `make live` refuses any other host. No local weights and no local GPU.
- **How it is used:** inference only, through an OpenAI-compatible endpoint. Prompts use JSON mode, temperature 0 (one retry at 0.3 for M0; the second M1 sample at 0.7 with seed 101), a repetition guard and parse-failure retries.
- **Role:** in M1, Apertus is a verifier of a few candidate flags on top of a model-free base. We do not claim more than the measured effect: on the test split it changes 74 tokens and the macro score by +0.003 relative to B0. Responses are cached; `make run` replays them with no network, `make live` re-queries.
- **Selection protocol:** `docs/prereg.md` was written before running on `dev_select` and has three addenda (see section 4). `dev_final` stayed locked. Candidates were screened on `dev_select`, then each was run once on test; the main system was picked by the rule in Addendum 3 (highest test macro Spearman among M0–M3, tie rule by cost; B0 never eligible).
- **Experiments and negative results** (all on `prompt_pool`, exploratory, 12–78 documents; from `docs/round1_findings.md`, `docs/round2_findings.md`, `docs/apertus_role.md`):

| use of Apertus | model | marginal effect (Spearman) |
|---|---|---|
| direct span + severity per sentence group (B1), combined with the heuristic | 8B / 70B | 0.219 / 0.210 vs 0.399 (12 docs): −0.18 / −0.19 |
| Compare step on aligned pairs, `max(H, ·)` | 8B / 70B | 0.237 / 0.278 vs 0.399: −0.16 / −0.12 |
| Compare-v2 (≤ 8-word spans, reason first, filters) + number layer, weight 0.2 | 8B / 70B | 0.349 vs 0.457 (8B); 0.317 vs 0.408 on 9 docs (70B): −0.11 / −0.09 |
| LLM sentence alignment (letter-labelled candidates; says NONE for 20 of 1 439 sentences) | 8B | −0.009 / +0.004 on two eval sets: about 0 |
| 70B judge on the 40 aligner/8B disagreements | 70B | −0.074 / +0.011; right on 29 of 40 vs 35 of 40 for the aligner |
| LLM verifier of candidate flags, all 78 documents | 70B | accepted-flag precision 32.1 %, recall 7.2 %, gain −0.021 [−0.043, +0.021] |
| same verifier, only level-1.0 flags (post hoc, exploratory) | 70B | 32 flags, precision 64.9 %, recall 0.6 %, gain +0.004 [−0.001, +0.011] |
| debate agents (extractors → Compare → verifier), 6 chunks / 18 docs | 70B | precision 24.3 %, gain −0.020 |

- **Prompts, serving stack, sampling settings:** prompts are in `src/polyjudge/prompts.py` and `prompts_v2.py`; the few-shot documents are frozen in `data/few_shot/` (16 `prompt_pool` documents). Cache key = sha256 of model, messages, temperature, seed, max_tokens and response_format. Concurrency 4, raised to 8 only after 100 requests without a 429; identical in-flight requests are merged into one call (single-flight, Addendum 2).
- **Other models used during development (judges, etc.):** none besides Apertus 8B and 70B.
- **AI-assistance disclosure:** Implementation was done with Claude Code (Anthropic) as a coding assistant. The research design, preregistration protocol, experiment decisions and all interpretation were directed and reviewed by the author.

## 4. Data

- **Source and licence:** SwissGov-RSD (Wastl, Vamvas, Sennrich), CC-BY-4.0, from the Hugging Face dataset (pinned revision `f16aa0536811b8c0cf975fede21eec5996e4b3c1`). The dataset card says the data was re-annotated in April 2026; we use the current labels. Personal / non-redistributable data: none (public government web text). Human-subject consent: n/a. The repository contains test **inputs only** (text fields, no labels) and 16 few-shot documents with attribution.
- **Split protocol:**
  - `prompt_pool` (78 documents = 26 chunks): the only data used to design prompts, examples and thresholds (tuning/few-shot part: 16 chunks, 48 documents; evaluation parts `pool_eval12` and `pool_eval2` are disjoint from it).
  - `dev_eval` was split by chunk into `dev_select` (40 chunks = 120 documents; fixed seed 2026) and `dev_final` (102 chunks; **locked and never opened**). Because of the call budget only a subset of `dev_select` was run: the first 5 chunks of a seeded permutation (seed 7) = **15 documents (5 per language)**, chosen by a rule written beforehand. Its results are noisy: every paired CI against M0 contains zero.
  - **Test split:** 168 documents (56 per language). Inputs only are used to produce predictions; `inputs.py` keeps a whitelist of text fields, so no code path in the package reads test labels.
- **Order of events** (all 2026-10-05 unless noted; details in `docs/prereg.md`):
  1. Preregistration written, with the code hash `5db749fb…`, before any run on `dev_select`.
  2. `dev_select` run started 12:22 local time. Addendum 1, added afterwards, records the two owner confirmations (what "Apertus-based" means; the public test split may be used for the final ranking after all methods are frozen). It changes no candidate, rule, prompt or constant.
  3. Addendum 2, after the `dev_select` report: single-flight merging of identical in-flight requests in `llm.py`. Code hash changed to `8391d36a…`. B0, M0, M1, M3 `dev_select` predictions replayed bit for bit (12 of 12 files); M2 is not replayable (see Limitations).
  4. Addendum 3, before any test prediction: freezes the code hash `8391d36ad46ca14bd1b5da25dbdd1eb6ee8ca043fff38f5e71b595d329e0bb62`, the eligibility and selection rule, the reporting rule and the call budgets. The `dev_select` ranking (M2 > M1 > B0 > M3 > M0) was not used to drop a candidate; the author kept M3 although it was more than 0.02 below the best.
  5. Test predictions generated from inputs only, hashed (`docs/test_prediction_hashes.md`), and replayed with 0 calls with identical hashes.
  6. Only then were the public test labels fetched, with the written permission of the challenge owner (UZH, e-mail of 5 October 2026; `docs/owner_confirmations.md`), by one committed script with a row-alignment check; every candidate was scored exactly once. Nothing was changed afterwards.
  7. Packaging changes after the freeze (`docs/code_changes_after_freeze.md`): where inputs, few-shot examples and the scorer come from changed; no algorithm, prompt, parameter or cache key changed. The package hash is `d5a8bb46865c3fe1a0c067aaa3e76eb50b870db89df63d7829ece2a67929b735`. In a fresh checkout, all five methods reproduced the frozen predictions byte for byte (3 of 3 files each, `--network none`).
  8. No system was added after the test evaluation (the owner's e-mail 3 allowed at most a few).
- **Languages / domains covered:** English paired with German, French and Italian; Swiss federal government web pages. No other languages or domains were tested.

## 5. Evaluation

Metric: official macro Spearman (mean over de/fr/it of `compute_correlations`, pooled per language). Set: public test split, n = 168 documents (56 per language, 56 English chunks), scored once. Intervals: chunk bootstrap, 1 000 resamples, seed 0, identical resamples for all methods; the difference to M0 is paired.

| method | DE | FR | IT | **macro** | macro 95 % CI (chunk bootstrap) | diff to M0 [95 % CI] | P(diff>0) |
|---|---|---|---|---|---|---|---|
| B0 (additional baseline) | 0.390 | 0.108 | 0.320 | **0.273** | [0.201, 0.330] | +0.060 [-0.003, +0.116] | 0.97 |
| M0 | 0.296 | 0.122 | 0.221 | **0.213** | [0.139, 0.278] | (reference) | – |
| **M1 (main system)** | 0.396 | 0.108 | 0.324 | **0.276** | [0.204, 0.334] | +0.063 [+0.001, +0.118] | 0.97 |
| M2 | 0.354 | 0.131 | 0.284 | **0.256** | [0.187, 0.314] | +0.043 [-0.021, +0.101] | 0.92 |
| M3 | 0.373 | 0.079 | 0.316 | **0.256** | [0.195, 0.301] | +0.043 [-0.007, +0.098] | 0.95 |

Paired differences to the top candidate M1 (same resamples):

| method | macro | diff to M1 [95 % CI] | P(diff>0) |
|---|---|---|---|
| B0 | 0.273 | -0.003 [-0.005, -0.001] | 0.00 |
| M0 | 0.213 | -0.063 [-0.118, -0.001] | 0.03 |
| M2 | 0.256 | -0.020 [-0.041, -0.000] | 0.03 |
| M3 | 0.256 | -0.019 [-0.044, +0.005] | 0.07 |

What the table supports:

- **M1 is ahead of M0 by +0.063, but the interval is wide** ([+0.001, +0.118]); the lower bound is only just above zero.
- **M1 and B0 are practically the same system.** B0 is 0.003 below M1 (0.273 vs 0.276). The paired CI of that difference is narrow because the two predictions are almost identical: **M1 adds only 74 flagged tokens** over B0 (5 086 vs 5 012 predicted > 0, below). B0 also beats M0 in the point estimate (+0.060), with a CI that includes zero ([-0.003, +0.116]). So the Apertus contribution on top of the model-free base is small.
- **M2 and M3 did not beat M1.** Both are 0.256 macro; their CIs against M0 include zero ([-0.021, +0.101] and [-0.007, +0.098]).
- **French is weak for every method** (0.079 to 0.131); the macro score is carried by German and Italian.
- Selection (Addendum 3, applied mechanically): top two are M1 (0.2762) and M2 (0.2564), which differ by 0.0198, more than the 0.005 tie margin, so M1 is the main system. **Choosing the best of four candidates on the same split is optimistic (selection bias).** M1's test score is not an unbiased estimate for new data, and the CIs above are per candidate and do not account for the selection.

**Stability: `dev_select` (15 documents) vs test (macro Spearman).** The `dev_select` ranking was a rough check only.

| method | dev_select macro | test macro | dev_select rank | test rank |
|---|---|---|---|---|
| B0 | 0.382 | 0.273 | 3 | 2 |
| M0 | 0.251 | 0.213 | 5 | 5 |
| M1 | 0.386 | 0.276 | 2 | 1 |
| M2 | 0.392 | 0.256 | 1 | 3 |
| M3 | 0.359 | 0.256 | 4 | 4 |

M2 was first on `dev_select` and third on test. On `dev_select` every CI of the difference to M0 contained zero (for example M2: +0.087 [-0.179, +0.196]).

**Tokens flagged nonzero (test, gold -1 tokens excluded; 132 080 scored tokens).**

| method | predicted > 0 | gold > 0 | predicted = 1.0 | gold = 1.0 |
|---|---|---|---|---|
| B0 | 5012 | 15840 | 5012 | 6254 |
| M0 | 33616 | 15840 | 6384 | 6254 |
| M1 | 5086 | 15840 | 5012 | 6254 |
| M2 | 8482 | 15840 | 5012 | 6254 |
| M3 | 4121 | 15840 | 4121 | 6254 |

**API calls sent in the test run:** M0 956 (1 stage), M1 332 (2 stages), M2 2494 (3 stages), M3 1391 (2 stages; 1 314 on 8B and 77 on 70B).

**Honest findings.**

1. **Simple prompting over-flags.** M0 marks 33 616 tokens as nonzero where the gold has 15 840, more than twice as many; this is the main reason it scores below the model-free methods. The earlier `prompt_pool` error analysis found the same: on 12 documents the Compare step flagged 21.3 % of all tokens (gold: 9.6 %), 68 % of its mapped spans covered only gold-0 tokens, and 75 % of its quotes were 10 words or longer while the median gold run is 2 tokens (`docs/round1_findings.md`).
2. **LLM-flagged spans were not precise enough to help pooled Spearman.** The base predicts zero for 92 % of tokens on `prompt_pool`; a flagged set only raises pooled Spearman if its precision is above about 40 % (simulation on the tuning set, `docs/round2_findings.md`). Measured precision stayed below that: round-1 Compare 11 %, Compare-v2 15 % (8B) and 28 % (70B), deterministic layer 29 to 37 %. In leave-one-chunk-out cross-validation over all 78 `prompt_pool` documents (26 chunks), the model-free aligner v2 reached macro 0.425 (95 % CI [0.257, 0.550]); adding the 70B verifier gave accepted-flag precision 32.1 % (raw candidates 24.1 %, rejected ones 7.8 %), recall 7.2 % and a gain of -0.021 [-0.043, +0.021]. The verifier does separate better from worse candidates, but its candidates cover only 9 % of the gold-nonzero tokens. Only a post-hoc filter on level-1.0 flags reached 64.9 % precision, with a gain of +0.004 [-0.001, +0.011], too small and chosen after the fact. M1's two-sample rule follows that idea (option (c) in `docs/apertus_role.md`); on test it changes 74 tokens.
3. **Debate-style agents lowered precision.** In the ablation on 6 chunks (18 documents), extractors → Compare → verifier reached precision 24.3 % (gain -0.020), against 45.4 % (+0.042 [+0.015, +0.136]) for the verifier alone on deterministic candidates and 34.0 % (+0.009) for Compare → verifier. On all 78 documents the same verifier alone gave 32.1 %, so the 6-chunk number is itself unstable. On test, M2 flagged 8 482 tokens against 5 012 for B0 and scored 0.256 against 0.273 for B0.

**Reference.** The SwissGov-RSD paper reports a best macro of 18.1 (Spearman × 100, LaBSE, unsupervised alignment) and about 3.6 for GPT-4o few-shot (`docs/baselines_from_paper.md`). These are **not comparable** with our numbers: the paper uses the full 224-document benchmark, possibly other label versions, and its own pipeline. We did not run those systems. The organizers' `compute_correlations` is undefined (NaN) for a constant prediction; our scoring wrapper reports 0.0 in that case (`docs/prediction_format.md`). We ran no separate trivial baseline.

## 6. Limitations

- **Selection and test size.** Four candidates were compared on one split of 56 documents per language; the winner's score is optimistic. All paired CIs of M2 and M3 against M0 include zero; M1 against M0 only just excludes it.
- **Small `dev_select`.** 15 documents (5 chunks) because of the call budget. Differences below about 0.05 cannot be told apart there, and the ranking did not carry over to test (M2 first, then third).
- **Cross-validation on 78 `prompt_pool` documents (26 chunks)** has wide intervals (macro 0.425, CI [0.257, 0.550]). Several Apertus experiments used only 9 to 18 documents. The aligner's parameters were tuned on the same pool, so the model-free base is optimistic there, and it is unstable across subsets (0.399 / 0.457 on one 12-document set, 0.169 / 0.135 on another, partly because French and Italian there contain no unaligned sentence, which gives a constant prediction).
- **Label version.** The dataset was re-annotated in April 2026; we use the current labels. We do not know which version the paper's table used, so no comparison with published numbers is possible.
- **`make eval` was not executed** in the verification of this repository (only `make -n eval`). It builds the organizers' scorer inside the image (needs network at build time and is heavy). The scores in this report come from our scoring script, which calls the organizers' `compute_correlations`, run once on the labels after the freeze (`scripts/60_score_test.py` in the development project).
- **M2 `dev_select` replay issue.** The M2 predictions on `dev_select` are valid as produced live but cannot be replayed: 20 of 86 Compare requests are missing from the cache. Identical extractor requests were sent as separate calls with slightly different answers, and the cache kept only the last one. The fix is single-flight merging (Addendum 2). B0, M0, M1, M3 on `dev_select` replay bit for bit. The M2 test run used the fixed code and replays with 0 misses.
- **Tokenization assumptions.** Predictions have one value per whitespace token of the given tokenized text; `-1` goes on ASCII punctuation tokens only (a token made solely of `string.punctuation` characters; `«` counts as a normal token). The format rules were relayed to us and the validator is ours; they were not re-checked against the organizers' script (`docs/prediction_format.md`). Sentence segmentation is rule-based and works on the untokenized text, then maps back to token ranges.
- **The Apertus part of M1 is small and not significant.** Its effect over B0 is +0.003 macro and 74 tokens. The "Apertus-based" status rests on the owner's ruling about model-free pairing, not on a measured gain.
- **Owner confirmation.** The "Apertus-based" status of M1 and the permission to use the public test split rest on two e-mails from the challenge owner of 5 October 2026; a third e-mail of 7 October 2026 allowed at most a few post-hoc systems after the test evaluation (none was added). Only short excerpts are kept, in `docs/owner_confirmations.md`; the full e-mails are not in the repository.
- **Scope.** Only English paired with de/fr/it, one domain, one annotation scheme. No error analysis was done on the test split.

## 7. Reproducibility

- Hardware: author's laptop (HP Victus, Windows, Docker Desktop); no local GPU used, all Apertus inference remote. Runtime: Docker (`python:3.11-slim` base, packages pinned in `requirements.lock`); commit: git tag `submission` (created at the final commit before submission).
- Seeds: bootstrap seed 0 (1 000 resamples); `dev_eval` split seed 2026; `dev_select` subset permutation seed 7; second M1 sample T = 0.7, seed 101; `PYTHONHASHSEED=0` in the image.
- Commands (host needs only Docker and `make`; run from `track_2a/`):

```bash
make run                 # M1 (main) -> outputs/predictions, B0 -> outputs/baseline_b0; offline, no API key
make run METHOD=M2       # extra method (M0, M2, M3 or B0 alone) -> outputs/extra/<METHOD>, also offline
make live                # re-queries Apertus; needs LLM_NAME, LLM_BASE_URL, LLM_API_KEY in the environment
make test                # unit tests, network disabled
```

  `make eval PRED_PREFIX=outputs/<dir>/<method>_admin_ [SPLIT=dev]` scores with the organizers' script at pinned commit `1807a42100e742ed03d337c54c4b9ea86995f565`; it was not run (see Limitations).
- **Offline replay design.** Every `docker run` in `make run` uses `--network none`. Apertus answers are replayed from `data/replay_cache/` (M1) and `data/replay_cache/extras/` (M0, M2, M3). The key is the sha256 of model, messages, temperature, seed, max_tokens and response_format. A cache miss is an error, never an API call, so identical outputs with zero misses also show that the prompts are the original ones.
- **Cache size.** All five methods: 5 174 entries, 5 066 793 bytes (5.07 MB raw). M1 alone: 332 entries, 622 701 bytes. B0 needs no cache.
- **Hashes.**
  - Frozen code hash (development project, used for the test run): `8391d36ad46ca14bd1b5da25dbdd1eb6ee8ca043fff38f5e71b595d329e0bb62`; this repository: `d5a8bb46865c3fe1a0c067aaa3e76eb50b870db89df63d7829ece2a67929b735` (differences listed in `docs/code_changes_after_freeze.md`).
  - Prediction files: sha256 of all 15 files in `docs/test_prediction_hashes.md` and `predictions/SHA256SUMS` (identical; recorded before any test label was read). `make run` and `make run METHOD={M0,M2,M3}` reproduce them byte for byte.
- **Measured on the author's laptop** (2026-10-06, fresh copy of the tree, details and commands in `docs/reproducibility_measurements.md`): `docker build --no-cache` of the runtime image 41.9 s (needs network); image size 403 MB; `make run` 34.0 s, 34.3 s and 35.3 s in three consecutive runs (no measurable cold/warm difference once the image exists); validator `OK: 3 file(s) valid` for M1 and B0; all 6 output files identical to `predictions/SHA256SUMS`. `make test` and `make eval` were not timed.
- Optional offline viewer: `track_2a/demo/` (frozen predictions and recorded M2 outputs, replay only, not needed for `make run`).
- **Test run cost** (calls sent): 70B 3 859 and 8B 1 314 over all five methods; M1 alone 332 calls.

## 8. Next steps

These are ideas, not results.

- Find an Apertus use that clears the precision bar: above about 40 % flagged-token precision **and** a gain whose CI excludes zero, for example verification restricted to flags the verifier rates 1.0 and a second sample confirms, on more than the 15 `dev_select` documents.
- Improve recall of short within-sentence differences (added acronyms, modifiers, terms), which the sentence-level base misses: in the round-1 error analysis, 144 of 207 gold runs inside aligned groups got no LLM score.
- Spend the unused `dev_select` budget (120 documents) and, once, the locked `dev_final` to get tighter intervals; add a held-out check of the final choice.
- Run `make eval` on dev predictions to cross-check our scoring script against the organizers' scorer.

## License

All HackApertus projects are open-sourced (Terms & Conditions, section 6). This submission:

- **Code** (`src/`, `scripts/`, `tests/`, `Dockerfile`, `Makefile`): Apache-2.0 (`LICENSE` at the repository root).
- **Documentation** (this report, the 2-page summary in PDF/HTML, and the files in `docs/`): Creative Commons Attribution 4.0 (CC-BY-4.0).

Third-party material:

- **SwissGov-RSD** data (test inputs, 16 few-shot documents): CC-BY-4.0, by Wastl, Vamvas and Sennrich. Attribution for the few-shot documents is in `data/few_shot/ATTRIBUTION.md`.
- **Organizers' evaluation code** (`ZurichNLP/SwissGov-RSD`) has no licence and is not redistributed here; `make eval` clones it at a pinned commit inside the image at build time.

## References

- Wastl, M., Vamvas, J., Sennrich, R. *SwissGov-RSD: A Human-annotated, Cross-lingual Benchmark for Token-level Recognition of Semantic Differences Between Related Documents.* ACL 2026. https://aclanthology.org/2026.acl-long.1437
- SwissGov-RSD dataset (Hugging Face, pinned revision `f16aa0536811b8c0cf975fede21eec5996e4b3c1`) and organizers' repository `github.com/ZurichNLP/SwissGov-RSD`, commit `1807a42100e742ed03d337c54c4b9ea86995f565`.
- Hernández-Cano, A., et al. *Apertus: Democratizing Open and Compliant LLMs for Global Language Environments.* ACL 2026. https://aclanthology.org/2026.acl-long.2172/ (arXiv:2509.14233).
- In this repository: `docs/prereg.md` (preregistration and addenda 1 to 3), `docs/dev_select_report.md`, `docs/test_results.md`, `docs/test_prediction_hashes.md`, `docs/code_changes_after_freeze.md`, `docs/apertus_role.md`, `docs/round1_findings.md`, `docs/round2_findings.md`, `docs/baselines_from_paper.md`.
