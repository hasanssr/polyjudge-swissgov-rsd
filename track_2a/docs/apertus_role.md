# Apertus role audit

Scope: what Apertus contributes to the **current best pipeline**, measured on `prompt_pool` only (dev_eval and TEST untouched). Scores are the organizers'
`compute_correlations`, pooled per language; CIs are bootstrap over chunks. Everything below is exploratory evidence from 18–78 documents.

## 1. The current best pipeline contains no Apertus component

Best variant by cross-validation (leave-one-chunk-out over all 26 prompt_pool chunks, simplest variant within the CI of the best): **model-free aligner v2**.
Macro Spearman **0.425, 95 % CI [0.257, 0.550]** (DE 0.528, FR 0.398, IT 0.347). Steps, all model-free:

| step | what it does |
|---|---|
| sentence segmentation (`segmentation.py`) | splits the untokenized text, maps sentences to gold token ranges, keeps −1 on ASCII punctuation |
| monotone sentence alignment (`aligner.py`, v2) | DP over 1-1/1-2/2-1/2-2/1-0/0-1 beads using length ratio, relative position, shared numbers/dates (months normalised across EN/DE/FR/IT), names, acronyms, and a penalty for weakly supported complex beads |
| presence score | tokens of sentences with no counterpart = 1.0, everything else 0 |

Variants that add model-free signals did not change this beyond noise (CV macro: v3 0.447, + number layer 0.446, + number & parenthesis layer 0.429;
paired differences to v3 all have CIs containing 0).

## 2. Every place Apertus was tried, and what it added

| use of Apertus | model | calls | marginal effect (Spearman, macro unless stated) |
|---|---|---|---|
| B1: direct span + severity per sentence group, combined with the heuristic | 8B / 70B | 79 / 77 | 0.219 / 0.210 vs 0.399 (pool_eval12) → **−0.18 / −0.19** |
| Compare step on aligned pairs (round-1 prompts), `max(H, ·)` | 8B / 70B | 78 / 75 | 0.237 / 0.278 vs 0.399 → **−0.16 / −0.12** |
| Compare-v2 (≤ 8-word spans, reason first, filters) + number layer, weight 0.2 | 8B / 70B | 78 / 36 | 0.349 vs 0.457 (8B); on 9 docs 0.317 vs 0.408 (70B) → **−0.11 / −0.09** |
| LLM sentence alignment (candidates → which match / NONE), presence 1 / 0.5 / 0 | 8B | 214 (+115 lost to a broken first prompt) | eval12 −0.009, eval2 +0.004 → **≈ 0** (it says NONE for 20 of 1 439 sentences) |
| 70B judge on the 40 aligner/8B disagreements | 70B | 6 | eval12 −0.074, eval2 +0.011; right on 29/40 vs 35/40 for the aligner |
| **LLM verifier of candidate flags** (number / parenthesis / untranslated / Compare-v2 candidates, aligned pairs only), accepted = "real", level from the model, added to the CV base | 70B | 90 | all 78 docs: accepted-flag token precision **32.1 %** (raw candidates 24.1 %, rejected 7.8 %); recall 7.2 %; gain **−0.021 [−0.043, +0.021]** → success threshold not reached |
| same verifier, only flags with level 1.0 (post-hoc, exploratory) | 70B | (same) | 32 flags, precision 64.9 %, recall 0.6 %, gain **+0.004 [−0.001, +0.011]** — too small and chosen after the fact |
| same verifier with 8B (12 docs) | 8B | 139 | 31–58 % of the batches unparseable even after a retry; precision 18–26 %; gain −0.011…−0.016 (12 docs) |
| debate agents: blind extractors (EN / target) → Compare → verifier, 6 chunks (18 docs) | 70B | 279 | precision 24.3 %, gain −0.020; verifier alone on deterministic candidates (same docs) 45.4 %, +0.042 [+0.015, +0.136]; Compare → verifier 34.0 %, +0.009 → **extra agents lower precision** |

Reading the numbers:
* **Marginal contribution of Apertus to the final score: none that can be distinguished from zero.** The best case is the post-hoc level-1.0 filter (+0.004, CI includes 0).
* The verifier **does discriminate**: accepted flags are ≈ 4× as precise as rejected ones (32.1 % vs 7.8 %), and its level 1.0 is informative (64.9 %). The problem is that the
  candidate set only contains 9 % of the gold-nonzero tokens and the precision of the accepted set is below the ≈ 40 % break-even for a base that is 92 % ties.
* On the 6-chunk debate subset the verifier alone looked successful (precision 45.4 %, +0.042); on all 78 docs the same verifier fails. The subset is small and was
  chosen by size, so the all-document result is the one to trust; it also shows how unstable these estimates are.
* Model size matters for robustness (batches unparseable after one retry: 70B 0–5 %, 8B 0–58 % depending on the task), not for over-flagging.

## 3. What the model-free parts do

All of the final prediction: sentence units, alignment, the "no counterpart ⇒ 1.0" rule, punctuation handling, and the scorer wrapper. The deterministic layers (numbers,
parentheses) and the aligner variants add ≤ ±0.02 with CIs that include 0.

## 4. Consequences for the report and the rules

* The challenge README states that submissions **must be built with Apertus**. A pipeline whose best variant has no Apertus component is at risk on that criterion even though
  it scores best. I have not found a rule that defines how large the Apertus part must be; this should be clarified with the organizers.
* Honest options: (a) report the model-free system as the main result and the Apertus experiments (alignment, verifier, debate) as a documented negative-result study with the
  cached, budget-guarded infrastructure; (b) ship the verifier (level ≥ 1.0 flags) as an optional Apertus layer with a weight tied to its measured, tiny effect, clearly labelled as
  not significant; (c) find an Apertus use that clears the precision bar (> 40 % **and** a positive CI), e.g. verification restricted to flags the verifier rates 1.0 *and* a second sample confirms.
* Debate-style agents (extractors → Compare) stay in the report only as a **negative result**: they did not improve precision (24.3 % vs 45.4 % / 34.0 %).
