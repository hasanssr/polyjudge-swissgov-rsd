# Round 2 — LLM alignment, span re-test, 70B check (exploratory; dev_eval and TEST untouched)

Evaluation sets (prompt_pool only): `pool_eval12` (4 chunks, 12 docs), `pool_eval2` (6 chunks, 18 docs); tuning / few-shot set = the other 16 chunks
(48 docs). Few-shot sources and aligner tuning exclude both evaluation sets (asserted in code, tested). Scores: organizers' `compute_correlations`, pooled per
language (constant predictions = 0.0, their function crashes there). Calls: 8B 674/700, 70B 194/200.

| variant | eval12 DE / FR / IT / macro | eval2 DE / FR / IT / macro |
|---|---|---|
| heuristic v2 (round-1 "heuristic") | 0.619 / 0.150 / 0.427 / **0.399** | 0.405 / −0.020 / 0.122 / **0.169** |
| improved heuristic v3 (tuned on the tuning set) | 0.619 / 0.275 / 0.475 / **0.457** | 0.405 / 0.000 / 0.000 / **0.135** |
| + 8B alignment, presence 1 / 0.5 / 0 | 0.616 / 0.260 / 0.468 / 0.448 | 0.424 / −0.014 / 0.007 / 0.139 |
| + 70B judge on the 40 disagreements | 0.619 / 0.275 / 0.254 / 0.383 | 0.417 / 0.000 / 0.021 / 0.146 |
| + deterministic number/parenthesis layer (w = 0.2) | 0.593 / 0.257 / 0.400 / 0.417 | 0.260 / 0.109 / 0.116 / 0.161 |
| + deterministic layer, numbers only | 0.598 / 0.232 / 0.444 / 0.425 | 0.381 / 0.044 / 0.108 / 0.178 |
| + Compare-v2 span layer 8B (w = 0.2) | 0.493 / 0.118 / 0.362 / 0.324 | not run (budget) |
| + Compare-v2 8B + deterministic layer | 0.521 / 0.182 / 0.344 / 0.349 | not run (budget) |
| round-1 Compare, 8B / 70B, max(H, ·) | 0.237 / 0.278 (macro) | not run |

Main findings
* **Precision break-even.** With a base that is 92 % ties, a flagged set splits the zero block and only raises pooled Spearman if its precision is
  above ≈ 40 % (simulation on the tuning set, 223 flagged tokens: 10 % → 0.42, 30 % → 0.476, 50 % → 0.530, base 0.518; random flags: 0.424). Measured precision:
  Compare 8B/70B round 1 11 %, Compare-v2 8B 15 % / 70B 28 %, deterministic layer 29–37 %. The weight `w` is irrelevant for a binary base (any w in (0, 1) gives the same ranking).
* **70B vs 8B (round-1 prompts, 12 docs):** parse failures 0 % vs 2.6–4.0 %; B1 flags 25.8 % vs 35.0 % of the tokens, Compare 20.8 % vs 21.3 %; hybrids 0.210 / 0.278 (70B) vs
  0.219 / 0.237 (8B) against 0.399 for the heuristic: 70B fixes robustness, not over-flagging.
* **LLM alignment:** the first prompt version was broken by a numbering confusion (NONE for 83 % / 78 % of the first two items of every batch); labelling candidates with
  letters removed it (anchor items 24/24, parse failures 2 %), but the 8B then almost never says NONE (20 of 1 439 sentences; 1 of the 22 truly unaligned sentences caught).
  Aligner and 8B agree on 97.2 % of the sentences; of the 40 disagreements the aligner is right in 35, the 8B in 5, the 70B judge in 29.
* The model-free heuristic itself is unstable: 0.399 / 0.457 on eval12 but 0.169 / 0.135 on eval2 (FR and IT of eval2 contain no unaligned sentence → constant prediction).
