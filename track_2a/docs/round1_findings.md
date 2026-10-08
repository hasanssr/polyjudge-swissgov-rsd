# Round 1 — first LLM layer on 12 prompt_pool documents (exploratory)

Model `swiss-ai/Apertus-v1.5-8B` (CSCS), temperature 0, one call per sentence group (≤ 6 beads / ≤ 1 400 chars), ≤ 1 retry.
Evaluation set: `splits/pool_eval12.json` (4 prompt_pool chunks × 3 languages, seed 41). Few-shot examples (3 per call, same language) come only from
the other 22 pool chunks (hard assertion in `prompts.select_shots`). `dev_eval` and the test split were not used (`POLYJUDGE_FORBID_SETS=dev_eval`).
Scores: the organizers' `compute_correlations`, pooled per language over 4 documents → **exploratory, very noisy**. The heuristic's hyper-parameters
(aligner v2) were tuned on the whole pool, i.e. including these 12 documents, so the heuristic is optimistic here.

| system | DE | FR | IT | macro | calls |
|---|---|---|---|---|---|
| heuristic only (unaligned = 1.0) | 0.619 | 0.150 | 0.427 | **0.399** | 0 |
| B1 alone (per group, direct spans) | 0.228 | 0.115 | 0.135 | 0.159 | 79 |
| Compare alone (aligned groups) | 0.042 | 0.020 | 0.048 | 0.037 | 78 |
| **Hybrid v1** = max(heuristic, Compare) | 0.335 | 0.078 | 0.299 | **0.237** | (same) |
| Δ hybrid − heuristic | −0.285 | −0.072 | −0.128 | **−0.162** | |
| *diagnostic:* max(heuristic, B1 on aligned) | 0.303 | 0.122 | 0.230 | 0.219 | |

Robustness: B1 first-attempt parse failures 2/77 (2.6 %, both truncations), after retry 0; Compare 3/75 (4.0 %), after retry 1 (1.3 %), 2 repetition-guard aborts.
Mapping (verbatim only): B1 184/188 records mapped (98 %); Compare 121/128 (95 %), +2 only with case/punctuation normalisation, 5 unmapped.
LLM-flagged tokens (Compare): 21.3 % of all tokens (gold: 9.6 % on these documents). Inside aligned groups: precision 0.11, recall 0.33.
Severity of Compare records: 0.6 in 100/128; of the 0.6 tokens only 13 % have gold > 0, the 0.2/0.4/0.8 tokens 0 %.

Error analysis (see `scripts/18_error_analysis.py`, `scripts/19_extra_counts.py`):
* False alarms: 82 of 121 mapped spans (68 %) cover only gold-0 tokens; 80 of them are `meaning_shift`. 75 % of all Compare quotes are ≥ 10 words
  (median 17), whereas gold runs have a median of 2 tokens (69 % ≤ 4 tokens): the model marks whole sentences. In 39 % of the records the model's own
  reason says the passages are the same / differ only in wording; 32 records copy the generic reason of the few-shot demonstrations.
  Of 10 sampled false alarms: 4 over-flagging of paraphrase / hallucinated mismatch, 6 translation-neutral wording (title translation, number format 3.043 vs 3,043,
  "programme" vs "génération", untranslated German titles on the English page); 0 mapping failures, 0 alignment errors.
* Misses: 144 of 207 gold runs inside aligned groups (70 %) get no LLM score; of 10 sampled: 6 short within-sentence differences (added acronym, modifier, term),
  1 number/metadata (file size), 3 alignment errors (extra content in a mismatched sentence or a menu line); 0 verified mapping failures.
* Group level: the LLM reports ≥ 1 span in 84 % of aligned groups, gold has a nonzero span in 63 %.

Proposed changes (not applied): (1) minimal-span, reason-first output with explicit rejection of "same information" reports; (2) few-shot with concrete
reasons and short gold spans, more empty demonstrations and an explicit base rate; (3) a deterministic number/metadata/parenthetical mismatch layer for the short-span misses.
