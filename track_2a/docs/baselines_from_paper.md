# Published baselines (SwissGov-RSD paper)

**Source:** Wastl, Vamvas, Sennrich, *SwissGov-RSD: A Human-annotated, Cross-lingual Benchmark for Token-level Recognition of
Semantic Differences Between Related Documents*, ACL 2026 (`aclanthology.org/2026.acl-long.1437`), **Table 3** ("Token-level
Spearman correlations with gold labels for three language pairs, comparing results on the iSTS-RSD and the SwissGov-RSD
dataset"; proceedings page 31139). Numbers below are the paper's, ×100, transcribed from the PDF text layer; the
Kendall-τ version is in the paper's Appendix I. I did not re-run any of these systems.

## SwissGov-RSD columns of Table 3 (Spearman × 100)

| System | EN-DE | EN-FR | EN-IT | macro-avg |
|---|---|---|---|---|
| **DiffAlign (unsupervised)** | | | | |
| LaBSE† | 20.6 | 8.2 | **25.6** | **18.1** |
| XLM-R + SimCSE† | 19.2 | 10.0 | 23.4 | 17.5 |
| Qwen3-Embedding 4B | 20.1 | 11.2 | 20.3 | 17.2 |
| Qwen3-Embedding 8B | 19.3 | **12.1** | 18.6 | 16.7 |
| EuroBERT 610M | 18.6 | 11.4 | 18.7 | 16.2 |
| EuroBERT 2.1B | 13.7 | 8.9 | 17.4 | 13.3 |
| bge-m3 | 14.8 | 9.9 | 13.6 | 12.8 |
| MmBERT-base | 14.3 | 8.6 | 15.6 | 12.8 |
| EuroBERT 210M | 11.5 | 8.4 | 16.1 | 12.0 |
| gte-multilingual-base | 10.7 | 5.9 | 12.8 | 9.8 |
| MmBERT-small | 10.3 | 5.5 | 11.0 | 8.9 |
| ModernBERT-large | 0.5 | 0.7 | 2.7 | 1.3 |
| **LLMs, few-shot prompting** | | | | |
| GPT-4o | 4.2 | 1.2 | 5.5 | 3.6 |
| Llama-3.1 405B Instruct | −0.8 | −2.2 | 7.4 | 1.5 |
| (Llama-3.1 8B, GPT-4o-mini, DeepSeek-R1, o3-mini-low) | – | – | – | not evaluated on SwissGov ("only the best-performing systems on iSTS-RSD" were) |
| **Fine-tuned LLMs** | | | | |
| GPT-4o-mini (fine-tuned) | 3.9 | 1.1 | 5.3 | 3.4 |
| **Fine-tuned encoders (projected labels\*)** | | | | |
| ModernBERT-large (EN-IT\*) | 7.5 | 4.8 | 12.4 | 8.2 |
| ModernBERT-large (EN-FR\*) | 5.4 | 5.2 | 8.7 | 6.4 |
| ModernBERT-large (multi\*) | 5.7 | 2.5 | 8.8 | 5.7 |
| ModernBERT-large (EN-DE\*) | 6.4 | 1.6 | 5.6 | 4.5 |
| ModernBERT-large (EN) | 5.7 | 1.3 | 4.0 | 3.7 |

† = inputs had to be split because of the model's length limit. Macro-avg is my arithmetic mean of the three columns.
(For comparison, the same table on the synthetic iSTS-RSD: fine-tuned GPT-4o-mini 78–82, fine-tuned ModernBERT 54–63 —
they collapse on SwissGov, which the authors attribute to overfitting to the augmentation patterns.)

## What this says about the bar

* Best published SwissGov-RSD macro-average ≈ **18** (LaBSE†, unsupervised DiffAlign); EN-FR is the hardest pair for every system
  (best 12.1). Few-shot LLM prompting is near zero (GPT-4o 3.6), fine-tuned LLMs and encoders ≤ 8 on average.
  The paper itself concludes that unsupervised encoder alignment is the strongest approach on this benchmark.
* Inter-annotator agreement (paper §4, Table 2) is the realistic ceiling; I have not transcribed it here.

## Are these numbers comparable to ours?

| aspect | paper Table 3 | ours | consequence |
|---|---|---|---|
| metric | token-level Spearman over **all** gold labels, equivalent tokens = 0 | the organizers' `compute_correlations`, pooled per language, gold −1 dropped | **same metric** (same code path as the organizers' repo) |
| data | "the full SwissGov-RSD benchmark" = 224 docs per language pair (Table 1) = dev 168 + test 56 | dev only: 168 docs/language (`dev_eval`: 142) | **different split**: their score mixes in the test portion; no number is directly comparable |
| labels | version used for the paper | HF dataset card says the data was **re-annotated in April 2026** (old annotations archived); we use the current HF/GitHub dev labels | possible shift; unknown whether the paper's table used the new labels |
| inputs | whole documents (†: split into segments for LaBSE / XLM-R) | our own sentence segmentation + alignment | not a like-for-like pipeline |
| systems | none of them fine-tuned on in-domain SwissGov labels | few-shot only (no training data allowed in our setting) | comparable in spirit to the "few-shot" and "unsupervised" rows |

Local reference points on **our** dev split (organizers' dev predictions, same scorer; see `scripts/09_baselines.py`):
their mmBERT-based DiffAlign variants reach 0.21–0.22 macro on full dev, and our model-free "unaligned sentence = 1.0" heuristic
reaches 0.23 (dev) — i.e. in the same range as the paper's best ≈ 0.18, which suggests the dev part is somewhat easier than the
full benchmark or that the labels changed; either way **the numbers to beat in the report are the dev numbers we compute ourselves, not
the paper's.**
