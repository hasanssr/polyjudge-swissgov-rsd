# Prediction format (as defined by the organizers' evaluation code)

Source of truth: `organizers_repo` (commit `1807a42100e742ed03d337c54c4b9ea86995f565`): `scripts/evaluate_predictions_admin.py`,
`evaluation/utils.py` (`load_predictions`), `evaluation/predictions.py` (`EncoderPrediction`), `rsd/recognizers/utils.py` (`DifferenceSample`).
I found no separate hackathon submission-format document (the Notion guide is not machine-readable for me), so **the organizers' script is the only specification
available; the exact test-submission file layout should still be confirmed with the organizers.**

## Files

The evaluator takes a *prefix* and opens, for each language pair `lang ∈ {de, fr, it}`:

```
<prefix><lang>.jsonl.jsonl          (prefix without "llm_predictions" in it; with it the suffix is ".jsonl")
```
Our convention: `<out_dir>/<METHOD>_admin_<lang>.jsonl.jsonl`, i.e. prefix `<out_dir>/<METHOD>_admin_`
(example: `output/pred_eval2/B0_admin_de.jsonl.jsonl`; written by `polyjudge.predict`, one file per language, UTF-8, `\n` line ends).
(`--short` appends `_short` to the language code; not used.)

## One JSON object per line

```json
{"id": "admin_de_0", "text_a": "<English, tokenized>", "text_b": "<German, tokenized>",
 "labels_a": [0.0, 0.0, 1.0, -1.0, ...], "labels_b": [0.0, 1.0, -1.0, ...]}
```
* `load_predictions` chooses the class by keys: `api_request` → LLM record, `prompt` → LLM record, **otherwise an `EncoderPrediction` with exactly
  `id, text_a, text_b, labels_a, labels_b`** — this is the layout we use (our files contain none of the LLM keys).
* `text_a` / `text_b` are the **tokenized** texts of the dataset fields of the same name (not the `*_untokenized` variants), copied unchanged from the input.
* **Order**: the evaluator `zip`s the predictions with the gold file of the same language; ids are *not* matched. The order must equal the dataset order
  (the order of the input file), and the number of lines must equal the number of gold rows (`assert len(predictions) == len(gold_samples)`).

## Tokens and label lengths

* A token is one whitespace-separated item: `tokens = text.split()` (`DifferenceSample.tokens_a = tuple(text_a.split())`).
* `labels_a` has one value per token of `text_a`, `labels_b` one per token of `text_b`.
  If a list is **shorter** than the gold, the evaluator appends `(0.0,) * k` — which raises `TypeError` for JSON lists (unless the prefix contains "Zurich"); if
  **longer** it is truncated. So: write exactly one label per token.

## Values

* Gold values: `-1` (ASCII punctuation), `0` (equivalent), `0.2, 0.4, 0.6, 0.8, 1.0` (difference level = annotator label / 5).
* Predictions may be any finite real number (the metric is rank based); we use `0.0` … `2.0`.
* **ASCII punctuation tokens** (every character in `string.punctuation`, e.g. `.` `,` `'` `(` `)` `-` `/` `%`) are `-1` in the gold and are **dropped from scoring**
  (`filtered = [p for p, g in zip(pred, gold) if g != -1]`). We write `-1.0` there, mirroring the gold. Non-ASCII punctuation tokens (`–`, `«`, `»`, `•`, `€`, `…`) carry real gold
  labels and must receive a prediction like any word.
* A prediction that is **constant** on all scored tokens makes their `compute_correlations` crash (NaN in the bootstrap); `polyjudge.official_eval` reports 0.0 for that case.

## What is scored

For each language, label vectors of **both sides are pooled** over all documents (side A always; side B when its gold is not all −1), tokens with gold −1 are removed, and
`compute_correlations` returns the global Spearman (and Kendall τ-b) correlation. The script prints one value per language; the macro average over de/fr/it is the mean of the
three (our primary metric). Predictions are never matched by id and never need the gold to produce them.

## Verification (zero API calls)

`scripts/41_verify_format.py`: B0 predictions for `pool_eval2` (18 documents) are written by `polyjudge.predict`, read back with the **organizers' own `load_predictions`**
(their heavy imports are bypassed by a stub for the `rsd.recognizers` package) and scored with their `compute_correlations`:

| | DE | FR | IT | macro |
|---|---|---|---|---|
| organizers' loader + official function | 0.405 | −0.020 | 0.122 | 0.169 |
| our reader + same function | 0.405 | −0.020 | 0.122 | 0.169 |

Token lists matched the dataset's gold tokens for every document and 1 652 punctuation positions carry −1.
