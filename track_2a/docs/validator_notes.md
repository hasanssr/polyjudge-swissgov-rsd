# Prediction format

Source: the organizers' evaluation script (as relayed by the team; re-verify against the script
before submission). Checked by `scripts/validate_predictions.py` (`make run` and `make validate`).

- **Files:** jsonl, one per language, named `<prefix>{de,fr,it}.jsonl.jsonl`. Row order is identical to the input file.
- **Fields per row:** exactly `id`, `text_a`, `text_b`, `labels_a`, `labels_b`. `id`, `text_a`, `text_b` are copied unchanged from the input.
- **Tokens:** whitespace-separated tokens of the tokenized text as given: `len(labels_a) == len(text_a.split())`, same for b.
- **Label values:** floats in [0, 1] (the gold uses steps of 0.2; steps are not enforced here), and `-1` on ASCII punctuation tokens only.
  The gold marks those tokens `-1` and they are dropped from scoring. An "ASCII punctuation token" is a token made solely of
  characters in Python's `string.punctuation`; non-ASCII punctuation such as `«` is a normal token.
- The validator rejects: wrong/missing/extra files, wrong row count or order, missing/extra fields, wrong label count,
  non-numeric/NaN/bool labels, values outside [0, 1], `-1` on a non-punctuation token, and non-`-1` on a punctuation token
  (relax the last with `--allow-punct-values` if the official script turns out to accept it).

Open point: the exact `<prefix>` is unknown here; pass `--prefix` to the validator once known.
