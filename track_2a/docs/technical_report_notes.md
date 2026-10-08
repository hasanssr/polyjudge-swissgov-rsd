# Notes for the technical report (working notes in this project; not the packaging repository)

Status: 2026-10-05. These notes collect statements the final `technical_report.md` must make; numbers are filled in when Task 5 is done.

## Rules of the challenge as confirmed by the organizers

Two confirmations from the challenge owner, **Michelle Wastl (UZH)**, by e-mail on **5 October 2026** (as relayed to the project; the e-mail is not stored in this repository). They are also recorded in the
addendum of `docs/prereg.md`; **neither changes the candidate list or the selection rules.**

1. **What "built on Apertus" means.** A hybrid counts as Apertus-based when the **sentence pairing is completely model-free**. Hence candidates **M1 and M2** (model-free sentence pairing + Apertus for the
   semantic decisions) qualify as Apertus-based, and **the aligner used inside M1 and M2 must not use any model**.
   *How the project meets it:* the pairing in M1/M2 is `polyjudge.aligner.HeuristicAlignerV2` (dynamic programming over sentence length ratio, relative position, shared numbers/dates/names/acronyms and a
   complexity penalty) on top of rule-based sentence segmentation. `tests/test_model_free_aligner.py` verifies, statically (import closure) and at runtime (clean subprocess), that this stack and the model-free
   flag generation (numbers, parentheses, untranslated text) load no LLM client, no embedding model and no network code, and that `run_m1` / `run_m2` never call the LLM alignment code that belongs to M3.
2. **Use of the public test split.** The public test split may be used for the **final ranking, after all methods are frozen.** The project's protocol: predictions for every remaining candidate are generated from the
   test *inputs* only, saved with file hashes, and only then are the test labels fetched by one committed script and every candidate scored exactly once (Task 5).

## What the report has to say about the candidates

* **B0** (model-free, no Apertus) is submitted as an *additional, documented, reproducible baseline* — it is not the main system.
* **M0** (one 70B B1 call per sentence group) is the simple Apertus prompting baseline; the quality claim is the improvement over M0.
* **M1, M2** qualify as Apertus-based because their pairing is model-free (confirmation 1). **M3** is Apertus-based by construction (8B alignment questions, 70B judge).
* The main system is the best Apertus-based candidate (M0–M3) on the final test ranking, chosen by the preregistered procedure.

## Honest-reporting reminders

* Earlier prompt_pool experiments found no component that beats the model-free baseline with a confidence interval excluding zero (`docs/apertus_role.md`); the report must state this and must not
  describe Apertus contributions that were not measured.
* The dev_select evaluation is limited to 15 documents (5 chunks) by the call budget; its confidence intervals are wide (`docs/prereg.md`, section "Cost-dependent evaluation subset").
* The test-run cost of all five candidates is large (see the dry-run estimate in the status report); any dropped candidate and the reason must be listed.
