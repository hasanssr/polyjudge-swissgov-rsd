# Confirmations from the challenge owner

Short verbatim excerpts from three e-mails from Michelle Wastl (UZH, challenge owner) to the author: e-mail 1 on 5 October 2026, e-mail 2 on 5 October 2026, e-mail 3 on 7 October 2026. The e-mails themselves are not included; only the passages the protocol relies on.

## E-mail 1 (5 October 2026): what the main system must be

> Since it's the HackApertus hackathon, I'm afraid the main system that you submit does have to build on Apertus.

> LLMs are generally quite bad at finding and scoring differences in SwissGov documents (see Table 3 here), so a method that is not based on Apertus is just as valuable.

> document the method that is not based on Apertus, keep it as reproducible as everything else, and submit it as an additional baseline. Run the Apertus-based methods on the test set once and then submit the best-performing one as your main system

> The overall quality and creativity will be considered in the judgment just as much as the Spearman correlation improvement over the simple Apertus prompting baseline, so low Spearman correlations with Apertus do not mean the submission is bad.

## E-mail 2 (5 October 2026): answers to the author's two questions

**Question 1.** Does a hybrid count as Apertus-based when the sentence pairing is computed without any model, but Apertus makes the scoring or verification decisions?

> If the sentence pairing is completely model free, then yes!

**Question 2.** May the public test labels be used for the final ranking after all methods are frozen?

> Yes, you may use the public test split for the final ranking.

## E-mail 3 (7 October 2026): post-hoc system after the test evaluation

**Question.** May one additional, clearly labeled post-hoc system be developed after the test evaluation?

> You can go ahead and do so in this case, but please refrain from adding many more systems after evaluating on the test set.

## How the project uses these

- B0 (model-free) is documented, reproducible and submitted as an additional baseline; M0 to M3 were each run on the test set once, and the best of M0 to M3 is the main system (`prereg.md`, Addendum 3).
- M1 and M2 count as Apertus-based because their sentence pairing is model-free; `tests/test_model_free_aligner.py` checks that the aligner and the pairing stack load no model or LLM code (`prereg.md`, Addendum 1).
- The public test labels were opened only after all predictions were frozen and hashed (`prereg.md`, Addendum 3; `test_prediction_hashes.md`).

This file was added after docs/prereg.md and docs/technical_report_notes.md were written; the 'relayed, not stored' wording in those earlier documents refers to the time before this file existed.
