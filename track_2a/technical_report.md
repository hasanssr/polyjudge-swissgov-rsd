# Technical report — `PolyJudge`

A deeper write-up than the README: what we built, how it works, and what the
numbers say. Also covers the specific submission requirements of the UZH challenge.

- **Track:** `Track 2A — UZH: Detecting Cross-Lingual Semantic Differences in Swiss Government Websites`
- **Event:** Online
- **Team:** `TODO team name` — `TODO members`
- **Demo:** `TODO link to video, deployment, or notebook (or "none")`

## 1. Summary

TODO: the problem, the approach, and the headline result in one paragraph (main system vs. the simple Apertus prompting baseline, with CI; no number until measured).

### Problem

TODO: task definition; input/output format (jsonl per language de/fr/it; fields `id, text_a, text_b, labels_a, labels_b`; per-token float in [0, 1], -1 on ASCII punctuation); the official metric (macro Spearman).

## 2. Architecture

Per the challenge owner's ruling, the **main system must build on Apertus**. The repo presents two systems, both reproduced by `make run` offline (replay cache, no network):

- **Main system** — `TODO method name` (Apertus-based). Candidates M0 (simple Apertus prompting baseline), M1, M2, M3 (`TODO` one-line description each) were each run on the test split once; the main system is the best by `TODO`.
- **B0, model-free alignment baseline** — additional and documented, not the main submission. `TODO`: pipeline steps, hyperparameters, how chosen (on `prompt_pool` only), runtime, dependencies.

Components, data flow and where each one runs: `TODO`. Diagrams go in `docs/`.

## 3. Use of Apertus

- **Model:** `TODO name and version (LLM_NAME)`
- **How it is used:** inference
- **Where it runs:** `TODO hosted endpoint / local weights`
- **Role:** core of the main system. Responses are cached; `make run` replays them with no network, `make live` re-queries.
- **Selection protocol:** `docs/prereg.md` (TODO) was written before running on `dev_select`; `dev_final` stayed locked; candidates ranked on `dev_select`; each surviving candidate run on test once. `TODO` results and any dropped candidates.
- **Experiments and negative results:** `TODO` (each: setup, dev result, why it did or did not help; include the ones that failed).
- **Prompts, serving stack, sampling settings:** `TODO`
- **Other models used during development (judges, etc.):** `TODO or "none"`
- **AI-assistance disclosure:** Implementation was done with Claude Code (Anthropic) as a coding assistant. The research design, preregistration protocol, experiment decisions and all interpretation were directed and reviewed by the author.

## 4. Data

- Source and licence: `TODO`. Personal / non-redistributable data: `TODO or "none"`. Human-subject consent: `TODO or "n/a"`.
- Split protocol:
  - `prompt_pool` (`TODO` size): the only data used to design prompts, examples, thresholds.
  - `dev_eval`, split by chunk into `dev_select` (`TODO` chunks) and `dev_final` (`TODO`; locked until the selection rule was fixed).
  - Test split: inputs only are used to produce predictions. Test labels were never read before the predictions were frozen. `TODO: how this was enforced`.
- Languages / domains covered: `TODO`.

## 5. Evaluation

Metric: `TODO official macro Spearman`. Set: `TODO`, n = `TODO`.

| Setup | Metric | Result | 95% CI | Method of CI |
|-------|--------|--------|--------|--------------|
| Trivial baseline | TODO | TODO | TODO | TODO (e.g. bootstrap, B=TODO, seed=TODO) |
| M0 simple Apertus prompting | TODO | TODO | TODO | TODO |
| M1 / M2 / M3 | TODO | TODO | TODO | TODO |
| **Main system** (`TODO`) | TODO | TODO | TODO (diff to M0) | paired chunk bootstrap |
| B0 model-free baseline | TODO | TODO | TODO | TODO |

Per-language breakdown: `TODO`.

## 6. Limitations

TODO: failure modes, languages not tested, label noise, small dev set, anything not verified.

## 7. Reproducibility

- Hardware: `TODO`; runtime: Docker, Python 3.12; seeds: `TODO`; commit: `TODO SHA`.
- `make run`: main system (replay cache) + B0, offline, no API key; `make live`: needs `LLM_NAME`, `LLM_BASE_URL`, `LLM_API_KEY`; replay cache in `data/replay_cache/`.
- `make eval`: dev scores; `make test`: tests.
- Wall-clock time of `make run`: `TODO`.

## 8. Next steps

TODO

## License

Creative Commons Attribution 4.0 (CC-BY-4.0). All HackApertus projects are open-sourced.

## References

TODO
