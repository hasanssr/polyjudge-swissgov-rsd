# PolyJudge: Track 2A submission (UZH: cross-lingual semantic differences in Swiss government websites)

> **AI-assistance disclosure.** Implementation was done with Claude Code (Anthropic) as a coding assistant. The research design, preregistration protocol, experiment decisions and all interpretation were directed and reviewed by the author.

Team: Solo, Hasan Sarı. Repository: https://github.com/hasanssr/polyjudge-swissgov-rsd

Links: [technical report](technical_report.md) | [2-page summary (PDF)](docs/PolyJudge_2page_summary.pdf) | [optional offline viewer](demo/README.md) | [docs/](docs/) (preregistration with addenda, test results, hashes, owner confirmations, measurements)

### How to run
Needs only Docker and `make` (no GPU, no API key, no model weights). `make run` is **offline**: it builds the image, then runs with `docker run --network none`; Apertus answers are replayed from a committed cache, and a cache miss is an error, never an API call.

Note: the first `make run` builds the Docker image and needs network access once, to download the pinned Python packages (`requirements.lock`). The prediction step itself runs with `--network none` and needs no API key.

```bash
git clone https://github.com/hasanssr/polyjudge-swissgov-rsd.git
cd polyjudge-swissgov-rsd/track_2a
make run                 # M1 (main) -> outputs/predictions, B0 -> outputs/baseline_b0; offline, validated, not scored
make run METHOD=M2       # an extra method (M0, M2, M3, or B0 alone) -> outputs/extra/<METHOD>, also offline
make live                # M1 against Apertus; needs LLM_NAME, LLM_BASE_URL, LLM_API_KEY in the environment
make test                # unit tests, network disabled
```

Measured on the author's laptop: image build from scratch 41.9 s, `make run` about 34 s, outputs byte-identical to `predictions/SHA256SUMS` (`docs/reproducibility_measurements.md`).

### Systems
| System | Role | Output folder | How it works offline |
|---|---|---|---|
| **M1**: model-free sentence alignment + Apertus-70B verifier (flags accepted only at level 1.0 and in 2 of 2 independent samples) | **main system** | `outputs/predictions/` | Apertus responses replayed from `data/replay_cache/` (332 entries) |
| **B0**: model-free alignment baseline | additional, documented, reproducible baseline | `outputs/baseline_b0/` | no model calls at all |
| M0: simple Apertus prompting | comparison method | `outputs/extra/M0/` | replay cache in `data/replay_cache/extras/` |
| M2: multi-agent (extractors, compare, verifier) | comparison method | `outputs/extra/M2/` | replay cache in `data/replay_cache/extras/` |
| M3: Apertus-judged alignment | comparison method | `outputs/extra/M3/` | replay cache in `data/replay_cache/extras/` |

Output files are `<method>_admin_{de,fr,it}.jsonl.jsonl`, one row per input document in input order, fields `id, text_a, text_b, labels_a, labels_b`; one float in [0, 1] per whitespace token, `-1` on ASCII punctuation tokens only (`docs/prediction_format.md`). The predictions submitted with this repository are in `predictions/` (M1, B0) and `predictions/extra/` (M0, M2, M3); SHA-256 values are in `predictions/SHA256SUMS` (identical to the hashes recorded before any test label was read), and `make run` reproduces them bit for bit. Results and caveats are in the technical report.

`make eval PRED_PREFIX=outputs/<dir>/<method>_admin_ [SPLIT=dev]` scores with the organizers' own script (the image build clones github.com/ZurichNLP/SwissGov-RSD at commit `1807a42100e742ed03d337c54c4b9ea86995f565`; their repo has no licence, so nothing of it is vendored). It needs network at build time, is heavy (their requirements include torch), and **was not executed** in the verification of this repository (only `make -n eval`). `make validate` runs the format validator on the host (python3 only).

### Layout
- `src/polyjudge/` code (runtime modules); `src/dev_only/cv.py` leave-one-chunk-out analysis, dev only (not in the image)
- `scripts/` format validator, cache manifest, few-shot export, secrets/label scan, report number checker (`check_report_numbers.py`)
- `requirements.lock` (pip freeze of the runtime image), `requirements.txt` (top-level pins)
- `data/test_input/` test inputs, **text only** (CC-BY-4.0, ZurichNLP/SwissGov-RSD; no labels); `data/splits/` split ids; `data/replay_cache/` and `data/replay_cache/extras/`; `data/few_shot/` the 16 frozen prompt-pool documents used as in-prompt examples by M0 and M2 (CC-BY-4.0, with attribution). Size of the tracked `data/` directory, measured with `git ls-files`: 7.16 MB in 5 186 files (limit: 100 MB)
- `predictions/` submitted predictions; `docs/` preregistration, test results, hashes, format notes, owner confirmations, measurements; `technical_report.md`

**Licences of this submission**
- Code (`src/`, `scripts/`, `tests/`, `Dockerfile`, `Makefile`): Apache-2.0 (the `LICENSE` file at the repository root).
- Documentation, the technical report and the 2-page summary (PDF/HTML): CC-BY-4.0.
- Third-party data (SwissGov-RSD test inputs and the few-shot documents, by Wastl, Vamvas and Sennrich): CC-BY-4.0; attribution for the few-shot documents is in `data/few_shot/ATTRIBUTION.md`.
- The organizers' evaluation code has no licence and is not redistributed.

---

*Challenge text from the hackathon template follows.*

# Academia Challenges

Submissions must use the Apertus model family.
For Track 2 this means that submitted solutions must be built with Apertus. Other open-weights models can be used to support development, e.g. as automatic judges during evaluation. Their role must be clearly described in the submission report.

💬 In case you have questions, join the conversation on Discord or send an email to “hello@hackapertus.ch”

## How it works
Pick from 5 academia challenges provided by Swiss academic institutions:

- **FHGR:** AI-Powered Job Interview Coach
- **OpenParlData:** Extracting Parliamentary Affairs from PDFs into One Common Structure
- **OST:** Multilingual Natural Language Inference over Swiss Official Voting Booklets
- **UZH:** Detecting Cross-Lingual Semantic Differences in Swiss Government Websites
- **ZHAW:** See It, Say It, Pick It: Vision-Language Grounding for a Real Robot Arm

The challenges incl. submission and judging criteria are described in our **Getting Started guide**:
https://hackapertus.notion.site/getting-started-guide-onlinehack

## Run it

Keep `track_2a/` as it is: don't rename it or move its files, just delete the
other track directories.

From the root of the project:

```bash
make run
```

Fill in the [Makefile](Makefile) so that it works on a clean checkout. It is
expected to run the project in a Docker container, since that is how the judges
will run it, without relying on anything already installed on your machine.

Requirements: Docker and `make`; no GPU, no API key and no model weights for `make run` (Apertus responses are replayed from the committed cache). `make live` needs `LLM_NAME`, `LLM_BASE_URL` and `LLM_API_KEY` for the CSCS Inference API.

## Data
The `data/` directory must not exceed 100 MB.


## 📦 Submission Requirements & Deliverables
❗️ Submissions are not handled on Devpost. Submit through our website only:
http://hackapertus.ch/online-hack/submissions

Requirements differ by challenge. See the description of the challenge you are entering for the exact deliverables.


## ⚖️ Judging Criteria
Judging criteria also differ by challenge. See the respective challenge description.


## Support

**Licensing requirements**
Please check our Terms & Conditions (6. What you build is open source):
https://hackapertus.ch/terms-and-conditions

## FAQ
💡 https://hackapertus.ch/faq

## Contact
💬 In case you have questions, join the conversation on Discord or send an email to “hello@hackapertus.ch”
