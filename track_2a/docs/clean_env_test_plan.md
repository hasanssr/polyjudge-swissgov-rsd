# Clean-environment test plan for `make run`

Goal: a judge can run `make run` on a fresh clone with no network at run time. Network is needed once, for `docker build`
(base image, `pip install -r requirements.lock`). Every `docker run` in the recipes uses `--network none`.

## Preconditions
- Docker running; no `.env`, no `LLM_*` variables (`env | grep LLM_` prints nothing); `make` installed (or use the docker:cli trick below).
- Work in a directory outside the working tree and clone the pushed branch (or, before the first push, copy `git ls-files -co --exclude-standard`).

## Steps
1. `git clone <repo-url> /tmp/fresh && cd /tmp/fresh/track_2a`; `ls -a` shows no `.env`, `outputs/`, `.cache/`.
2. Optional cold start: `docker rmi -f polyjudge-2a:local; docker builder prune -af`.
3. `make run` — expect exit 0, two lines `OK: 3 file(s) valid` (M1 in `outputs/predictions/`, B0 in `outputs/baseline_b0/`). Record wall-clock time and `docker images polyjudge-2a:local`.
4. Network really off: `docker run --rm --network none polyjudge-2a:local python -c "import urllib.request as u; u.urlopen('http://1.1.1.1', timeout=3)"` must fail.
5. Byte-identical check: compare `sha256sum` of `outputs/**/*.jsonl.jsonl` with `predictions/SHA256SUMS`.
6. All methods: `for m in M0 M2 M3 B0; do make run METHOD=$m; done` (outputs in `outputs/extra/<m>` or `outputs/baseline_b0`), then repeat step 5 for `predictions/extra/`.
7. `make test` — expect all tests pass with the network disabled.
8. `make live` without `LLM_*` exits 1 with a one-line message and prints no secret.
9. Validator negative checks on a copy of `outputs/predictions`: truncate a label list, swap two rows, rename a file; `make validate` must exit 1 each time.
10. NOT part of this plan: `make eval` (clones the organizers' repo and installs torch; needs network and many GB). Run it separately if scoring is wanted.

No `make` on the host (e.g. Windows)? Run the real Makefile through a container that has the docker CLI and the same path mounted:
```bash
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v "$F:$F" -w "$F/track_2a" docker:cli \
  sh -c "apk add -q make && make run"      # $F = absolute path of the fresh checkout
```

## Repo checks before publishing
- `python scripts/scan_secrets.py .` prints `0 finding(s)`.
- `git ls-files | grep -E '(^|/)\.env|outputs/|\.cache/|__pycache__'` is empty.
- `data/test_input/*.jsonl` keys are exactly `id, subset, text_a, text_a_untokenized, text_b, text_b_untokenized` (tests/test_inputs_no_labels.py).
- `du -sh data` is far below 100 MB.

## Record
Date, commit SHA, OS, Docker version, time of `make run`, pass/fail per step.
