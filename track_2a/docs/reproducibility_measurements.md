# Reproducibility measurements

Measured on 2026-10-06 (local time, UTC+3) on the author's laptop: HP Victus 15-fa0xxx, 12th Gen Intel Core i5-12450H, 15.7 GB RAM, Windows 11 Home, Docker Desktop (client and server 29.8.1), GNU Make 4.4.1. No local GPU was used; `make run` makes no Apertus calls at all (replay cache, `--network none`).

## Method

1. Copy the current tree (`git ls-files -co --exclude-standard`, 5 272 files, includes uncommitted files, excludes anything ignored) to a temporary folder outside the repository.
2. In `track_2a/` of that copy: `docker rmi -f polyjudge-2a:local`, then build the runtime image from scratch:

```bash
docker build --no-cache --target runtime -t polyjudge-2a:local .
```

3. Run `make run` three times in a row (host `make`; each `docker run` in the recipe uses `--network none`). Before run 2 the `outputs/` folder was deleted, so every run regenerates the files. Wall-clock time from the shell, `date +%s%N` before and after.
4. Compare the sha256 of the six output files (M1 and B0, three languages each) with `predictions/SHA256SUMS`.

## Numbers

| quantity | value |
|---|---|
| `docker build --no-cache` of the runtime image (needs network: base image and `pip install -r requirements.lock`) | 41.9 s (41 871 ms) |
| image size | 403 MB (`docker images`); 402 562 111 bytes (`docker image inspect`; two earlier builds of the same tree gave 402 562 430 and 402 562 630 bytes) |
| `make run`, first run after the build (image already built, so `make build` is a cache hit) | 34.0 s (33 971 ms) |
| `make run`, second run (`outputs/` deleted before) | 34.3 s (34 263 ms) |
| `make run`, third run | 35.3 s (35 258 ms) |
| validator output | `OK: 3 file(s) valid` for M1 and for B0, in every run |
| output hashes vs `predictions/SHA256SUMS` | 6 of 6 files identical (0 hashes not found in the file) |

Cold and warm runs of `make run` differ by less than a second here, because the replay cache is small (M1: 332 entries); we did not profile where the time goes. The only cold step is the image build.

A first attempt at this measurement (same tree, same commands) lost its timings because `bc` is not installed on the host; it gave the same image size and identical output hashes, and the full sequence took about 2 minutes.

Not measured: `make test` and `make eval` (the latter needs network and installs the organizers' requirements, including torch).
