# PolyJudge viewer (offline)

Single self-contained HTML app built from `viewer_data.json` (schema: `README_viewer_data.md`).
No CDN, no external fonts/scripts, no API calls; it only reads the bundle, never `C:\Projects\PolyJudge` or label data.

## Open
Double-click `dist/polyjudge_viewer.html` (any modern browser, works from `file://`).

## Rebuild
```
python build_viewer.py            # stdlib only; writes dist/polyjudge_viewer.html
python build_viewer.py --full-res # embed the avatar PNGs at original resolution (much larger file)
python tests/smoke_test.py        # checks the BUILT html
```
Inputs: `viewer_data.json`, `assets/char_de.png|char_fr.png|char_it.png|char_judge.png` (a missing image becomes a placeholder
avatar), `src/template.html`. By default the 1254 px portraits are box-downscaled 3x (pure stdlib) since they are shown at <= ~110 px.

## Smoke test
`tests/smoke_test.py` parses the built file and fails if any `split=test` document has non-null gold, if prediction lengths do not
match token counts, if the embedded data differs from `viewer_data.json`, if external resources are referenced, or if required
honesty texts are missing. If Edge/Chrome is installed it also opens the file headless (`#selftest`), renders every document,
method, gold mode and PolyJudge scenes, and fails on any console/page error; otherwise that step is skipped.

## Contents
* **Viewer**: filters (language, split, id search, transcript only), methods B0/M0/M1 (main)/M2/M3, aligned sentence blocks
  (colour per alignment pair, dashed = unaligned), token heatmap (hover = exact value, punctuation -1 never coloured),
  gold | prediction | side-by-side for dev_select only, per-document summary (flagged = value > 0).
* **PolyJudge tab (default)**: replay of recorded M2 transcripts (not live); characters by role (extractor_en -> char_de, extractor_tgt -> char_fr,
  compare -> char_it, verifier -> char_judge). Phases: reading, speech bubbles (typewriter), verdict panel, span highlight.
  Verdict spans are the contiguous runs of the recorded `pred.M2` values > 0 inside the group's sentences. Transcript texts are
  cut at 400 chars by the exporter, so JSON is often truncated; the prettifier extracts fields tolerantly and falls back to raw text.
  Only the selected group's sentences are shown, so the scene fits 1920x1080 without scrolling. Controls: play group, play all groups (1.5 s pause between groups), step, speed, group selector, presentation mode (button or key P; Esc exits). `prefers-reduced-motion` disables animations and typing.
* **Scoreboard**: static table from the task description.

Navigation: full-width black tubelight bar fixed at the top (keys 1/2/3 = Viewer / PolyJudge / Scoreboard; P = presentation mode, which hides the bar; move the mouse to the top edge to show it). The app opens on the PolyJudge tab. Document picker (toolbar, both tabs): search, language/split chips, arrow keys + Enter, `[` / `]` for previous/next document; the PolyJudge tab lists only the 174 documents with an M2 transcript, the Viewer tab all 183. Speech bubbles have a fixed size and scroll internally.
