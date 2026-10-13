# PolyJudge viewer (optional demo)

An offline viewer of the **frozen predictions** (B0, M0, M1, M2, M3) and the **recorded M2 outputs** (the multi-agent method, shown as a replay). It is **not live**: nothing is computed, no model is called, no network is used. It is optional and not needed for `make run`.

## Open
Double-click `dist/polyjudge_viewer.html` (any modern browser; it works from `file://`).

`dist/polyjudge_viewer.html` is included in the repository (LF line endings): 7 633 773 bytes (7.28 MiB), sha256 `f3acc182a99906da04630238794d9c2324e6241f7357e8d78aca41f94677357d`. It can be rebuilt with `python build_viewer.py`. On Windows the build writes CRLF line endings (7 634 615 bytes, sha256 `66b301695f07e97df4f7fe5635a708cc2ecb377dadcd689843f44dafc8d66b7c`); after converting CRLF to LF the file is byte-identical to the committed one (checked), and git stores it with LF.

Press **F11** for full screen and **P** for presentation mode (Esc leaves it). The page opens on the PolyJudge tab; keys 1/2/3 switch between Viewer, PolyJudge and Scoreboard.

## Rebuild
```bash
python build_viewer.py            # standard library only; writes dist/polyjudge_viewer.html
python tests/smoke_test.py        # checks the built file (also opens it in headless Edge/Chrome if installed)
```
The rebuild from the files in this folder gives the same `dist/polyjudge_viewer.html`, apart from the line-ending difference described above (checked by sha256). The smoke test also confirms that none of the 168 test documents carries gold labels.

## Where the data comes from
`viewer_data.json` (schema: `README_viewer_data.md`) is derived from the frozen prediction files and the replay cache by an export script that is **not part of this repository**. It contains the inputs, sentence alignment, the five methods' predictions and, for the M2 replay, truncated recorded agent messages. The bundle contains gold labels only for 15 documents of the public SwissGov-RSD validation (dev) split, used under CC-BY-4.0; test documents carry no labels (`gold = null`, checked by the smoke test).

`README_viewer.md` is the README of the UI project this folder was copied from; the schema and rebuild notes there still apply.

## Size
About 21.2 MB in total (the built HTML is 7.6 MB, the four character images 7.2 MB, `viewer_data.json` 6.3 MB). The 100 MB limit applies to `track_2a/data/`, which this folder is not part of.

## Attribution
Data: SwissGov-RSD, CC-BY-4.0 (Wastl, Vamvas, Sennrich). Character illustrations were generated with an AI image generator.
