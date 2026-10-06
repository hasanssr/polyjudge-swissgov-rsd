#!/usr/bin/env python3
"""One-off exporter: freeze the few-shot example documents used by M0 and M2 into data/few_shot/ (provenance + checks).

Needs the pinned dev split of ZurichNLP/SwissGov-RSD (HF cache or network), so run it on a tree WITHOUT data/few_shot/
(otherwise the frozen files short-circuit the original selection). It runs the original selection code:
  M0: prompts.frozen_shots(lang, "b1")            (splits/shots_round1.json)
  M2: prompts_v2.select_shots2(lang, fewshot_chunks())
and writes only those documents (prompt_pool records of the dev split, CC-BY-4.0), checking that none lies in an evaluation
chunk (pool_eval*, dev_select, dev_final) or among the test inputs.
usage (from an /app-like tree):  python scripts/export_few_shot.py [--out data/few_shot]
"""
import argparse
import json
from pathlib import Path

from polyjudge.data import LANGS, ROOT, chunk_key, load_split, load_split2
from polyjudge.evalsets import fewshot_chunks
from polyjudge.prompts import eval_chunks, frozen_shots
from polyjudge.prompts_v2 import select_shots2

ATTRIBUTION = """# Attribution

The few-shot example documents in `few_shot_rows.jsonl` are records of the **dev split** (prompt_pool part) of the
SwissGov-RSD dataset, pinned revision `f16aa0536811b8c0cf975fede21eec5996e4b3c1`
(https://huggingface.co/datasets/ZurichNLP/SwissGov-RSD), licensed **CC-BY-4.0**.

Citation: Wastl, Vamvas, Sennrich. SwissGov-RSD (ACL 2026), ZurichNLP.

Changes: only the {n_docs} documents used as in-prompt examples by methods M0 and M2 are included, unmodified (all fields).
`m2_shots.json` records which documents and beads the M2 selection picked; the M0 selection is `data/splits/shots_round1.json`.
No document comes from an evaluation set (pool_eval*, dev_select, dev_final) or from the test split (checked by
`scripts/export_few_shot.py`).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "few_shot")
    a = ap.parse_args()
    assert not (a.out / "few_shot_rows.jsonl").exists(), "frozen files already exist: run on a tree without data/few_shot/"
    pool_chunks = set(load_split()["prompt_pool"])
    s2 = load_split2()
    forbidden = eval_chunks() | set(s2["dev_select"]) | set(s2["dev_final"])
    test_ids = {json.loads(l)["id"] for p in (ROOT / "data" / "test_input").glob("test_*.jsonl") for l in p.open(encoding="utf-8") if l.strip()}

    rows, m2 = {}, {}
    for lang in LANGS:
        for row, g in frozen_shots(lang, "b1"):                              # M0
            rows[row["id"]] = row
        m2[lang] = []
        for row, g, _out in select_shots2(lang, fewshot_chunks()):          # M2
            rows[row["id"]] = row
            m2[lang].append({"doc": row["id"], "beads": g.bead_ids})
    for r in rows.values():
        assert chunk_key(r) in pool_chunks, f"{r['id']} is not a prompt_pool record"
        assert chunk_key(r) not in forbidden, f"{r['id']} lies in an evaluation chunk"
        assert r["id"] not in test_ids, f"{r['id']} is a test input"
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "few_shot_rows.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for i in sorted(rows):
            f.write(json.dumps(rows[i], ensure_ascii=False, separators=(",", ":")) + "\n")
    (a.out / "m2_shots.json").write_text(json.dumps(m2, indent=1), encoding="utf-8")
    (a.out / "ATTRIBUTION.md").write_text(ATTRIBUTION.format(n_docs=len(rows)), encoding="utf-8")
    print(json.dumps({"docs": len(rows), "by_lang": {l: sum(1 for i in rows if i.startswith(f"admin_{l}_")) for l in LANGS}}))


if __name__ == "__main__":
    main()
