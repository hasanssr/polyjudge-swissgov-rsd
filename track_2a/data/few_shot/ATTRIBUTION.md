# Attribution

The few-shot example documents in `few_shot_rows.jsonl` are records of the **dev split** (prompt_pool part) of the
SwissGov-RSD dataset, pinned revision `f16aa0536811b8c0cf975fede21eec5996e4b3c1`
(https://huggingface.co/datasets/ZurichNLP/SwissGov-RSD), licensed **CC-BY-4.0**.

Citation: Wastl, Vamvas, Sennrich. SwissGov-RSD (ACL 2026), ZurichNLP.

Changes: only the 16 documents used as in-prompt examples by methods M0 and M2 are included, unmodified (all fields).
`m2_shots.json` records which documents and beads the M2 selection picked; the M0 selection is `data/splits/shots_round1.json`.
No document comes from an evaluation set (pool_eval*, dev_select, dev_final) or from the test split (checked by
`scripts/export_few_shot.py`).
