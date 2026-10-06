"""The input loader must not be able to expose label fields. Uses only the repo's own test inputs (no organizers data, no network)."""
import asyncio
import dataclasses
import json

import pytest

from polyjudge import inputs as I
from polyjudge.inputs import FORBIDDEN_KEYS, INPUT_KEYS, Input, _from_raw, load_inputs
from polyjudge.predict import predict

ROW = dict(id="admin_de_1", subset="admin_de", text_a="a b", text_a_untokenized="a b", text_b="c d", text_b_untokenized="c d")


def test_input_has_exactly_the_whitelisted_fields_and_is_frozen():
    assert tuple(f.name for f in dataclasses.fields(Input)) == INPUT_KEYS
    assert not set(FORBIDDEN_KEYS) & set(INPUT_KEYS)
    obj = _from_raw(ROW)
    with pytest.raises(dataclasses.FrozenInstanceError):
        obj.text_a = "x"
    for k in FORBIDDEN_KEYS:
        assert not hasattr(obj, k)


def test_from_raw_drops_label_and_annotator_fields():
    raw = {**ROW, "labels_a": [0.2, 0.4], "labels_b": [0.0, 1.0], "annotator_tag": "x", "chunk_id": "c"}
    obj = _from_raw(raw)
    assert set(vars(obj)) == set(INPUT_KEYS) and set(obj.as_row()) == set(INPUT_KEYS)
    assert not any(k in json.dumps(obj.as_row()) for k in FORBIDDEN_KEYS)


def test_shipped_test_inputs_contain_no_label_keys():
    for lang in ("de", "fr", "it"):
        for line in I._test_file(lang).read_text(encoding="utf-8").splitlines():
            keys = set(json.loads(line))
            assert keys == set(INPUT_KEYS), keys ^ set(INPUT_KEYS)


def test_load_inputs_test_returns_only_input_objects():
    docs = load_inputs("test")
    assert len(docs) == 168 and all(type(d) is Input for d in docs)
    assert {d.lang for d in docs} == {"de", "fr", "it"}
    for d in docs:
        assert set(vars(d)) == set(INPUT_KEYS) and not any(hasattr(d, k) for k in FORBIDDEN_KEYS)


def test_loader_discards_labels_even_if_the_file_contains_them(tmp_path, monkeypatch):
    (tmp_path / "data" / "test_input").mkdir(parents=True)
    for lang in ("de", "fr", "it"):
        row = {**ROW, "id": f"admin_{lang}_1", "subset": f"admin_{lang}", "labels_a": [0.2, 0.4], "labels_b": [0.0, 1.0], "annotator_tag": "t"}
        (tmp_path / "data" / "test_input" / f"test_{lang}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(I, "ROOT", tmp_path)
    docs = load_inputs("test")
    assert len(docs) == 3 and all(set(vars(d)) == set(INPUT_KEYS) for d in docs)


def test_predict_refuses_anything_but_input_objects():
    with pytest.raises(TypeError):
        asyncio.run(predict("B0", [dict(ROW, labels_a=[0.0])], "offline"))
