import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import validate_predictions as vp  # noqa: E402

SRC = [
    {"id": "1", "text_a": "Der Bund hilft , sofort .", "text_b": "La Confédération aide"},
    {"id": "2", "text_a": "Hallo Welt", "text_b": "Bonjour ( oui )"},
]
IN_NAME, OUT_NAME = "swissde.jsonl", "swissde.jsonl.jsonl"


def write(d: Path, name: str, rows):
    d.mkdir(exist_ok=True)
    (d / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def good_pred():
    return [
        {**SRC[0], "labels_a": [0.0, 0.2, 1.0, -1, 0.4, -1], "labels_b": [0, 0.6, 1.0]},
        {**SRC[1], "labels_a": [0.0, 0.2], "labels_b": [1.0, -1, 0.0, -1]},
    ]


def run(tmp_path, pred_rows, pred_name=OUT_NAME, extra=()):
    write(tmp_path / "in", IN_NAME, SRC)
    write(tmp_path / "out", pred_name, pred_rows)
    return vp.main(["--input-dir", str(tmp_path / "in"), "--pred-dir", str(tmp_path / "out"), *extra])


def test_valid_continuous_and_minus_one_on_punct(tmp_path):
    assert run(tmp_path, good_pred()) == 0


def test_valid_non_step_values(tmp_path):
    rows = good_pred()
    rows[0]["labels_a"][0] = 0.137
    assert run(tmp_path, rows) == 0


def test_wrong_label_count(tmp_path):
    rows = good_pred()
    rows[0]["labels_a"] = rows[0]["labels_a"][:-1]
    assert run(tmp_path, rows) == 1


def test_row_order_changed(tmp_path):
    assert run(tmp_path, good_pred()[::-1]) == 1


def test_missing_and_extra_field(tmp_path):
    rows = good_pred()
    del rows[1]["labels_b"]
    assert run(tmp_path, rows) == 1
    rows = good_pred()
    rows[0]["score"] = 0.5
    assert run(tmp_path, rows) == 1


def test_wrong_file_name(tmp_path):
    assert run(tmp_path, good_pred(), pred_name="swissde.jsonl") == 1
    assert run(tmp_path, good_pred(), pred_name="swissxx.jsonl.jsonl") == 1


def test_prefix_enforced(tmp_path):
    assert run(tmp_path, good_pred(), extra=["--prefix", "swiss"]) == 0
    assert run(tmp_path, good_pred(), extra=["--prefix", "other"]) == 1


def test_missing_row(tmp_path):
    assert run(tmp_path, good_pred()[:1]) == 1


def test_out_of_range_values(tmp_path):
    for bad in (1.2, -0.1, float("nan")):
        rows = good_pred()
        rows[0]["labels_a"][0] = bad
        assert run(tmp_path, rows) == 1


def test_minus_one_only_on_ascii_punct(tmp_path):
    rows = good_pred()
    rows[0]["labels_a"][0] = -1  # 'Der' is not punctuation
    assert run(tmp_path, rows) == 1


def test_punct_needs_minus_one_unless_relaxed(tmp_path):
    rows = good_pred()
    rows[0]["labels_a"][3] = 0.5  # ',' token
    assert run(tmp_path, rows) == 1
    assert run(tmp_path, rows, extra=["--allow-punct-values"]) == 0


def test_non_numeric_and_bool_labels(tmp_path):
    for bad in ("0.5", True, None):
        rows = good_pred()
        rows[0]["labels_a"][0] = bad
        assert run(tmp_path, rows) == 1


def test_non_ascii_punctuation_is_a_normal_token(tmp_path):
    src = [{"id": "1", "text_a": "« Bund »", "text_b": "x"}]
    write(tmp_path / "in", IN_NAME, src)
    ok = [{**src[0], "labels_a": [0.5, 0.0, 0.2], "labels_b": [0.0]}]
    write(tmp_path / "out", OUT_NAME, ok)
    args = ["--input-dir", str(tmp_path / "in"), "--pred-dir", str(tmp_path / "out")]
    assert vp.main(args) == 0
    bad = [{**src[0], "labels_a": [-1, 0.0, 0.2], "labels_b": [0.0]}]
    write(tmp_path / "out", OUT_NAME, bad)
    assert vp.main(args) == 1
