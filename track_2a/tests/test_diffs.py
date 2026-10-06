import asyncio
import json
from types import SimpleNamespace

import pytest

from polyjudge.config import LLMConfig
from polyjudge.diffs import extract_json, normalize_side, parse_b1, parse_compare
from polyjudge.llm import CallBudget, LLMClient, is_degenerate


@pytest.mark.parametrize("raw,expected", [("en", "a"), ("EN", "a"), ("English", "a"), ("a", "a"), ("tgt", "b"), ("target", "b"), ("de", "b"),
                                          ("fr", "b"), ("it", "b"), ("French", "b"), (" 'it' ", "b"), ("xx", None), (None, None), (3, None)])
def test_normalize_side(raw, expected):
    assert normalize_side(raw) == expected


def test_b1_accepts_en_de_fr_it_and_tgt_sides():
    txt = ('{"differences": [{"side":"en","quote":"A","severity":0.6,"before":""},{"side":"fr","quote":"B","severity":"0.8","before":"x y"},'
           '{"side":"de","quote":"C","severity":0.2},{"side":"it","quote":"D","severity":1.0},{"side":"tgt","quote":"E","severity":0.4},'
           '{"side":"zz","quote":"F","severity":0.4},{"side":"en","quote":"","severity":0.4},{"side":"en","quote":"G","severity":"high"}]}')
    r = parse_b1(txt)
    assert r.ok and r.n_raw == 8 and r.n_dropped == 3
    assert [(x["side"], x["quote"], x["severity"]) for x in r.records] == [("a", "A", 0.6), ("b", "B", 0.8), ("b", "C", 0.2), ("b", "D", 1.0), ("b", "E", 0.4)]


def test_compare_types_repair_and_language_keys():
    txt = ('{"differences": ['
           '{"type":"omission","en":{"quote":"only EN","before":"x"},"tgt":{"quote":"should be dropped","before":""},"severity":1.0,"reason":"r1"},'
           '{"type":"addition","en":{"quote":"dropped","before":""},"fr":{"quote":"only FR","before":""},"severity":0.8,"reason":"r2"},'
           '{"type":"meaning_shift","en":{"quote":"E","before":""},"tgt":{"quote":"T","before":"a b"},"severity":0.4,"reason":"r3"},'
           '{"type":"meaning_shift","en":null,"tgt":null,"severity":0.4}]}')
    r = parse_compare(txt)
    assert r.ok and r.n_raw == 4 and r.n_dropped == 1
    assert [(x["side"], x["quote"], x["severity"], x["type"]) for x in r.records] == [
        ("a", "only EN", 1.0, "omission"), ("b", "only FR", 0.8, "addition"), ("a", "E", 0.4, "meaning_shift"), ("b", "T", 0.4, "meaning_shift")]


def test_garbage_and_wrapped_json():
    assert not parse_b1("not json at all").ok
    assert not parse_b1('{"foo": 1}').ok
    assert parse_b1('Sure! ```json\n{"differences": []}\n```').ok
    assert parse_b1('{"differences": [{"side":"en","quote":"q","severity":0.2}]} trailing text').records[0]["quote"] == "q"
    assert extract_json('prefix {"a": "}{"} suffix') == {"a": "}{"}


def test_is_degenerate():
    loop = '{"differences": [' + '{"side": "en", "quote": "the same quote again", "severity": 0.6}, ' * 5
    assert is_degenerate(loop)
    assert not is_degenerate('{"differences": [{"side":"en","quote":"alpha beta","severity":0.6},{"side":"tgt","quote":"gamma delta","severity":0.4}]}')


# ---- streaming repetition guard with a fake SDK ----
def _chunk(text=None, finish=None, usage=None):
    ch = [SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=finish)] if (text is not None or finish) else []
    return SimpleNamespace(choices=ch, usage=usage)


class FakeStream:
    def __init__(self, chunks):
        self.chunks, self.closed, self.consumed = chunks, False, 0

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self.chunks:
            self.consumed += 1
            yield c

    async def close(self):
        self.closed = True


def _client(tmp_path, stream):
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=None)))

    async def create(**kw):
        assert kw["stream"] is True and kw["stream_options"]["include_usage"] is True
        return stream
    sdk.chat.completions.create = create
    b = CallBudget(tmp_path / "b.json")
    b.reset(10)
    return LLMClient(LLMConfig("swiss-ai/Apertus-v1.5-8B", "https://x.invalid/v1", "SECRET"), sdk_client=sdk, cache_dir=tmp_path / "c", budget=b, log_path=None), b


def test_repetition_guard_aborts_early_and_the_abort_is_cached(tmp_path):
    unit = '{"side": "en", "quote": "the same quote again", "severity": 0.6}, '
    chunks = [_chunk('{"differences": [')] + [_chunk(unit) for _ in range(200)] + [_chunk(finish="stop")]
    st = FakeStream(chunks)
    c, b = _client(tmp_path, st)
    r = asyncio.run(c.chat([{"role": "user", "content": "x"}], rep_guard=True))
    assert r.aborted and r.finish_reason == "repetition_abort" and st.closed and st.consumed < 20 and b.state()["used"] == 1
    files = list((tmp_path / "c").rglob("*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text(encoding="utf-8"))["aborted"] is True
    r2 = asyncio.run(c.chat([{"role": "user", "content": "x"}], rep_guard=True))   # served from the cache: still aborted, no budget used
    assert r2.cached and r2.aborted and b.state()["used"] == 1


def test_stream_normal_completion_is_cached(tmp_path):
    chunks = [_chunk('{"differences": '), _chunk("[]}"), _chunk(finish="stop"), _chunk(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=5))]
    c, b = _client(tmp_path, FakeStream(chunks))
    r = asyncio.run(c.chat([{"role": "user", "content": "x"}], rep_guard=True))
    assert not r.aborted and r.text == '{"differences": []}' and (r.prompt_tokens, r.completion_tokens) == (7, 5) and r.finish_reason == "stop"
    assert len(list((tmp_path / "c").rglob("*.json"))) == 1
