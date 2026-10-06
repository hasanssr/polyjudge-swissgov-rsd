"""Offline tests for polyjudge.llm using a fake SDK client (no network, no API key)."""
import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest

from polyjudge.config import LLMConfig
from polyjudge.llm import BudgetExceeded, CallBudget, LLMClient, cache_key

SECRET = "SECRET-KEY-123-DO-NOT-LEAK"
PROMPT = "PROMPT-TEXT-XYZ-SHOULD-NOT-BE-LOGGED"
CFG = LLMConfig("swiss-ai/Apertus-v1.5-8B", "https://example.invalid/v1", SECRET)


class FakeRaw:
    def __init__(self, text, headers=None):
        self._text, self.headers = text, headers or httpx.Headers({"x-ratelimit-remaining": "9", "authorization": "Bearer " + SECRET})

    def parse(self):
        msg = SimpleNamespace(content=self._text)
        return SimpleNamespace(model="swiss-ai/Apertus-v1.5-8B", choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3))


class FakeSDK:
    """scripted: list of exceptions or strings, consumed per call."""
    def __init__(self, script=None, delay=0.0):
        self.script, self.calls, self.inflight, self.max_inflight, self.delay = list(script or []), 0, 0, 0, delay
        outer = self

        class _Raw:
            async def create(self, **kw):
                outer.calls += 1
                outer.inflight += 1
                outer.max_inflight = max(outer.max_inflight, outer.inflight)
                try:
                    await asyncio.sleep(outer.delay)
                    item = outer.script.pop(0) if outer.script else "ok"
                    if isinstance(item, Exception):
                        raise item
                    return FakeRaw(item)
                finally:
                    outer.inflight -= 1
        self.chat = SimpleNamespace(completions=SimpleNamespace(with_raw_response=_Raw()))


def err(status, retry_after=None):
    h = {"retry-after": str(retry_after)} if retry_after is not None else {}
    resp = httpx.Response(status, headers=h, request=httpx.Request("POST", "http://x"))
    cls = openai.RateLimitError if status == 429 else openai.InternalServerError
    return cls("boom", response=resp, body=None)


def mk(tmp_path, sdk, limit=50, **kw):
    b = CallBudget(tmp_path / "budget.json")
    b.reset(limit)
    return LLMClient(CFG, sdk_client=sdk, cache_dir=tmp_path / "cache", budget=b, log_path=tmp_path / "log.jsonl", **kw), b


def msgs(i=0):
    return [{"role": "user", "content": f"{PROMPT} {i}"}]


def test_cache_hit_costs_nothing(tmp_path):
    sdk = FakeSDK()
    c, b = mk(tmp_path, sdk)

    async def go():
        r1 = await c.chat(msgs(), seed=1)
        r2 = await c.chat(msgs(), seed=1)
        return r1, r2
    r1, r2 = asyncio.run(go())
    assert not r1.cached and r2.cached and r2.text == r1.text
    assert sdk.calls == 1 and b.state()["used"] == 1 and c.stats.cache_hits == 1
    assert cache_key("m", msgs(), 0, 1) != cache_key("m", msgs(), 0, 2)


def test_429_retry_after_then_success_counts_every_attempt(tmp_path):
    sdk = FakeSDK([err(429, retry_after=0), "done"])
    c, b = mk(tmp_path, sdk)
    r = asyncio.run(c.chat(msgs()))
    assert r.text == "done" and r.attempts == 2 and c.stats.n429 == 1 and sdk.calls == 2 and b.state()["used"] == 2


def test_5xx_backoff_and_giveup(tmp_path, monkeypatch):
    async def fast(_):  # no real sleeping
        return None
    monkeypatch.setattr(asyncio, "sleep", fast)
    sdk = FakeSDK([err(500)] * 10)
    c, b = mk(tmp_path, sdk, max_retries=2)
    with pytest.raises(Exception) as ei:
        asyncio.run(c.chat(msgs()))
    assert "500" in str(ei.value) and sdk.calls == 3 and b.state()["used"] == 3


def test_budget_aborts_before_sending(tmp_path):
    sdk = FakeSDK()
    c, b = mk(tmp_path, sdk, limit=2)

    async def go():
        await c.chat(msgs(1))
        await c.chat(msgs(2))
        await c.chat(msgs(3))
    with pytest.raises(BudgetExceeded):
        asyncio.run(go())
    assert sdk.calls == 2 and b.state() == {"limit": 2, "used": 2}


def test_default_budget_is_zero(tmp_path):
    sdk = FakeSDK()
    b = CallBudget(tmp_path / "none.json")
    c = LLMClient(CFG, sdk_client=sdk, cache_dir=tmp_path / "c", budget=b, log_path=None)
    with pytest.raises(BudgetExceeded):
        asyncio.run(c.chat(msgs()))
    assert sdk.calls == 0


def test_dry_run_sends_nothing(tmp_path):
    sdk = FakeSDK()
    c, b = mk(tmp_path, sdk, dry_run=True)

    async def go():
        return [await c.chat(msgs(i)) for i in range(3)]
    rs = asyncio.run(go())
    assert all(r.dry_run for r in rs) and sdk.calls == 0 and c.stats.planned == 3 and b.state()["used"] == 0


def test_bounded_concurrency(tmp_path):
    sdk = FakeSDK(delay=0.02)
    c, _ = mk(tmp_path, sdk, concurrency=3)

    async def go():
        await asyncio.gather(*(c.chat(msgs(i)) for i in range(12)))
    asyncio.run(go())
    assert sdk.max_inflight <= 3 and sdk.calls == 12


def test_thinking_models_refused(tmp_path):
    c, _ = mk(tmp_path, FakeSDK())
    with pytest.raises(ValueError):
        asyncio.run(c.chat(msgs(), model="swiss-ai/Apertus-v1.5-8B-thinking"))


def test_secrets_and_prompts_never_in_log_or_cache(tmp_path):
    sdk = FakeSDK([err(429, retry_after=0), "RESPONSE-OK"])
    c, _ = mk(tmp_path, sdk)
    asyncio.run(c.chat(msgs()))
    blobs = [(tmp_path / "log.jsonl").read_text(encoding="utf-8")] + [p.read_text(encoding="utf-8") for p in (tmp_path / "cache").rglob("*.json")]
    assert blobs and all(SECRET not in b and PROMPT not in b and "Bearer" not in b and "authorization" not in b.lower() for b in blobs)
    assert "ratelimit" in blobs[0]  # rate-limit headers are kept (whitelist)
    assert SECRET not in repr(CFG) and SECRET not in str(CFG)


def test_budget_write_survives_transient_permission_errors(tmp_path, monkeypatch):
    import os as _os
    real, calls = _os.replace, {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError("locked by OneDrive")
        return real(a, b)
    monkeypatch.setattr("polyjudge.llm.os.replace", flaky)
    b = CallBudget(tmp_path / "b.json")
    b.reset(5)
    assert b.reserve() == 1 and b.state() == {"limit": 5, "used": 1} and calls["n"] >= 4
