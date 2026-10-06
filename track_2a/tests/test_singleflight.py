"""Single-flight (identical in-flight requests -> ONE API call, same cache key, same result for every waiter), adaptive concurrency, chained budgets."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from polyjudge.config import LLMConfig
from polyjudge.llm import BudgetExceeded, CallBudget, ChainBudget, LLMClient, LLMError

CFG = LLMConfig("swiss-ai/Apertus-v1.5-8B", "https://example.invalid/v1", "SECRET")


class FakeRaw:
    def __init__(self, text):
        self._text, self.headers = text, httpx.Headers({})

    def parse(self):
        msg = SimpleNamespace(content=self._text)
        return SimpleNamespace(model="m", choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2))


def _chunk(text=None, finish=None, usage=None):
    ch = [SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=finish)] if (text is not None or finish) else []
    return SimpleNamespace(choices=ch, usage=usage)


class FakeStream:
    def __init__(self, text):
        self.parts = [_chunk(text), _chunk(finish="stop"), _chunk(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2))]

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for p in self.parts:
            await asyncio.sleep(0)
            yield p

    async def close(self):
        pass


class FakeSDK:
    """Answers depend on the request content AND on the call number, so two separate calls for the same prompt would give different texts."""
    def __init__(self, delay=0.05, fail_status=None, rate_limit_first=0):
        self.calls, self.delay, self.fail_status, self.rate_limit_first = 0, delay, fail_status, rate_limit_first
        self.inflight = self.max_inflight = 0
        outer = self

        class _Raw:
            async def create(self, **kw):
                return await outer._handle(kw, stream=False)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create, with_raw_response=_Raw()))

    async def _create(self, **kw):
        return await self._handle(kw, stream=kw.get("stream", False))

    async def _handle(self, kw, stream):
        self.calls += 1
        n = self.calls
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await asyncio.sleep(self.delay)
            if self.fail_status:
                resp = httpx.Response(self.fail_status, request=httpx.Request("POST", "http://x"))
                raise openai.BadRequestError("bad", response=resp, body=None)
            if n <= self.rate_limit_first:
                resp = httpx.Response(429, headers={"retry-after": "0"}, request=httpx.Request("POST", "http://x"))
                raise openai.RateLimitError("slow down", response=resp, body=None)
            text = json.dumps({"call": n, "user": kw["messages"][-1]["content"]})
            return FakeStream(text) if stream else FakeRaw(text)
        finally:
            self.inflight -= 1


def mk(tmp_path, sdk, limit=1000, **kw):
    b = CallBudget(tmp_path / "b.json")
    b.reset(limit)
    return LLMClient(CFG, sdk_client=sdk, cache_dir=tmp_path / "c", budget=b, log_path=None, **kw), b


def msg(x="same"):
    return [{"role": "user", "content": x}]


@pytest.mark.parametrize("rep_guard", [False, True])
def test_concurrent_identical_requests_make_one_api_call(tmp_path, rep_guard):
    sdk = FakeSDK()
    c, b = mk(tmp_path, sdk, concurrency=8)

    async def go():
        return await asyncio.gather(*(c.chat(msg(), rep_guard=rep_guard) for _ in range(5)))
    rs = asyncio.run(go())
    assert sdk.calls == 1 and b.state()["used"] == 1 and c.stats.coalesced == 4
    assert len({r.text for r in rs}) == 1 and len({r.key for r in rs}) == 1          # same key, same result for all waiters
    assert len(list((tmp_path / "c").rglob("*.json"))) == 1
    r2 = asyncio.run(c.chat(msg(), rep_guard=rep_guard))                              # afterwards: a plain cache hit
    assert r2.cached and r2.text == rs[0].text and sdk.calls == 1


def test_distinct_requests_are_not_merged_and_use_cache_false_is_never_merged(tmp_path):
    sdk = FakeSDK()
    c, _ = mk(tmp_path, sdk, concurrency=8)

    async def go():
        a = await asyncio.gather(*(c.chat(msg(f"q{i}")) for i in range(4)))
        b = await asyncio.gather(*(c.chat(msg("same"), use_cache=False) for _ in range(3)))
        return a, b
    a, b = asyncio.run(go())
    assert sdk.calls == 4 + 3 and c.stats.coalesced == 0 and len({r.text for r in a}) == 4


def test_leader_error_reaches_every_waiter_and_nothing_stays_in_flight(tmp_path):
    sdk = FakeSDK(fail_status=400)
    c, b = mk(tmp_path, sdk, concurrency=8)

    async def go():
        return await asyncio.gather(*(c.chat(msg()) for _ in range(4)), return_exceptions=True)
    out = asyncio.run(go())
    assert all(isinstance(e, LLMError) for e in out) and sdk.calls == 1 and b.state()["used"] == 1
    assert not c._inflight


def test_budget_exhaustion_is_propagated_once(tmp_path):
    sdk = FakeSDK()
    c, b = mk(tmp_path, sdk, limit=0, concurrency=8)

    async def go():
        return await asyncio.gather(*(c.chat(msg()) for _ in range(3)), return_exceptions=True)
    out = asyncio.run(go())
    assert all(isinstance(e, BudgetExceeded) for e in out) and sdk.calls == 0


def test_replay_and_cache_still_work_with_merging(tmp_path):
    sdk = FakeSDK()
    c, _ = mk(tmp_path, sdk, concurrency=8)
    rs = asyncio.run(_gather(c))
    rep = LLMClient(None, model="swiss-ai/Apertus-v1.5-8B", replay=True, cache_dir=tmp_path / "c", log_path=None)
    again = asyncio.run(_gather(rep))
    assert [r.text for r in again] == [r.text for r in rs] and sdk.calls == 1


async def _gather(c):
    return await asyncio.gather(*(c.chat(msg(), rep_guard=True) for _ in range(3)))


# ----------------------------------------------------------------------------- adaptive concurrency
def test_concurrency_rises_from_base_to_max_only_without_429(tmp_path):
    sdk = FakeSDK(delay=0.01)
    c, _ = mk(tmp_path, sdk, concurrency=2, max_concurrency=6, raise_after=4)

    async def go():
        await asyncio.gather(*(c.chat(msg(f"q{i}")) for i in range(12)))
    asyncio.run(go())
    assert c.sem.limit == 6 and sdk.max_inflight <= 6 and sdk.max_inflight >= 3          # it did use the higher limit

    sdk2 = FakeSDK(delay=0.01, rate_limit_first=1)
    c2, _ = mk(tmp_path / "x", sdk2, concurrency=2, max_concurrency=6, raise_after=4)

    async def go2():
        await asyncio.gather(*(c2.chat(msg(f"r{i}")) for i in range(12)))
    asyncio.run(go2())
    assert c2.stats.n429 == 1 and c2.sem.limit == 2 and sdk2.max_inflight <= 2           # a 429 pins the base concurrency


# ----------------------------------------------------------------------------- chained budgets
def test_chain_budget_reserves_on_all_or_none(tmp_path):
    method, glob = CallBudget(tmp_path / "method.json"), CallBudget(tmp_path / "global.json")
    method.reset(2)
    glob.reset(10)
    ch = ChainBudget(method, glob)
    ch.reserve()
    ch.reserve()
    with pytest.raises(BudgetExceeded):
        ch.reserve()                                                                       # method limit reached
    assert method.state()["used"] == 2 and glob.state()["used"] == 2                       # nothing was reserved by the failed call
    method.reset(5)
    glob.reset(2)
    glob._write({"limit": 2, "used": 2})
    with pytest.raises(BudgetExceeded):
        ch.reserve()                                                                       # global limit reached
    assert method.state()["used"] == 0
