"""Async LLM client for the OpenAI-compatible Apertus endpoint (CSCS / HF router).

* bounded concurrency (default 4), exponential backoff with full jitter on 429 / 5xx / connection errors,
  `Retry-After` honoured
* disk cache keyed by sha256(model, messages, temperature, seed, max_tokens, response_format)
* persistent global call budget: **every HTTP attempt (retries included) is counted; the call that would exceed the limit
  raises `BudgetExceeded` before anything is sent**. The default limit is 0, so nothing can run until a limit is set:
      python -m polyjudge.llm budget --reset --limit 100
* dry-run mode: no network, reports how many calls would be made / served from cache
* per-call JSONL log (latency, token counts, status, attempt, rate-limit header values). Never logs the API key,
  request headers, prompt text or response text.
* the "-thinking" model variants are refused.
"""
from __future__ import annotations

import argparse
import asyncio
import email.utils
import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import openai

from .config import LLMConfig, load_llm_config

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output"
CACHE_DIR = ROOT / "cache"
BUDGET_PATH = OUT_DIR / "llm_budget.json"
LOG_PATH = OUT_DIR / "llm_calls.jsonl"
_RL_PREFIXES = ("x-ratelimit", "ratelimit", "retry-after")


def budget_for(model: str) -> "CallBudget":
    """Separate persistent call budgets per model size (8B / 70B)."""
    tag = "70b" if "70B" in model else "8b"
    return CallBudget(OUT_DIR / f"llm_budget_{tag}.json")


class CacheMiss(RuntimeError):
    """Replay mode: the request is not in the cache. Nothing was sent."""


class BudgetExceeded(RuntimeError):
    pass


class LLMError(RuntimeError):
    def __init__(self, kind: str, status: int | None = None):
        super().__init__(f"{kind}" + (f" (HTTP {status})" if status else ""))
        self.kind, self.status = kind, status


# ------------------------------------------------------------------------------------------ budget
class CallBudget:
    """File-backed counter shared by all processes/scripts. Counts attempts, not logical calls."""

    def __init__(self, path: Path = BUDGET_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"limit": 0, "used": 0}

    def _write(self, st: dict) -> None:
        """Atomic replace, retried: on Windows (OneDrive / antivirus) the target is briefly locked -> PermissionError.
        If it still fails the exception propagates and the request is NOT sent (fail closed)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st), encoding="utf-8")
        for i in range(40):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                if i == 39:
                    raise
                time.sleep(0.05 * (1 + i // 5))

    def state(self) -> dict:
        return self._read()

    def reset(self, limit: int) -> None:
        with self._lock:
            self._write({"limit": int(limit), "used": 0})

    def reserve(self, n: int = 1) -> int:
        with self._lock:
            st = self._read()
            if st["used"] + n > st["limit"]:
                raise BudgetExceeded(f"call budget exhausted: used {st['used']} of {st['limit']}")
            st["used"] += n
            self._write(st)
            return st["used"]


class ChainBudget:
    """Several budgets that must ALL have room (e.g. a per-method sub-limit and the global limit of a model). A request is reserved on every budget or on none."""

    def __init__(self, *budgets: CallBudget):
        self.budgets = budgets
        self._lock = threading.Lock()

    def reserve(self, n: int = 1) -> int:
        with self._lock:
            for b in self.budgets:
                st = b.state()
                if st["used"] + n > st["limit"]:
                    raise BudgetExceeded(f"call budget '{b.path.stem}' exhausted: used {st['used']} of {st['limit']}")
            return [b.reserve(n) for b in self.budgets][-1]

    def state(self) -> dict:
        return {b.path.stem: b.state() for b in self.budgets}


class AdaptiveLimiter:
    """Concurrency limiter whose limit can change while requests are in flight."""

    def __init__(self, limit: int):
        self.limit = limit
        self._active = 0
        self._cond = asyncio.Condition()

    async def __aenter__(self):
        async with self._cond:
            await self._cond.wait_for(lambda: self._active < self.limit)
            self._active += 1

    async def __aexit__(self, *exc):
        async with self._cond:
            self._active -= 1
            self._cond.notify_all()

    def set_limit(self, n: int) -> None:
        self.limit = n
        try:
            asyncio.get_running_loop().create_task(self._wake())
        except RuntimeError:
            pass

    async def _wake(self):
        async with self._cond:
            self._cond.notify_all()


# ------------------------------------------------------------------------------------------ results
@dataclass
class LLMResult:
    text: str
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float = 0.0
    cached: bool = False
    dry_run: bool = False
    attempts: int = 0
    key: str = ""
    aborted: bool = False  # stopped early by the repetition guard (cached, so replay reproduces the abort)


@dataclass
class Stats:
    calls: int = 0           # logical requests
    coalesced: int = 0       # identical in-flight requests merged into another request's single API call
    cache_hits: int = 0
    attempts: int = 0        # HTTP attempts actually sent
    retries: int = 0
    n429: int = 0
    n5xx: int = 0
    planned: int = 0         # dry-run: would be sent
    events: list = field(default_factory=list)  # (t_start, latency, status) per attempt, for throughput analysis


def cache_key(model, messages, temperature, seed, max_tokens=None, response_format=None) -> str:
    blob = json.dumps([model, messages, temperature, seed, max_tokens, response_format], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


_REPEAT_TAIL = re.compile(r"(.{20,300}?)\1{2,}\s*$", re.S)


def is_degenerate(text: str, tail_chars: int = 900) -> bool:
    """True if the output ends in a loop: a unit of 20-300 chars repeated >= 3 times back to back."""
    return bool(_REPEAT_TAIL.search(text[-tail_chars:]))


def _retry_after(headers) -> float | None:
    v = None if headers is None else headers.get("retry-after")
    if not v:
        return None
    try:
        return max(0.0, float(v))
    except ValueError:
        try:
            return max(0.0, email.utils.parsedate_to_datetime(v).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


def _rl_headers(headers) -> dict:
    if headers is None:
        return {}
    return {k.lower(): v for k, v in headers.items() if k.lower().startswith(_RL_PREFIXES)}


class LLMClient:
    def __init__(self, cfg: LLMConfig | None = None, *, model: str | None = None, concurrency: int = 4,
                 cache_dir: Path = CACHE_DIR, budget: CallBudget | None = None, dry_run: bool = False, replay: bool = False,
                 max_retries: int = 5, timeout: float = 180.0, log_path: Path | None = LOG_PATH, sdk_client=None,
                 max_concurrency: int | None = None, raise_after: int = 100):
        self.cfg = cfg if cfg is not None else (None if (dry_run or replay or sdk_client is not None) else load_llm_config())
        self.default_model = model or (self.cfg.name if self.cfg else None)
        self.sem = AdaptiveLimiter(concurrency)
        self._base_conc, self._max_conc, self._raise_after, self._no_raise = concurrency, max_concurrency or concurrency, raise_after, False
        self._inflight: dict = {}      # cache key -> Future of the single request in flight
        self.used_keys: set[str] = set()   # every cache key this client was asked for (hits, merged and sent)
        self.cache_dir, self.budget = Path(cache_dir), budget or (budget_for(self.default_model) if self.default_model else CallBudget())
        self.dry_run, self.max_retries, self.log_path = dry_run, max_retries, (Path(log_path) if log_path else None)
        self.replay = replay
        self.stats = Stats()
        self._sdk = sdk_client
        if self._sdk is None and not dry_run and not replay:
            self._sdk = openai.AsyncOpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key, max_retries=0, timeout=timeout)

    # -- helpers
    @staticmethod
    def _guard_model(model: str) -> None:
        if not model or "thinking" in model.lower():
            raise ValueError("model id missing or a '-thinking' variant (not allowed)")

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cache_get(self, key: str) -> LLMResult | None:
        p = self._cache_path(key)
        if not p.is_file():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return LLMResult(**{**d, "cached": True, "key": key})

    def _cache_put(self, key: str, r: LLMResult) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        d = {k: getattr(r, k) for k in ("text", "model", "finish_reason", "prompt_tokens", "completion_tokens", "latency_s", "aborted")}
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")  # response + usage only: no prompt, no key
        os.replace(tmp, p)

    def _log(self, **rec) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        rec["ts"] = round(time.time(), 3)
        with self.log_path.open("a", encoding="utf-8") as f:  # never: key, headers (except rate-limit), prompts, responses
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # -- main entry
    async def chat(self, messages: list[dict], *, model: str | None = None, temperature: float = 0.0, seed: int | None = None,
                   max_tokens: int = 1024, json_mode: bool = False, use_cache: bool = True, tag: str = "",
                   rep_guard: bool = False) -> LLMResult:
        model = model or self.default_model
        self._guard_model(model)
        response_format = {"type": "json_object"} if json_mode else None
        key = cache_key(model, messages, temperature, seed, max_tokens, response_format)
        self.used_keys.add(key)
        self.stats.calls += 1
        if use_cache and (hit := self._cache_get(key)):
            self.stats.cache_hits += 1
            self._log(event="cache_hit", model=model, tag=tag, prompt_tokens=hit.prompt_tokens, completion_tokens=hit.completion_tokens)
            return hit
        if self.replay:
            raise CacheMiss(f"replay mode: no cached response for a '{tag or 'untagged'}' request on {model} (key {key[:12]}); nothing was sent")
        if self.dry_run:
            self.stats.planned += 1
            return LLMResult(text="", model=model, dry_run=True, key=key)
        if not use_cache:
            return await self._send(messages, model, temperature, seed, max_tokens, response_format, use_cache, tag, rep_guard, key)
        if key in self._inflight:        # single-flight: an identical request is already on its way -> wait for ITS result (same answer for every waiter)
            self.stats.coalesced += 1
            self._log(event="coalesced", model=model, tag=tag)
            return await self._inflight[key]
        fut = asyncio.get_running_loop().create_future()
        fut.add_done_callback(lambda f: None if f.cancelled() else f.exception())      # no "exception never retrieved" noise when nobody waits
        self._inflight[key] = fut
        try:
            result = await self._send(messages, model, temperature, seed, max_tokens, response_format, use_cache, tag, rep_guard, key)
        except asyncio.CancelledError:
            fut.cancel()
            raise
        except BaseException as e:
            fut.set_exception(e)         # waiters fail with the same error
            raise
        else:
            fut.set_result(result)
            return result
        finally:
            self._inflight.pop(key, None)

    def _maybe_raise(self) -> None:
        """Concurrency 4 -> 8 only after `raise_after` attempts without a single 429."""
        if (not self._no_raise and self._max_conc > self.sem.limit and self.stats.attempts >= self._raise_after and self.stats.n429 == 0):
            self.sem.set_limit(self._max_conc)

    async def _send(self, messages, model, temperature, seed, max_tokens, response_format, use_cache, tag, rep_guard, key) -> LLMResult:
        kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if seed is not None:
            kwargs["seed"] = seed
        if response_format:
            kwargs["response_format"] = response_format
        async with self.sem:
            for attempt in range(self.max_retries + 1):
                self.budget.reserve()  # raises BudgetExceeded before anything is sent
                self.stats.attempts += 1
                t0 = time.perf_counter()
                started = time.time()
                try:
                    if rep_guard:
                        r = await self._stream_once(kwargs, key, attempt, t0)
                        self.stats.events.append((started, r.latency_s, 200))
                        self._log(event="aborted" if r.aborted else "ok", model=model, tag=tag, attempt=attempt, status=200, latency_s=r.latency_s,
                                  prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens, finish_reason=r.finish_reason, rl={})
                        if use_cache:
                            self._cache_put(key, r)   # aborted answers are cached too: replay must follow the same path
                        self._maybe_raise()
                        return r
                    raw = await self._sdk.chat.completions.with_raw_response.create(**kwargs)
                    resp, headers = raw.parse(), raw.headers
                except (openai.APIConnectionError, openai.APITimeoutError) as e:
                    status, headers, kind = None, None, type(e).__name__
                except openai.APIStatusError as e:
                    status, headers, kind = e.status_code, e.response.headers, type(e).__name__
                    if status < 500 and status != 429:  # client error: do not retry
                        self._log(event="error", model=model, tag=tag, attempt=attempt, status=status, kind=kind, latency_s=round(time.perf_counter() - t0, 3))
                        self.stats.events.append((started, time.perf_counter() - t0, status))
                        raise LLMError(kind, status) from None
                else:
                    dt = time.perf_counter() - t0
                    u = resp.usage
                    r = LLMResult(text=resp.choices[0].message.content or "", model=resp.model or model,
                                  finish_reason=resp.choices[0].finish_reason, prompt_tokens=getattr(u, "prompt_tokens", None),
                                  completion_tokens=getattr(u, "completion_tokens", None), latency_s=round(dt, 3), attempts=attempt + 1, key=key)
                    self.stats.events.append((started, dt, 200))
                    self._log(event="ok", model=model, tag=tag, attempt=attempt, status=200, latency_s=r.latency_s, prompt_tokens=r.prompt_tokens,
                              completion_tokens=r.completion_tokens, finish_reason=r.finish_reason, rl=_rl_headers(headers))
                    if use_cache:
                        self._cache_put(key, r)
                    self._maybe_raise()
                    return r
                dt = time.perf_counter() - t0
                self.stats.events.append((started, dt, status or 0))
                if status == 429:
                    self.stats.n429 += 1
                    self._no_raise = True
                    self.sem.set_limit(self._base_conc)      # never above the base concurrency after a 429
                elif status and status >= 500:
                    self.stats.n5xx += 1
                ra = _retry_after(headers)
                delay = max(ra, 0.5) if ra is not None else random.uniform(0.25, min(60.0, 1.0 * 2 ** attempt))
                self._log(event="retry" if attempt < self.max_retries else "giveup", model=model, tag=tag, attempt=attempt, status=status, kind=kind,
                          latency_s=round(dt, 3), retry_after=ra, sleep_s=round(delay, 2), rl=_rl_headers(headers))
                if attempt == self.max_retries:
                    raise LLMError(kind, status)
                self.stats.retries += 1
                await asyncio.sleep(delay)
        raise LLMError("unreachable")


    async def _stream_once(self, kwargs: dict, key: str, attempt: int, t0: float) -> LLMResult:
        """Streamed completion with an early abort when the output starts to loop (see `is_degenerate`)."""
        kw = dict(kwargs, stream=True, stream_options={"include_usage": True})
        stream = await self._sdk.chat.completions.create(**kw)
        parts, finish, usage, aborted, checked = [], None, None, False, 0
        try:
            async for ch in stream:
                if getattr(ch, "usage", None):
                    usage = ch.usage
                if not ch.choices:
                    continue
                c0 = ch.choices[0]
                parts.append(c0.delta.content or "")
                if c0.finish_reason:
                    finish = c0.finish_reason
                total = sum(map(len, parts))
                if total - checked >= 40:
                    checked = total
                    if is_degenerate("".join(parts)):
                        aborted, finish = True, "repetition_abort"
                        break
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                res = close()
                if asyncio.iscoroutine(res):
                    await res
        text = "".join(parts)
        pt = getattr(usage, "prompt_tokens", None)
        ct = getattr(usage, "completion_tokens", None)
        if ct is None:  # no usage in a stream that was cut: rough estimate (chars/4)
            ct = max(1, len(text) // 4)
        return LLMResult(text=text, model=kwargs["model"], finish_reason=finish, prompt_tokens=pt, completion_tokens=ct,
                         latency_s=round(time.perf_counter() - t0, 3), attempts=attempt + 1, key=key, aborted=aborted)

    async def list_models(self) -> list[str]:
        """GET /models -> ids only (counts as one call against the budget)."""
        self.budget.reserve()
        t0 = time.perf_counter()
        page = await self._sdk.models.list()
        ids = sorted(m.id for m in page.data)
        self._log(event="models", status=200, latency_s=round(time.perf_counter() - t0, 3), n=len(ids))
        return ids

    def summary(self) -> str:
        s = self.stats
        return (f"requests={s.calls} cache_hits={s.cache_hits} attempts_sent={s.attempts} retries={s.retries} "
                f"429s={s.n429} 5xx={s.n5xx} coalesced={s.coalesced} conc={self.sem.limit}" + (f" | DRY-RUN would send {s.planned} calls" if self.dry_run else ""))

    async def aclose(self) -> None:
        if self._sdk is not None and hasattr(self._sdk, "close"):
            await self._sdk.close()


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m polyjudge.llm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("budget", help="show or reset the global call budget")
    b.add_argument("--reset", action="store_true")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--model", default=None, help="8B / 70B (default: the legacy shared budget)")
    a = ap.parse_args()
    cb = budget_for(a.model) if a.model else CallBudget()
    if a.cmd == "budget":
        if a.reset:
            if a.limit is None:
                ap.error("--reset needs --limit")
            cb.reset(a.limit)
        print(cb.state())


if __name__ == "__main__":
    main()
