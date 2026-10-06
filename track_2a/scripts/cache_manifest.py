#!/usr/bin/env python3
"""Record which replay-cache keys each method reads on the test inputs (replay mode, no network, no API).

Run inside the image (or any tree laid out like /app):
    python scripts/cache_manifest.py [--cache /app/cache] [--out manifest.json]
Output: {method: sorted list of cache keys}. Used once to split data/replay_cache/ (M1) from data/replay_cache/extras/
(M0, M2, M3); also a regression check: every method must replay with zero cache misses.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from polyjudge import predict as P
from polyjudge.inputs import load_inputs
from polyjudge.llm import CACHE_DIR, LLMClient


async def main_async(cache: Path, methods):
    inputs = load_inputs("test")
    seen: dict = {}
    orig = LLMClient._cache_get

    def spy(self, key):
        seen.setdefault(self._method, set()).add(key)
        return orig(self, key)

    LLMClient._cache_get = spy
    out = {}
    for m in methods:
        P_cl = P.make_clients("replay", cache_dir=cache) if m != "B0" else P.Clients()
        for c in (P_cl.c8, P_cl.c70):
            if c is not None:
                c._method = m
        seen[m] = set()
        await P.predict(m, inputs, "replay" if m != "B0" else "offline", clients=P_cl)
        await P_cl.aclose()
        out[m] = sorted(seen[m])
        print(m, len(out[m]), "keys", file=sys.stderr)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    ap.add_argument("--out", type=Path, default=Path("manifest.json"))
    ap.add_argument("--methods", default="B0,M0,M1,M2,M3")
    a = ap.parse_args()
    res = asyncio.run(main_async(a.cache, a.methods.split(",")))
    a.out.write_text(json.dumps(res), encoding="utf-8")
