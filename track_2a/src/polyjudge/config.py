"""LLM configuration from track_2a/.env (resolved relative to this package, never the working directory).

The .env file is read as utf-8-sig (a BOM is stripped) and overrides variables already present in the process
environment. Secret values are never printed or logged: `LLMConfig` masks the key, shows only scheme+host of the URL.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import dotenv_values, load_dotenv

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"  # track_2a/.env
REQUIRED = ("LLM_NAME", "LLM_BASE_URL", "LLM_API_KEY")


@dataclass(frozen=True)
class LLMConfig:
    name: str
    base_url: str
    api_key: str

    def __repr__(self) -> str:  # keep secrets out of logs and tracebacks
        u = urlparse(self.base_url)
        return f"LLMConfig(name={self.name!r}, endpoint='{u.scheme}://{u.netloc}', api_key='***')"

    __str__ = __repr__


def load_env(path: Path = ENV_PATH) -> bool:
    """Load `path` into os.environ (override=True, BOM-safe). Returns True if the file existed."""
    if not path.is_file():
        return False
    load_dotenv(dotenv_path=path, override=True, encoding="utf-8-sig")
    return True


def env_status(path: Path = ENV_PATH) -> dict:
    """Diagnostic that is safe to print: file facts and key NAMES only, never values."""
    info = {"path": str(path), "exists": path.is_file()}
    if info["exists"]:
        raw = path.read_bytes()
        info.update(bytes=len(raw), utf8_bom=raw.startswith(b"\xef\xbb\xbf"), crlf=raw.count(b"\r\n"),
                    keys_parsed={k: bool(v) for k, v in dotenv_values(path, encoding="utf-8-sig").items()})
    return info


def load_llm_config(base_url: str | None = None, name: str | None = None) -> LLMConfig:
    """Config from .env; `base_url` / `name` (e.g. from a CLI flag) override the file without editing it."""
    found = load_env()
    values = {k: (os.environ.get(k) or "").strip() for k in REQUIRED}
    if base_url:
        values["LLM_BASE_URL"] = base_url.strip()
    if name:
        values["LLM_NAME"] = name.strip()
    missing = [k for k, v in values.items() if not v]
    if missing:
        where = f"{ENV_PATH} ({'found' if found else 'NOT found'})"
        raise RuntimeError(f"Missing/empty: {', '.join(missing)}. Looked in {where} and the process environment.")
    return LLMConfig(values["LLM_NAME"], values["LLM_BASE_URL"], values["LLM_API_KEY"])
