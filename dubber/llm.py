"""LLM access behind one interface. Only text is ever sent, never audio or video.

OpenAICompatible talks to any /v1 endpoint (FreeLLMAPI locally, Groq/Google on the demo Space).
Every response is cached on disk by request hash; 429/5xx back off 1,2,4,8,16 s, then the next
model in the fallback chain is tried. The API key is read from config and never logged.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from . import config
from .schema import sha1

log = logging.getLogger("dubber.llm")
_THINK = re.compile(r"<(think|thinking)>.*?(</\1>|$)\s*", re.S)  # an unclosed tag means the answer was cut off
STATS = {"calls": 0}  # network calls made in this process, for report.json


class LLMError(RuntimeError):
    def __init__(self, msg: str, retryable: bool = False, status: int = 0):
        super().__init__(msg)
        self.retryable = retryable
        self.status = status


def parse_json(text: str):
    """Tolerant JSON extraction: strips fences/thinking, takes the outermost {...} or [...]."""
    t = _THINK.sub("", text).strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        t = t.rsplit("```", 1)[0]
    candidates = []
    for o, c in (("{", "}"), ("[", "]")):
        i, j = t.find(o), t.rfind(c)
        if i != -1 and j > i:
            candidates.append((i, t[i:j + 1]))
    for _, chunk in sorted(candidates):
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            continue
    raise LLMError(f"no JSON object in response: {t[:200]!r}")


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], model: str, json_mode: bool = False,
             temperature: float = config.LLM_TEMPERATURE, max_tokens: Optional[int] = None,
             fallback: list[str] = ()) -> str:
        """Return the assistant message text. Raises LLMError when every model in the chain fails."""

    def chat_json(self, messages: list[dict], model: str, **kw):
        return parse_json(self.chat(messages, model, json_mode=True, **kw))


class OpenAICompatible(LLMProvider):
    def __init__(self, base_url: str = config.LLM_BASE_URL, key: str = config.LLM_KEY,
                 cache_dir: Path = config.CACHE_DIR / "llm", backoff: list[float] = config.LLM_BACKOFF_S,
                 timeout: float = config.LLM_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self._key = key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.backoff = list(backoff)
        self.timeout = timeout
        self.calls = 0  # network calls actually made (cache misses)

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._key:
            h["Authorization"] = f"Bearer {self._key}"
        return h

    def models(self) -> list[str]:
        req = urllib.request.Request(self.base_url + "/models", headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return [m["id"] for m in json.load(r)["data"]]

    def _post(self, model: str, messages: list[dict], json_mode: bool, temperature: float,
              max_tokens: Optional[int]) -> tuple[str, str]:
        body = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens:
            body["max_tokens"] = max_tokens
        # ponytail: no response_format. Several free routes reject or ignore it (NVIDIA answers 502
        # "format_ignored"); the prompts demand JSON and parse_json is tolerant, which is enough.
        req = urllib.request.Request(self.base_url + "/chat/completions",
                                     data=json.dumps(body).encode("utf-8"), headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.load(r)
                via = r.headers.get("X-Routed-Via", "")
        except urllib.error.HTTPError as e:
            detail = e.read(300).decode("utf-8", "replace")
            # 413 from FreeLLMAPI means "no route with enough token quota left right now", i.e. a rate limit
            raise LLMError(f"{model}: HTTP {e.code} {detail}", retryable=e.code in (429, 413) or e.code >= 500,
                           status=e.code)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LLMError(f"{model}: {type(e).__name__}: {e}", retryable=True)
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"{model}: malformed response {str(data)[:200]}", retryable=True)
        return _THINK.sub("", content).strip(), via

    def chat(self, messages, model, json_mode=False, temperature=config.LLM_TEMPERATURE,
             max_tokens=None, fallback=()):
        key = sha1(messages, model, json_mode, temperature, max_tokens)
        cache = self.cache_dir / f"{key}.json"
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))["content"]
        last: Optional[LLMError] = None
        chain = [model, *[m for m in fallback if m != model]]
        dead: set[str] = set()  # models that answered with a non-retryable error (400, 401 ...)
        # Backoff is per round over the whole chain: a rate-limited route costs one request, not 31 s,
        # and the chain is retried 1,2,4,8,16 s later while quotas reset.
        for attempt, delay in enumerate([0, *self.backoff]):
            if len(dead) == len(chain):
                break
            if delay:
                time.sleep(delay)
            for m in chain:
                if m in dead:
                    continue
                try:
                    self.calls += 1
                    STATS["calls"] += 1
                    content, via = self._post(m, messages, json_mode, temperature, max_tokens)
                    if not content:
                        raise LLMError(f"{m}: empty content", retryable=True)
                    if json_mode:
                        try:
                            parse_json(content)  # never cache an answer the caller cannot use
                        except LLMError as e:
                            raise LLMError(f"{m}: {e}", retryable=True)
                    cache.write_text(json.dumps({"content": content, "model": m, "via": via}, ensure_ascii=False),
                                     encoding="utf-8")
                    if m != model:
                        log.warning("fell back to %s (%s)", m, via)
                    return content
                except LLMError as e:
                    last = e
                    if not e.retryable:
                        dead.add(m)
                    log.info("llm round %d on %s failed: %s", attempt + 1, m, str(e)[:160])
        raise LLMError(f"all models failed; last error: {last}")


def default_provider() -> OpenAICompatible:
    return OpenAICompatible()


if __name__ == "__main__":  # self-check of the parser only (no network)
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('<think>hmm</think>Sure: [{"id": 0, "translation": "x"}] done') == [{"id": 0, "translation": "x"}]
    assert parse_json('text {"a": {"b": [1,2]}} tail') == {"a": {"b": [1, 2]}}
    print("llm parser ok")
