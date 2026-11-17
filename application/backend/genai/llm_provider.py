"""
LLM client for OpenAI and Google Gemini over plain REST (httpx).

Provider selection (Req 14.6, 15.5):
    OPENAI_API_KEY set           -> "openai"  (also when GEMINI_API_KEY is set)
    only GEMINI_API_KEY set      -> "gemini"
    neither                      -> "none"
A key that is empty or only whitespace counts as not set. The environment is
read on every call, so keys loaded after import are picked up.

Models: $OPENAI_MODEL (default gpt-4o) and $GEMINI_MODEL (comma-separated list, default
"gemini-3.5-flash,gemini-3.1-flash-lite"). When a Gemini model is retired (404), overloaded
(503) or rate-limited (429) the next model in the list is tried automatically.

Security rules:
    - The Gemini key is sent in the `x-goog-api-key` header, the OpenAI key in
      `Authorization: Bearer`. No key is ever placed in a URL.
    - Failures are logged by exception type name only, never by message, so no
      key, response body or request detail reaches a log line.
    - Raised `LLMError` / `LLMStreamInterrupted` messages are fixed texts.

Tests can route all traffic through `httpx.MockTransport` by setting the
module attribute `_transport` (a MockTransport with a sync handler works for
both `complete_text` and `stream_text`).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from typing import AsyncIterator, Literal
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_GEMINI_MODELS = ("gemini-3.5-flash", "gemini-3.1-flash-lite")
_GEMINI_RETRY_STATUS = {404, 429, 500, 503}
GEMINI_MIN_OUTPUT_TOKENS = 4096  # Gemini 3.x counts "thinking" tokens in the output budget
DEFAULT_GEMINI_THINKING = "low"  # keeps thinking short so the visible answer is never cut off
STREAM_TEMPERATURE = 0.3
INTERRUPTED_MESSAGE = "The response was interrupted. Please try again."

# Optional transport override (httpx.MockTransport in tests). None = real network.
_transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None

# complete_text runs the blocking request in a worker so the caller's total
# wait is bounded by `timeout` (Req 13.15), even if a slow server trickles bytes.
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm-complete")


class LLMError(Exception):
    """Any failure of a non-streaming call, or of a stream before its first token."""


class LLMStreamInterrupted(Exception):
    """A stream failed after at least one token was delivered."""


# ── configuration ─────────────────────────────────────────────────────────────

def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def configured_provider() -> Literal["openai", "gemini", "none"]:
    if _env("OPENAI_API_KEY"):
        return "openai"
    if _env("GEMINI_API_KEY"):
        return "gemini"
    return "none"


def _openai_model() -> str:
    return _env("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL


def _gemini_models() -> list[str]:
    raw = _env("GEMINI_MODEL")
    models = [m.strip() for m in raw.split(",") if m.strip()] if raw else list(DEFAULT_GEMINI_MODELS)
    return [m[len("models/"):] if m.startswith("models/") else m for m in models]


def _gemini_model() -> str:
    return _gemini_models()[0]


def _gemini_url(method: str, model: str | None = None) -> str:
    # The model name is path-quoted; the API key is never part of the URL.
    url = f"{GEMINI_BASE_URL}/{quote(model or _gemini_model(), safe='-._')}:{method}"
    if method == "streamGenerateContent":
        url += "?alt=sse"
    return url


def _client_kwargs(timeout: float) -> dict:
    kwargs: dict = {"timeout": httpx.Timeout(timeout)}
    if _transport is not None:
        kwargs["transport"] = _transport
    return kwargs


def _log_failure(provider: str, what: str, exc: BaseException) -> None:
    logger.warning("LLM %s via %s failed: %s", what, provider, type(exc).__name__)


# ── request builders ──────────────────────────────────────────────────────────

def _openai_headers() -> dict:
    return {"Authorization": f"Bearer {_env('OPENAI_API_KEY')}", "Content-Type": "application/json"}


def _gemini_headers() -> dict:
    return {"x-goog-api-key": _env("GEMINI_API_KEY"), "Content-Type": "application/json"}


def _split_messages(system: str, messages: list[dict]) -> tuple[str, list[dict]]:
    """Fold any `system` entries into the system text and keep user/assistant turns."""
    system_parts = [system] if system else []
    turns: list[dict] = []
    for msg in messages or []:
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        turns.append({"role": "assistant" if role == "assistant" else "user", "content": content})
    return "\n\n".join(system_parts), turns


def _openai_body(system: str, turns: list[dict], *, max_tokens: int, temperature: float,
                 stream: bool, json_mode: bool = False) -> dict:
    msgs = ([{"role": "system", "content": system}] if system else []) + turns
    body: dict = {
        "model": _openai_model(),
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


def _gemini_body(system: str, turns: list[dict], *, max_tokens: int, temperature: float,
                 json_mode: bool = False) -> dict:
    gen_config: dict = {"maxOutputTokens": max(max_tokens, GEMINI_MIN_OUTPUT_TOKENS), "temperature": temperature}
    if json_mode:
        gen_config["responseMimeType"] = "application/json"
    body: dict = {
        "contents": [
            {"role": "model" if t["role"] == "assistant" else "user", "parts": [{"text": t["content"]}]}
            for t in turns
        ],
        "generationConfig": gen_config,
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    return body


def _for_model(body: dict, model: str) -> dict:
    """Add a bounded thinking level for Gemini 3.x models (older models reject the field)."""
    if not model.startswith("gemini-3"):
        return body
    level = _env("GEMINI_THINKING_LEVEL") or DEFAULT_GEMINI_THINKING
    out = dict(body)
    out["generationConfig"] = {**body.get("generationConfig", {}), "thinkingConfig": {"thinkingLevel": level}}
    return out


def _gemini_text(payload: dict) -> str:
    if isinstance(payload.get("error"), dict):
        raise LLMError("Gemini returned an error")
    parts: list[str] = []
    for cand in payload.get("candidates") or []:
        for part in (cand.get("content") or {}).get("parts") or []:
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
        break  # first candidate only
    return "".join(parts)


def _openai_message_text(payload: dict) -> str:
    if isinstance(payload.get("error"), dict):
        raise LLMError("OpenAI returned an error")
    choices = payload.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content")
    return content if isinstance(content, str) else ""


def _openai_delta_text(payload: dict) -> str:
    if isinstance(payload.get("error"), dict):
        raise LLMError("OpenAI returned an error")
    choices = payload.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("delta") or {}).get("content")
    return content if isinstance(content, str) else ""


# ── non-streaming completion ──────────────────────────────────────────────────

def _complete_blocking(provider: str, system: str, user: str, *, timeout: float,
                       max_tokens: int, temperature: float, json_mode: bool) -> str:
    if json_mode and "json" not in f"{system}\n{user}".lower():
        # OpenAI JSON mode rejects prompts that never mention JSON.
        user = f"{user}\n\nRespond with a single JSON object."
    turns = [{"role": "user", "content": user}]
    with httpx.Client(**_client_kwargs(timeout)) as client:
        if provider == "openai":
            resp = client.post(
                OPENAI_URL,
                headers=_openai_headers(),
                json=_openai_body(system, turns, max_tokens=max_tokens, temperature=temperature,
                                  stream=False, json_mode=json_mode),
            )
            resp.raise_for_status()
            return _openai_message_text(resp.json())
        body = _gemini_body(system, turns, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode)
        models = _gemini_models()
        for i, model in enumerate(models):
            resp = client.post(_gemini_url("generateContent", model), headers=_gemini_headers(), json=_for_model(body, model))
            if resp.status_code in _GEMINI_RETRY_STATUS and i < len(models) - 1:
                logger.warning("Gemini model %s returned HTTP %s; trying %s", model, resp.status_code, models[i + 1])
                continue
            resp.raise_for_status()
            return _gemini_text(resp.json())
        return ""


def complete_text(system: str, user: str, *, timeout: float = 20.0, max_tokens: int = 1200,
                  temperature: float = 0.3, json_mode: bool = False) -> tuple[str, str]:
    """Return `(text, provider)`. Raises `LLMError` on any failure, empty output,
    no configured provider, or when the call takes longer than `timeout` seconds."""
    provider = configured_provider()
    if provider == "none":
        raise LLMError("No LLM provider is configured")
    future = _EXECUTOR.submit(
        _complete_blocking, provider, system, user, timeout=timeout,
        max_tokens=max_tokens, temperature=temperature, json_mode=json_mode,
    )
    try:
        text = future.result(timeout=timeout)
    except _FutureTimeout as exc:
        future.cancel()  # the worker stops on its own httpx timeout
        _log_failure(provider, "completion", exc)
        raise LLMError("The LLM request timed out") from None
    except Exception as exc:
        _log_failure(provider, "completion", exc)
        raise LLMError("The LLM request failed") from None
    if not text.strip():
        logger.warning("LLM completion via %s failed: EmptyResponse", provider)
        raise LLMError("The LLM returned an empty response")
    return text, provider


# ── streaming ─────────────────────────────────────────────────────────────────

async def _sse_payloads(resp: httpx.Response) -> AsyncIterator[dict]:
    async for line in resp.aiter_lines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if not data:
            continue
        if data == "[DONE]":
            return
        yield json.loads(data)


async def _openai_stream(system: str, turns: list[dict], *, max_tokens: int,
                         timeout: float) -> AsyncIterator[str]:
    body = _openai_body(system, turns, max_tokens=max_tokens, temperature=STREAM_TEMPERATURE, stream=True)
    async with httpx.AsyncClient(**_client_kwargs(timeout)) as client:
        async with client.stream("POST", OPENAI_URL, headers=_openai_headers(), json=body) as resp:
            resp.raise_for_status()
            async for payload in _sse_payloads(resp):
                text = _openai_delta_text(payload)
                if text:
                    yield text


async def _gemini_stream(system: str, turns: list[dict], *, max_tokens: int,
                         timeout: float) -> AsyncIterator[str]:
    body = _gemini_body(system, turns, max_tokens=max_tokens, temperature=STREAM_TEMPERATURE)
    models = _gemini_models()
    async with httpx.AsyncClient(**_client_kwargs(timeout)) as client:
        for i, model in enumerate(models):
            async with client.stream("POST", _gemini_url("streamGenerateContent", model),
                                     headers=_gemini_headers(), json=_for_model(body, model)) as resp:
                if resp.status_code in _GEMINI_RETRY_STATUS and i < len(models) - 1:
                    logger.warning("Gemini model %s returned HTTP %s; trying %s", model, resp.status_code, models[i + 1])
                    continue  # nothing has been yielded yet, so switching models is safe
                resp.raise_for_status()
                async for payload in _sse_payloads(resp):
                    text = _gemini_text(payload)
                    if text:
                        yield text
                return


async def stream_text(system: str, messages: list[dict], *, first_token_timeout: float = 20.0,
                      gap_timeout: float = 20.0, max_tokens: int = 600) -> AsyncIterator[str]:
    """Yield text chunks from the configured provider.

    Each step of the provider stream is bounded with `asyncio.wait_for`
    (`first_token_timeout` until the first token, `gap_timeout` after it).
    Any failure or timeout before the first token raises `LLMError`; after it,
    `LLMStreamInterrupted(INTERRUPTED_MESSAGE)`. A stream that ends without any
    text also raises `LLMError`, so callers can fall back.
    """
    provider = configured_provider()
    if provider == "none":
        raise LLMError("No LLM provider is configured")

    system_text, turns = _split_messages(system, messages)
    http_timeout = max(first_token_timeout, gap_timeout) + 5.0  # wait_for is the real bound
    source = (_openai_stream if provider == "openai" else _gemini_stream)(
        system_text, turns, max_tokens=max_tokens, timeout=http_timeout
    )

    got_token = False
    try:
        while True:
            step_timeout = gap_timeout if got_token else first_token_timeout
            try:
                chunk = await asyncio.wait_for(source.__anext__(), timeout=step_timeout)
            except StopAsyncIteration:
                break
            except Exception as exc:  # includes TimeoutError, httpx errors, bad JSON
                _log_failure(provider, "stream", exc)
                if got_token:
                    raise LLMStreamInterrupted(INTERRUPTED_MESSAGE) from None
                raise LLMError("The LLM stream failed before the first token") from None
            got_token = True
            yield chunk
    finally:
        await source.aclose()

    if not got_token:
        logger.warning("LLM stream via %s failed: EmptyResponse", provider)
        raise LLMError("The LLM returned an empty response")
