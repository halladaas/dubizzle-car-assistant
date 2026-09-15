"""Thin, traced wrapper around LiteLLM. Every call (data-prep, filter
extraction, intent classification, response generation, eval judging) goes
through here so it's logged to SQLite for the traceability story in the
README, and so the model can be swapped via env vars alone.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import litellm
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
TRACE_DB_PATH = DATA_DIR / "traces.db"

# Small, cheap model for structured/classification tasks (filter extraction,
# intent classification, data-prep enrichment). Large model reserved for the
# final conversational response. Both are Gemini free-tier via Google AI
# Studio; override with env vars if rate-limited or newer tiers ship.
#
# Both default to the same flash-lite tier: as of this build, Gemini 3.x's
# non-lite "flash" models are thinking models that spend tens-to-hundreds of
# reasoning tokens on every call by default (confirmed empirically -- a
# 1-word reply burned 107 reasoning tokens), which directly fights the
# latency/cost goals for a task this size. Flash-lite answers immediately
# with no reasoning overhead. Bump GEMINI_LARGE_MODEL to a thinking-tier
# model if richer conversational responses are worth the added latency.
SMALL_MODEL = os.getenv("GEMINI_SMALL_MODEL", "gemini/gemini-3.5-flash-lite")
LARGE_MODEL = os.getenv("GEMINI_LARGE_MODEL", "gemini/gemini-3.5-flash-lite")


def _init_trace_db() -> None:
    conn = sqlite3.connect(TRACE_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            component TEXT NOT NULL,
            model TEXT NOT NULL,
            latency_ms REAL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            request_json TEXT,
            response_text TEXT,
            error TEXT
        )
        """
    )
    conn.commit()
    conn.close()


_init_trace_db()


def _log_trace(
    component: str,
    model: str,
    latency_ms: float,
    usage: dict | None,
    request: dict,
    response_text: str | None,
    error: str | None,
) -> None:
    conn = sqlite3.connect(TRACE_DB_PATH)
    conn.execute(
        """INSERT INTO traces
           (ts, component, model, latency_ms, prompt_tokens, completion_tokens,
            request_json, response_text, error)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            component,
            model,
            round(latency_ms, 1),
            (usage or {}).get("prompt_tokens"),
            (usage or {}).get("completion_tokens"),
            json.dumps(request)[:4000],
            (response_text or "")[:4000],
            error,
        ),
    )
    conn.commit()
    conn.close()


def _parse_retry_delay(message: str) -> float | None:
    match = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', message)
    return float(match.group(1)) + 1 if match else None


def _is_rate_limit_error(exc: Exception) -> bool:
    return "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc)


def with_rate_limit_retry(fn, *args, retries: int = 5, **kwargs):
    """The Gemini free tier has tight per-minute request caps (as low as 15
    RPM for chat completion on some models), easily hit by a single
    multi-turn conversation (intent + filter extraction + response
    generation per turn) or a batch job. Retry with the API's suggested
    delay rather than surface a 429 to the user/caller."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - litellm maps 429s inconsistently across code paths
            if not _is_rate_limit_error(exc) or attempt == retries - 1:
                raise
            last_error = exc
            wait = _parse_retry_delay(str(exc)) or (10 * (attempt + 1))
            time.sleep(wait)
    raise RuntimeError(f"Rate limited after {retries} retries") from last_error


def call_llm(
    messages: list[dict],
    component: str,
    model: str | None = None,
    tools: list[dict] | None = None,
    tool_choice: Any = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
):
    """Call the LLM and log the call. Raises on failure (caller decides
    fallback behavior — we never swallow errors silently here)."""
    model = model or LARGE_MODEL
    start = time.perf_counter()
    response = None
    error = None
    try:
        response = with_rate_limit_retry(
            litellm.completion,
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response
    except Exception as exc:  # noqa: BLE001 - we log and re-raise
        error = str(exc)
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        usage = None
        content_preview = None
        if response is not None:
            usage_obj = getattr(response, "usage", None)
            if usage_obj is not None:
                usage = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                    "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                }
            try:
                msg = response.choices[0].message
                content_preview = msg.content or json.dumps(
                    [tc.function.arguments for tc in (msg.tool_calls or [])]
                )
            except Exception:  # noqa: BLE001
                content_preview = None
        _log_trace(
            component=component,
            model=model,
            latency_ms=latency_ms,
            usage=usage,
            request={"messages": messages, "has_tools": bool(tools)},
            response_text=content_preview,
            error=error,
        )


def call_with_tool(
    messages: list[dict],
    tool_schema: dict,
    component: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict | None:
    """Force a single function call; return parsed JSON args or None if the
    model didn't comply (caller should have a non-LLM fallback)."""
    try:
        response = call_llm(
            messages=messages,
            component=component,
            model=model or SMALL_MODEL,
            tools=[tool_schema],
            tool_choice={
                "type": "function",
                "function": {"name": tool_schema["function"]["name"]},
            },
            temperature=temperature,
        )
    except Exception:  # noqa: BLE001
        return None

    try:
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
    except Exception:  # noqa: BLE001
        return None
