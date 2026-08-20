"""Tracing hooks for multi-agent execution.

Supports:
1. LangSmith tracing (automatically configured via environment variables)
2. In-memory and structured JSON span tracking
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

_TRACES_DIR = Path("reports") / "traces"


def init_tracing() -> None:
    """Configure LangSmith environment variables if credentials are present."""
    from multi_agent_research_lab.core.config import get_settings

    settings = get_settings()
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        logger.info("LangSmith tracing enabled for project: %s", settings.langsmith_project)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager for tracing an execution span.

    Records start, duration, metadata attributes, and logs the span.
    """
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": None,
        "error": None,
    }
    try:
        yield span
    except Exception as exc:
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = round(perf_counter() - started, 4)
        logger.debug("TraceSpan [%s] finished in %.4fs (attrs: %s)", name, span["duration_seconds"], span["attributes"])


def save_trace_log(run_name: str, trace_events: list[dict[str, Any]]) -> Path:
    """Save execution trace events to a JSON file in reports/traces/."""
    _TRACES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_name)
    filepath = _TRACES_DIR / f"trace_{safe_name}_{timestamp}.json"

    with filepath.open("w", encoding="utf-8") as f:
        json.dump({"run_name": run_name, "events": trace_events}, f, indent=2, ensure_ascii=False)

    logger.info("Saved trace log to: %s", filepath)
    return filepath

