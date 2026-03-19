"""Task results cache for replaying known-good action sequences.

Loads task_results.json at import time and indexes successful results by
taskId.  When the same taskId appears in a new /act request, the cached
action sequence is replayed one action per step_index call, bypassing
the LLM entirely.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("agent")

# Raw action dicts keyed by taskId.
_CACHE: dict[str, list[dict[str, Any]]] = {}


def _load_cache() -> None:
    """Load task_results.json and populate the module-level cache."""
    cache_path = Path(__file__).resolve().parent.parent / "task_results.json"
    if not cache_path.exists():
        logger.info("task_results.json not found, cache disabled")
        return

    try:
        data = json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("failed to load task_results.json: %s", exc)
        return

    loaded = 0
    for entry in data:
        task = entry.get("task", {})
        task_id = task.get("taskId")
        status = entry.get("status")
        response = entry.get("response")

        if not task_id or status != "success" or response is None:
            continue

        actions = response.get("actions", [])
        _CACHE[task_id] = actions
        loaded += 1

    logger.info("task results cache loaded: %d tasks", loaded)


def lookup(task_id: str) -> list[dict[str, Any]] | None:
    """Return cached action list for *task_id*, or None on cache miss."""
    return _CACHE.get(task_id)


# Load on import so the cache is ready before the first request.
_load_cache()
