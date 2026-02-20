"""Per-task state tracking and loop detection.

Tracks the last action signature and URL per task to detect when the agent
is stuck in a loop (same action + URL repeated 2+ times). Sends a
course-correction hint to the LLM when a loop is detected.
"""

from __future__ import annotations


# Module-level process-local state keyed by task_id.
_TASK_STATE: dict[str, dict] = {}


def get_action_signature(decision: dict) -> str:
    """Build a string signature from an LLM decision dict.

    Returns:
        ``"{action}:{candidate_id}"`` for click/type/select,
        ``"{action}:{url}"`` for navigate,
        ``"{action}"`` for scroll/done/wait.
    """
    action = decision.get("action", "unknown")
    if action in ("click", "type", "select"):
        cid = decision.get("candidate_id", "?")
        return f"{action}:{cid}"
    if action == "navigate":
        url = decision.get("url", "")
        return f"{action}:{url}"
    return action


def check_loop(task_id: str, url: str, action_sig: str) -> str | None:
    """Track action+URL for a task and detect repetition loops.

    When the same ``action_sig + url`` pair repeats (repeat count >= 2),
    returns a course-correction hint string. Otherwise returns ``None``.
    The repeat counter resets when the action_sig or url changes.
    """
    if task_id not in _TASK_STATE:
        _TASK_STATE[task_id] = {
            "last_sig": None,
            "last_url": None,
            "repeat_count": 0,
        }

    state = _TASK_STATE[task_id]
    combo = f"{action_sig}@{url}"
    last_combo = (
        f"{state['last_sig']}@{state['last_url']}"
        if state["last_sig"] is not None
        else None
    )

    if combo == last_combo:
        state["repeat_count"] += 1
    else:
        state["last_sig"] = action_sig
        state["last_url"] = url
        state["repeat_count"] = 1

    if state["repeat_count"] >= 2:
        return "You are repeating the same action. Try something different."
    return None


def clear_task_state(task_id: str) -> None:
    """Remove a task from the state dict (cleanup after done or error)."""
    _TASK_STATE.pop(task_id, None)
