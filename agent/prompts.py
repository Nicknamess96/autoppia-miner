"""System and user prompt construction for the web automation agent.

Builds structured prompts that present the task, page state, interactive
elements, and action history to the LLM for decision-making.
"""

from __future__ import annotations

from agent.classifier import TaskType


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a web automation agent. You receive a task, page state, and interactive elements. Choose ONE action and return JSON only.

Actions:
- {"action":"click","candidate_id":N}
- {"action":"type","candidate_id":N,"text":"..."}
- {"action":"select","candidate_id":N,"text":"..."}
- {"action":"navigate","url":"http://localhost/..."}
- {"action":"scroll_down"} or {"action":"scroll_up"}
- {"action":"done"}

Rules:
1. candidate_id must match an [N] from INTERACTIVE ELEMENTS.
2. For navigate, keep all query params (especially ?seed=).
3. Use exact credential values from the task.
4. Return valid JSON only. No markdown or commentary.
5. If a confirmation dialog appears, click Confirm/OK/Yes. If a success message appears, respond done.

Example:
Input: [0] button "Log In" (id=login-btn) [1] input[text] "Username" (name=user)
Task: Log in with username admin
Output: {"action":"type","candidate_id":1,"text":"admin"}
"""

# ---------------------------------------------------------------------------
# Task-type-specific STRATEGY hints
# ---------------------------------------------------------------------------

_TASK_HINTS: dict[str, str] = {
    TaskType.ADD_FILM.value: (
        "STRATEGY:\n"
        "Navigate to the add/create film page. "
        "Fill in all required fields from the task. "
        "Click submit/save."
    ),
    TaskType.EDIT_FILM.value: (
        "STRATEGY:\n"
        "Find the target film in the list, click to its detail page. "
        "Click edit/modify. Update the specified fields. "
        "Save changes."
    ),
    TaskType.DELETE_FILM.value: (
        "STRATEGY:\n"
        "Find the target film, click to its detail page. "
        "Click delete/remove. Confirm the dialog. "
        "Done."
    ),
    TaskType.EDIT_USER.value: (
        "STRATEGY:\n"
        "Log in with the given credentials. "
        "Navigate to profile/settings page. "
        "Update the specified fields. Save changes."
    ),
    TaskType.ADD_TO_WATCHLIST.value: (
        "STRATEGY:\n"
        "Find the target film by browsing or searching. "
        "Click to its detail page. "
        "Click the add-to-watchlist/wishlist button."
    ),
    TaskType.REMOVE_FROM_WATCHLIST.value: (
        "STRATEGY:\n"
        "Navigate to the watchlist/wishlist page. "
        "Find the target film. "
        "Click remove to remove it from the watchlist."
    ),
    TaskType.FILM_DETAIL.value: (
        "STRATEGY:\n"
        "Find the target film by browsing or searching. "
        "Click the film title or poster to reach the detail page. "
        "Done."
    ),
    TaskType.FILTER_FILM.value: (
        "STRATEGY:\n"
        "Find the filter/sort controls on the movie list page. "
        "Select the specified criteria. "
        "Apply the filter. Done."
    ),
}


# ---------------------------------------------------------------------------
# History formatting
# ---------------------------------------------------------------------------

def format_history_entry(
    step: int,
    action_type: str,
    element_text: str,
    result: str,
    url_changed: str | None = None,
) -> str:
    """Format a single history entry for the LLM prompt.

    Returns:
        ``"Step {step}: {action_type} on '{element_text}' -> {result}"``
        with optional ``" (now at {url_changed})"`` suffix.
    """
    line = f"Step {step}: {action_type} on '{element_text}' -> {result}"
    if url_changed:
        line += f" (now at {url_changed})"
    return line


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_system_prompt(task_type: TaskType = TaskType.UNKNOWN) -> str:
    """Return the system prompt, optionally appending a STRATEGY hint for *task_type*.

    If *task_type* has an entry in ``_TASK_HINTS``, the hint is appended after
    the base prompt.  Otherwise the bare ``SYSTEM_PROMPT`` is returned.
    """
    hint = _TASK_HINTS.get(task_type.value)
    if hint is not None:
        return SYSTEM_PROMPT + "\n\n" + hint
    return SYSTEM_PROMPT


def build_user_prompt(
    *,
    task_prompt: str,
    page_ir: str,
    history_lines: list[str],
    steps_remaining: int,
    loop_hint: str | None = None,
) -> str:
    """Build the user message for the LLM.

    Includes task description, page IR (URL, title, page structure,
    interactive elements), action history, steps remaining with urgency,
    and optional loop-detection warning.
    """
    history_text = "\n".join(history_lines) if history_lines else "No actions yet"

    parts = [
        f"TASK: {task_prompt}",
        "",
        page_ir,
        "",
        "HISTORY:",
        history_text,
        "",
    ]

    # Steps remaining with urgency at 3 or fewer
    steps_line = f"STEPS REMAINING: {steps_remaining}"
    if steps_remaining <= 3:
        steps_line += " -- Take the most direct action to complete the task."
    parts.append(steps_line)

    if loop_hint:
        parts.append("")
        parts.append(f"WARNING: {loop_hint}")

    parts.append("")
    parts.append("Choose your next action. Return JSON only.")

    return "\n".join(parts)
