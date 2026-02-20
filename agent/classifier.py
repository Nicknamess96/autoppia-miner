"""Task-type classification and hard-coded login action sequence.

Classifies task prompts by type (login, registration, navigation, etc.)
using regex patterns, detects login form fields from page candidates,
and generates deterministic login actions to eliminate LLM calls for
predictable login tasks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parsing.candidates import Candidate


class TaskType(str, Enum):
    """Known task categories derived from prompt text."""

    LOGIN = "login"
    REGISTRATION = "registration"
    NAVIGATION = "navigation"
    CRUD = "crud"
    FORM = "form"
    UNKNOWN = "unknown"


# Patterns ordered by specificity -- LOGIN first (most valuable to optimize).
_TASK_PATTERNS: list[tuple[re.Pattern[str], TaskType]] = [
    (
        re.compile(
            r"\blog\s*in\b|\bsign\s*in\b|\blogin\b|\bauthenticat",
            re.IGNORECASE,
        ),
        TaskType.LOGIN,
    ),
    (
        re.compile(
            r"\bregist(er|ration)\b|\bsign\s*up\b|\bcreate\s+account\b",
            re.IGNORECASE,
        ),
        TaskType.REGISTRATION,
    ),
    (
        re.compile(
            r"\bnavig(ate|ation)\b|\bgo\s+to\b|\bvisit\b|\bbrowse\s+to\b",
            re.IGNORECASE,
        ),
        TaskType.NAVIGATION,
    ),
    (
        re.compile(
            r"\b(create|delete|remove|edit|update|add)\b.*"
            r"\b(item|record|entry|product|movie|book|booking|reservation)\b",
            re.IGNORECASE,
        ),
        TaskType.CRUD,
    ),
    (
        re.compile(
            r"\bfill\b.*\bform\b|\bsubmit\b.*\bform\b|\bcomplete\b.*\bform\b",
            re.IGNORECASE,
        ),
        TaskType.FORM,
    ),
]


def classify_task(prompt: str) -> TaskType:
    """Classify a task prompt into a TaskType using keyword patterns.

    Scans the prompt against ``_TASK_PATTERNS`` in order and returns
    the first match.  Returns ``TaskType.UNKNOWN`` if no pattern matches.
    """
    for pattern, task_type in _TASK_PATTERNS:
        if pattern.search(prompt):
            return task_type
    return TaskType.UNKNOWN


@dataclass
class LoginFields:
    """Identified login form fields with their candidate IDs."""

    username_id: int
    password_id: int
    submit_id: int


def detect_login_fields(candidates: list[Candidate]) -> LoginFields | None:
    """Detect username, password, and submit fields in candidates.

    Returns ``LoginFields`` if all three are found, ``None`` otherwise.

    Detection rules:
    - **Username:** ``<input>`` with ``input_type`` in (text, email, "") AND
      label containing 'user', 'email', or 'login' (case-insensitive).
    - **Password:** ``<input>`` with ``input_type == "password"``.
    - **Submit:** ``<button>`` or ``<input type="submit">`` with label
      containing 'log in', 'sign in', 'login', 'submit', or 'enter'.
      Fallback: button in the same ``parent_form`` as the password field.
    """
    username_id: int | None = None
    password_id: int | None = None
    submit_id: int | None = None
    password_form: str | None = None

    for c in candidates:
        # Password field (most distinctive -- check first)
        if c.input_type == "password" and password_id is None:
            password_id = c.id
            password_form = c.parent_form
            continue

        # Username field
        if c.tag == "input" and c.input_type in ("text", "email", ""):
            label_lower = (c.label or "").lower()
            if any(kw in label_lower for kw in ("user", "email", "login")):
                if username_id is None:
                    username_id = c.id
                    continue

    # Submit button: prefer button with matching label, fallback to same form
    for c in candidates:
        if c.tag == "button" or (c.tag == "input" and c.input_type == "submit"):
            label_lower = (c.label or c.text or "").lower()
            if any(
                kw in label_lower
                for kw in ("log in", "sign in", "login", "submit", "enter")
            ):
                submit_id = c.id
                break
            # Fallback: button in same form as password field
            if (
                password_form
                and c.parent_form == password_form
                and submit_id is None
            ):
                submit_id = c.id

    if (
        username_id is not None
        and password_id is not None
        and submit_id is not None
    ):
        return LoginFields(
            username_id=username_id,
            password_id=password_id,
            submit_id=submit_id,
        )
    return None


def get_login_action(step_index: int, fields: LoginFields) -> dict | None:
    """Return the hard-coded action dict for a login sequence step.

    Steps:
        0: Type ``<username>`` into the username field.
        1: Type ``<password>`` into the password field.
        2: Click the submit button.

    Returns ``None`` when ``step_index >= 3`` (sequence complete; caller
    should fall through to the LLM for post-login evaluation).
    """
    if step_index == 0:
        return {
            "action": "type",
            "candidate_id": fields.username_id,
            "text": "<username>",
        }
    if step_index == 1:
        return {
            "action": "type",
            "candidate_id": fields.password_id,
            "text": "<password>",
        }
    if step_index == 2:
        return {
            "action": "click",
            "candidate_id": fields.submit_id,
        }
    return None
