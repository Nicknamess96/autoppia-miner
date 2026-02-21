"""Task-type classification and hard-coded action sequences.

Classifies task prompts by type (login, logout, registration, contact, etc.)
using regex patterns, detects form fields from page candidates, and generates
deterministic action sequences to eliminate LLM calls for predictable tasks.
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

    LOGOUT = "logout"
    LOGIN = "login"
    REGISTRATION = "registration"
    CONTACT = "contact"
    NAVIGATION = "navigation"
    CRUD = "crud"
    FORM = "form"
    UNKNOWN = "unknown"


# Patterns ordered by specificity -- LOGOUT before LOGIN to avoid false matches.
_TASK_PATTERNS: list[tuple[re.Pattern[str], TaskType]] = [
    (
        re.compile(
            r"\blog\s*out\b|\blogout\b|\bsign\s*out\b",
            re.IGNORECASE,
        ),
        TaskType.LOGOUT,
    ),
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
    # Compound CONTACT: navigation verbs + "contact" (must precede NAVIGATION)
    (
        re.compile(
            r"\bcontact\b.*\b(form|fill|submit|message|send)\b"
            r"|\b(go\s+to|visit|navigat(e|ion)|browse\s+to)\b.*\bcontact\b",
            re.IGNORECASE,
        ),
        TaskType.CONTACT,
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
    (
        re.compile(
            r"\bcontact\b",
            re.IGNORECASE,
        ),
        TaskType.CONTACT,
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


# =========================================================================
# Logout shortcut
# =========================================================================


@dataclass
class LogoutTarget:
    """Identified logout button/link candidate ID."""

    button_id: int


def detect_logout_target(candidates: list[Candidate]) -> LogoutTarget | None:
    """Find a logout/sign-out button or link in candidates.

    Returns ``LogoutTarget`` if found, ``None`` otherwise.
    """
    for c in candidates:
        if c.tag not in ("button", "a", "input"):
            continue
        label_lower = (c.label or c.text or "").lower()
        if any(kw in label_lower for kw in ("logout", "log out", "sign out")):
            return LogoutTarget(button_id=c.id)
    return None


def get_logout_action(step_index: int, target: LogoutTarget) -> dict | None:
    """Return the hard-coded action dict for a logout sequence step.

    Steps:
        0: Click the logout button/link.

    Returns ``None`` when ``step_index >= 1``.
    """
    if step_index == 0:
        return {"action": "click", "candidate_id": target.button_id}
    return None


# =========================================================================
# Registration shortcut
# =========================================================================


@dataclass
class RegistrationFields:
    """Identified registration form fields with their candidate IDs."""

    username_id: int
    email_id: int | None  # Some forms combine user/email into one field
    password_id: int
    confirm_password_id: int | None
    submit_id: int


def detect_registration_fields(candidates: list[Candidate]) -> RegistrationFields | None:
    """Detect registration form fields in candidates.

    Returns ``RegistrationFields`` if at least username, password, and submit
    are found.  ``email_id`` and ``confirm_password_id`` may be ``None``.
    """
    username_id: int | None = None
    email_id: int | None = None
    password_ids: list[int] = []
    submit_id: int | None = None

    for c in candidates:
        label_lower = (c.label or c.text or "").lower()

        # Password fields (collect all -- first is password, second is confirm)
        if c.input_type == "password":
            password_ids.append(c.id)
            continue

        # Email field (explicit email type or label)
        if c.tag == "input" and c.input_type == "email":
            if email_id is None:
                email_id = c.id
                continue

        # Username field
        if c.tag == "input" and c.input_type in ("text", ""):
            if any(kw in label_lower for kw in ("user", "name", "login")):
                if username_id is None:
                    username_id = c.id
                    continue
            # Email in a text field
            if "email" in label_lower and email_id is None:
                email_id = c.id
                continue

    # Submit button
    for c in candidates:
        if c.tag == "button" or (c.tag == "input" and c.input_type == "submit"):
            label_lower = (c.label or c.text or "").lower()
            if any(
                kw in label_lower
                for kw in ("register", "sign up", "create", "submit")
            ):
                submit_id = c.id
                break
            # Fallback: any button after password fields
            if password_ids and submit_id is None:
                submit_id = c.id

    if not password_ids or submit_id is None:
        return None

    # If no separate username found, use email as username
    if username_id is None and email_id is not None:
        username_id = email_id
        email_id = None

    if username_id is None:
        return None

    return RegistrationFields(
        username_id=username_id,
        email_id=email_id,
        password_id=password_ids[0],
        confirm_password_id=password_ids[1] if len(password_ids) > 1 else None,
        submit_id=submit_id,
    )


def get_registration_action(step_index: int, fields: RegistrationFields) -> dict | None:
    """Return the hard-coded action dict for a registration sequence step.

    Steps are dynamically built based on which fields exist:
        - Type username
        - Type email (if separate field)
        - Type password
        - Type confirm password (if exists)
        - Click submit

    Returns ``None`` when sequence is complete.
    """
    steps: list[dict] = [
        {"action": "type", "candidate_id": fields.username_id, "text": "<username>"},
    ]
    if fields.email_id is not None:
        steps.append(
            {"action": "type", "candidate_id": fields.email_id, "text": "<email>"}
        )
    steps.append(
        {"action": "type", "candidate_id": fields.password_id, "text": "<password>"}
    )
    if fields.confirm_password_id is not None:
        steps.append(
            {"action": "type", "candidate_id": fields.confirm_password_id, "text": "<password>"}
        )
    steps.append({"action": "click", "candidate_id": fields.submit_id})

    if step_index < len(steps):
        return steps[step_index]
    return None


# =========================================================================
# Contact shortcut
# =========================================================================


@dataclass
class ContactFields:
    """Identified contact form fields with their candidate IDs."""

    name_id: int | None
    email_id: int | None
    message_id: int | None
    submit_id: int


def detect_contact_fields(candidates: list[Candidate]) -> ContactFields | None:
    """Detect contact form fields in candidates.

    Returns ``ContactFields`` if at least a submit button is found along
    with at least one typeable field (name, email, or message).
    """
    name_id: int | None = None
    email_id: int | None = None
    message_id: int | None = None
    submit_id: int | None = None

    for c in candidates:
        label_lower = (c.label or c.text or "").lower()

        if c.tag == "textarea" and message_id is None:
            message_id = c.id
            continue

        if c.tag == "input" and c.input_type in ("email",):
            if email_id is None:
                email_id = c.id
                continue

        if c.tag == "input" and c.input_type in ("text", ""):
            if any(kw in label_lower for kw in ("name", "your name", "full name")):
                if name_id is None:
                    name_id = c.id
                    continue
            if "email" in label_lower and email_id is None:
                email_id = c.id
                continue
            if "subject" in label_lower or "message" in label_lower:
                continue  # skip non-essential text fields
            # Generic text input -- use as name if none found yet
            if name_id is None:
                name_id = c.id
                continue

    # Submit button
    for c in candidates:
        if c.tag == "button" or (c.tag == "input" and c.input_type == "submit"):
            label_lower = (c.label or c.text or "").lower()
            if any(
                kw in label_lower
                for kw in ("send", "submit", "contact")
            ):
                submit_id = c.id
                break
            if submit_id is None:
                submit_id = c.id

    if submit_id is None:
        return None
    # Need at least one typeable field
    if name_id is None and email_id is None and message_id is None:
        return None

    return ContactFields(
        name_id=name_id,
        email_id=email_id,
        message_id=message_id,
        submit_id=submit_id,
    )


def get_contact_action(step_index: int, fields: ContactFields) -> dict | None:
    """Return the hard-coded action dict for a contact form sequence step.

    Steps are dynamically built based on which fields exist:
        - Type name (if exists)
        - Type email (if exists)
        - Type message (if textarea exists)
        - Click submit

    Returns ``None`` when sequence is complete.
    """
    steps: list[dict] = []
    if fields.name_id is not None:
        steps.append(
            {"action": "type", "candidate_id": fields.name_id, "text": "Test User"}
        )
    if fields.email_id is not None:
        steps.append(
            {"action": "type", "candidate_id": fields.email_id, "text": "test@example.com"}
        )
    if fields.message_id is not None:
        steps.append(
            {"action": "type", "candidate_id": fields.message_id, "text": "Hello, this is a test message."}
        )
    steps.append({"action": "click", "candidate_id": fields.submit_id})

    if step_index < len(steps):
        return steps[step_index]
    return None
