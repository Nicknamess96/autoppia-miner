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

    # Fine-grained film/user task types (v1.1)
    ADD_FILM = "add_film"
    EDIT_FILM = "edit_film"
    DELETE_FILM = "delete_film"
    EDIT_USER = "edit_user"
    ADD_TO_WATCHLIST = "add_to_watchlist"
    REMOVE_FROM_WATCHLIST = "remove_from_watchlist"
    FILM_DETAIL = "film_detail"
    FILTER_FILM = "filter_film"
    SEARCH_FILM = "search_film"
    SHARE_MOVIE = "share_movie"


# Patterns ordered by specificity -- first match wins.
# CRITICAL: More specific patterns MUST precede more general ones.
# Order rationale:
#   1. LOGOUT before LOGIN (avoid "logout" matching "log in")
#   2. EDIT_USER before LOGIN (login+edit prompts are user edits, not logins)
#   3. LOGIN/REGISTRATION/CONTACT-compound (existing)
#   4. REMOVE_FROM_WATCHLIST before DELETE_FILM (watchlist removal != film deletion)
#   5. ADD_TO_WATCHLIST before ADD_FILM (watchlist addition != film addition)
#   6. DELETE_FILM/ADD_FILM/EDIT_FILM (specific CRUD before generic CRUD)
#   7. FILTER_FILM/SEARCH_FILM/SHARE_MOVIE (discovery types)
#   8. FILM_DETAIL before NAVIGATION (film page nav != generic nav)
#   9. NAVIGATION/CRUD/FORM/CONTACT-fallback (generic catchalls)
_TASK_PATTERNS: list[tuple[re.Pattern[str], TaskType]] = [
    # --- Position 1: LOGOUT (most specific auth action) ---
    (
        re.compile(
            r"\blog\s*out\b|\blogout\b|\bsign\s*out\b",
            re.IGNORECASE,
        ),
        TaskType.LOGOUT,
    ),
    # --- Position 2: EDIT_USER (login + edit/update profile fields; must precede LOGIN) ---
    (
        re.compile(
            r"\blogin\b.*\b(edit|modify|update|change)\b.*"
            r"\b(profile|first\s*name|last\s*name|bio|location|website|genre|name)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        TaskType.EDIT_USER,
    ),
    # --- Position 3: LOGIN ---
    (
        re.compile(
            r"\blog\s*in\b|\bsign\s*in\b|\blogin\b|\bauthenticat",
            re.IGNORECASE,
        ),
        TaskType.LOGIN,
    ),
    # --- Position 4: REGISTRATION ---
    (
        re.compile(
            r"\bregist(er|ration)\b|\bsign\s*up\b|\bcreate\s+account\b",
            re.IGNORECASE,
        ),
        TaskType.REGISTRATION,
    ),
    # --- Position 5: Compound CONTACT (navigation verbs + "contact") ---
    (
        re.compile(
            r"\bcontact\b.*\b(form|fill|submit|message|send)\b"
            r"|\b(go\s+to|visit|navigat(e|ion)|browse\s+to)\b.*\bcontact\b",
            re.IGNORECASE,
        ),
        TaskType.CONTACT,
    ),
    # --- Position 6: REMOVE_FROM_WATCHLIST (must precede DELETE_FILM) ---
    (
        re.compile(
            r"\b(remove|delete)\b.*\b(watchlist|wishlist)\b",
            re.IGNORECASE,
        ),
        TaskType.REMOVE_FROM_WATCHLIST,
    ),
    # --- Position 7: ADD_TO_WATCHLIST (must precede ADD_FILM) ---
    (
        re.compile(
            r"\badd\b.*\b(wishlist|watchlist)\b",
            re.IGNORECASE,
        ),
        TaskType.ADD_TO_WATCHLIST,
    ),
    # --- Position 8: DELETE_FILM ---
    (
        re.compile(
            r"\b(remove|delete|erase|discard|permanently\s+delete)\b.*\b(film|movie|records|database|system)\b",
            re.IGNORECASE,
        ),
        TaskType.DELETE_FILM,
    ),
    # --- Position 9: ADD_FILM ---
    (
        re.compile(
            r"\b(add|insert|register)\b.*\b(film|movie)\b",
            re.IGNORECASE,
        ),
        TaskType.ADD_FILM,
    ),
    # --- Position 10: EDIT_FILM ---
    (
        re.compile(
            r"\b(update|modify|change|edit)\b.*"
            r"\b(director|release\s+year|year|genre|rating|duration|cast|film|movie)\b",
            re.IGNORECASE,
        ),
        TaskType.EDIT_FILM,
    ),
    # --- Position 11: FILTER_FILM ---
    (
        re.compile(
            r"\bfilter\b",
            re.IGNORECASE,
        ),
        TaskType.FILTER_FILM,
    ),
    # --- Position 12: SEARCH_FILM ---
    (
        re.compile(
            r"\b(search\s+for|look\s+for|look\s+up|find)\b.*\b(film|movie)\b",
            re.IGNORECASE,
        ),
        TaskType.SEARCH_FILM,
    ),
    # --- Position 13: SHARE_MOVIE ---
    (
        re.compile(
            r"\bshare\b.*\b(film|movie|details)\b",
            re.IGNORECASE,
        ),
        TaskType.SHARE_MOVIE,
    ),
    # --- Position 14: FILM_DETAIL (must precede generic NAVIGATION) ---
    (
        re.compile(
            r"\b(navigate|go)\b.*\b(details?\s*page|movie\s*page|film\s*page|film\s*details)\b",
            re.IGNORECASE,
        ),
        TaskType.FILM_DETAIL,
    ),
    # --- Position 15: NAVIGATION (generic) ---
    (
        re.compile(
            r"\bnavig(ate|ation)\b|\bgo\s+to\b|\bvisit\b|\bbrowse\s+to\b",
            re.IGNORECASE,
        ),
        TaskType.NAVIGATION,
    ),
    # --- Position 16: CRUD (generic fallback, after specific CRUD types) ---
    (
        re.compile(
            r"\b(create|delete|remove|edit|update|add)\b.*"
            r"\b(item|record|entry|product|movie|book|booking|reservation)\b",
            re.IGNORECASE,
        ),
        TaskType.CRUD,
    ),
    # --- Position 17: FORM ---
    (
        re.compile(
            r"\bfill\b.*\bform\b|\bsubmit\b.*\bform\b|\bcomplete\b.*\bform\b",
            re.IGNORECASE,
        ),
        TaskType.FORM,
    ),
    # --- Position 18: CONTACT fallback ---
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


# =========================================================================
# Film name extraction (shared helper)
# =========================================================================


def extract_film_name_from_prompt(prompt: str) -> str | None:
    """Extract a film/movie name from the task prompt.

    Handles patterns like:
    - ``"...the movie 'The Matrix'..."`` -> ``"The Matrix"``
    - ``"...film 'Whiplash'..."`` -> ``"Whiplash"``
    - ``"Update the director of The Matrix to..."`` -> ``"The Matrix"``
    - ``"Remove The Matrix, a film..."`` -> ``"The Matrix"``
    - ``"Remove The Matrix from the database"`` -> ``"The Matrix"``

    Returns ``None`` if no film name can be identified.
    """
    if not prompt:
        return None

    # Pattern 1: Quoted name after movie/film keyword
    m = re.search(r"\b(?:movie|film)\s+['\"]([^'\"]+)['\"]", prompt, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Pattern 2: Any quoted name
    m = re.search(r"['\"]([^'\"]+)['\"]", prompt)
    if m:
        return m.group(1).strip()

    # Pattern 3: "of <Film Name> to" (edit patterns)
    m = re.search(r"\bof\s+(.+?)\s+to\b", prompt, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Pattern 4: "Remove/Delete/Erase/Discard <Film Name>, a ..." or
    #            "Remove <Film Name> from ..." or "... that ..."
    m = re.search(
        r"\b(?:remove|delete|erase|discard)\s+(.+?)(?:,\s*(?:a\s+|which\s+)|\s+from\b|\s+that\b)",
        prompt,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().strip("'\"")

    # Pattern 5: "Add <Film Name> to your watchlist/wishlist"
    m = re.search(
        r"\badd\s+(.+?)\s+to\s+(?:your\s+)?(?:watchlist|wishlist)\b",
        prompt,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().strip("'\"")

    # Pattern 6: "details/page of/for <Film Name>" (FILM_DETAIL prompts)
    m = re.search(
        r"\b(?:details?|page|info)\s+(?:of|for|about)\s+(?:the\s+)?(?:movie\s+|film\s+)?(.+?)$",
        prompt,
        re.IGNORECASE,
    )
    if m:
        name = m.group(1).strip().strip("'\"")
        # Reject generic descriptions (not film names)
        if name and name[0].isupper() and name.lower() not in (
            "movie", "film", "page", "details", "newest movie",
            "latest movie", "oldest movie", "newest film", "latest film",
        ):
            return name

    # Pattern 7: "Navigate/Go to the <Film Name> movie/film page"
    m = re.search(
        r"\b(?:navigate|go)\s+to\s+(?:the\s+)?([A-Z][^.!?]+?)\s+(?:movie|film)\s+(?:page|details)\b",
        prompt,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().strip("'\"")

    return None


# =========================================================================
# ADD_FILM shortcut
# =========================================================================


@dataclass
class AddFilmFields:
    """Form fields for the add-film page."""

    name_id: int | None
    director_id: int | None
    year_id: int | None
    genre_id: int | None  # May be a <select> dropdown
    duration_id: int | None
    cast_id: int | None
    submit_id: int


def detect_add_film_fields(candidates: list[Candidate]) -> AddFilmFields | None:
    """Detect add-film form fields in candidates.

    Returns ``AddFilmFields`` if at least a submit button and one
    film-related input are found, ``None`` otherwise.
    """
    name_id: int | None = None
    director_id: int | None = None
    year_id: int | None = None
    genre_id: int | None = None
    duration_id: int | None = None
    cast_id: int | None = None
    submit_id: int | None = None

    for c in candidates:
        label_lower = (c.label or c.text or "").lower()
        placeholder_lower = (c.placeholder or "").lower()
        name_attr = c.attrs.get("name", "").lower()

        searchable = f"{label_lower} {placeholder_lower} {name_attr}"

        if c.tag == "input":
            if name_id is None and any(
                k in searchable
                for k in ("title", "movie name", "film name")
            ):
                name_id = c.id
                continue
            # "name" keyword -- but exclude common non-film fields
            if name_id is None and "name" in searchable and not any(
                k in searchable for k in ("user", "your", "full", "last", "first")
            ):
                name_id = c.id
                continue
            if director_id is None and "director" in searchable:
                director_id = c.id
                continue
            if year_id is None and any(
                k in searchable for k in ("year", "release")
            ):
                year_id = c.id
                continue
            if duration_id is None and any(
                k in searchable for k in ("duration", "runtime", "length")
            ):
                duration_id = c.id
                continue
            if cast_id is None and any(
                k in searchable for k in ("cast", "actor", "starring")
            ):
                cast_id = c.id
                continue

        if c.tag == "select":
            if genre_id is None and "genre" in searchable:
                genre_id = c.id
                continue

        if c.tag == "textarea":
            if cast_id is None and any(
                k in searchable for k in ("cast", "description")
            ):
                cast_id = c.id
                continue

    # Submit button
    for c in candidates:
        if c.tag == "button" or (c.tag == "input" and c.input_type == "submit"):
            lbl = (c.label or c.text or "").lower()
            if any(k in lbl for k in ("add", "create", "save", "submit")):
                submit_id = c.id
                break
            if submit_id is None:
                submit_id = c.id

    if submit_id is None:
        return None
    # Need at least one film-related field
    if all(
        f is None
        for f in [name_id, director_id, year_id, genre_id, duration_id, cast_id]
    ):
        return None

    return AddFilmFields(
        name_id=name_id,
        director_id=director_id,
        year_id=year_id,
        genre_id=genre_id,
        duration_id=duration_id,
        cast_id=cast_id,
        submit_id=submit_id,
    )


def extract_add_film_values_from_prompt(prompt: str) -> dict[str, str]:
    """Extract film attribute values from an ADD_FILM task prompt.

    Returns a dict with keys ``name``, ``director``, ``year``, ``genre``
    (only keys that are actually mentioned in the prompt).
    """
    values: dict[str, str] = {}

    # Name
    name = extract_film_name_from_prompt(prompt)
    if name:
        values["name"] = name

    # Director: "directed by 'Wes Anderson'" or "directed by Wes Anderson"
    m = re.search(
        r"directed\s+by\s+['\"]?([^'\"]+?)['\"]?(?:\s*$|\s*,|\s*\.|\s+released|\s+with|\s+in\b)",
        prompt,
        re.IGNORECASE,
    )
    if m:
        values["director"] = m.group(1).strip()

    # Year: "released in 2014" or "year 2014"
    m = re.search(r"(?:released\s+in|year)\s+(\d{4})", prompt, re.IGNORECASE)
    if m:
        values["year"] = m.group(1)

    # Genre: "with genres Animation, Fantasy, and Adventure" or "genre Action"
    m = re.search(r"genres?\s+(.+?)(?:\s*$|\s*\.)", prompt, re.IGNORECASE)
    if m:
        values["genre"] = m.group(1).strip()

    return values


def get_add_film_action(
    step_index: int, fields: AddFilmFields, values: dict[str, str]
) -> dict | None:
    """Return the action dict for a step in the add-film sequence.

    Builds a dynamic step list from fields that have matching values.
    For each field with an extracted value, generates a type (or select) action.
    The final step is a click on the submit button.

    Returns ``None`` when ``step_index >= len(steps)`` (sequence complete).
    """
    # Field order: name, director, year, genre, duration, cast
    field_map: list[tuple[str, int | None, str]] = [
        ("name", fields.name_id, "type"),
        ("director", fields.director_id, "type"),
        ("year", fields.year_id, "type"),
        ("genre", fields.genre_id, "select"),
        ("duration", fields.duration_id, "type"),
        ("cast", fields.cast_id, "type"),
    ]

    steps: list[dict] = []
    for key, field_id, action_type in field_map:
        if field_id is not None and key in values:
            steps.append({
                "action": action_type,
                "candidate_id": field_id,
                "text": values[key],
            })

    steps.append({"action": "click", "candidate_id": fields.submit_id})

    if step_index < len(steps):
        return steps[step_index]
    return None


# =========================================================================
# Shared navigation helpers
# =========================================================================


def find_film_details_link(
    candidates: list[Candidate], film_name: str
) -> int | None:
    """Find the Details link for a specific film in the movie list.

    Scans candidates for ``<a>`` tags whose href points to a movie page
    (``/movies/``) and whose card context contains the target film name
    (case-insensitive substring match).  Falls back to link text matching
    "detail" or "info".  Returns the candidate ID or ``None``.
    """
    film_name_lower = film_name.lower()

    # Pass 1: links pointing to /movies/ whose context mentions the film
    for c in candidates:
        if c.tag != "a":
            continue
        href = c.attrs.get("href", "")
        if "/movies/" not in href:
            continue
        context_lower = (c.context or "").lower()
        if film_name_lower in context_lower:
            return c.id

    # Pass 2: links with "detail" or "info" text whose context mentions the film
    for c in candidates:
        if c.tag != "a":
            continue
        label_lower = (c.label or c.text or "").lower()
        if "detail" not in label_lower and "info" not in label_lower:
            continue
        context_lower = (c.context or "").lower()
        if film_name_lower in context_lower:
            return c.id

    return None


def find_button_by_label(
    candidates: list[Candidate], keywords: tuple[str, ...]
) -> int | None:
    """Find a button/link/input matching any keyword in its label.

    Scans candidates with tag ``button``, ``a``, or ``input`` for any
    keyword (case-insensitive) in the label text.  Returns the first
    matching candidate ID, or ``None``.
    """
    for c in candidates:
        if c.tag not in ("button", "a", "input"):
            continue
        label_lower = (c.label or c.text or "").lower()
        if any(kw in label_lower for kw in keywords):
            return c.id
    return None


def find_confirm_button(candidates: list[Candidate]) -> int | None:
    """Find a confirmation button (Confirm, OK, Yes) in candidates.

    Used after the initial Delete button has been clicked, so the
    candidates now represent the confirmation dialog.  Returns the
    candidate ID or ``None``.
    """
    return find_button_by_label(candidates, ("confirm", "ok", "yes"))


# =========================================================================
# EDIT_FILM shortcut
# =========================================================================


@dataclass
class EditFilmFields:
    """Form fields for the edit-film page."""

    name_id: int | None
    director_id: int | None
    year_id: int | None
    genre_id: int | None
    rating_id: int | None
    duration_id: int | None
    submit_id: int


def detect_edit_film_fields(candidates: list[Candidate]) -> EditFilmFields | None:
    """Detect edit-film form fields in candidates.

    Same heuristic as ``detect_add_film_fields`` but also detects a
    ``rating`` field.  Returns ``EditFilmFields`` if at least a submit
    button and one editable field are found, ``None`` otherwise.
    """
    name_id: int | None = None
    director_id: int | None = None
    year_id: int | None = None
    genre_id: int | None = None
    rating_id: int | None = None
    duration_id: int | None = None
    submit_id: int | None = None

    for c in candidates:
        label_lower = (c.label or c.text or "").lower()
        placeholder_lower = (c.placeholder or "").lower()
        name_attr = c.attrs.get("name", "").lower()

        searchable = f"{label_lower} {placeholder_lower} {name_attr}"

        if c.tag in ("input", "select", "textarea"):
            if name_id is None and any(
                k in searchable
                for k in ("title", "movie name", "film name")
            ):
                name_id = c.id
                continue
            if name_id is None and "name" in searchable and not any(
                k in searchable for k in ("user", "your", "full", "last", "first")
            ):
                name_id = c.id
                continue
            if director_id is None and "director" in searchable:
                director_id = c.id
                continue
            if year_id is None and any(
                k in searchable for k in ("year", "release")
            ):
                year_id = c.id
                continue
            if rating_id is None and "rating" in searchable:
                rating_id = c.id
                continue
            if duration_id is None and any(
                k in searchable for k in ("duration", "runtime", "length")
            ):
                duration_id = c.id
                continue
            if c.tag == "select" and genre_id is None and "genre" in searchable:
                genre_id = c.id
                continue

    # Submit button
    for c in candidates:
        if c.tag == "button" or (c.tag == "input" and c.input_type == "submit"):
            lbl = (c.label or c.text or "").lower()
            if any(k in lbl for k in ("save", "update", "submit", "edit", "modify")):
                submit_id = c.id
                break
            if submit_id is None:
                submit_id = c.id

    if submit_id is None:
        return None
    if all(
        f is None
        for f in [name_id, director_id, year_id, genre_id, rating_id, duration_id]
    ):
        return None

    return EditFilmFields(
        name_id=name_id,
        director_id=director_id,
        year_id=year_id,
        genre_id=genre_id,
        rating_id=rating_id,
        duration_id=duration_id,
        submit_id=submit_id,
    )


# Mapping from prompt field names to EditFilmFields attribute names
_EDIT_FIELD_ALIASES: dict[str, str] = {
    "director": "director",
    "release year": "year",
    "year": "year",
    "rating": "rating",
    "duration": "duration",
    "genre": "genre",
    "name": "name",
    "title": "name",
}


def extract_edit_values_from_prompt(prompt: str) -> dict[str, str]:
    """Extract the target field and new value from an EDIT_FILM task prompt.

    Parses patterns like:
    - ``"Update the director of The Matrix to Christopher Nolan"``
    - ``"Modify the release year of Pulp Fiction to 1994"``
    - ``"Change the rating of Interstellar to 4.8"``
    - ``"Edit the duration of The Godfather to 175 minutes"``

    Returns a dict with keys ``field`` (normalized), ``value``, and
    optionally ``film_name``.
    """
    values: dict[str, str] = {}

    # Film name
    film_name = extract_film_name_from_prompt(prompt)
    if film_name:
        values["film_name"] = film_name

    # Pattern: "Update/Modify/Change/Edit the {field_phrase} of {film} to {value}"
    m = re.search(
        r"\b(?:update|modify|change|edit)\s+the\s+(.+?)\s+of\s+.+?\s+to\s+(\d+(?:\.\d+)?|\S+(?:\s+\S+)*)(?:\s+minutes|\s+min|\s*$)",
        prompt,
        re.IGNORECASE,
    )
    if m:
        raw_field = m.group(1).strip().lower()
        raw_value = m.group(2).strip()

        # Normalize field name
        normalized = _EDIT_FIELD_ALIASES.get(raw_field, raw_field)
        values["field"] = normalized
        values["value"] = raw_value

    return values


def get_edit_film_action(
    step_index: int, fields: EditFilmFields, values: dict[str, str]
) -> dict | None:
    """Return the action dict for a step in the edit-film sequence.

    Only types into the single field matching ``values["field"]``, then
    clicks submit.

    Returns ``None`` when ``step_index >= len(steps)`` (sequence complete).
    """
    target_field = values.get("field")
    new_value = values.get("value")

    if not target_field or not new_value:
        return None

    # Map field name -> candidate ID
    field_id_map: dict[str, int | None] = {
        "name": fields.name_id,
        "director": fields.director_id,
        "year": fields.year_id,
        "genre": fields.genre_id,
        "rating": fields.rating_id,
        "duration": fields.duration_id,
    }

    target_id = field_id_map.get(target_field)

    steps: list[dict] = []
    if target_id is not None:
        steps.append({
            "action": "type",
            "candidate_id": target_id,
            "text": new_value,
        })
    steps.append({"action": "click", "candidate_id": fields.submit_id})

    if step_index < len(steps):
        return steps[step_index]
    return None


# =========================================================================
# EDIT_USER: credential extraction, profile fields, value extraction, action gen
# =========================================================================


def extract_credentials_from_prompt(prompt: str) -> tuple[str, str] | None:
    """Extract username and password from an EDIT_USER task prompt.

    Handles patterns like:
    - ``"Login where username equals user1 and password equals pass123. ..."``
    - ``"Login for the following username:filmfan and password:pass456. ..."``
    - ``"Login where username equals 'user1' and password equals 'pass123'."``

    Returns ``(username, password)`` tuple or ``None``.
    """
    m = re.search(
        r"username\s*(?:equals|:)\s*['\"]?(\S+?)['\"]?\s+and\s+password\s*(?:equals|:)\s*['\"]?(\S+?)['\"]?(?:\s*[.,]|\s+|\s*$)",
        prompt,
        re.IGNORECASE,
    )
    if m:
        return (m.group(1).strip("'\""), m.group(2).strip("'\""))
    return None


@dataclass
class ProfileFields:
    """Form fields for the user profile edit page."""

    first_name_id: int | None
    last_name_id: int | None
    email_id: int | None
    bio_id: int | None
    location_id: int | None
    website_id: int | None
    genres_id: int | None
    submit_id: int


def detect_profile_fields(candidates: list[Candidate]) -> ProfileFields | None:
    """Detect profile edit form fields in candidates.

    Uses label/placeholder/name-attribute/id-attribute keyword matching
    (same pattern as ``detect_add_film_fields``).  Returns ``ProfileFields``
    if at least a submit button and one profile-related input are found,
    ``None`` otherwise.
    """
    first_name_id: int | None = None
    last_name_id: int | None = None
    email_id: int | None = None
    bio_id: int | None = None
    location_id: int | None = None
    website_id: int | None = None
    genres_id: int | None = None
    submit_id: int | None = None

    for c in candidates:
        label_lower = (c.label or c.text or "").lower()
        placeholder_lower = (c.placeholder or "").lower()
        name_attr = c.attrs.get("name", "").lower()
        id_attr = c.attrs.get("id", "").lower()

        searchable = f"{label_lower} {placeholder_lower} {name_attr} {id_attr}"

        if c.tag == "input":
            if first_name_id is None and "first" in searchable and "name" in searchable:
                first_name_id = c.id
                continue
            if last_name_id is None and "last" in searchable and "name" in searchable:
                last_name_id = c.id
                continue
            if email_id is None and "email" in searchable:
                email_id = c.id
                continue
            if location_id is None and "location" in searchable:
                location_id = c.id
                continue
            if website_id is None and any(
                k in searchable for k in ("website", "url", "site")
            ):
                website_id = c.id
                continue
            if genres_id is None and any(
                k in searchable for k in ("genre", "favorite")
            ):
                genres_id = c.id
                continue

        if c.tag == "textarea":
            if bio_id is None and any(
                k in searchable for k in ("bio", "about", "description")
            ):
                bio_id = c.id
                continue

    # Submit button
    for c in candidates:
        if c.tag == "button" or (c.tag == "input" and c.input_type == "submit"):
            lbl = (c.label or c.text or "").lower()
            if any(k in lbl for k in ("save", "update", "submit")):
                submit_id = c.id
                break
            if submit_id is None:
                submit_id = c.id

    if submit_id is None:
        return None
    if all(
        f is None
        for f in [
            first_name_id, last_name_id, email_id, bio_id,
            location_id, website_id, genres_id,
        ]
    ):
        return None

    return ProfileFields(
        first_name_id=first_name_id,
        last_name_id=last_name_id,
        email_id=email_id,
        bio_id=bio_id,
        location_id=location_id,
        website_id=website_id,
        genres_id=genres_id,
        submit_id=submit_id,
    )


def extract_profile_values_from_prompt(prompt: str) -> dict[str, str]:
    """Extract profile field values from an EDIT_USER task prompt.

    Handles patterns like:
    - ``"Update your first name to John"``
    - ``"Modify your bio to passionate filmmaker"``
    - ``"Change your location to New York, USA"``
    - ``"Edit your website to https://myfilmblog.example.com"``
    - ``"Update your favorite genre to Sci-Fi"``
    - ``"ensure that your website contains 'https://cinephileworld.example.org'"``

    Returns dict with keys matching profile field names.
    """
    values: dict[str, str] = {}

    # Pattern: "{verb} your {field} to {value}"
    for m in re.finditer(
        r"(?:update|modify|change|edit|set)\s+(?:your\s+)?(.+?)\s+to\s+(.+?)(?:\s*\.(?:\s|$)|\s+and\s+|\s*$)",
        prompt,
        re.IGNORECASE,
    ):
        raw_field = m.group(1).strip().lower()
        raw_value = m.group(2).strip()

        if "first" in raw_field and "name" in raw_field:
            values["first_name"] = raw_value
        elif "last" in raw_field and "name" in raw_field:
            values["last_name"] = raw_value
        elif "bio" in raw_field:
            values["bio"] = raw_value
        elif "location" in raw_field:
            values["location"] = raw_value
        elif "website" in raw_field:
            values["website"] = raw_value
        elif "genre" in raw_field:
            values["genres"] = raw_value

    # Also handle "contains" pattern for constraint-style prompts
    # "ensure that your website contains 'https://cinephileworld.example.org'"
    for m in re.finditer(
        r"(?:your\s+)(\w+)\s+(?:does\s+NOT\s+)?contains?\s+['\"]?(.+?)['\"]?(?:\s+and\s+|\s*$)",
        prompt,
        re.IGNORECASE,
    ):
        raw_field = m.group(1).strip().lower()
        raw_value = m.group(2).strip()

        if "location" in raw_field:
            values["location"] = raw_value
        elif "website" in raw_field:
            values["website"] = raw_value
        elif "bio" in raw_field:
            values["bio"] = raw_value

    return values


def get_profile_action(
    step_index: int, fields: ProfileFields, values: dict[str, str]
) -> dict | None:
    """Return the action dict for a step in the profile-edit sequence.

    Builds a dynamic step list from fields that have matching values
    (same pattern as ``get_add_film_action``).  Field order:
    first_name, last_name, email, bio, location, website, genres.
    All use ``type`` action.  Final step: click submit.

    Returns ``None`` when ``step_index >= len(steps)`` (sequence complete).
    """
    field_map: list[tuple[str, int | None]] = [
        ("first_name", fields.first_name_id),
        ("last_name", fields.last_name_id),
        ("email", fields.email_id),
        ("bio", fields.bio_id),
        ("location", fields.location_id),
        ("website", fields.website_id),
        ("genres", fields.genres_id),
    ]

    steps: list[dict] = []
    for key, field_id in field_map:
        if field_id is not None and key in values:
            steps.append({
                "action": "type",
                "candidate_id": field_id,
                "text": values[key],
            })

    steps.append({"action": "click", "candidate_id": fields.submit_id})

    if step_index < len(steps):
        return steps[step_index]
    return None


# =========================================================================
# Watchlist helpers: button detection, tab navigation, remove button
# =========================================================================


def find_watchlist_button(candidates: list[Candidate]) -> int | None:
    """Find the watchlist/save button on a movie detail page.

    Uses a two-pass approach:
    1. First pass: specific keywords (``watchlist``, ``wishlist``,
       ``add to list``, ``save to list``).
    2. Second pass: generic keywords (``save``, ``add``) only if the
       first pass found nothing.

    Returns the first matching candidate ID, or ``None``.
    """
    _SPECIFIC_KW = ("watchlist", "wishlist", "add to list", "save to list")
    _GENERIC_KW = ("save", "add")

    # First pass: specific keywords
    for c in candidates:
        if c.tag not in ("button", "a", "input"):
            continue
        label_lower = (c.label or c.text or "").lower()
        if any(kw in label_lower for kw in _SPECIFIC_KW):
            return c.id

    # Second pass: generic keywords (fallback)
    for c in candidates:
        if c.tag not in ("button", "a", "input"):
            continue
        label_lower = (c.label or c.text or "").lower()
        if any(kw in label_lower for kw in _GENERIC_KW):
            return c.id

    return None


def find_tab_button(
    candidates: list[Candidate], keywords: tuple[str, ...]
) -> int | None:
    """Find a tab button matching any keyword.

    Looks for ``button`` elements (or elements with ``role="tab"``)
    whose label matches any keyword (case-insensitive).

    Returns the first matching candidate ID, or ``None``.
    """
    for c in candidates:
        is_tab = c.attrs.get("role") == "tab" or c.tag == "button"
        if not is_tab:
            continue
        label_lower = (c.label or c.text or "").lower()
        if any(kw in label_lower for kw in keywords):
            return c.id
    return None


def find_remove_button_for_film(
    candidates: list[Candidate], film_name: str
) -> int | None:
    """Find the remove button for a specific film in the watchlist.

    Scans candidates for ``button``/``a`` tags with ``"remove"`` in their
    label text.  For each such button, checks if *film_name*
    (case-insensitive) appears in the button's ``context``.

    Returns the candidate ID of the first matching remove button, or
    ``None``.
    """
    film_name_lower = film_name.lower()

    for c in candidates:
        if c.tag not in ("button", "a", "input"):
            continue
        label_lower = (c.label or c.text or "").lower()
        if "remove" not in label_lower:
            continue
        context_lower = (c.context or "").lower()
        if film_name_lower in context_lower:
            return c.id

    return None


# =========================================================================
# FILTER_FILM helpers: criteria extraction, dropdown detection
# =========================================================================

# Known genres from the autocinema webapp (lowercase for case-insensitive matching)
_KNOWN_GENRES = frozenset({
    "action", "adventure", "animation", "comedy", "crime",
    "documentary", "drama", "fantasy", "horror", "mystery",
    "romance", "sci-fi", "thriller", "war", "western", "screen",
})


@dataclass
class FilterDropdowns:
    """Identified genre and year filter dropdowns."""

    genre_id: int | None
    year_id: int | None


def extract_filter_criteria_from_prompt(prompt: str) -> dict[str, str]:
    """Extract genre and/or year filter criteria from a FILTER_FILM prompt.

    Handles patterns like:
    - ``"Filter movies released in the year 1994"`` -> ``{"year": "1994"}``
    - ``"Filter for Action movies"`` -> ``{"genre": "Action"}``
    - ``"Browse films from 2010 in the Drama genre"`` -> ``{"genre": "Drama", "year": "2010"}``
    - ``"Filter movie list to Sci-Fi genre"`` -> ``{"genre": "Sci-Fi"}``

    Returns dict with optional keys ``"genre"`` and ``"year"``.
    """
    criteria: dict[str, str] = {}

    # Year: first 4-digit number in the 1900-2099 range
    m = re.search(r"\b(19|20)\d{2}\b", prompt)
    if m:
        criteria["year"] = m.group(0)

    # Genre: scan for any known genre name (case-insensitive)
    prompt_lower = prompt.lower()
    for genre in _KNOWN_GENRES:
        if genre in prompt_lower:
            # Reconstruct original-case genre from the prompt
            idx = prompt_lower.index(genre)
            criteria["genre"] = prompt[idx : idx + len(genre)]
            break

    return criteria


def detect_filter_dropdowns(
    candidates: list[Candidate],
) -> FilterDropdowns | None:
    """Detect genre and year ``<select>`` dropdowns on the search page.

    Genre dropdown: ``<select>`` whose options contain non-numeric text
    (genre names like "Action", "Comedy") excluding "All..." placeholders.

    Year dropdown: ``<select>`` whose options contain 4-digit year numbers.

    Returns ``None`` if neither genre nor year dropdown is found.
    """
    genre_id: int | None = None
    year_id: int | None = None

    for c in candidates:
        if c.tag != "select":
            continue
        if not c.options:
            continue

        # Check if options contain years (4-digit numbers)
        year_options = [o for o in c.options if re.match(r"^\d{4}$", o.strip())]
        if year_options:
            year_id = c.id
            continue

        # Check if options contain genre-like text (non-numeric, not "All...")
        text_options = [
            o
            for o in c.options
            if o.strip()
            and not o.strip().startswith("All")
            and not re.match(r"^\d+$", o.strip())
        ]
        if text_options:
            genre_id = c.id
            continue

    if genre_id is None and year_id is None:
        return None
    return FilterDropdowns(genre_id=genre_id, year_id=year_id)
