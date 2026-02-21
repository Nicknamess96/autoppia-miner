"""LLM decision to IWA action conversion and URL normalization.

Converts the JSON dict returned by the LLM into typed Pydantic action
models that conform to the IWA evaluator contract.  Includes pre-validation
with field inference for missing text on credential and select fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from models.actions import (
    ActionUnion,
    ClickAction,
    NavigateAction,
    ScrollAction,
    SelectDropDownOptionAction,
    TypeAction,
    WaitAction,
)
from models.selectors import (
    AttributeValueSelector,
    SelectorUnion,
    TagContainsSelector,
    XpathSelector,
)

if TYPE_CHECKING:
    from parsing.candidates import Candidate


# ---------------------------------------------------------------------------
# Same-URL detection
# ---------------------------------------------------------------------------

def _same_path_query(url_a: str, url_b: str) -> bool:
    """Return True if two URLs have the same path and query (ignoring scheme/host/fragment)."""
    if not url_a or not url_b:
        return False
    a = urlsplit(url_a)
    b = urlsplit(url_b)
    return a.path == b.path and a.query == b.query


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------

def normalize_url(raw_url: str) -> str:
    """Rewrite a URL to ``http://localhost`` while preserving path/query/fragment.

    - If no scheme, prepend ``http://localhost``.
    - If has scheme, replace scheme+netloc with ``http://localhost``.
    - Returns empty string for empty input.
    """
    if not raw_url:
        return ""

    if not raw_url.startswith(("http://", "https://", "//")):
        # Relative path -- prepend localhost
        if raw_url.startswith("/"):
            return f"http://localhost{raw_url}"
        return f"http://localhost/{raw_url}"

    parts = urlsplit(raw_url)
    return urlunsplit(("http", "localhost", parts.path, parts.query, parts.fragment))


def preserve_seed(target_url: str, current_url: str) -> str:
    """Ensure target URL carries seed, web_agent_id, and validator_id params.

    Copies missing params from ``current_url`` to ``target_url``.
    If target already has a param with the same value, it is left unchanged.
    """
    if not target_url or not current_url:
        return target_url

    target_parts = urlsplit(target_url)
    current_parts = urlsplit(current_url)

    target_params = parse_qs(target_parts.query, keep_blank_values=True)
    current_params = parse_qs(current_parts.query, keep_blank_values=True)

    propagate_keys = ("seed", "web_agent_id", "validator_id")
    modified = False

    for key in propagate_keys:
        if key in current_params and key not in target_params:
            target_params[key] = current_params[key]
            modified = True

    if not modified:
        return target_url

    new_query = urlencode(target_params, doseq=True)
    return urlunsplit((
        target_parts.scheme,
        target_parts.netloc,
        target_parts.path,
        new_query,
        target_parts.fragment,
    ))


# ---------------------------------------------------------------------------
# Selector conversion helper
# ---------------------------------------------------------------------------

def _selector_from_dict(d: dict) -> SelectorUnion:
    """Convert a dict (from Candidate.selector) to the appropriate Pydantic selector model."""
    sel_type = d.get("type", "")

    if sel_type == "attributeValueSelector":
        return AttributeValueSelector(
            type="attributeValueSelector",
            attribute=d.get("attribute", ""),
            value=d.get("value", ""),
            case_sensitive=d.get("case_sensitive", False),
        )
    if sel_type == "tagContainsSelector":
        return TagContainsSelector(
            type="tagContainsSelector",
            value=d.get("value", ""),
            case_sensitive=d.get("case_sensitive", False),
        )
    if sel_type == "xpathSelector":
        return XpathSelector(
            type="xpathSelector",
            value=d.get("value", ""),
            case_sensitive=d.get("case_sensitive", False),
        )

    # Fallback -- use tagContainsSelector with whatever value we have
    return TagContainsSelector(
        type="tagContainsSelector",
        value=d.get("value", "unknown"),
    )


# ---------------------------------------------------------------------------
# Validation with field inference
# ---------------------------------------------------------------------------

def validate_and_fix(
    decision: dict, candidates: list[Candidate]
) -> dict | None:
    """Validate an LLM decision and infer missing fields when obvious.

    For actions that do not require a candidate (done, scroll_down,
    scroll_up, navigate), returns the decision unchanged.

    For candidate-based actions (click, type, select):
    - Validates candidate_id is a parseable int within range.
    - For "type" with missing text: infers credential placeholders
      (password -> "<password>", email/user label -> "<username>").
    - For "select" with missing text: infers first option from candidate.
    - Returns ``None`` for ambiguous or unrecoverable cases.
    """
    action = decision.get("action", "")

    # No-candidate actions pass through
    if action in ("done", "scroll_down", "scroll_up", "navigate"):
        return decision

    # Candidate-based actions
    if action in ("click", "type", "select"):
        raw_cid = decision.get("candidate_id", -1)
        try:
            cid = int(raw_cid)
        except (ValueError, TypeError):
            return None

        if cid < 0 or cid >= len(candidates):
            return None  # No fuzzy matching per locked decision

        if action == "click":
            return decision

        candidate = candidates[cid]

        if action == "type":
            text = decision.get("text", "")
            if not text:
                # Infer credential placeholders
                if candidate.input_type == "password":
                    return {**decision, "text": "<password>"}
                label_lower = (candidate.label or "").lower()
                if "user" in label_lower or "email" in label_lower:
                    return {**decision, "text": "<username>"}
                # Ambiguous -- discard
                return None
            return decision

        if action == "select":
            text = decision.get("text", "")
            if not text:
                if candidate.options:
                    return {**decision, "text": candidate.options[0]}
                return None
            return decision

    # Unknown action type
    return None


# ---------------------------------------------------------------------------
# Action building
# ---------------------------------------------------------------------------

def build_action(
    decision: dict,
    candidates: list[Candidate],
    current_url: str,
    *,
    step_index: int = 99,
) -> ActionUnion | None:
    """Convert an LLM decision dict to an IWA ActionUnion.

    Uses ``validate_and_fix()`` for pre-validation and field inference.
    Returns ``None`` only for "done" (caller returns empty actions list).
    Returns ``ScrollAction(down=True)`` as a safe fallback for invalid
    decisions instead of ``None`` (avoids re-prompt LLM call).

    When *step_index* < 5 and candidates exist, falls back to clicking
    the first candidate instead of scrolling (more likely to make progress
    in early steps).
    """
    # Pre-validate and infer missing fields
    fixed = validate_and_fix(decision, candidates)
    if fixed is None:
        # Smarter fallback: click first candidate in early steps
        if step_index < 5 and candidates:
            selector = _selector_from_dict(candidates[0].selector)
            return ClickAction(type="ClickAction", selector=selector)
        # Late steps: scroll is safer
        return ScrollAction(type="ScrollAction", down=True)

    action = fixed.get("action", "")

    # --- Done signal ---
    if action == "done":
        return None

    # --- Scroll actions (no candidate needed) ---
    if action == "scroll_down":
        return ScrollAction(type="ScrollAction", down=True)
    if action == "scroll_up":
        return ScrollAction(type="ScrollAction", up=True)

    # --- Navigate action ---
    if action == "navigate":
        raw_url = fixed.get("url", "")
        normalized = normalize_url(raw_url)
        final_url = preserve_seed(normalized, current_url)
        # Guard: navigating to the same path+query causes chrome-error loops
        if _same_path_query(final_url, current_url):
            return ScrollAction(type="ScrollAction", down=True)
        return NavigateAction(type="NavigateAction", url=final_url)

    # --- Actions requiring candidate_id ---
    if action in ("click", "type", "select"):
        cid = int(fixed["candidate_id"])
        selector = _selector_from_dict(candidates[cid].selector)

        if action == "click":
            return ClickAction(type="ClickAction", selector=selector)

        if action == "type":
            text = fixed.get("text", "")
            return TypeAction(type="TypeAction", selector=selector, text=text)

        if action == "select":
            text = fixed.get("text", "")
            return SelectDropDownOptionAction(
                type="SelectDropDownOptionAction",
                selector=selector,
                text=text,
            )

    # --- Unknown action type -> safe fallback ---
    return ScrollAction(type="ScrollAction", down=True)
