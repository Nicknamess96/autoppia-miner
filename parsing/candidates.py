"""Candidate extraction from HTML snapshots.

Extracts all interactive elements (buttons, links, inputs, textareas,
selects, role=button, role=link) into structured ``Candidate`` dataclass
instances for downstream LLM consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from parsing.filtering import is_disabled, is_hidden
from parsing.labels import infer_label
from parsing.selectors import build_selector

INTERACTIVE_SELECTORS = [
    "button",
    "a[href]",
    "input",
    "textarea",
    "select",
    "[role='button']",
    "[role='link']",
]


@dataclass
class Candidate:
    """An interactive element extracted from HTML."""

    id: int
    tag: str
    text: str
    selector: dict
    attrs: dict[str, str]
    label: str = ""
    parent_form: str | None = None
    input_type: str = ""
    placeholder: str = ""
    checked: bool = False
    selected: bool = False
    disabled: bool = False
    current_value: str = ""
    options: list[str] = field(default_factory=list)
    context: str = ""


def _attrs_to_str_map(attrs: dict) -> dict[str, str]:
    """Convert BS4 attribute dict to a flat str:str map.

    BS4 may return list values for attributes like ``class``.
    These are joined with a space.
    """
    result: dict[str, str] = {}
    for k, v in attrs.items():
        if isinstance(v, list):
            result[k] = " ".join(str(item) for item in v)
        else:
            result[k] = str(v)
    return result


def _get_parent_form(el) -> str | None:  # noqa: ANN001
    """Walk up the DOM to find an enclosing ``<form>``.

    Returns the form's ``id`` or ``name`` attribute if found, else ``None``.
    """
    form = el.find_parent("form")
    if form is None:
        return None
    return form.get("id") or form.get("name") or None


def _get_select_options(el) -> list[str]:  # noqa: ANN001
    """For ``<select>`` elements, return a list of ``<option>`` text values."""
    return [
        opt.get_text(strip=True)
        for opt in el.find_all("option")
        if opt.get_text(strip=True)
    ]


def _norm_context_ws(s: str) -> str:
    """Collapse whitespace in context text."""
    import re as _re
    return _re.sub(r"\s+", " ", s).strip()


def _pick_context_container(el) -> str:  # noqa: ANN001
    """Walk up the DOM (max 8 levels) to find a card-like container.

    Looks for ``li``, ``tr``, ``article``, or a ``div`` whose visible text
    is between 50 and 900 characters.  Returns the container's text
    truncated to 180 chars, or empty string if no suitable container found.
    """
    _CARD_TAGS = {"li", "tr", "article"}
    node = el.parent
    for _ in range(8):
        if node is None or node.name in (None, "body", "html", "[document]"):
            break
        tag = node.name
        text = _norm_context_ws(node.get_text(" ", strip=True))
        text_len = len(text)
        if tag in _CARD_TAGS and 20 < text_len < 900:
            return text[:180]
        if tag == "div" and 50 < text_len < 900:
            return text[:180]
        node = node.parent
    return ""


def extract_candidates(
    html: str, *, soup: BeautifulSoup | None = None
) -> list[Candidate]:
    """Extract all interactive elements from HTML.

    Parses the HTML with ``BeautifulSoup`` (lxml backend) and iterates
    over interactive element selectors. Hidden, disabled, and
    ``input[type=hidden]`` elements are excluded. Candidates are
    deduplicated by selector signature.

    Args:
        html: Raw HTML string. Ignored when *soup* is provided.
        soup: Optional pre-parsed ``BeautifulSoup`` object (e.g. from
            ``prune_html()``).  When supplied, the *html* argument is
            not parsed again, avoiding double-parsing overhead.

    Returns a list of ``Candidate`` objects with sequential IDs starting
    at 0.
    """
    if soup is None:
        soup = BeautifulSoup(html, "lxml")
    candidates: list[Candidate] = []
    seen_sigs: set[tuple[str, str, str]] = set()

    for css_sel in INTERACTIVE_SELECTORS:
        for el in soup.select(css_sel):
            tag = el.name
            attrs = _attrs_to_str_map(el.attrs)

            # Skip input[type=hidden]
            if tag == "input" and attrs.get("type", "").lower() == "hidden":
                continue

            # Skip hidden or disabled elements
            if is_hidden(attrs) or is_disabled(attrs):
                continue

            # Infer label and build selector
            label = infer_label(soup, el, attrs)
            selector = build_selector(tag, attrs, text=label)

            # Deduplicate by selector signature
            sig = (
                selector.get("type", ""),
                selector.get("attribute", ""),
                selector.get("value", ""),
            )
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)

            # Build candidate with full context
            context = ""
            if tag in ("a", "button"):
                context = _pick_context_container(el)

            candidate = Candidate(
                id=len(candidates),
                tag=tag,
                text=label,
                selector=selector,
                attrs=attrs,
                label=label,
                parent_form=_get_parent_form(el),
                input_type=attrs.get("type", "") if tag == "input" else "",
                placeholder=attrs.get("placeholder", ""),
                checked="checked" in attrs,
                current_value=attrs.get("value", ""),
                options=_get_select_options(el) if tag == "select" else [],
                context=context,
            )
            candidates.append(candidate)

    return candidates
