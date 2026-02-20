"""Label inference for interactive HTML elements."""

from __future__ import annotations

import re


def _norm_ws(s: str) -> str:
    """Collapse whitespace runs into a single space and strip."""
    return re.sub(r"\s+", " ", (s or "")).strip()


def infer_label(soup, el, attrs: dict[str, str]) -> str:  # noqa: ANN001
    """Infer a human-readable label for an element.

    Priority order:
    1. For ``a`` and ``button``: inner text (truncated to 120 chars)
    2. ``aria-label`` attribute
    3. ``placeholder`` attribute
    4. ``title`` attribute
    5. ``aria-labelledby`` reference (first ID, look up element text)
    6. Associated ``<label for="id">`` matching element's ``id``
    7. Parent ``<label>`` wrapper
    8. Empty string if nothing found
    """
    tag = el.name

    # Buttons and links (including role=button/role=link): use inner text first
    role = attrs.get("role", "")
    if tag in ("a", "button") or role in ("button", "link"):
        t = _norm_ws(el.get_text(" ", strip=True))
        if t:
            return t[:120]

    # Attribute-based labels
    for key in ("aria-label", "placeholder", "title"):
        if attrs.get(key):
            return _norm_ws(attrs[key])[:120]

    # aria-labelledby reference
    if attrs.get("aria-labelledby"):
        lid = attrs["aria-labelledby"].split()[0]
        lab = soup.find(id=lid)
        if lab:
            t = _norm_ws(lab.get_text(" ", strip=True))
            if t:
                return t[:120]

    # Associated <label for="id">
    if attrs.get("id"):
        lab = soup.find("label", attrs={"for": attrs["id"]})
        if lab:
            t = _norm_ws(lab.get_text(" ", strip=True))
            if t:
                return t[:120]

    # Parent <label> wrapper
    parent_label = el.find_parent("label")
    if parent_label:
        t = _norm_ws(parent_label.get_text(" ", strip=True))
        if t:
            return t[:120]

    return ""
