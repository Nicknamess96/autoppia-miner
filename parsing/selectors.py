"""Selector priority chain for building IWA-compatible selectors.

Priority order (ACT-10):
    id > data-testid > href > aria-label > name > placeholder > title > text fallback

Never uses CSS class-based selectors (Tailwind breaks Playwright).
"""

from __future__ import annotations

from models.selectors import sel_attr, sel_text


def build_selector(tag: str, attrs: dict[str, str], text: str = "") -> dict:
    """Build the best IWA-compatible selector for an element.

    Returns a dict suitable for inclusion in an IWA action payload.
    """
    if attrs.get("id"):
        return sel_attr("id", attrs["id"]).model_dump()

    if attrs.get("data-testid"):
        return sel_attr("data-testid", attrs["data-testid"]).model_dump()

    if tag == "a" and attrs.get("href"):
        href = attrs["href"]
        if not href.lower().startswith("javascript:"):
            return sel_attr("href", href).model_dump()

    if attrs.get("aria-label"):
        return sel_attr("aria-label", attrs["aria-label"]).model_dump()

    if attrs.get("name"):
        return sel_attr("name", attrs["name"]).model_dump()

    if attrs.get("placeholder"):
        return sel_attr("placeholder", attrs["placeholder"]).model_dump()

    if attrs.get("title"):
        return sel_attr("title", attrs["title"]).model_dump()

    if text and tag in {"button", "a"}:
        return sel_text(text).model_dump()

    # Last resort: tag-only custom selector
    return sel_attr("custom", tag).model_dump()
