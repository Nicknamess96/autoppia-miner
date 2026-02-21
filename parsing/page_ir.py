"""Compact Page IR (Intermediate Representation) builder.

Produces a two-section plain-text representation of a web page for LLM
consumption:

    Section 1 -- **Page Context:** URL, title, heading hierarchy, and a
    visible-text summary.

    Section 2 -- **Interactive Elements:** Numbered one-liner per
    candidate element with tag, label, and key metadata.

The output respects a hard token budget (default 1200 tokens) enforced
via a 4-char/token heuristic with a 10 % safety margin (effective
character cap = ``max_tokens * 4 * 0.9``).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from parsing.pruning import strip_presentation_attrs

if TYPE_CHECKING:
    from bs4 import BeautifulSoup

    from parsing.candidates import Candidate


# ---------------------------------------------------------------------------
# Compact element formatting
# ---------------------------------------------------------------------------


def _format_candidate_compact(c: Candidate) -> str:
    """Format a single candidate as a compact one-liner.

    Format::

        [id] tag "label" (key=value, key=value)

    Examples::

        [0] button "Log In" (id=login-btn, form=loginForm)
        [3] input[password] "Enter password" (name=password, form=loginForm)
        [5] select "Country" (name=country, options=[USA, Canada, UK])
    """
    parts: list[str] = [f"[{c.id}]"]

    # Tag with input_type suffix for inputs
    if c.input_type:
        parts.append(f"{c.tag}[{c.input_type}]")
    else:
        parts.append(c.tag)

    # Label (truncated to 60 chars)
    label = c.label or c.text
    if label:
        parts.append(f'"{label[:60]}"')

    # Parenthetical metadata (only present items, in order)
    meta: list[str] = []

    sel_attr = c.selector.get("attribute", "")
    sel_val = c.selector.get("value", "")
    if sel_attr and sel_val and sel_attr != "custom":
        meta.append(f"{sel_attr}={sel_val[:40]}")

    if c.parent_form:
        meta.append(f"form={c.parent_form}")

    if c.current_value:
        meta.append(f"val={c.current_value[:30]}")

    if c.options:
        opts = ", ".join(c.options[:5])
        meta.append(f"options=[{opts}]")

    if c.disabled:
        meta.append("disabled")

    if meta:
        parts.append(f"({', '.join(meta)})")

    # Context suffix for links/buttons — helps LLM distinguish repeated labels
    if c.tag in ("a", "button") and c.context:
        ctx = c.context
        # Don't append if context is just the label repeated
        if ctx.strip().lower() != (c.label or c.text or "").strip().lower():
            parts.append(f'-> "{ctx[:120]}"')

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Page IR builder
# ---------------------------------------------------------------------------


def build_page_ir(
    soup: BeautifulSoup,
    url: str,
    title: str,
    candidates: list[Candidate],
    *,
    max_tokens: int = 1200,
) -> str:
    """Build a compact page intermediate representation under token budget.

    Operates on a **copy** of *soup* so that the original (which may
    still be needed for selector building) is not mutated by attribute
    stripping.

    Args:
        soup: Pruned ``BeautifulSoup`` object (from ``prune_html()``).
        url: Current page URL.
        title: Page ``<title>`` text.
        candidates: List of ``Candidate`` objects from ``extract_candidates()``.
        max_tokens: Hard token cap for the output (default 1200).

    Returns:
        A plain-text string with two labelled sections.
    """
    char_limit = int(max_tokens * 4 * 0.9)

    # Work on a copy so the caller's soup is untouched
    soup_copy = copy.copy(soup)
    strip_presentation_attrs(soup_copy)

    # ---- Section 1: Page Context ----
    lines: list[str] = [f"URL: {url}", f"TITLE: {title}", ""]

    # Headings
    lines.append("PAGE STRUCTURE:")
    for h in soup_copy.find_all(["h1", "h2", "h3"]):
        text = h.get_text(strip=True)[:80]
        if text:
            lines.append(f"  {h.name}: {text}")

    # Body visible text summary
    body_text = soup_copy.get_text(" ", strip=True)[:500]
    if body_text:
        lines.append(f"TEXT: {body_text}")

    lines.append("")

    # ---- Section 2: Interactive Elements ----
    lines.append("INTERACTIVE ELEMENTS:")
    for c in candidates:
        lines.append(_format_candidate_compact(c))

    ir_text = "\n".join(lines)

    # Enforce token cap
    if len(ir_text) > char_limit:
        ir_text = _truncate_ir(lines, candidates, char_limit)

    return ir_text


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def _truncate_ir(
    lines: list[str],
    candidates: list[Candidate],
    char_limit: int,
) -> str:
    """Truncate the Page IR to fit within *char_limit*.

    Truncation priority (keep first, cut last):

    - **ALWAYS keep:** URL, TITLE lines
    - **ALWAYS keep:** "INTERACTIVE ELEMENTS:" header + element lines
      (priority: form inputs > buttons/submits > links > others)
    - **CUT FIRST:** body text summary (``TEXT:`` line)
    - **CUT SECOND:** page structure headings
    - **CUT LAST:** interactive element lines (starting from end of list),
      appending a ``"... (N more elements truncated)"`` notice.
    """
    # Separate lines into buckets
    header_lines: list[str] = []   # URL, TITLE
    heading_lines: list[str] = []  # "  h1: ...", "  h2: ...", etc.
    text_line: str | None = None   # "TEXT: ..."
    element_lines: list[str] = []  # "[N] tag ..."

    for line in lines:
        if line.startswith("URL:") or line.startswith("TITLE:"):
            header_lines.append(line)
        elif line.startswith("  h") and ":" in line[:8]:
            heading_lines.append(line)
        elif line.startswith("TEXT:"):
            text_line = line
        elif line.startswith("["):
            element_lines.append(line)

    # Build result: always start with URL + TITLE
    result: list[str] = list(header_lines)
    result.append("")
    budget = char_limit - sum(len(l) + 1 for l in result)

    # Add interactive elements header
    ie_header = "INTERACTIVE ELEMENTS:"
    result.append(ie_header)
    budget -= len(ie_header) + 1

    # Add element lines until budget runs low
    included_elements = 0
    for el_line in element_lines:
        cost = len(el_line) + 1
        # Reserve space for potential truncation notice + heading section
        if budget - cost > 80:
            result.append(el_line)
            budget -= cost
            included_elements += 1
        else:
            remaining = len(element_lines) - included_elements
            notice = f"... ({remaining} more elements truncated)"
            result.append(notice)
            budget -= len(notice) + 1
            break

    # Fill remaining budget with page structure if space allows
    if budget > 100 and heading_lines:
        result.append("")
        ps_header = "PAGE STRUCTURE:"
        result.append(ps_header)
        budget -= len(ps_header) + 2
        for h_line in heading_lines:
            cost = len(h_line) + 1
            if budget - cost > 0:
                result.append(h_line)
                budget -= cost
            else:
                break

    # Add body text only if plenty of space left
    if budget > 200 and text_line:
        # Truncate text to remaining budget
        available = budget - 1
        if len(text_line) > available:
            text_line = text_line[:available]
        result.append(text_line)

    return "\n".join(result)
