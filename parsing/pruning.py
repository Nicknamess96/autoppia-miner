"""HTML pruning pipeline for token-efficient page representation.

Strips non-semantic content (scripts, styles, SVGs, comments) from HTML
while preserving all functional attributes needed for candidate extraction
and selector building.

Two-stage design:
    1. ``prune_html()`` -- removes tag subtrees and comments but keeps
       class/style/data-* attributes intact so that ``filtering.py`` can
       still detect hidden elements.
    2. ``strip_presentation_attrs()`` -- removes class, style, and data-*
       attributes (except ``data-testid``).  Called later by ``page_ir.py``
       on a *copy* of the soup, after candidate extraction is complete.
"""

from __future__ import annotations

from bs4 import BeautifulSoup, Comment

STRIP_TAGS = {"script", "style", "svg", "noscript", "iframe"}

# Attributes to preserve even when stripping presentation attrs.
# data-testid is kept because it is used in the selector priority chain.
_PRESERVE_DATA_ATTRS = {"data-testid"}


def prune_html(raw_html: str) -> BeautifulSoup:
    """Parse and prune HTML, removing non-semantic tag subtrees and comments.

    Decomposes ``<script>``, ``<style>``, ``<svg>``, ``<noscript>``, and
    ``<iframe>`` tags entirely.  Removes all HTML comments.

    Class, style, and data-* attributes are **not** stripped here -- they
    are still needed by ``parsing.filtering`` to detect hidden elements.

    Returns:
        A ``BeautifulSoup`` object with unwanted tags decomposed and
        comments removed.
    """
    soup = BeautifulSoup(raw_html, "lxml")

    # 1. Remove entire tag subtrees
    for tag_name in STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # 2. Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    return soup


def strip_presentation_attrs(soup: BeautifulSoup) -> None:
    """Remove presentation-only attributes from all elements in *soup*.

    Strips ``class``, ``style``, and ``data-*`` attributes (except
    ``data-testid`` which is used in the selector priority chain).

    This function **mutates** the soup in place.  It should be called on
    a *copy* of the soup after candidate extraction so that filtering
    (which relies on class/style) is not affected.
    """
    for tag in soup.find_all(True):
        attrs_to_remove = []
        for attr in tag.attrs:
            if attr in ("class", "style"):
                attrs_to_remove.append(attr)
            elif attr.startswith("data-") and attr not in _PRESERVE_DATA_ATTRS:
                attrs_to_remove.append(attr)
        for attr in attrs_to_remove:
            del tag[attr]
