"""Hidden and disabled element detection for HTML candidate extraction."""


def is_hidden(attrs: dict[str, str]) -> bool:
    """Check if an element should be excluded as hidden.

    Checks all six hiding methods:
    1. ``hidden`` attribute present
    2. ``aria-hidden="true"``
    3. ``style`` containing ``display:none`` (case-insensitive)
    4. ``style`` containing ``visibility:hidden`` (case-insensitive)
    5. ``class`` containing tokens ``hidden``, ``sr-only``, or ``invisible``
    """
    if "hidden" in attrs:
        return True
    if attrs.get("aria-hidden", "").lower() == "true":
        return True
    style = attrs.get("style", "").lower()
    if "display:none" in style or "display: none" in style:
        return True
    if "visibility:hidden" in style or "visibility: hidden" in style:
        return True
    classes = attrs.get("class", "").lower()
    hidden_tokens = {"hidden", "sr-only", "invisible"}
    for token in classes.split():
        if token in hidden_tokens:
            return True
    return False


def is_disabled(attrs: dict[str, str]) -> bool:
    """Check if an element is disabled.

    Checks ``disabled`` attribute and ``aria-disabled="true"``.
    """
    if "disabled" in attrs:
        return True
    if attrs.get("aria-disabled", "").lower() == "true":
        return True
    return False
