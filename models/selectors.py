"""IWA selector types as Pydantic v2 models with discriminated union."""

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class AttributeValueSelector(BaseModel):
    """Matches elements by HTML attribute name and value."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["attributeValueSelector"]
    attribute: str
    value: str
    case_sensitive: bool = False


class TagContainsSelector(BaseModel):
    """Matches elements containing text."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tagContainsSelector"]
    value: str
    case_sensitive: bool = False


class XpathSelector(BaseModel):
    """Matches elements by XPath expression."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["xpathSelector"]
    attribute: Optional[str] = None
    value: str
    case_sensitive: bool = False


SelectorUnion = Annotated[
    Union[AttributeValueSelector, TagContainsSelector, XpathSelector],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Helper factory functions (convenience for Phase 2+)
# ---------------------------------------------------------------------------


def sel_attr(
    attribute: str, value: str, case_sensitive: bool = False
) -> AttributeValueSelector:
    """Create an attributeValueSelector."""
    return AttributeValueSelector(
        type="attributeValueSelector",
        attribute=attribute,
        value=value,
        case_sensitive=case_sensitive,
    )


def sel_text(value: str, case_sensitive: bool = False) -> TagContainsSelector:
    """Create a tagContainsSelector."""
    return TagContainsSelector(
        type="tagContainsSelector",
        value=value,
        case_sensitive=case_sensitive,
    )


def sel_xpath(value: str) -> XpathSelector:
    """Create an xpathSelector."""
    return XpathSelector(
        type="xpathSelector",
        value=value,
    )
