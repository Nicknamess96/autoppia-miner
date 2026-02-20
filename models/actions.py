"""All 6 IWA action types as Pydantic v2 models with discriminated union."""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from models.selectors import SelectorUnion


class ClickAction(BaseModel):
    """Click an element identified by a selector."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["ClickAction"]
    selector: SelectorUnion


class TypeAction(BaseModel):
    """Type text into an element identified by a selector."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["TypeAction"]
    selector: SelectorUnion
    text: str


class SelectDropDownOptionAction(BaseModel):
    """Select a dropdown option by visible text."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["SelectDropDownOptionAction"]
    selector: SelectorUnion
    text: str
    timeout_ms: int = 4000


class NavigateAction(BaseModel):
    """Navigate to a URL or go back/forward in browser history."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["NavigateAction"]
    url: str
    go_back: bool = False
    go_forward: bool = False


class ScrollAction(BaseModel):
    """Scroll the page up or down."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["ScrollAction"]
    down: bool = False
    up: bool = False


class WaitAction(BaseModel):
    """Wait for a specified duration (seconds)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["WaitAction"]
    time_seconds: float = 1.0


ActionUnion = Annotated[
    Union[
        ClickAction,
        TypeAction,
        SelectDropDownOptionAction,
        NavigateAction,
        ScrollAction,
        WaitAction,
    ],
    Field(discriminator="type"),
]
