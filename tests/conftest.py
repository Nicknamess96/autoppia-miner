"""Shared test fixtures for autoppia-miner tests."""
from __future__ import annotations

import pytest
from parsing.candidates import Candidate


def make_candidate(
    id: int,
    tag: str = "input",
    text: str = "",
    label: str = "",
    input_type: str = "",
    placeholder: str = "",
    parent_form: str | None = None,
    **kwargs,
) -> Candidate:
    """Factory for creating Candidate instances in tests."""
    return Candidate(
        id=id,
        tag=tag,
        text=text,
        selector={"attribute": "id", "value": f"el-{id}"},
        attrs={},
        label=label,
        parent_form=parent_form,
        input_type=input_type,
        placeholder=placeholder,
        **kwargs,
    )


@pytest.fixture
def search_page_candidates() -> list[Candidate]:
    """Candidates for a typical /search page with a text input and submit button."""
    return [
        make_candidate(0, tag="a", text="Home", label="Home"),
        make_candidate(1, tag="input", input_type="text", label="Search", placeholder="Search films..."),
        make_candidate(2, tag="button", text="Search", label="Search"),
        make_candidate(3, tag="a", text="Login", label="Login"),
    ]


@pytest.fixture
def login_page_candidates() -> list[Candidate]:
    """Candidates for a typical login page."""
    return [
        make_candidate(0, tag="input", input_type="text", label="Username"),
        make_candidate(1, tag="input", input_type="password", label="Password"),
        make_candidate(2, tag="button", text="Log In", label="Log In"),
    ]
