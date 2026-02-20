"""ActResponse Pydantic model with typed action list."""

from pydantic import BaseModel

from models.actions import ActionUnion


class ActResponse(BaseModel):
    """Response body for the POST /act endpoint.

    The actions list uses the typed ActionUnion discriminated union,
    ensuring all responses are fully validated on construction.
    """

    actions: list[ActionUnion]
