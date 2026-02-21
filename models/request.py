"""ActRequest Pydantic model with strict validation (extra=forbid)."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class ActRequest(BaseModel):
    """Incoming request body for the POST /act endpoint.

    Fields match the IWA evaluator payload exactly.
    Extra fields are rejected with a 422 response.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str
    prompt: str
    snapshot_html: str
    screenshot: Optional[str] = None
    url: str
    step_index: int
    history: list[dict[str, Any]] = []
    web_project_id: Optional[str] = None
