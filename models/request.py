"""ActRequest Pydantic model — extra fields silently ignored for forward-compat."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class ActRequest(BaseModel):
    """Incoming request body for the POST /act endpoint.

    Fields match the IWA evaluator payload.
    Extra fields (e.g. ``model``) are silently ignored for forward-compat.
    """

    model_config = ConfigDict(extra="ignore")

    task_id: str
    prompt: str
    snapshot_html: str
    screenshot: Optional[str] = None
    url: str
    step_index: int
    history: list[dict[str, Any]] = []
    web_project_id: Optional[str] = None
