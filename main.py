"""FastAPI application for the Autoppia web automation agent.

Exports ``app`` for use with ``uvicorn main:app``.

Build: v1.1.2 -- force re-evaluation (cache bust).
"""

import json
import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent.loop import decide
from models.request import ActRequest
from models.response import ActResponse


# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------

class StructuredFormatter(logging.Formatter):
    """JSON-lines formatter for structured log output."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Include optional extra fields when present
        for key in ("task_id", "url", "step_index", "action_type"):
            val = getattr(record, key, None)
            if val is not None:
                log_data[key] = val
        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


_handler = logging.StreamHandler()
_handler.setFormatter(StructuredFormatter())

logger = logging.getLogger("agent")
logger.addHandler(_handler)
logger.setLevel(logging.INFO)
# Prevent propagation to root logger to avoid duplicate output
logger.propagate = False


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Autoppia Web Agent")


# ---------------------------------------------------------------------------
# Global exception handler -- agent must NEVER crash
# ---------------------------------------------------------------------------

SAFE_WAIT_RESPONSE = {"actions": [{"type": "WaitAction", "time_seconds": 1.0}]}


@app.exception_handler(Exception)
async def catch_all_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch all unhandled exceptions and return a safe WaitAction."""
    logger.error(
        "Unhandled exception: %s: %s\n%s",
        type(exc).__name__,
        exc,
        traceback.format_exc(),
    )
    return JSONResponse(status_code=200, content=SAFE_WAIT_RESPONSE)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """Health check -- must respond instantly (20s sandbox timeout)."""
    return {"status": "healthy"}


@app.post("/act", response_model=ActResponse)
async def act(request: ActRequest) -> ActResponse:
    """Handle an IWA act request.

    Delegates to the agent decision loop which parses the page HTML,
    calls the LLM for action decisions, and returns typed IWA actions.
    """
    logger.info(
        "act request",
        extra={
            "task_id": request.task_id,
            "url": request.url,
            "step_index": request.step_index,
        },
    )

    response = decide(request)

    action_type = (
        type(response.actions[0]).__name__ if response.actions else "done"
    )
    logger.info(
        "act response",
        extra={"action_type": action_type},
    )

    return response
