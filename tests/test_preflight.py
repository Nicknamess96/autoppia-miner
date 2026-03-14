"""Pre-flight validation tests for container readiness (VALID-01, VALID-02, VALID-03).

These tests verify every property needed for sandbox success without Docker:
- VALID-01: /act contract round-trip with typed actions
- VALID-02: Read-only filesystem safety (no writes in code, Dockerfile ENV)
- VALID-03: Endpoint availability (/health, /act) and timing
"""

import re
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MINER_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_LOGIN_PAYLOAD = {
    "task_id": "preflight-001",
    "prompt": "Log in with username admin and password secret",
    "snapshot_html": (
        "<html><body><form>"
        "<label for='username'>Username</label>"
        "<input type='text' id='username'>"
        "<label for='password'>Password</label>"
        "<input type='password' id='password'>"
        "<button type='submit'>Log In</button>"
        "</form></body></html>"
    ),
    "url": "http://localhost/login?seed=42&web_agent_id=99&validator_id=test",
    "step_index": 0,
    "history": [],
}

VALID_ACTION_TYPES = {
    "ClickAction",
    "TypeAction",
    "SelectDropDownOptionAction",
    "NavigateAction",
    "ScrollAction",
    "WaitAction",
}

# Patterns that indicate filesystem write operations
FS_WRITE_PATTERNS = [
    r"""open\s*\([^)]*['"][wa]['"]""",       # open(..., 'w') or open(..., 'a')
    r"""\.write\s*\(""",                      # .write(...)
    r"""\.mkdir\s*\(""",                      # .mkdir(...)
    r"""Path\s*\([^)]*\)\.write_""",          # Path(...).write_text / write_bytes
    r"""FileHandler\s*\(""",                  # logging.FileHandler(...)
    r"""RotatingFileHandler\s*\(""",          # logging.handlers.RotatingFileHandler(...)
]


# ---------------------------------------------------------------------------
# VALID-03: Health endpoint
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHealthEndpoint:
    """Verify /health endpoint returns 200 with status healthy."""

    def test_health_endpoint(self):
        """GET /health returns 200 with {"status": "healthy"}."""
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"status": "healthy"}


# ---------------------------------------------------------------------------
# VALID-01: /act contract tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestActContract:
    """Verify /act endpoint contract: valid request/response, action types, timing."""

    def test_act_returns_valid_response(self):
        """POST /act with login task returns 200 with actions list."""
        client = TestClient(app)
        resp = client.post("/act", json=SAMPLE_LOGIN_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        assert "actions" in data, "Response must contain 'actions' key"
        assert isinstance(data["actions"], list), "'actions' must be a list"
        assert len(data["actions"]) > 0, "Response must contain at least one action"
        for action in data["actions"]:
            assert "type" in action, "Each action must have a 'type' field"

    def test_act_action_types_valid(self):
        """Every action type in response is one of the 6 valid ActionUnion types."""
        client = TestClient(app)
        resp = client.post("/act", json=SAMPLE_LOGIN_PAYLOAD)
        data = resp.json()
        for action in data["actions"]:
            assert action["type"] in VALID_ACTION_TYPES, (
                f"Invalid action type '{action['type']}', "
                f"expected one of {VALID_ACTION_TYPES}"
            )

    def test_act_empty_html_returns_actions(self):
        """POST /act with minimal/empty HTML still returns 200 (never crashes).

        With empty HTML the hardcoded login handler cannot find fields and
        falls through to the LLM path which may fail without API keys.
        The global exception handler must catch any error and return a safe
        WaitAction response (status 200).  We disable raise_server_exceptions
        so the TestClient returns the response instead of re-raising.
        """
        client = TestClient(app, raise_server_exceptions=False)
        payload = {
            **SAMPLE_LOGIN_PAYLOAD,
            "snapshot_html": "<html><body></body></html>",
        }
        resp = client.post("/act", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "actions" in data
        assert isinstance(data["actions"], list)

    def test_act_request_extra_fields_ignored(self):
        """POST /act with extra unknown fields still returns 200 (forward compat)."""
        client = TestClient(app)
        payload = {
            **SAMPLE_LOGIN_PAYLOAD,
            "unknown_future_field": "some_value",
            "another_field": 42,
        }
        resp = client.post("/act", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "actions" in data

    def test_hardcoded_handler_timing(self):
        """Hardcoded handler response time is under 500ms (generous CI margin)."""
        client = TestClient(app)
        start = time.monotonic()
        resp = client.post("/act", json=SAMPLE_LOGIN_PAYLOAD)
        elapsed = time.monotonic() - start
        assert resp.status_code == 200
        assert elapsed < 0.5, (
            f"Hardcoded handler took {elapsed:.3f}s, expected under 0.5s"
        )


# ---------------------------------------------------------------------------
# VALID-02: Read-only filesystem safety
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestReadOnlyFilesystem:
    """Verify no filesystem writes in miner code and Dockerfile safety settings."""

    def test_no_filesystem_writes(self):
        """Scan all *.py files (excluding tests/ and __pycache__) for write patterns."""
        violations = []
        py_files = sorted(MINER_ROOT.rglob("*.py"))
        for py_file in py_files:
            rel = py_file.relative_to(MINER_ROOT)
            # Skip tests and cache directories and venv
            parts = rel.parts
            if any(
                p in ("tests", "__pycache__", ".venv", "venv")
                for p in parts
            ):
                continue
            content = py_file.read_text()
            for pattern in FS_WRITE_PATTERNS:
                matches = re.findall(pattern, content)
                if matches:
                    violations.append(
                        f"{rel}: matched pattern {pattern!r} ({len(matches)} hit(s))"
                    )
        assert violations == [], (
            "Filesystem write patterns found in miner code:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_dockerfile_readonly_safety(self):
        """Dockerfile contains required ENV settings for read-only safety."""
        dockerfile = MINER_ROOT / "Dockerfile"
        content = dockerfile.read_text()
        assert "PYTHONDONTWRITEBYTECODE=1" in content, (
            "Dockerfile missing PYTHONDONTWRITEBYTECODE=1"
        )
        assert "HOME=/tmp" in content, "Dockerfile missing HOME=/tmp"
        assert "PIP_NO_CACHE_DIR=1" in content, (
            "Dockerfile missing PIP_NO_CACHE_DIR=1"
        )
        assert "--workers 1" in content, "Dockerfile missing --workers 1"

    def test_dockerfile_base_image(self):
        """Dockerfile uses python:3.11.14-slim (matching sandbox base)."""
        dockerfile = MINER_ROOT / "Dockerfile"
        content = dockerfile.read_text()
        assert "python:3.11.14-slim" in content, (
            "Dockerfile should use python:3.11.14-slim as base image"
        )

    def test_logging_uses_stream_handler(self):
        """main.py uses StreamHandler (not FileHandler) for logging."""
        main_py = MINER_ROOT / "main.py"
        content = main_py.read_text()
        assert "StreamHandler" in content, (
            "main.py should use logging.StreamHandler for stdout/stderr logging"
        )
        assert "FileHandler" not in content, (
            "main.py should NOT use FileHandler (filesystem writes in sandbox)"
        )
