"""LLM gateway HTTP client with IWA-Task-ID header and retry logic.

Sends requests to the OpenAI-compatible gateway inside the IWA validator
sandbox. Uses httpx for HTTP and tenacity for retry-on-error.
"""

from __future__ import annotations

import os

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient gateway errors that should be retried."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503)
    if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout)):
        return True
    return False


class LLMClient:
    """Synchronous client for the OpenAI-compatible LLM gateway.

    Reads ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY`` from the environment.
    Sends the mandatory ``IWA-Task-ID`` header on every request and retries
    on 429 / 5xx errors with exponential backoff.
    """

    def __init__(self, timeout: float = 25.0) -> None:
        self.base_url: str = os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.timeout = timeout
        self._client: httpx.Client = httpx.Client(timeout=self.timeout)

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
        retry=retry_if_exception(_is_retryable),
    )
    def chat_completions(
        self,
        *,
        task_id: str,
        messages: list[dict],
        model: str = "gpt-5.2",
        temperature: float = 0.2,
        max_tokens: int = 300,
    ) -> dict:
        """Send a chat-completions request to the LLM gateway.

        Args:
            task_id: IWA task identifier (sent as ``IWA-Task-ID`` header).
            messages: OpenAI-style message list.
            model: Model name (default ``gpt-5.2``).
            temperature: Sampling temperature (ignored for GPT-5.x).
            max_tokens: Maximum tokens in the completion.

        Returns:
            The parsed JSON response dict from the gateway.

        Raises:
            httpx.HTTPStatusError: On non-retryable HTTP errors.
        """
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "IWA-Task-ID": task_id,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body: dict = {"model": model, "messages": messages}

        # GPT-5.x uses max_completion_tokens and does not accept temperature.
        if model.startswith("gpt-5"):
            body["max_completion_tokens"] = max_tokens
        else:
            body["temperature"] = temperature
            body["max_tokens"] = max_tokens

        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        if self._client is not None:
            self._client.close()
