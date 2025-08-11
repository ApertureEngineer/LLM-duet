"""Client utilities for interacting with the OpenAI API."""
from __future__ import annotations

from typing import Optional

try:  # pragma: no cover - import is trivial
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - handled in generate
    OpenAI = None  # type: ignore


class OpenAIClient:
    """Minimal wrapper around the OpenAI chat completions API.

    Parameters
    ----------
    api_key:
        API key used for authentication. If omitted, the ``OPENAI_API_KEY``
        environment variable is used.
    base_url:
        Optional alternative base URL for the API.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        if OpenAI is None:  # pragma: no cover - network library missing
            raise ImportError("The 'openai' package is required to call the OpenAI API")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, model: str, prompt: str, stream: bool = False, **_: object) -> str:
        """Generate a completion from the OpenAI API."""

        if stream:  # pragma: no cover - streaming not implemented
            raise NotImplementedError("Streaming responses are not supported")
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

