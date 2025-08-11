"""Client utilities for interacting with a local OLLAMA server."""
from __future__ import annotations

import os
from typing import List, Dict, Optional

try:  # pragma: no cover - import is trivial
    import requests  # type: ignore
except Exception:  # pragma: no cover - handled in generate
    requests = None  # type: ignore


class OllamaClient:
    """A small wrapper around the OLLAMA HTTP API.

    Parameters
    ----------
    base_url:
        URL where the OLLAMA server is accessible. If omitted the constructor
        reads the ``OLLAMA_HOST`` environment variable and falls back to
        ``"http://localhost:11434"`` when the variable is not set.
    """

    def __init__(self, base_url: str | None = None) -> None:
        if base_url is None:
            base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs: object,
    ) -> str:
        """Generate a completion from a model using the /api/generate endpoint.

        Parameters
        ----------
        model:
            The name of the model to query.
        prompt:
            Prompt text to send to the model.
        stream:
            Whether to use streaming responses. Streaming is disabled by default
            because this client collects the full response before returning.
        kwargs:
            Additional keyword arguments.
            options:
                Additional options passed directly to the API.
            timeout:
                Timeout for the HTTP request in seconds. Defaults to 60.
        """

        url = f"{self.base_url}/api/generate"
        payload: Dict = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        options: Dict | None = kwargs.get("options")  # type: ignore[assignment]
        timeout = int(kwargs.get("timeout", 60))
        if options:
            payload["options"] = options

        if requests is None:  # pragma: no cover - network library missing
            raise ImportError("The 'requests' package is required to call the OLLAMA API")

        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
