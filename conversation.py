"""Utilities to orchestrate a back-and-forth between two language models.

The default implementation uses a local OLLAMA server, but any client object
providing a ``generate`` method can be supplied, enabling use of remote APIs
such as OpenAI's chat completions.
"""
from __future__ import annotations

from typing import List, Tuple, Protocol, runtime_checkable

from ollama_client import OllamaClient


@runtime_checkable
class LLMClient(Protocol):
    """Protocol describing the methods a language model client must provide."""

    def generate(self, model: str, prompt: str, stream: bool = False, **kwargs: object) -> str:
        """Return a completion for ``prompt`` from ``model``."""


def have_conversation(
    model_a: str,
    model_b: str,
    prompt: str,
    turns: int = 4,
    client: LLMClient | None = None,
) -> List[Tuple[str, str]]:
    """Have two models converse by generating responses alternately.

    Parameters
    ----------
    model_a, model_b:
        Names of the models participating in the conversation.
    prompt:
        Initial text to seed the conversation.
    turns:
        Number of turns in the conversation. Each turn represents a single
        model response. Thus, ``turns`` of 4 will produce two responses from each
        model.
    client:
        Optional client implementing :class:`LLMClient`. If omitted a new
        :class:`OllamaClient` will be created.

    Returns
    -------
    list of tuple
        A list of ``(model_name, response)`` pairs in the order they were
        produced.
    """

    if client is None:
        client = OllamaClient()

    history: List[Tuple[str, str]] = []
    conversation_context = prompt
    current_model = model_a

    for _ in range(turns):
        response = client.generate(current_model, conversation_context, stream=False)
        history.append((current_model, response))
        conversation_context += f"\n{current_model}: {response}"
        current_model = model_b if current_model == model_a else model_a

    return history


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Make two OLLAMA models talk to each other.")
    parser.add_argument("prompt", help="Initial prompt to start the conversation")
    parser.add_argument("--model-a", default="llama2", help="Name of the first model")
    parser.add_argument("--model-b", default="llama2", help="Name of the second model")
    parser.add_argument("--turns", type=int, default=4, help="Number of turns in the conversation")
    args = parser.parse_args()

    history = have_conversation(args.model_a, args.model_b, args.prompt, args.turns)
    for model, text in history:
        print(f"{model}: {text}")


if __name__ == "__main__":
    main()
