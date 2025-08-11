"""Utilities to orchestrate a back-and-forth between two language models.

The default implementation uses a local OLLAMA server, but any client object
providing a ``generate`` method can be supplied, enabling use of remote APIs
such as OpenAI's chat completions.
"""
from __future__ import annotations

from typing import List, Tuple, Protocol, runtime_checkable, cast

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
    system_a: str | None = None,
    system_b: str | None = None,
) -> List[Tuple[str, str]]:
    """Have two models converse by generating responses alternately.

    Parameters
    ----------
    model_a, model_b:
        Names of the models participating in the conversation.
    prompt:
        Initial text sent to both models.
    turns:
        Number of model responses to generate. ``turns`` of ``4`` yields two
        responses from each model.
    client:
        Optional client implementing :class:`LLMClient`. If omitted a new
        :class:`OllamaClient` is created.
    system_a, system_b:
        Optional system prompts prepended to each model's conversation history.

    Returns
    -------
    list of tuple
        ``(model_name, response)`` pairs in the order produced.
    """

    client = cast(LLMClient, client or OllamaClient())

    history: List[Tuple[str, str]] = []
    history_a: List[str] = []
    history_b: List[str] = []
    if system_a:
        history_a.append(system_a)
    if system_b:
        history_b.append(system_b)
    history_a.append(f"user: {prompt}")
    history_b.append(f"user: {prompt}")

    current_model = model_a
    for _ in range(turns):
        current_history = history_a if current_model == model_a else history_b
        response = client.generate(current_model, "\n".join(current_history), stream=False)
        history.append((current_model, response))
        line = f"{current_model}: {response}"
        if current_model == model_a:
            history_b.append(line)
        else:
            history_a.append(line)
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
