# LLM-duet

Utilities for letting two language models carry on a conversation with each
other. The initial topic is provided by a human and the models take turns
producing responses.

## Features

- `conversation.py` orchestrates a back-and-forth between two models.
- Supports local [OLLAMA](https://ollama.ai) servers via `OllamaClient`.
- Includes an `OpenAIClient` for running the duet through the OpenAI API.

## Usage

### Command line (OLLAMA)

With an OLLAMA server running locally you can start a conversation directly
from the command line:

```bash
python conversation.py "Discuss the future of robotics" --model-a llama2 --model-b mistral --turns 4
```

### From Python using the OpenAI API

```python
from conversation import have_conversation
from openai_client import OpenAIClient

client = OpenAIClient()  # uses the OPENAI_API_KEY environment variable
history = have_conversation("gpt-3.5-turbo", "gpt-4", "Debate the future of AI", client=client)
for model, text in history:
    print(f"{model}: {text}")
```

## Development

The repository contains only a few Python files. See `AGENTS.md` for coding
guidelines.

## License

Distributed under the terms of the MIT license. See `LICENSE` for details.
