"""Answer-generation client abstraction. Swap providers (hosted Claude, a local
Ollama model, ...) by implementing AnswerGenerator - same pattern as the Embedder
interface in ingest/embed.py. Provider is chosen by LLM_PROVIDER in .env.

Usage:
    from llm import get_generator, LLMError
    generator = get_generator()
    answer = generator.generate(system_prompt, user_content)
"""

import logging
from abc import ABC, abstractmethod

from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)


class LLMError(Exception):
    """Raised by any AnswerGenerator on failure. `status_code` is the HTTP status
    the API layer should return for it."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


class AnswerGenerator(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_content: str) -> str:
        ...


class ClaudeGenerator(AnswerGenerator):
    """Hosted Claude via the Anthropic API. Highest answer quality; requires
    ANTHROPIC_API_KEY and network access. Default provider - see README Decisions."""

    def __init__(self, model: str = CLAUDE_MODEL, api_key: str | None = ANTHROPIC_API_KEY):
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set", status_code=500)
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_content: str) -> str:
        import anthropic

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                output_config={"effort": "low"},  # simple grounded Q&A, not deep reasoning
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
        except anthropic.AuthenticationError:
            raise LLMError("Invalid ANTHROPIC_API_KEY", status_code=500)
        except anthropic.RateLimitError:
            raise LLMError("Claude API rate limited, try again shortly", status_code=429)
        except anthropic.APIStatusError as e:
            raise LLMError(f"Claude API error: {e.message}", status_code=502)
        except anthropic.APIConnectionError:
            raise LLMError("Could not reach Claude API", status_code=502)

        if response.stop_reason == "refusal":
            raise LLMError("Claude declined to answer this request", status_code=502)

        return next((b.text for b in response.content if b.type == "text"), "")


class OllamaGenerator(AnswerGenerator):
    """Local model via Ollama (https://ollama.com). Free, no API key, no network
    dependency, lower answer quality than Claude. Requires `ollama serve` running
    and the model already pulled (`ollama pull <model>`)."""

    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, system_prompt: str, user_content: str) -> str:
        import requests

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
        }
        logging.info("Ollama request -> %s/api/chat: %s", self.base_url, payload)

        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        except requests.exceptions.ConnectionError:
            raise LLMError(
                f"Could not reach Ollama at {self.base_url} - is `ollama serve` running?",
                status_code=502,
            )

        if response.status_code == 404:
            raise LLMError(
                f"Ollama model '{self.model}' not found - run `ollama pull {self.model}`",
                status_code=500,
            )
        if not response.ok:
            raise LLMError(f"Ollama error: {response.text}", status_code=502)

        return response.json().get("message", {}).get("content", "")


def get_generator() -> AnswerGenerator:
    """Factory: returns the AnswerGenerator configured by LLM_PROVIDER in .env."""
    if LLM_PROVIDER == "ollama":
        return OllamaGenerator()
    if LLM_PROVIDER == "anthropic":
        return ClaudeGenerator()
    raise LLMError(
        f"Unknown LLM_PROVIDER '{LLM_PROVIDER}' - use 'anthropic' or 'ollama'", status_code=500
    )
