"""Common interface for all LLM provider clients."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


class LLMError(RuntimeError):
    """Raised when a provider returns an error or invalid response."""


@dataclass
class LLMResponse:
    text: str
    latency_ms: int
    tokens_in: int | None = None
    tokens_out: int | None = None
    model: str = ""
    provider: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class BaseClient:
    """Abstract base. Each provider subclass implements `generate`."""

    provider_name: str = ""
    available_models: list[str] = []

    def generate(
        self,
        model: str,
        question: str,
        *,
        system: str | None = None,
        max_tokens: int = 800,
        temperature: float = 0.0,
        timeout: float = 60.0,
        response_schema: dict | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        raise NotImplementedError

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)
