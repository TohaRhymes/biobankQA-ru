"""Anthropic client used as a CANDIDATE model in the "external" run.

Compares Russia-accessible LLMs against Claude. NOT a judge — Claude is
evaluated head-to-head with the Russian models. Default model is the latest
Anthropic flagship as of 2026-06.
"""

from __future__ import annotations

import json
import os

import anthropic
from anthropic import APIError as AnthropicAPIError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import BaseClient, LLMError, LLMResponse


AVAILABLE_MODELS = ["claude-opus-4-7"]
DEFAULT_MODEL = "claude-opus-4-7"


class AnthropicClient(BaseClient):
    provider_name = "anthropic"
    available_models = AVAILABLE_MODELS

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]
        self._client = anthropic.Anthropic(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type((AnthropicAPIError, LLMError)),
        reraise=True,
    )
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
        del reasoning_effort  # parity; not used here
        kwargs: dict = dict(
            model=model,
            messages=[{"role": "user", "content": question}],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        if system:
            kwargs["system"] = system
        if response_schema is not None:
            # Force structured output via tool-use: Anthropic returns the JSON
            # in `tool_use.input`. We then render it as a JSON string into
            # `LLMResponse.text` so the downstream parser sees a JSON object.
            kwargs["tools"] = [
                {
                    "name": "answer",
                    "description": "Submit the structured MCQ answer.",
                    "input_schema": response_schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": "answer"}
        t0 = self._now_ms()
        resp = self._client.messages.create(**kwargs)
        latency = self._now_ms() - t0
        try:
            if response_schema is not None:
                tool_block = next(
                    (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
                    None,
                )
                if tool_block is None:
                    raise LLMError(
                        f"Anthropic returned no tool_use block for {model}: {resp}"
                    )
                text = json.dumps(tool_block.input, ensure_ascii=False)
            else:
                text = "".join(
                    block.text for block in resp.content if getattr(block, "type", None) == "text"
                )
        except (AttributeError, TypeError) as e:
            raise LLMError(f"Anthropic malformed response for {model}: {resp}") from e
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text.strip(),
            latency_ms=latency,
            tokens_in=getattr(usage, "input_tokens", None) if usage else None,
            tokens_out=getattr(usage, "output_tokens", None) if usage else None,
            model=model,
            provider=self.provider_name,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )
