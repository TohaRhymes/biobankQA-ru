"""OpenRouter client (OpenAI-compatible chat completions).

OpenRouter exposes the standard OpenAI chat-completions API at
`https://openrouter.ai/api/v1`. Model IDs are namespaced by vendor:
`{vendor}/{model}` and may include a date suffix.
"""

from __future__ import annotations

import os

from openai import OpenAI
from openai import APIError as OpenAIAPIError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import BaseClient, LLMError, LLMResponse


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Verified against https://openrouter.ai/api/v1/models on 2026-06-14.
# Trailing date suffixes are omitted in favor of stable aliases where supported
# by the catalog; if an alias is not recognized, use the dated id explicitly.
OPENROUTER_MODELS = [
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3.7-max",
    "qwen/qwen3.6-35b-a3b",
]


class OpenRouterClient(BaseClient):
    provider_name = "openrouter"
    available_models = OPENROUTER_MODELS

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=OPENROUTER_BASE_URL,
        )

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type((OpenAIAPIError, LLMError)),
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
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": question})
        create_kwargs: dict = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if response_schema is not None:
            # Strict JSON-schema guided generation. OpenRouter forwards this to
            # the underlying provider when supported and gracefully degrades.
            create_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "mcq_answer",
                    "schema": response_schema,
                    "strict": True,
                },
            }
        # Route around AkashML which rejects standards-compliant JSON Schemas
        # (`502 Grammar error: Unimplemented keys: ['uniqueItems']`). Other
        # OpenRouter upstreams handle the schema fine, so we let them serve.
        # Merged with the reasoning-effort passthrough (DeepSeek V4, Qwen3.x);
        # providers that don't support `reasoning` silently drop it.
        extra_body: dict = {"provider": {"ignore": ["AkashML"], "allow_fallbacks": True}}
        if reasoning_effort is not None:
            extra_body["reasoning"] = {"effort": reasoning_effort}
        create_kwargs["extra_body"] = extra_body
        t0 = self._now_ms()
        resp = self._client.chat.completions.create(**create_kwargs)
        latency = self._now_ms() - t0
        # OpenRouter sometimes returns HTTP 200 with `choices=None` when an
        # upstream provider drops the request (rate limit / circuit break /
        # provider error). Surface this as LLMError so tenacity retries
        # instead of crashing with `TypeError: 'NoneType' object is not
        # subscriptable`.
        if resp.choices is None or len(resp.choices) == 0:
            raw_snippet = repr(resp)[:500]
            raise LLMError(
                f"OpenRouter no choices for {model} "
                f"(provider={getattr(resp, 'provider', '?')}; raw={raw_snippet})"
            )
        try:
            text = resp.choices[0].message.content or ""
        except (AttributeError, IndexError) as e:
            raise LLMError(f"OpenRouter malformed response for {model}: {resp}") from e
        # OpenRouter occasionally returns empty content with finish_reason=stop
        # (no exception). Raise so the retry decorator can take another shot.
        if not text.strip():
            try:
                fr = resp.choices[0].finish_reason
            except Exception:
                fr = "?"
            raise LLMError(f"OpenRouter empty content for {model} (finish_reason={fr})")
        usage = resp.usage
        return LLMResponse(
            text=text.strip(),
            latency_ms=latency,
            tokens_in=getattr(usage, "prompt_tokens", None),
            tokens_out=getattr(usage, "completion_tokens", None),
            model=model,
            provider=self.provider_name,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )
