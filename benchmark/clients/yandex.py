"""Yandex AI Studio client (OpenAI-compatible).

Uses the OpenAI-compatible endpoint exposed by Yandex AI Studio at
`https://ai.api.cloud.yandex.net/v1`. Model identifiers are URIs of the form
`gpt://{folder_id}/{model_name}/{version}`.

We use the Responses API (`client.responses.create`) — that's what the working
example from the user shows, and it's the documented path for Yandex.
"""

from __future__ import annotations

import os
import sys

from openai import OpenAI
from openai import APIError as OpenAIAPIError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import BaseClient, LLMError, LLMResponse


# Module-level flag so we only log the json_object-rejection notice once per run.
_YANDEX_JSON_FALLBACK_WARNED = False


YANDEX_BASE_URL = "https://ai.api.cloud.yandex.net/v1"

# Verified on 2026-06-15 by chat.completions probing of the target folder.
# Model URIs (the part after `gpt://{folder}/`).
# Llama / Qwen / DeepSeek were NOT available on this account at probe time —
# only the YandexGPT family and the GPT-OSS open-weight models are exposed.
YANDEX_MODELS = [
    "yandexgpt/latest",          # YandexGPT 5.1 Pro (default alias)
    "yandexgpt-lite/latest",     # YandexGPT 5.1 Lite
    "yandexgpt-32k/latest",      # YandexGPT long-context variant
    "gpt-oss-120b/latest",       # OpenAI GPT-OSS 120B (open weights, hosted by Yandex)
    "gpt-oss-20b/latest",        # OpenAI GPT-OSS 20B
]


class YandexClient(BaseClient):
    provider_name = "yandex"
    available_models = YANDEX_MODELS

    def __init__(
        self,
        api_key: str | None = None,
        folder_id: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ["YANDEX_API_KEY"]
        self.folder_id = folder_id or os.environ["YANDEX_CLOUD_FOLDER"]
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=YANDEX_BASE_URL,
            project=self.folder_id,
        )

    def _full_model_uri(self, model: str) -> str:
        return f"gpt://{self.folder_id}/{model}"

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
        # Yandex's gateway rejects `reasoning` passthrough (verified). Accept
        # the kwarg for interface parity with the OpenRouter client; ignored.
        del reasoning_effort
        # Yandex's OpenAI-compat layer does NOT accept strict json_schema, but
        # the looser `{"type": "json_object"}` is sometimes honoured. Try with
        # it when schema is requested; on any failure, retry once without so we
        # never break Yandex over a guided-generation feature.
        base_kwargs: dict = dict(
            model=self._full_model_uri(model),
            temperature=temperature,
            instructions=system or "",
            input=question,
            max_output_tokens=max_tokens,
            timeout=timeout,
        )
        t0 = self._now_ms()
        if response_schema is not None:
            try:
                resp = self._client.responses.create(
                    **base_kwargs,
                    response_format={"type": "json_object"},
                )
            except Exception as e:  # noqa: BLE001
                global _YANDEX_JSON_FALLBACK_WARNED
                if not _YANDEX_JSON_FALLBACK_WARNED:
                    print(
                        f"  [yandex] response_format=json_object rejected "
                        f"({type(e).__name__}); falling back to plain prompt.",
                        file=sys.stderr,
                    )
                    _YANDEX_JSON_FALLBACK_WARNED = True
                resp = self._client.responses.create(**base_kwargs)
        else:
            resp = self._client.responses.create(**base_kwargs)
        latency = self._now_ms() - t0
        text = (resp.output_text or "").strip()
        usage = getattr(resp, "usage", None)
        tokens_in = getattr(usage, "input_tokens", None) if usage else None
        tokens_out = getattr(usage, "output_tokens", None) if usage else None
        return LLMResponse(
            text=text,
            latency_ms=latency,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=model,
            provider=self.provider_name,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )
