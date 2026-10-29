"""Sber GigaChat client.

Auth flow (OAuth 2.0 client_credentials):
    1. POST https://ngw.devices.sberbank.ru:9443/api/v2/oauth
       Headers: Authorization: Basic {SBER_AUTH_KEY}, RqUID: {uuid4}
       Body: scope={SBER_SCOPE}
       -> {"access_token": "...", "expires_at": <ms>}
    2. POST https://gigachat.devices.sberbank.ru/api/v1/chat/completions
       Headers: Authorization: Bearer {access_token}
       Body: OpenAI-style chat completion

The Russian Trusted Root CA cert must already be appended to certifi (one-time
setup performed by the user via `curl ... >> $(python3 -m certifi)`).
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import BaseClient, LLMError, LLMResponse


GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"

# Verified via GET /api/v1/models on 2026-06-14. In gen-2 the "Lite" tier
# is simply named `GigaChat-2` (no `-Lite` suffix).
AVAILABLE_MODELS = ["GigaChat-2", "GigaChat-2-Pro", "GigaChat-2-Max"]


class GigaChatClient(BaseClient):
    provider_name = "gigachat"
    available_models = AVAILABLE_MODELS

    def __init__(
        self,
        auth_key: str | None = None,
        scope: str | None = None,
    ) -> None:
        self.auth_key = auth_key or os.environ["SBER_AUTH_KEY"]
        self.scope = scope or os.environ["SBER_SCOPE"]
        self._token: str | None = None
        self._token_expires_at_ms: int = 0
        self._http = httpx.Client(timeout=60.0)

    # ----- OAuth -----
    def _token_valid(self) -> bool:
        return self._token is not None and self._token_expires_at_ms - 30_000 > int(time.time() * 1000)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=120),
        retry=retry_if_exception_type((httpx.HTTPError, LLMError)),
        reraise=True,
    )
    def _refresh_token(self) -> None:
        headers = {
            "Authorization": f"Basic {self.auth_key}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        resp = self._http.post(
            GIGACHAT_OAUTH_URL,
            headers=headers,
            data={"scope": self.scope},
        )
        if resp.status_code != 200:
            raise LLMError(f"GigaChat OAuth failed: {resp.status_code} {resp.text[:300]}")
        data = resp.json()
        self._token = data["access_token"]
        self._token_expires_at_ms = int(data.get("expires_at", int(time.time() * 1000) + 30 * 60 * 1000))

    def _ensure_token(self) -> str:
        if not self._token_valid():
            self._refresh_token()
        assert self._token is not None
        return self._token

    # ----- Chat -----
    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=120),
        retry=retry_if_exception_type((httpx.HTTPError, LLMError)),
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
        # GigaChat does not support structured/guided generation via response_format.
        # Accept the kwarg for interface parity and ignore it; the system prompt
        # still asks for JSON output and benchmark.parsing.parse_mcq_letters
        # tolerates either JSON or bare-letter responses.
        # `reasoning_effort` is also accepted for parity but unused (no Sber knob).
        del response_schema, reasoning_effort
        if model not in self.available_models:
            raise LLMError(f"Unknown GigaChat model: {model}. Available: {self.available_models}")
        token = self._ensure_token()
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": question})
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        t0 = self._now_ms()
        resp = self._http.post(
            f"{GIGACHAT_BASE_URL}/chat/completions",
            headers=headers,
            json=body,
            timeout=timeout,
        )
        latency = self._now_ms() - t0
        if resp.status_code == 401:
            # token may have just expired — refresh and retry once more via tenacity
            self._token = None
            raise LLMError(f"GigaChat 401: {resp.text[:200]}")
        if resp.status_code != 200:
            raise LLMError(f"GigaChat {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"GigaChat malformed response: {data}") from e
        usage = data.get("usage", {}) or {}
        return LLMResponse(
            text=text,
            latency_ms=latency,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            model=model,
            provider=self.provider_name,
            raw=data,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> GigaChatClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
