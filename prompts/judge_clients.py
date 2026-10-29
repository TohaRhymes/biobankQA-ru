"""OpenAI client used as LLM-judge for open-ended biobanking answers.

NOT a competitor model — judges other models' responses against a structured
rubric. Default model is `gpt-5` per project requirement (2026-06).
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI
from openai import APIError as OpenAIAPIError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from benchmark.prompts import (
    JUDGE_SYSTEM,
    judge_user_template,
    JUDGE_REASONING_EFFORT,
    JUDGE_MAX_TOKENS,
)

from .base import BaseClient, LLMError, LLMResponse


JUDGE_MODELS = ["gpt-5", "gpt-5-mini"]
DEFAULT_JUDGE_MODEL = "gpt-5"


class OpenAIJudgeClient(BaseClient):
    provider_name = "openai-judge"
    available_models = JUDGE_MODELS

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_JUDGE_MODEL,
        reasoning_effort: str = JUDGE_REASONING_EFFORT,
    ) -> None:
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.model = model
        # GPT-5 spends output budget on chain-of-thought first; "minimal" reserves
        # most of the budget for the actual completion text.
        self.reasoning_effort = reasoning_effort
        self._client = OpenAI(api_key=self.api_key)

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
    ) -> LLMResponse:
        """Plain text completion via Responses API (used for smoke-test).

        GPT-5 family does not accept the `temperature` parameter — omit it for those.
        When `response_schema` is provided, the Responses API is asked to emit
        strict JSON matching the schema (used by external candidate-mode MCQ).
        """
        kwargs: dict = dict(
            model=model,
            instructions=system or "",
            input=question,
            max_output_tokens=max_tokens,
            timeout=timeout,
        )
        if model.startswith("gpt-5"):
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        else:
            kwargs["temperature"] = temperature
        if response_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "mcq_answer",
                    "schema": response_schema,
                    "strict": True,
                },
            }
        t0 = self._now_ms()
        resp = self._client.responses.create(**kwargs)
        latency = self._now_ms() - t0
        text = (resp.output_text or "").strip()
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            latency_ms=latency,
            tokens_in=getattr(usage, "input_tokens", None) if usage else None,
            tokens_out=getattr(usage, "output_tokens", None) if usage else None,
            model=model,
            provider=self.provider_name,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type((OpenAIAPIError, LLMError)),
        reraise=True,
    )
    def judge_open_answer(
        self,
        question: str,
        model_answer: str,
        rubric_hint: str | None = None,
    ) -> dict[str, Any]:
        """Score an open-ended biobanking answer along 5 axes. Returns JSON dict."""
        judge_kwargs: dict = dict(
            model=self.model,
            instructions=JUDGE_SYSTEM,
            input=judge_user_template(question, model_answer, rubric_hint),
            max_output_tokens=JUDGE_MAX_TOKENS,
        )
        if self.model.startswith("gpt-5"):
            judge_kwargs["reasoning"] = {"effort": self.reasoning_effort}
        else:
            judge_kwargs["temperature"] = 0.0
        resp = self._client.responses.create(**judge_kwargs)
        text = (resp.output_text or "").strip()
        # Strip code fences if present
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"Judge returned non-JSON: {text[:300]}") from e
