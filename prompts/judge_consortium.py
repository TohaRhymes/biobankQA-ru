"""Consortium-of-four judge for open-ended biobanking answers.

Per reviewer requirement: judging is done by a consortium of the top
Russia-accessible models plus qwen3.7-max (top candidate on MCQ direct F1 =
0.842), not by an external single judge.

Judges:
  - Sber GigaChat-2-Max
  - Yandex YandexGPT 5.1 Pro (yandexgpt/latest)
  - DeepSeek V4 Pro via OpenRouter
  - Qwen3.7-max via OpenRouter

Each judge scores along the 5-axis rubric independently; we keep all four
sets and report the mean. If any judge fails to return parseable JSON, that
judge is excluded for that item; aggregation uses the survivors.

Self-cell exclusion (candidate = judge) is enforced downstream in
final_benchmark.py aggregation, now covering four model IDs.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from benchmark.prompts import JUDGE_SYSTEM, judge_user_template, JUDGE_MAX_TOKENS

from .gigachat import GigaChatClient
from .yandex import YandexClient
from .openrouter import OpenRouterClient


# (display_name, factory, model_id)
JUDGE_PANEL: list[tuple[str, Any, str]] = [
    ("gigachat:GigaChat-2-Max", GigaChatClient, "GigaChat-2-Max"),
    ("yandex:yandexgpt/latest", YandexClient, "yandexgpt/latest"),
    ("openrouter:deepseek/deepseek-v4-pro", OpenRouterClient, "deepseek/deepseek-v4-pro"),
    ("openrouter:qwen/qwen3.7-max", OpenRouterClient, "qwen/qwen3.7-max"),
]

# v3 rubric (iter4): axes renamed to match the human xlsx column names.
# Old names (v2): actionability, uncertainty_handling.
# New names (v3): sop_applicability, no_uncertainty.
# use_in_work is a new 0/1 binary axis (parallel to the human binary).
SCORE_KEYS = (
    "completeness",
    "safety",
    "sop_applicability",
    "no_uncertainty",
    "hallucination_severity",
    "use_in_work",
)


def _parse_judge_json(text: str) -> dict | None:
    """Extract a JSON object from a judge response, tolerating code fences and
    pre/post chatter."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip("\n").strip()
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _has_all_score_keys(d: dict | None) -> bool:
    if not isinstance(d, dict):
        return False
    return all(k in d for k in SCORE_KEYS)


class JudgeConsortium:
    """Holds long-lived judge clients and exposes a single `judge_open_answer`."""

    def __init__(self) -> None:
        # Warm up openai submodules SYNCHRONOUSLY before any thread runs.
        # yandex uses `client.responses.create`, openrouter uses
        # `client.chat.completions.create`; if two threads first-touch these
        # different submodules at the same time Python's import lock can
        # deadlock (CPython 3.11 ModuleLock race). Importing here removes the
        # race entirely.
        import openai.resources.chat  # noqa: F401
        import openai.resources.responses  # noqa: F401

        self._clients = []
        for name, factory, model in JUDGE_PANEL:
            self._clients.append((name, factory(), model))
        self._executor = ThreadPoolExecutor(max_workers=len(self._clients))

    def _one_judge(self, name: str, client, model: str, question: str, answer: str) -> tuple[str, dict | None, str | None]:
        try:
            resp = client.generate(
                model=model,
                question=judge_user_template(question, answer),
                system=JUDGE_SYSTEM,
                max_tokens=JUDGE_MAX_TOKENS,
                temperature=0.0,
                timeout=90.0,
            )
        except Exception as e:  # noqa: BLE001
            return name, None, f"{type(e).__name__}: {str(e)[:200]}"
        parsed = _parse_judge_json(resp.text)
        if not _has_all_score_keys(parsed):
            return name, None, f"unparseable_or_missing_keys: {(resp.text or '')[:150]!r}"
        return name, parsed, None

    def judge_open_answer(self, question: str, model_answer: str) -> dict:
        """Run the judge panel in parallel and aggregate the scores.

        Panel size is defined by JUDGE_PANEL (currently 4 judges).
        """
        futures = {
            self._executor.submit(self._one_judge, n, c, m, question, model_answer): n
            for (n, c, m) in self._clients
        }
        individual: dict[str, dict | None] = {}
        errors: dict[str, str] = {}
        for fut in as_completed(futures):
            name, parsed, err = fut.result()
            individual[name] = parsed
            if err:
                errors[name] = err

        ok = [v for v in individual.values() if v]
        if not ok:
            return {
                "judges": [n for (n, _, _) in JUDGE_PANEL],
                "individual": individual,
                "errors": errors,
                "aggregated": None,
                "n_successful_judges": 0,
            }

        agg: dict[str, float] = {}
        for k in SCORE_KEYS:
            vals = [v[k] for v in ok if isinstance(v.get(k), (int, float))]
            agg[k] = round(sum(vals) / len(vals), 2) if vals else None
        notes = " | ".join(v.get("notes", "") for v in ok if v.get("notes"))

        return {
            "judges": [n for (n, _, _) in JUDGE_PANEL],
            "individual": individual,
            "errors": errors,
            "aggregated": {**agg, "notes": notes[:600]},
            "n_successful_judges": len(ok),
        }

    def close(self) -> None:
        self._executor.shutdown(wait=False)
