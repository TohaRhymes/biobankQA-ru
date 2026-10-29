"""Pre-flight: do GigaChat-2-Max, YandexGPT/latest and DeepSeek-V4-Pro
actually emit `[src:N]` tags when instructed?

Three hand-crafted (question, fake_context) tuples are sent through the
`RAG_OPEN_SYSTEM` prompt. We measure per-provider compliance — at least
one `[src:N]` or `[src:нет]` token in the response.

Pass criterion: >= 80% of responses include at least one citation.

Output: `benchmark/outputs/rag/citation_preflight.json`.

Usage::

    python -m benchmark.scripts.citation_preflight                  # full
    python -m benchmark.scripts.citation_preflight --skip-yandex    # only Gi+DS
    python -m benchmark.scripts.citation_preflight --dry-run        # render prompts only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from benchmark.rag.prompts import RAG_OPEN_SYSTEM, rag_user_template


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "benchmark" / "outputs" / "rag" / "citation_preflight.json"


# ---------------------------------------------------------------------------
# 3 hand-crafted (question, context_chunks) tuples
# ---------------------------------------------------------------------------
PREFLIGHT_CASES: list[dict] = [
    {
        "case_id": "preflight_consent",
        "question": (
            "Что должен сделать координатор биобанка, если донор отозвал "
            "информированное добровольное согласие после сдачи биоматериала?"
        ),
        "chunks": [
            {
                "text": (
                    "При отзыве ИДС донором биобанк обязан в течение 30 дней "
                    "приостановить любые исследования, использующие данный "
                    "биообразец, и зафиксировать отзыв в реестре согласий."
                ),
                "source_short": "program-ru-2025",
                "section": "Модуль 4: согласия",
                "page": 1,
            },
            {
                "text": (
                    "Биообразец, полученный по отозванному ИДС, подлежит "
                    "уничтожению либо обезличиванию в соответствии с СОП "
                    "учреждения; решение принимает руководитель биобанка."
                ),
                "source_short": "isber-ru-4ed",
                "section": "Глава 4. Этика",
                "page": 42,
            },
        ],
    },
    {
        "case_id": "preflight_temperature_excursion",
        "question": (
            "Морозильник -80°C в архиве дал отказ на 6 часов, температура "
            "поднялась до -55°C. Что делать в первую очередь?"
        ),
        "chunks": [
            {
                "text": (
                    "При температурной экскурсии вне диапазона –80 ± 10°C "
                    "лаборант фиксирует время отклонения, сообщает дежурному "
                    "инженеру и не открывает дверцу до восстановления режима."
                ),
                "source_short": "isber-en-5ed",
                "section": "10.3 Storage Equipment Failure",
                "page": 88,
            },
            {
                "text": (
                    "Все биообразцы, подвергшиеся экскурсии, должны быть "
                    "отмечены в LIMS флагом 'temperature_excursion' с указанием "
                    "длительности и максимальной температуры."
                ),
                "source_short": "isber-en-5ed",
                "section": "10.5 Sample Annotation",
                "page": 91,
            },
        ],
    },
    {
        "case_id": "preflight_no_context",
        "question": (
            "Какой марки морозильник официально рекомендован программой курса "
            "НИИ Отта для хранения сывороток?"
        ),
        "chunks": [
            {
                "text": (
                    "Программа охватывает темы ИДС, СОП, преаналитические "
                    "процедуры, документирование и контроль качества."
                ),
                "source_short": "program-ru-2025",
                "section": "Модуль 1: введение",
                "page": 1,
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Providers to test
# ---------------------------------------------------------------------------
DEFAULT_TARGETS = [
    ("gigachat", "GigaChat-2-Max"),
    ("yandex", "yandexgpt/latest"),
    ("openrouter", "deepseek/deepseek-v4-pro"),
]


CITATION_RE = re.compile(r"\[src:(?:\d+|нет|none)\]", re.IGNORECASE)


def _build_client(provider: str):
    """Lazy import per provider so a missing key for one doesn't kill the others."""
    if provider == "gigachat":
        from benchmark.clients import GigaChatClient
        return GigaChatClient()
    if provider == "yandex":
        from benchmark.clients import YandexClient
        return YandexClient()
    if provider == "openrouter":
        from benchmark.clients import OpenRouterClient
        return OpenRouterClient()
    raise ValueError(f"unknown provider: {provider}")


def _render_prompt(case: dict) -> tuple[str, str]:
    user = rag_user_template(
        case["question"],
        options=None,
        context_chunks=case["chunks"],
        context_found=True,
    )
    return RAG_OPEN_SYSTEM, user


def _has_citation(text: str) -> bool:
    if not text:
        return False
    return bool(CITATION_RE.search(text))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-yandex", action="store_true")
    p.add_argument("--skip-gigachat", action="store_true")
    p.add_argument("--skip-openrouter", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print rendered prompts and exit")
    p.add_argument("--max-tokens", type=int, default=600)
    return p.parse_args()


def main() -> int:
    load_dotenv()
    args = _parse_args()

    targets = list(DEFAULT_TARGETS)
    if args.skip_gigachat:
        targets = [t for t in targets if t[0] != "gigachat"]
    if args.skip_yandex:
        targets = [t for t in targets if t[0] != "yandex"]
    if args.skip_openrouter:
        targets = [t for t in targets if t[0] != "openrouter"]

    if args.dry_run:
        for case in PREFLIGHT_CASES:
            system, user = _render_prompt(case)
            print(f"=== {case['case_id']} ===")
            print("--- SYSTEM ---")
            print(system)
            print("--- USER ---")
            print(user)
            print()
        return 0

    results: list[dict] = []
    for provider, model in targets:
        try:
            client = _build_client(provider)
        except Exception as e:
            print(f"WARN: cannot init {provider}: {e}", file=sys.stderr)
            continue
        for case in PREFLIGHT_CASES:
            system, user = _render_prompt(case)
            t0 = time.time()
            err: str | None = None
            text = ""
            try:
                resp = client.generate(
                    model=model,
                    question=user,
                    system=system,
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                    timeout=90.0,
                )
                text = resp.text
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:300]}"
            results.append(
                {
                    "provider": provider,
                    "model": model,
                    "case_id": case["case_id"],
                    "has_citation": _has_citation(text),
                    "response": text,
                    "error": err,
                    "elapsed_s": round(time.time() - t0, 2),
                }
            )
            ok = "ok" if _has_citation(text) else "MISS"
            tag = ok if err is None else "ERR"
            print(f"  [{provider}/{model}] {case['case_id']}: {tag}", file=sys.stderr)

    # Per-provider compliance rate
    per_provider: dict[str, dict] = {}
    for r in results:
        p = r["provider"]
        slot = per_provider.setdefault(p, {"hits": 0, "n": 0, "errors": 0})
        if r["error"]:
            slot["errors"] += 1
            continue
        slot["n"] += 1
        if r["has_citation"]:
            slot["hits"] += 1
    for p, slot in per_provider.items():
        slot["rate"] = round(slot["hits"] / slot["n"], 3) if slot["n"] else None

    overall_n = sum(s["n"] for s in per_provider.values())
    overall_hits = sum(s["hits"] for s in per_provider.values())
    overall_rate = (overall_hits / overall_n) if overall_n else 0.0
    passed = overall_rate >= 0.80

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "ts": int(time.time()),
                "passed": passed,
                "threshold": 0.80,
                "overall_rate": round(overall_rate, 3),
                "per_provider": per_provider,
                "responses": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(file=sys.stderr)
    for p, slot in per_provider.items():
        print(f"  {p}: {slot['hits']}/{slot['n']} ({slot.get('rate')})", file=sys.stderr)
    print(
        f"OVERALL: {overall_hits}/{overall_n} = {overall_rate:.2%}  threshold 80% — "
        f"{'PASS' if passed else 'FAIL'}",
        file=sys.stderr,
    )
    print(f"WROTE {OUT_PATH}", file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
