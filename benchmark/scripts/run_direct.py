"""Direct-mode sweep: every model answers every dataset item, no extra context.

Output: `benchmark/outputs/raw/raw_outputs.jsonl` (append-only, idempotent).

Concurrency:
- per-provider semaphore (Yandex/Sber API throttling differs from OpenRouter).
- on-disk cache `outputs/raw/cache/{provider}__{model_safe}__{item_id}.json`.

Usage:
    python -m benchmark.scripts.run_direct
    python -m benchmark.scripts.run_direct --limit 5      # smoke
    python -m benchmark.scripts.run_direct --models GigaChat-2,yandexgpt/latest
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

from benchmark.clients import (
    GigaChatClient,
    OpenRouterClient,
    YandexClient,
)
from benchmark.clients.base import BaseClient
from benchmark.prompts import (
    MCQ_SYSTEM,
    OPEN_SYSTEM,
    mcq_user_template,
    open_user_template,
    TEMPERATURE,
    MCQ_MAX_TOKENS,
    MCQ_MAX_TOKENS_REASONING,
    OPEN_MAX_TOKENS,
    OPEN_MAX_TOKENS_REASONING,
    MCQ_ANSWER_SCHEMA,
)


# Reasoning-model classification: which model prefixes need a higher max-token
# budget (to fit hidden CoT) and which support `reasoning.effort` on OpenRouter.
REASONING_MCQ_PREFIXES = ("gpt-oss-", "qwen/qwen3.6-")  # qwen3.6 emits hidden CoT even on MCQ
REASONING_OPEN_PREFIXES = ("gpt-oss-", "deepseek/deepseek-v4-", "qwen/qwen3.6-")
OPENROUTER_REASONING_EFFORT = "low"  # try lower effort first; if quality drops we revisit


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "data" / "processed" / "biobankqa_v0_executable.jsonl"
OUT_DIR = REPO_ROOT / "benchmark" / "outputs" / "raw"
RAW_FILE = OUT_DIR / "raw_outputs.jsonl"
CACHE_DIR = OUT_DIR / "cache"

# Concurrency per provider — tuned for free/light tariffs.
CONCURRENCY = {
    "gigachat": 3,
    "yandex": 3,
    "openrouter": 6,
}

@dataclass
class TaskSpec:
    item: dict
    provider: str
    model: str

    @property
    def cache_key(self) -> str:
        model_safe = re.sub(r"[^A-Za-z0-9._-]", "_", self.model)
        return f"{self.provider}__{model_safe}__{self.item['id']}.json"


def _build_prompt(item: dict) -> tuple[str, str]:
    """Returns (system, user_prompt). MCQ uses the SHUFFLED option layout so
    that every model sees the same de-biased presentation."""
    if item["type"] == "mcq":
        opts = item.get("options_shuffled") or item.get("options") or {}
        return MCQ_SYSTEM, mcq_user_template(item["question"], opts)
    return OPEN_SYSTEM, open_user_template(item["question"])


def _load_dataset(limit: int | None) -> list[dict]:
    items = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        items = items[:limit]
    return items


def _build_tasks(items: list[dict], filter_models: set[str] | None) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    providers: list[tuple[str, list[str]]] = [
        ("gigachat", GigaChatClient.available_models),
        ("yandex", YandexClient.available_models),
        ("openrouter", OpenRouterClient.available_models),
    ]
    for provider, models in providers:
        for model in models:
            if filter_models and model not in filter_models:
                continue
            for item in items:
                tasks.append(TaskSpec(item=item, provider=provider, model=model))
    return tasks


def _run_one_sync(client: BaseClient, task: TaskSpec) -> dict:
    cache_path = CACHE_DIR / task.cache_key
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    system, user_prompt = _build_prompt(task.item)
    if task.item["type"] == "mcq":
        is_reasoning = task.model.startswith(REASONING_MCQ_PREFIXES)
        max_tokens = MCQ_MAX_TOKENS_REASONING if is_reasoning else MCQ_MAX_TOKENS
    else:
        is_reasoning = task.model.startswith(REASONING_OPEN_PREFIXES)
        max_tokens = OPEN_MAX_TOKENS_REASONING if is_reasoning else OPEN_MAX_TOKENS

    reasoning_effort = (
        OPENROUTER_REASONING_EFFORT
        if task.provider == "openrouter" and is_reasoning
        else None
    )

    try:
        resp = client.generate(
            model=task.model,
            question=user_prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=TEMPERATURE,
            timeout=90.0,
            response_schema=MCQ_ANSWER_SCHEMA if task.item["type"] == "mcq" else None,
            reasoning_effort=reasoning_effort,
        )
        rec = {
            "item_id": task.item["id"],
            "item_type": task.item["type"],
            "provider": task.provider,
            "model": task.model,
            "mode": "direct",
            "response": resp.text,
            "latency_ms": resp.latency_ms,
            "tokens_in": resp.tokens_in,
            "tokens_out": resp.tokens_out,
            "error": None,
            "ts": int(time.time()),
        }
    except Exception as e:  # noqa: BLE001
        rec = {
            "item_id": task.item["id"],
            "item_type": task.item["type"],
            "provider": task.provider,
            "model": task.model,
            "mode": "direct",
            "response": None,
            "latency_ms": None,
            "tokens_in": None,
            "tokens_out": None,
            "error": f"{type(e).__name__}: {str(e)[:300]}",
            "ts": int(time.time()),
        }
    # Cache ONLY on success. Errors (rate limits, timeouts, etc.) are returned
    # but not persisted, so the next run will retry them as a cache miss.
    if rec["error"] is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        sys.stderr.write(".")
    else:
        sys.stderr.write("e")
    sys.stderr.flush()
    return rec


async def _run_all(tasks: list[TaskSpec], clients: dict[str, BaseClient]) -> list[dict]:
    semaphores = {p: asyncio.Semaphore(n) for p, n in CONCURRENCY.items()}
    pool = ThreadPoolExecutor(max_workers=sum(CONCURRENCY.values()))
    loop = asyncio.get_running_loop()
    results: list[dict] = []
    done_count = 0

    async def wrapped(task: TaskSpec) -> dict:
        nonlocal done_count
        sem = semaphores[task.provider]
        async with sem:
            rec = await loop.run_in_executor(
                pool, _run_one_sync, clients[task.provider], task
            )
        done_count += 1
        if done_count % 10 == 0 or done_count == len(tasks):
            ok = "OK" if rec["error"] is None else "FAIL"
            print(
                f"  [{done_count}/{len(tasks)}] {rec['provider']}/{rec['model']}/{rec['item_id']} {ok}",
                file=sys.stderr,
            )
        return rec

    results = await asyncio.gather(*[wrapped(t) for t in tasks])
    pool.shutdown(wait=True)
    return results


def _load_cache_records() -> list[dict]:
    """Read every *.json file in CACHE_DIR and return the parsed records."""
    if not CACHE_DIR.exists():
        return []
    out: list[dict] = []
    for p in sorted(CACHE_DIR.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: failed to read cache {p.name}: {e}", file=sys.stderr)
    return out


def _write_aggregate(results: list[dict]) -> int:
    """Write RAW_FILE as the union of this-run results and on-disk cache.
    Dedupe by (provider, model, item_id); prefer the this-run record."""
    cached = _load_cache_records()
    merged: dict[tuple[str, str, str], dict] = {}
    for rec in cached:
        key = (rec.get("provider"), rec.get("model"), rec.get("item_id"))
        merged[key] = rec
    for rec in results:
        key = (rec.get("provider"), rec.get("model"), rec.get("item_id"))
        merged[key] = rec  # this-run wins on collision
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with RAW_FILE.open("w", encoding="utf-8") as f:
        for rec in merged.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(merged)


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Use only first N items (smoke)")
    parser.add_argument("--models", type=str, default="", help="Comma-separated model subset")
    parser.add_argument(
        "--from-cache-only",
        action="store_true",
        help="Don't call any API; build raw_outputs.jsonl from the existing cache only.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # --from-cache-only: no clients, no API calls, no dataset traversal needed.
    if args.from_cache_only:
        cached = _load_cache_records()
        with RAW_FILE.open("w", encoding="utf-8") as f:
            for rec in cached:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"Wrote {len(cached)} cached records, no API calls made.",
            file=sys.stderr,
        )
        print(f"WROTE {RAW_FILE}", file=sys.stderr)
        return 0

    filter_models = {m.strip() for m in args.models.split(",") if m.strip()} or None

    items = _load_dataset(args.limit)
    tasks = _build_tasks(items, filter_models)
    print(f"Dataset: {len(items)} items; total tasks: {len(tasks)}", file=sys.stderr)

    # Resume status: how many tasks already have a cache hit?
    n_total = len(tasks)
    n_cached = sum(1 for t in tasks if (CACHE_DIR / t.cache_key).exists())
    print(
        f"Resume status: {n_cached}/{n_total} items already in cache, "
        f"{n_total - n_cached} to run",
        file=sys.stderr,
    )

    clients: dict[str, BaseClient] = {
        "gigachat": GigaChatClient(),
        "yandex": YandexClient(),
        "openrouter": OpenRouterClient(),
    }

    t0 = time.time()
    interrupted = False
    results: list[dict] = []
    try:
        results = asyncio.run(_run_all(tasks, clients))
    except KeyboardInterrupt:
        interrupted = True
        n_cached_now = sum(1 for p in CACHE_DIR.glob("*.json"))
        print(
            f"\nInterrupted. {n_cached_now} items in cache. "
            f"Re-run the same command tomorrow to resume.",
            file=sys.stderr,
        )
    elapsed = time.time() - t0

    n_written = _write_aggregate(results)

    n_ok = sum(1 for r in results if r["error"] is None)
    print(file=sys.stderr)
    print(f"WROTE {RAW_FILE} ({n_written} records, union of run + cache)", file=sys.stderr)
    print(f"  this run: {n_ok}/{len(results)} successful  |  elapsed {elapsed:.1f}s", file=sys.stderr)

    if interrupted:
        return 130  # SIGINT convention
    return 0 if n_ok > 0 or n_written > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
