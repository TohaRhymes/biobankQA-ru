"""RAG-mode sweep: every model answers every dataset item with retrieved context.

Mirrors `run_direct.py` structure but:
- builds a per-item retrieval cache ONCE at the start of the run (the
  same context is reused across all 12 candidate LLMs — retrieval is
  LLM-independent).
- uses the RAG_*_SYSTEM prompts from `benchmark.rag.prompts`.
- writes to a separate cache namespace and raw file.

Output: `benchmark/outputs/raw/rag_outputs.jsonl` (append-only, idempotent).

Usage::

    python -m benchmark.scripts.run_rag                       # full sweep
    python -m benchmark.scripts.run_rag --limit 5             # smoke
    python -m benchmark.scripts.run_rag --models GigaChat-2,yandexgpt/latest
    python -m benchmark.scripts.run_rag --build-retrieval-only   # populate the cache, no LLM calls
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from benchmark.clients import (
    GigaChatClient,
    OpenRouterClient,
    YandexClient,
)
from benchmark.clients.base import BaseClient
from benchmark.prompts import (
    TEMPERATURE,
    MCQ_MAX_TOKENS,
    MCQ_MAX_TOKENS_REASONING,
    OPEN_MAX_TOKENS,
    OPEN_MAX_TOKENS_REASONING,
)
from benchmark.rag.prompts import (
    RAG_MCQ_ANSWER_SCHEMA,
    RAG_MCQ_SYSTEM,
    RAG_OPEN_ANSWER_SCHEMA,
    RAG_OPEN_SYSTEM,
    rag_user_template,
)
from benchmark.rag.retrieve import (
    FINAL_TOPK_DEFAULT,
    load_cached,
    query_hash,
    retrieve_for_item,
)


# Same reasoning-prefix lists as run_direct (kept in sync by hand).
REASONING_MCQ_PREFIXES = ("gpt-oss-", "qwen/qwen3.6-")
REASONING_OPEN_PREFIXES = ("gpt-oss-", "deepseek/deepseek-v4-", "qwen/qwen3.6-")
OPENROUTER_REASONING_EFFORT = "low"


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "data" / "processed" / "biobankqa_v0_executable.jsonl"
OUT_DIR = REPO_ROOT / "benchmark" / "outputs" / "raw"
RAW_FILE = OUT_DIR / "rag_outputs.jsonl"
CACHE_DIR = OUT_DIR / "cache_rag"

CONCURRENCY = {
    "gigachat": 3,
    "yandex": 3,
    "openrouter": 6,
}


# DA review dd382f8, fix #2c/d: post-hoc citation audit.
#
# Matches `src:N`, `[src:N]`, `src:нет`, `[src:нет]` anywhere in the raw
# response text. We do this against `resp.text` (not the parsed schema)
# so we catch citations that appear in the answer string itself, not just
# the structured citations[] field.
_CITATION_RE = re.compile(r"\[?src:(\d+|нет)\]?")


def _audit_citations(text: str | None, n_chunks: int) -> tuple[bool, list[str]]:
    """Return (has_citation, invalid_citations).

    `has_citation`     — True iff at least one `src:N` / `src:нет` token
                         appears in the raw response.
    `invalid_citations` — list of `src:N` strings where N is out of range
                          [1..n_chunks]. `src:нет` is always valid.
    """
    if not text:
        return False, []
    hits = _CITATION_RE.findall(text)
    if not hits:
        return False, []
    invalid: list[str] = []
    seen_invalid: set[str] = set()
    for h in hits:
        if h == "нет":
            continue
        try:
            n = int(h)
        except ValueError:
            continue
        if n < 1 or n > n_chunks:
            tag = f"src:{h}"
            if tag not in seen_invalid:
                seen_invalid.add(tag)
                invalid.append(tag)
    return True, invalid


@dataclass
class TaskSpec:
    item: dict
    provider: str
    model: str

    @property
    def cache_key(self) -> str:
        model_safe = re.sub(r"[^A-Za-z0-9._-]", "_", self.model)
        return f"{self.provider}__{model_safe}__{self.item['id']}.json"


def _build_query_for_item(item: dict) -> str:
    # IMPORTANT (DA review dd382f8, fix #1): MCQ retrieval uses ONLY the
    # question text — never `options_shuffled`. Concatenating shuffled
    # options made the query-hash dependent on the shuffle salt, which
    # busted the retrieval cache between reseeds and (worse) leaked the
    # distractor surface forms into BM25/dense scoring. Open-ended items
    # already used the bare question; both branches now agree.
    return item["question"]


def _build_prompt(item: dict, retrieval: dict) -> tuple[str, str]:
    """Returns (system, user_prompt) for the RAG run.

    `retrieval` is the cache record returned by `retrieve.retrieve(...)`.
    """
    chunks = retrieval.get("chunks", [])
    context_found = bool(retrieval.get("context_found"))
    if item["type"] == "mcq":
        opts = item.get("options_shuffled") or item.get("options") or {}
        user = rag_user_template(item["question"], opts, chunks, context_found=context_found)
        return RAG_MCQ_SYSTEM, user
    user = rag_user_template(item["question"], None, chunks, context_found=context_found)
    return RAG_OPEN_SYSTEM, user


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


def _prebuild_retrieval(items: list[dict], k: int) -> dict[str, dict]:
    """Build (or reuse) the per-item retrieval cache before the LLM sweep.

    Returns: mapping `item_id -> retrieval record`.
    """
    by_item: dict[str, dict] = {}
    miss = 0
    short_q_warned = 0
    for it in items:
        q = _build_query_for_item(it)
        # DA review dd382f8, fix #1: defensive length check. Now that MCQ
        # queries are question-only (no option-text padding), a degenerate
        # 5-char question would silently produce a near-empty BM25 query.
        # Warn loudly but don't crash — the retrieval will still run.
        if it["type"] == "mcq" and len(q.strip()) < 10:
            print(
                f"  WARN: MCQ item {it.get('id')} has short retrieval query "
                f"(len={len(q.strip())}): {q!r}",
                file=sys.stderr,
            )
            short_q_warned += 1
        qhash = query_hash(q)
        cached = load_cached(qhash)
        if cached is None:
            cached = retrieve_for_item(it, k=k)
            miss += 1
        by_item[it["id"]] = cached
    print(
        f"Retrieval cache: {len(items) - miss}/{len(items)} hit, {miss} freshly computed.",
        file=sys.stderr,
    )
    if short_q_warned:
        print(
            f"  ({short_q_warned} MCQ item(s) had query length < 10 chars)",
            file=sys.stderr,
        )
    return by_item


def _run_one_sync(client: BaseClient, task: TaskSpec, retrieval: dict) -> dict:
    cache_path = CACHE_DIR / task.cache_key
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    system, user_prompt = _build_prompt(task.item, retrieval)
    if task.item["type"] == "mcq":
        is_reasoning = task.model.startswith(REASONING_MCQ_PREFIXES)
        max_tokens = MCQ_MAX_TOKENS_REASONING if is_reasoning else MCQ_MAX_TOKENS
        # RAG MCQ asks for letters + citations -> bump the non-reasoning ceiling
        # so short JSON-only ceiling doesn't truncate the citation field.
        if not is_reasoning:
            max_tokens = max(max_tokens, 200)
        response_schema = RAG_MCQ_ANSWER_SCHEMA
    else:
        is_reasoning = task.model.startswith(REASONING_OPEN_PREFIXES)
        max_tokens = OPEN_MAX_TOKENS_REASONING if is_reasoning else OPEN_MAX_TOKENS
        # DA review dd382f8, fix #2b: open items now get a structured schema
        # too (answer + citations[]) so citation presence is enforced upstream
        # of the post-hoc regex check.
        response_schema = RAG_OPEN_ANSWER_SCHEMA

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
            response_schema=response_schema,
            reasoning_effort=reasoning_effort,
        )
        n_chunks_shown = len(retrieval.get("chunks", []))
        has_cit, invalid_cits = _audit_citations(resp.text, n_chunks_shown)
        rec = {
            "item_id": task.item["id"],
            "item_type": task.item["type"],
            "provider": task.provider,
            "model": task.model,
            "mode": "rag",
            "response": resp.text,
            "latency_ms": resp.latency_ms,
            "tokens_in": resp.tokens_in,
            "tokens_out": resp.tokens_out,
            "retrieved_chunk_ids": [c["chunk_id"] for c in retrieval.get("chunks", [])],
            "retrieval_context_found": bool(retrieval.get("context_found")),
            "retrieval_top_score": retrieval.get("top_score"),
            "retrieval_query_hash": retrieval.get("query_hash"),
            # DA review dd382f8, fix #2c/d: persist citation audit alongside
            # the raw response so the judge / reporter can split metrics by
            # citation presence without re-parsing.
            "has_citation": has_cit,
            "invalid_citations": invalid_cits,
            "error": None,
            "ts": int(time.time()),
        }
    except Exception as e:  # noqa: BLE001
        rec = {
            "item_id": task.item["id"],
            "item_type": task.item["type"],
            "provider": task.provider,
            "model": task.model,
            "mode": "rag",
            "response": None,
            "latency_ms": None,
            "tokens_in": None,
            "tokens_out": None,
            "retrieved_chunk_ids": [c["chunk_id"] for c in retrieval.get("chunks", [])],
            "retrieval_context_found": bool(retrieval.get("context_found")),
            "retrieval_top_score": retrieval.get("top_score"),
            "retrieval_query_hash": retrieval.get("query_hash"),
            # Keep the field schema consistent across success/failure records.
            "has_citation": False,
            "invalid_citations": [],
            "error": f"{type(e).__name__}: {str(e)[:300]}",
            "ts": int(time.time()),
        }
    if rec["error"] is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        sys.stderr.write(".")
    else:
        sys.stderr.write("e")
    sys.stderr.flush()
    return rec


async def _run_all(
    tasks: list[TaskSpec],
    clients: dict[str, BaseClient],
    by_item: dict[str, dict],
) -> list[dict]:
    semaphores = {p: asyncio.Semaphore(n) for p, n in CONCURRENCY.items()}
    pool = ThreadPoolExecutor(max_workers=sum(CONCURRENCY.values()))
    loop = asyncio.get_running_loop()
    done_count = 0

    async def wrapped(task: TaskSpec) -> dict:
        nonlocal done_count
        sem = semaphores[task.provider]
        retrieval = by_item[task.item["id"]]
        async with sem:
            rec = await loop.run_in_executor(
                pool, _run_one_sync, clients[task.provider], task, retrieval
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
    cached = _load_cache_records()
    merged: dict[tuple[str, str, str], dict] = {}
    for rec in cached:
        key = (rec.get("provider"), rec.get("model"), rec.get("item_id"))
        merged[key] = rec
    for rec in results:
        key = (rec.get("provider"), rec.get("model"), rec.get("item_id"))
        merged[key] = rec
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
    parser.add_argument("--k", type=int, default=FINAL_TOPK_DEFAULT, help="Top-K retrieval")
    parser.add_argument(
        "--build-retrieval-only",
        action="store_true",
        help="Populate the retrieval cache for every item, then exit (no LLM calls).",
    )
    parser.add_argument(
        "--from-cache-only",
        action="store_true",
        help="Build rag_outputs.jsonl from on-disk LLM cache, no API calls.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
    print(f"Dataset: {len(items)} items.", file=sys.stderr)

    # Step 1: per-item retrieval cache (LLM-independent).
    by_item = _prebuild_retrieval(items, k=args.k)
    if args.build_retrieval_only:
        print("Retrieval cache built; exiting (--build-retrieval-only).", file=sys.stderr)
        return 0

    # Step 2: LLM sweep, reusing the retrieval cache across all candidates.
    tasks = _build_tasks(items, filter_models)
    print(f"Total LLM tasks: {len(tasks)}", file=sys.stderr)
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
        results = asyncio.run(_run_all(tasks, clients, by_item))
    except KeyboardInterrupt:
        interrupted = True
        n_cached_now = sum(1 for p in CACHE_DIR.glob("*.json"))
        print(
            f"\nInterrupted. {n_cached_now} items in cache. "
            f"Re-run the same command to resume.",
            file=sys.stderr,
        )
    elapsed = time.time() - t0

    n_written = _write_aggregate(results)

    n_ok = sum(1 for r in results if r["error"] is None)
    print(file=sys.stderr)
    print(f"WROTE {RAW_FILE} ({n_written} records, union of run + cache)", file=sys.stderr)
    print(f"  this run: {n_ok}/{len(results)} successful  |  elapsed {elapsed:.1f}s", file=sys.stderr)

    if interrupted:
        return 130
    return 0 if n_ok > 0 or n_written > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
