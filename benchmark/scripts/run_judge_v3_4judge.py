"""4-judge consortium scoring on the full iter3 open-cell set.

Panel: GigaChat-2-Max, YandexGPT/latest, DeepSeek-V4-Pro, Qwen3.7-max.
Scope: every open cell in iter3 (12 candidate models × 2 modes × 14 tasks =
336 cells). This is the canonical auto-judge dataset the paper's §5 numbers
now rest on.

Source outputs (identical to full-bench responses per verification):
  data/processed/expert_review/iterations/iter3/raw_outputs.jsonl  (direct)
  data/processed/expert_review/iterations/iter3/rag_outputs.jsonl  (rag)

Outputs:
  data/processed/expert_review/iterations/iter3/scored_judge_v3_4judge.jsonl
  data/processed/expert_review/iterations/iter3/judge_cache_v3_4judge/*.json
  data/processed/expert_review/iterations/iter3/judge_v3_4judge_errors.jsonl

The cache dir is distinct from `judge_cache_v3/` (3-judge) so both artifacts
can coexist for the audit comparison in the paper.

Run from repo root:
    .venv/bin/python -m benchmark.scripts.run_judge_v3_4judge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from benchmark.clients import JudgeConsortium


REPO_ROOT = Path(__file__).resolve().parents[2]
ITER3_DIR = REPO_ROOT / "data" / "processed" / "expert_review" / "iterations" / "iter3"
DATASET = REPO_ROOT / "data" / "processed" / "biobankqa_v0_executable.jsonl"
RAW_DIRECT = ITER3_DIR / "raw_outputs.jsonl"
RAW_RAG = ITER3_DIR / "rag_outputs.jsonl"

SCORED_OUT = ITER3_DIR / "scored_judge_v3_4judge.jsonl"
JUDGE_CACHE = ITER3_DIR / "judge_cache_v3_4judge"
ERROR_LOG = ITER3_DIR / "judge_v3_4judge_errors.jsonl"

JUDGE_CONCURRENCY = 6

EXPECTED_JUDGES = 4


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _judge_cache_key(rec: dict) -> str:
    mode = rec.get("mode") or "direct"
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", rec["model"])
    return f"{rec['provider']}__{safe_model}__{rec['item_id']}__{mode}.json"


def _cache_is_complete(cached: dict) -> bool:
    """Cache entry counts as complete only if all EXPECTED_JUDGES returned a
    non-null score dict. Legacy 3-judge cache and partial-panel entries (where
    some judge hit 402/429 mid-run) are treated as incomplete so they get
    re-scored on the next pass."""
    ind = cached.get("open_scores_individual") or {}
    non_null = sum(1 for v in ind.values() if v)
    return non_null >= EXPECTED_JUDGES


def _judge_one(
    consortium: JudgeConsortium,
    rec: dict,
    item: dict,
    cache_dir: Path,
    error_log_path: Path,
) -> dict:
    cache_path = cache_dir / _judge_cache_key(rec)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if _cache_is_complete(cached):
            return cached
    upstream_empty = rec.get("error") is not None or not rec.get("response")
    if upstream_empty:
        out = {
            "open_scores": None,
            "open_scores_individual": None,
            "n_successful_judges": 0,
            "judge_notes": "model_error_or_empty",
            "error": rec.get("error"),
        }
        sys.stderr.write("u")
        sys.stderr.flush()
        return out
    try:
        verdict = consortium.judge_open_answer(
            question=item["question"],
            model_answer=rec["response"],
        )
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:200]}"
        with error_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": _judge_cache_key(rec), "error": err}) + "\n")
        sys.stderr.write("E")
        sys.stderr.flush()
        return {
            "open_scores": None,
            "open_scores_individual": None,
            "n_successful_judges": 0,
            "judge_notes": None,
            "error": err,
        }
    agg = verdict.get("aggregated")
    out = {
        "open_scores": agg,
        "open_scores_individual": verdict.get("individual"),
        "n_successful_judges": verdict.get("n_successful_judges", 0),
        "judge_notes": (agg or {}).get("notes", ""),
        "judge_errors": verdict.get("errors"),
        "error": None,
    }
    if out["n_successful_judges"] >= 1:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        sys.stderr.write(".")
    else:
        with error_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": _judge_cache_key(rec), "judge_errors": out["judge_errors"]}) + "\n")
        sys.stderr.write("F")
    sys.stderr.flush()
    return out


async def _run_all(
    open_pairs: list[tuple[dict, dict]],
    consortium: JudgeConsortium,
    cache_dir: Path,
    error_log_path: Path,
) -> list[dict]:
    pool = ThreadPoolExecutor(max_workers=JUDGE_CONCURRENCY)
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)
    done = 0

    async def wrapped(pair: tuple[dict, dict]) -> dict:
        nonlocal done
        rec, item = pair
        async with sem:
            scored = await loop.run_in_executor(
                pool, _judge_one, consortium, rec, item, cache_dir, error_log_path
            )
        done += 1
        if done % 20 == 0 or done == len(open_pairs):
            print(f"  judged {done}/{len(open_pairs)}", file=sys.stderr)
        return {**rec, **scored}

    out = await asyncio.gather(*[wrapped(p) for p in open_pairs])
    pool.shutdown(wait=True)
    return out


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-cache-only",
        action="store_true",
        help="Don't call the judge API; assemble scored file from cache.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of cells to judge (smoke test).",
    )
    args = parser.parse_args()

    JUDGE_CACHE.mkdir(parents=True, exist_ok=True)

    dataset = {i["id"]: i for i in _load_jsonl(DATASET)}

    raw_recs: list[dict] = _load_jsonl(RAW_DIRECT) + _load_jsonl(RAW_RAG)
    open_pairs: list[tuple[dict, dict]] = []
    for rec in raw_recs:
        if rec.get("item_type") != "open":
            continue
        item = dataset.get(rec["item_id"])
        if item is None:
            continue
        open_pairs.append((rec, item))
    print(f"Open cells to judge: {len(open_pairs)}", file=sys.stderr)
    if args.limit is not None:
        open_pairs = open_pairs[: args.limit]
        print(f"Limit applied → {len(open_pairs)} cells will be processed.", file=sys.stderr)

    def _cache_has_full_panel(rec: dict) -> bool:
        cp = JUDGE_CACHE / _judge_cache_key(rec)
        if not cp.exists():
            return False
        try:
            cached = json.loads(cp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return _cache_is_complete(cached)

    n_cached_full = sum(1 for rec, _ in open_pairs if _cache_has_full_panel(rec))
    print(
        f"cache: {n_cached_full}/{len(open_pairs)} cells have full 4-judge panel; "
        f"{len(open_pairs) - n_cached_full} to (re)run.",
        file=sys.stderr,
    )

    if args.from_cache_only:
        scored: list[dict] = []
        n_judged = 0
        for rec, _ in open_pairs:
            cp = JUDGE_CACHE / _judge_cache_key(rec)
            if cp.exists():
                judged = json.loads(cp.read_text(encoding="utf-8"))
                scored.append({**rec, **judged})
                n_judged += 1
            else:
                scored.append({**rec, "open_scores": None, "judge_notes": None})
        SCORED_OUT.parent.mkdir(parents=True, exist_ok=True)
        with SCORED_OUT.open("w", encoding="utf-8") as f:
            for s in scored:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(
            f"Assembled {len(scored)} records ({n_judged} from cache). WROTE {SCORED_OUT}",
            file=sys.stderr,
        )
        return 0

    interrupted = False
    consortium = JudgeConsortium()
    t0 = time.time()
    try:
        scored = asyncio.run(_run_all(open_pairs, consortium, JUDGE_CACHE, ERROR_LOG))
    except KeyboardInterrupt:
        interrupted = True
        n_now = sum(1 for _ in JUDGE_CACHE.glob("*.json"))
        print(
            f"\nInterrupted. {n_now} judge results in cache. Re-run to resume.",
            file=sys.stderr,
        )
        scored = []
        for rec, _ in open_pairs:
            cp = JUDGE_CACHE / _judge_cache_key(rec)
            if cp.exists():
                judged = json.loads(cp.read_text(encoding="utf-8"))
                scored.append({**rec, **judged})
            else:
                scored.append({**rec, "open_scores": None, "judge_notes": None})
    finally:
        consortium.close()

    SCORED_OUT.parent.mkdir(parents=True, exist_ok=True)
    with SCORED_OUT.open("w", encoding="utf-8") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nJudge elapsed: {time.time() - t0:.1f}s", file=sys.stderr)
    n_success = sum(1 for s in scored if s.get("open_scores"))
    print(
        f"WROTE {SCORED_OUT} ({len(scored)} records, {n_success} successfully judged)",
        file=sys.stderr,
    )
    if interrupted:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
