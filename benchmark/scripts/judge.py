"""Score raw model outputs.

MCQ: extract A-E letters from response, decode shuffled→canonical, F1 against
gold set (partial credit for multi-correct).

Open: judge consortium of GigaChat-2-Max + YandexGPT/latest + DeepSeek-V4-Pro
each scores along a 5-axis rubric; we keep all three sets and aggregate by
mean. No external (OpenAI/Anthropic) models in the default panel.

Output: `benchmark/outputs/scored/scored.jsonl`.
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
from benchmark.parsing import parse_mcq_letters


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "data" / "processed" / "biobankqa_v0_executable.jsonl"
RAW_DIRECT = REPO_ROOT / "benchmark" / "outputs" / "raw" / "raw_outputs.jsonl"
RAW_RAG = REPO_ROOT / "benchmark" / "outputs" / "raw" / "rag_outputs.jsonl"
SCORED_DIR = REPO_ROOT / "benchmark" / "outputs" / "scored"
SCORED_FILE_DEFAULT = SCORED_DIR / "scored.jsonl"
JUDGE_CACHE = SCORED_DIR / "judge_cache"

JUDGE_CONCURRENCY = 6

MCQ_LETTER_RE = re.compile(r"[ABCDE]")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _extract_mcq_letters(text: str | None) -> list[str]:
    """Pull the set of A-E letters the model emitted, deduplicated, in order.

    Delegates to benchmark.parsing.parse_mcq_letters, which tries the
    structured-JSON ``{"letters": [...]}`` path first and falls back to a
    plain A-E regex. Handles 'B', 'A,C,E', 'A, C, E', 'A C', 'Ответ: B',
    JSON objects, and code-fenced JSON.
    """
    return parse_mcq_letters(text)


def _f1(answer: set[str], gold: set[str]) -> float:
    if not answer and not gold:
        return 1.0
    if not answer or not gold:
        return 0.0
    inter = len(answer & gold)
    if inter == 0:
        return 0.0
    precision = inter / len(answer)
    recall = inter / len(gold)
    return 2 * precision * recall / (precision + recall)


def _score_mcq(rec: dict, item: dict) -> dict:
    """Score an MCQ answer. The model saw the SHUFFLED layout; we decode its
    letters back to canonical letters and compare to the red-marked gold set
    using F1 (partial credit for multi-correct items).
    """
    if rec["error"] is not None:
        return {
            "mcq_letters_shown": None,
            "mcq_letters_canonical": None,
            "mcq_score": None,
            "mcq_strict_correct": None,
            "mcq_invalid_format": True,
        }
    shown_letters = _extract_mcq_letters(rec.get("response"))
    if not shown_letters:
        return {
            "mcq_letters_shown": [],
            "mcq_letters_canonical": [],
            "mcq_score": None,
            "mcq_strict_correct": None,
            "mcq_invalid_format": True,
        }
    shuffle_map = item.get("shuffle_map_to_canonical") or {}
    canonical_letters = [shuffle_map.get(l, l) for l in shown_letters]

    # Build gold set: prefer multi-letter canonical list when present
    gold_letters = item.get("gold_letters_canonical") or item.get("gold_letters") or []
    if not gold_letters and item.get("gold_answer_canonical"):
        gold_letters = [item["gold_answer_canonical"]]
    elif not gold_letters and item.get("gold_answer"):
        gold_letters = [item["gold_answer"]]

    if not gold_letters:
        return {
            "mcq_letters_shown": shown_letters,
            "mcq_letters_canonical": canonical_letters,
            "mcq_score": None,
            "mcq_strict_correct": None,
            "mcq_invalid_format": False,
        }

    answer_set = set(canonical_letters)
    gold_set = set(gold_letters)
    score = _f1(answer_set, gold_set)
    strict = (answer_set == gold_set)
    return {
        "mcq_letters_shown": shown_letters,
        "mcq_letters_canonical": canonical_letters,
        "mcq_score": round(score, 3),
        "mcq_strict_correct": strict,
        "mcq_invalid_format": False,
    }


def _judge_cache_key(rec: dict) -> str:
    """Cache key MUST include mode — direct vs RAG produce different answers
    for the same (provider, model, item_id), and judge results are not
    interchangeable. Without mode in the key, a RAG run would silently
    reuse the direct judgment of an entirely different answer text.
    """
    mode = rec.get("mode") or "direct"
    safe_model = re.sub(r'[^A-Za-z0-9._-]', '_', rec['model'])
    return f"{rec['provider']}__{safe_model}__{rec['item_id']}__{mode}.json"


def _judge_open_sync(consortium: JudgeConsortium, rec: dict, item: dict) -> dict:
    cache_path = JUDGE_CACHE / _judge_cache_key(rec)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    upstream_empty = rec["error"] is not None or not rec.get("response")
    if upstream_empty:
        out = {
            "open_scores": None,
            "open_scores_individual": None,
            "n_successful_judges": 0,
            "judge_notes": "model_error_or_empty",
            "error": rec.get("error"),
        }
    else:
        try:
            verdict = consortium.judge_open_answer(
                question=item["question"],
                model_answer=rec["response"],
            )
            agg = verdict.get("aggregated")
            out = {
                "open_scores": agg,
                "open_scores_individual": verdict.get("individual"),
                "n_successful_judges": verdict.get("n_successful_judges", 0),
                "judge_notes": (agg or {}).get("notes", ""),
                "judge_errors": verdict.get("errors"),
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            out = {
                "open_scores": None,
                "open_scores_individual": None,
                "n_successful_judges": 0,
                "judge_notes": None,
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            }
    # Cache ONLY on success. Success = at least one judge produced a score.
    # Failures (n_successful_judges == 0 or upstream empty) are returned but
    # not persisted so the next run will retry them.
    successful = (not upstream_empty) and out.get("n_successful_judges", 0) >= 1
    if successful:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        sys.stderr.write(".")
    else:
        sys.stderr.write("e")
    sys.stderr.flush()
    return out


async def _judge_all_open(open_recs: list[tuple[dict, dict]], consortium: JudgeConsortium) -> list[dict]:
    pool = ThreadPoolExecutor(max_workers=JUDGE_CONCURRENCY)
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)
    done = 0

    async def wrapped(rec_item: tuple[dict, dict]) -> dict:
        nonlocal done
        rec, item = rec_item
        async with sem:
            scored = await loop.run_in_executor(pool, _judge_open_sync, consortium, rec, item)
        done += 1
        if done % 5 == 0 or done == len(open_recs):
            print(f"  judged {done}/{len(open_recs)}", file=sys.stderr)
        return {**rec, **scored}

    out = await asyncio.gather(*[wrapped(p) for p in open_recs])
    pool.shutdown(wait=True)
    return out


def _write_scored(scored: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def _build_scored_from_cache(raw_recs: list[dict], dataset: dict[str, dict]) -> tuple[list[dict], int]:
    """For each raw record, look up its judge cache (if any) and attach.
    MCQ records are scored deterministically (no API). Returns (scored, n_judged)."""
    scored: list[dict] = []
    n_judged = 0
    for rec in raw_recs:
        item = dataset.get(rec["item_id"])
        if item is None:
            continue
        if rec["item_type"] == "mcq":
            scored.append({**rec, **_score_mcq(rec, item), "open_scores": None, "judge_notes": None})
            continue
        cache_path = JUDGE_CACHE / _judge_cache_key(rec)
        if cache_path.exists():
            try:
                judged = json.loads(cache_path.read_text(encoding="utf-8"))
                scored.append({**rec, **judged})
                n_judged += 1
                continue
            except Exception as e:  # noqa: BLE001
                print(f"  WARN: failed to read judge cache {cache_path.name}: {e}", file=sys.stderr)
        # No cached judgment: emit a placeholder so the record still appears.
        scored.append({**rec, "open_scores": None, "judge_notes": None})
    return scored, n_judged


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-judge-open", action="store_true", help="Skip consortium judging of open answers")
    parser.add_argument(
        "--from-cache-only",
        action="store_true",
        help="Don't call any API; build scored.jsonl from the existing judge cache only.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DIRECT,
        help=f"Raw outputs JSONL to score. Default: {RAW_DIRECT.name}. Use {RAW_RAG.name} for the RAG sweep.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCORED_FILE_DEFAULT,
        help=f"Scored output path. Default: {SCORED_FILE_DEFAULT.name}.",
    )
    args = parser.parse_args()

    SCORED_DIR.mkdir(parents=True, exist_ok=True)
    JUDGE_CACHE.mkdir(parents=True, exist_ok=True)

    dataset = {i["id"]: i for i in _load_jsonl(DATASET)}
    raw_recs = _load_jsonl(args.input)
    scored_file = args.output
    print(f"Loaded {len(raw_recs)} raw outputs covering {len(dataset)} items.", file=sys.stderr)

    # --from-cache-only: deterministic MCQ scoring + cached open judgments, no API.
    if args.from_cache_only:
        scored, n_judged = _build_scored_from_cache(raw_recs, dataset)
        _write_scored(scored, scored_file)
        print(
            f"Wrote {len(scored)} records, {n_judged} judged from cache.",
            file=sys.stderr,
        )
        print(f"WROTE {scored_file}", file=sys.stderr)
        return 0

    scored: list[dict] = []
    open_pairs: list[tuple[dict, dict]] = []

    for rec in raw_recs:
        item = dataset.get(rec["item_id"])
        if item is None:
            continue
        if rec["item_type"] == "mcq":
            scored.append({**rec, **_score_mcq(rec, item), "open_scores": None, "judge_notes": None})
        else:
            open_pairs.append((rec, item))

    n_mcq = len(scored)
    print(f"MCQ scored: {n_mcq}", file=sys.stderr)
    print(f"Open to judge: {len(open_pairs)}", file=sys.stderr)

    # Resume status for open-answer judging.
    n_open = len(open_pairs)
    n_cached_open = sum(1 for rec, _ in open_pairs if (JUDGE_CACHE / _judge_cache_key(rec)).exists())
    print(
        f"Judge resume status: {n_cached_open}/{n_open} open-answer judgments "
        f"already cached, {n_open - n_cached_open} to run.",
        file=sys.stderr,
    )

    interrupted = False
    if not args.no_judge_open and open_pairs:
        t0 = time.time()
        consortium = JudgeConsortium()
        try:
            open_scored = asyncio.run(_judge_all_open(open_pairs, consortium))
            scored.extend(open_scored)
        except KeyboardInterrupt:
            interrupted = True
            n_cached_now = sum(1 for _ in JUDGE_CACHE.glob("*.json"))
            print(
                f"\nInterrupted. {n_cached_now} judge results in cache. "
                f"Re-run the same command tomorrow to resume.",
                file=sys.stderr,
            )
            # Fall back to cache-based assembly so the partial run still
            # produces a usable scored.jsonl.
            scored, n_judged_from_cache = _build_scored_from_cache(raw_recs, dataset)
            print(
                f"  Assembled {len(scored)} records ({n_judged_from_cache} judged from cache).",
                file=sys.stderr,
            )
        finally:
            consortium.close()
        print(f"Judge elapsed: {time.time() - t0:.1f}s", file=sys.stderr)
    else:
        for rec, _ in open_pairs:
            scored.append({**rec, "open_scores": None, "judge_notes": None})

    _write_scored(scored, scored_file)
    print(f"WROTE {scored_file} ({len(scored)} records)", file=sys.stderr)
    if interrupted:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
