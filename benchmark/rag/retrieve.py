"""Hybrid retrieval: BM25 + bge-m3 dense, fused via RRF, then bge-reranker-v2-m3.

Design constraints (v2 architecture):

- In-process numpy cosine + rank_bm25 — no Qdrant. The corpus is ~2k
  chunks so brute-force is faster than any vector DB.
- Per-query disk cache keyed by `sha256(normalize(query))`. Cache files
  live under `benchmark/outputs/rag/retrieval_cache/{hash}.json` and are
  re-used across all 12 candidate LLMs (retrieval is LLM-independent).
- Per-item pointer file `benchmark/outputs/rag/by_item.json` maps
  `item_id -> query_hash` so callers can look up by either key.
- Reranker model loaded lazily and held as a process-global singleton so
  every call within one run pays the load cost exactly once.
- If `max(retrieval_score) < SCORE_THRESHOLD`, the cache record marks
  `"context_found": false`. The prompt builder respects this and emits a
  `Источник: нет` marker so the judge can split metrics later.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import threading
from pathlib import Path
from typing import Any, Sequence

from .build_index import (
    BGE_M3_MODEL,
    BGE_M3_REVISION,
    BGE_RERANKER_MODEL,
    BGE_RERANKER_REVISION,
    BM25_PATH,
    CHUNK_IDS_PATH,
    EMBEDDINGS_PATH,
    META_PATH,
)
from .lemmatize import lemmatize
from .normalize import normalize_for_query


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "benchmark" / "outputs" / "rag" / "retrieval_cache"
BY_ITEM_FILE = REPO_ROOT / "benchmark" / "outputs" / "rag" / "by_item.json"

# Hybrid knobs.
BM25_TOPN = 50
DENSE_TOPN = 50
RRF_K = 60          # Reciprocal Rank Fusion constant
PRE_RERANK_TOPN = 20
FINAL_TOPK_DEFAULT = 5

# "No context found" threshold. We use the *post-rerank* score (cross-encoder
# logit). bge-reranker-v2-m3 typically sits in [-10, 10]; anything below
# this is effectively "model says these aren't relevant".
SCORE_THRESHOLD = -2.0


# ---------------------------------------------------------------------------
# Lazy global state — singletons, thread-safe init
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "bm25": None,
    "chunk_ids": None,
    "meta": None,
    "embeddings": None,
    "dense_model": None,
    "reranker": None,
}


def _load_index() -> None:
    """Populate `_state` with the on-disk index. Idempotent and lock-guarded."""
    if _state["chunk_ids"] is not None:
        return
    with _state_lock:
        if _state["chunk_ids"] is not None:
            return
        import numpy as np

        if not CHUNK_IDS_PATH.exists():
            raise RuntimeError(
                f"Retrieval index missing at {CHUNK_IDS_PATH}. "
                f"Run `python -m benchmark.rag.build_index` first."
            )
        _state["chunk_ids"] = json.loads(CHUNK_IDS_PATH.read_text(encoding="utf-8"))
        _state["meta"] = json.loads(META_PATH.read_text(encoding="utf-8"))

        if BM25_PATH.exists():
            with BM25_PATH.open("rb") as f:
                _state["bm25"] = pickle.load(f)
        if EMBEDDINGS_PATH.exists():
            _state["embeddings"] = np.load(EMBEDDINGS_PATH)


def _get_dense_model():
    if _state["dense_model"] is None:
        with _state_lock:
            if _state["dense_model"] is None:
                from sentence_transformers import SentenceTransformer

                _state["dense_model"] = SentenceTransformer(
                    BGE_M3_MODEL, revision=BGE_M3_REVISION
                )
    return _state["dense_model"]


def _get_reranker():
    if _state["reranker"] is None:
        with _state_lock:
            if _state["reranker"] is None:
                from sentence_transformers import CrossEncoder

                _state["reranker"] = CrossEncoder(
                    BGE_RERANKER_MODEL, revision=BGE_RERANKER_REVISION
                )
    return _state["reranker"]


# ---------------------------------------------------------------------------
# Query hashing + per-query cache I/O
# ---------------------------------------------------------------------------
def query_hash(query: str) -> str:
    norm = normalize_for_query(query)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]


def _cache_path(qhash: str) -> Path:
    return CACHE_DIR / f"{qhash}.json"


def load_cached(qhash: str) -> dict | None:
    p = _cache_path(qhash)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_cache(qhash: str, payload: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(qhash).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def update_by_item(item_id: str, qhash: str) -> None:
    BY_ITEM_FILE.parent.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    if BY_ITEM_FILE.exists():
        try:
            mapping = json.loads(BY_ITEM_FILE.read_text(encoding="utf-8"))
        except Exception:
            mapping = {}
    mapping[item_id] = qhash
    BY_ITEM_FILE.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Retrieval primitives
# ---------------------------------------------------------------------------
def _bm25_topn(query: str, n: int) -> list[tuple[int, float]]:
    """Returns list of (chunk_row, bm25_score), descending by score.

    Caller is responsible for normalising `query` (see `retrieve()`); this
    primitive only lemmatises. Re-normalising here would double-expand any
    abbreviation (DA #53 finding #1): `СОП` → `СОП (стандартная …) (…)`.
    """
    _load_index()
    bm25 = _state["bm25"]
    if bm25 is None:
        return []
    q_tokens = lemmatize(query, drop_stopwords=True)
    if not q_tokens:
        return []
    scores = bm25.get_scores(q_tokens)
    # argsort -- take top-n by score
    import numpy as np
    order = np.argsort(-scores)[:n]
    return [(int(i), float(scores[int(i)])) for i in order if scores[int(i)] > 0]


def _dense_topn(query: str, n: int) -> list[tuple[int, float]]:
    """Returns list of (chunk_row, cosine_similarity), descending.

    Caller is responsible for normalising `query`. We do NOT call
    `normalize_for_query` here — doing so would double-expand abbreviations
    (DA #53 finding #1), shifting the query distribution off the indexed
    one and inflating IDF weight of the expansion tokens.
    """
    _load_index()
    embeddings = _state["embeddings"]
    if embeddings is None:
        return []
    import numpy as np
    model = _get_dense_model()
    q_emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype("float32")
    # embeddings already L2-normalised at build time, so cosine == dot.
    sims = embeddings @ q_emb
    order = np.argsort(-sims)[:n]
    return [(int(i), float(sims[int(i)])) for i in order]


def _rrf_fuse(
    bm25: list[tuple[int, float]],
    dense: list[tuple[int, float]],
    k: int = RRF_K,
) -> list[tuple[int, float, int | None, int | None]]:
    """Reciprocal Rank Fusion.

    Returns: list of (chunk_row, rrf_score, bm25_rank_or_None, dense_rank_or_None),
    sorted by rrf_score desc.
    """
    bm25_rank = {row: r + 1 for r, (row, _) in enumerate(bm25)}
    dense_rank = {row: r + 1 for r, (row, _) in enumerate(dense)}
    rows = set(bm25_rank) | set(dense_rank)
    fused: list[tuple[int, float, int | None, int | None]] = []
    for row in rows:
        s = 0.0
        if row in bm25_rank:
            s += 1.0 / (k + bm25_rank[row])
        if row in dense_rank:
            s += 1.0 / (k + dense_rank[row])
        fused.append((row, s, bm25_rank.get(row), dense_rank.get(row)))
    fused.sort(key=lambda x: -x[1])
    return fused


def _rerank(query: str, candidates: Sequence[int]) -> list[tuple[int, float]]:
    """Returns (chunk_row, cross-encoder score), descending.

    Caller is responsible for normalising `query`. We do NOT call
    `normalize_for_query` here — same reason as `_dense_topn` above.
    """
    if not candidates:
        return []
    reranker = _get_reranker()
    meta = _state["meta"]
    chunk_ids = _state["chunk_ids"]
    pairs = [
        (query, meta[chunk_ids[row]]["text"])
        for row in candidates
    ]
    scores = reranker.predict(pairs)
    out = list(zip(candidates, [float(s) for s in scores]))
    out.sort(key=lambda x: -x[1])
    return out


# ---------------------------------------------------------------------------
# Public retrieval entry point
# ---------------------------------------------------------------------------
def retrieve(
    query: str,
    k: int = FINAL_TOPK_DEFAULT,
    *,
    item_id: str | None = None,
    use_cache: bool = True,
    use_reranker: bool = True,
) -> dict:
    """Retrieve top-k chunks for a query. Returns the full cache record dict::

        {
          "query": <normalised>,
          "query_hash": <hex>,
          "item_id": <optional>,
          "context_found": bool,
          "top_score": float,
          "chunks": [
            {
              "rank": 1,
              "chunk_id": ...,
              "text": ...,
              "source_short": ...,
              "section": ...,
              "page": ...,
              "bm25_rank": int|None,
              "dense_rank": int|None,
              "final_rank": int,
              "rerank_score": float|None,
              "rrf_score": float,
            }, ...
          ],
        }
    """
    norm = normalize_for_query(query)
    qhash = query_hash(query)

    if use_cache:
        cached = load_cached(qhash)
        if cached is not None:
            if item_id:
                update_by_item(item_id, qhash)
            return cached

    _load_index()
    chunk_ids = _state["chunk_ids"]
    meta = _state["meta"]

    bm25 = _bm25_topn(norm, BM25_TOPN)
    dense = _dense_topn(norm, DENSE_TOPN)
    fused = _rrf_fuse(bm25, dense)

    candidates = [row for row, _, _, _ in fused[:PRE_RERANK_TOPN]]

    if use_reranker and candidates:
        reranked = _rerank(norm, candidates)
        rerank_scores = {row: score for row, score in reranked}
        ordered = [row for row, _ in reranked][:k]
    else:
        rerank_scores = {}
        ordered = candidates[:k]

    # Build per-chunk payload
    rrf_lookup = {row: (rrf, bm_r, ds_r) for row, rrf, bm_r, ds_r in fused}
    chunks_out: list[dict] = []
    for final_rank, row in enumerate(ordered, start=1):
        cid = chunk_ids[row]
        m = meta[cid]
        rrf, bm_r, ds_r = rrf_lookup.get(row, (0.0, None, None))
        chunks_out.append(
            {
                "rank": final_rank,
                "chunk_id": cid,
                "text": m.get("text", ""),
                "source_short": m.get("source_short"),
                "article_id": m.get("article_id"),
                "section": m.get("section"),
                "page": m.get("page"),
                "bm25_rank": bm_r,
                "dense_rank": ds_r,
                "final_rank": final_rank,
                "rerank_score": rerank_scores.get(row),
                "rrf_score": rrf,
            }
        )

    top_score = max(
        (c["rerank_score"] for c in chunks_out if c["rerank_score"] is not None),
        default=None,
    )
    if top_score is None:
        # No reranker used: fall back to RRF score. Threshold is on a
        # different scale, so we approximate via the existence of any hit.
        context_found = bool(chunks_out)
        top_score_for_record = max((c["rrf_score"] for c in chunks_out), default=0.0)
    else:
        context_found = top_score >= SCORE_THRESHOLD
        top_score_for_record = top_score

    payload = {
        "query": norm,
        "raw_query": query,
        "query_hash": qhash,
        "item_id": item_id,
        "context_found": context_found,
        "top_score": top_score_for_record,
        "score_threshold": SCORE_THRESHOLD,
        "reranker_used": bool(use_reranker and chunks_out),
        "chunks": chunks_out,
    }

    if use_cache:
        save_cache(qhash, payload)
    if item_id:
        update_by_item(item_id, qhash)

    return payload


def retrieve_for_item(item: dict, k: int = FINAL_TOPK_DEFAULT) -> dict:
    """Convenience: retrieve using the item's question text only.

    DA review dd382f8, fix #1: previously we concatenated MCQ options into
    the query to give BM25 the option-specific terminology, but that made
    `query_hash` shuffle-salt dependent (cache misses on every reseed) and
    leaked distractor surface forms into retrieval scoring. Both MCQ and
    open-ended now retrieve on the bare question text; the candidate
    options are presented to the LLM only inside the prompt, not the index.
    """
    query = item["question"]
    return retrieve(query, k=k, item_id=item.get("id"))


__all__ = [
    "BY_ITEM_FILE",
    "CACHE_DIR",
    "FINAL_TOPK_DEFAULT",
    "RRF_K",
    "SCORE_THRESHOLD",
    "load_cached",
    "query_hash",
    "retrieve",
    "retrieve_for_item",
    "save_cache",
    "update_by_item",
]
