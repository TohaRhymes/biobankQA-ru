"""Build the hybrid RAG index from `data/processed/materials/chunks.jsonl`.

Outputs (in `benchmark/rag/index/`):
- `embeddings.npy`        — float32 (N, 1024) bge-m3 dense vectors, L2-normalised
- `bm25.pkl`              — pickled rank_bm25.BM25Okapi over LEMMATISED docs
- `chunk_ids.json`        — list of N chunk_ids aligned with embeddings rows
- `chunks_meta.json`      — list of dicts (chunk_id -> text, page, section, ...)
- `manifest.json`         — content hash of chunks.jsonl + model revisions + counts

Resumable: if `manifest.json` exists and the input chunk-set hash matches,
the script exits 0 without rebuilding. Forced rebuild via `--rebuild`.

NB: This script *does not* run during the validation step — it only needs
to be importable and parse `--help` without side-effects. Calls to bge-m3
happen at the bottom of `main()` which we never trigger in CI / lint.

Usage::

    python -m benchmark.rag.build_index           # build if missing/stale
    python -m benchmark.rag.build_index --rebuild # force
    python -m benchmark.rag.build_index --dry-run # plan only, no model load
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

from .lemmatize import batch_lemmatize
from .normalize import normalize_for_index


# ---------------------------------------------------------------------------
# Pinned model revisions
# ---------------------------------------------------------------------------
# Resolved via `huggingface_hub.HfApi().model_info(...)` on 2026-06-16.
# Pinning by commit SHA so a silent upstream re-tokenisation cannot change
# our retrieval distribution between runs.
BGE_M3_MODEL = "BAAI/bge-m3"
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
BGE_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

# Library versions we depend on. The actually-installed versions are also
# captured in manifest.json at build time; this is the contract floor.
DEP_FLOOR = {
    "sentence-transformers": "3.0",
    "rank_bm25": "0.2.2",
    "pymorphy3": "2.0",
    "pymorphy3-dicts-ru": "2.4",
}


REPO_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "materials" / "chunks.jsonl"
INDEX_DIR = REPO_ROOT / "benchmark" / "rag" / "index"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
BM25_PATH = INDEX_DIR / "bm25.pkl"
CHUNK_IDS_PATH = INDEX_DIR / "chunk_ids.json"
META_PATH = INDEX_DIR / "chunks_meta.json"
MANIFEST_PATH = INDEX_DIR / "manifest.json"

DENSE_BATCH = 32
# (Removed dead `DENSE_MAX_LEN = 1024` — DA review dd382f8, fix #7. It was
# never threaded into `model.encode(...)`, and sentence-transformers honours
# the model's own `max_seq_length` anyway. If we later need to override it,
# do `model.max_seq_length = N` before `.encode(...)` and document why.)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def _load_chunks(path: Path) -> list[dict]:
    chunks: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def _normalize_chunks(chunks: list[dict]) -> list[str]:
    return [normalize_for_index(c.get("text", "")) for c in chunks]


def _build_bm25(normalised_texts: list[str]):
    """Build BM25Okapi over LEMMATISED documents."""
    from rank_bm25 import BM25Okapi  # local import to keep top-level light

    tokenised = batch_lemmatize(normalised_texts, drop_stopwords=True)
    bm25 = BM25Okapi(tokenised)
    return bm25, tokenised


def _build_dense(normalised_texts: list[str], cache_dir: Path | None = None):
    """Embed every chunk with bge-m3 (sentence-transformers wrapper)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        BGE_M3_MODEL,
        revision=BGE_M3_REVISION,
        cache_folder=str(cache_dir) if cache_dir else None,
    )
    embs = model.encode(
        normalised_texts,
        batch_size=DENSE_BATCH,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return embs.astype("float32")


def _installed_versions() -> dict[str, str]:
    """Return resolved versions for sanity in the manifest."""
    out: dict[str, str] = {}
    try:
        from importlib.metadata import PackageNotFoundError, version

        for pkg in ("sentence-transformers", "rank_bm25", "pymorphy3", "pymorphy3-dicts-ru", "numpy"):
            try:
                out[pkg] = version(pkg)
            except PackageNotFoundError:
                out[pkg] = "missing"
    except Exception as e:
        out["_error"] = str(e)
    return out


def _write_manifest(
    chunks_hash: str,
    n_chunks: int,
    embedding_dim: int,
    extra: dict | None = None,
) -> None:
    manifest = {
        "chunks_jsonl_sha256": chunks_hash,
        "n_chunks": n_chunks,
        "embedding_dim": embedding_dim,
        "embedding_model": BGE_M3_MODEL,
        "embedding_revision": BGE_M3_REVISION,
        "reranker_model": BGE_RERANKER_MODEL,
        "reranker_revision": BGE_RERANKER_REVISION,
        "installed_versions": _installed_versions(),
        "dep_floor": DEP_FLOOR,
    }
    if extra:
        manifest.update(extra)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def is_index_fresh(chunks_path: Path = CHUNKS_PATH) -> bool:
    """True iff index exists and points at the same chunks.jsonl content."""
    if not MANIFEST_PATH.exists():
        return False
    if not (EMBEDDINGS_PATH.exists() and BM25_PATH.exists() and CHUNK_IDS_PATH.exists()):
        return False
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    expected = _sha256_file(chunks_path)
    return manifest.get("chunks_jsonl_sha256") == expected


def build_index(
    *,
    rebuild: bool = False,
    dry_run: bool = False,
    cache_dir: Path | None = None,
) -> int:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if not CHUNKS_PATH.exists():
        print(f"FATAL: chunks file not found at {CHUNKS_PATH}", file=sys.stderr)
        return 2

    if not rebuild and is_index_fresh():
        print(f"Index already up to date at {INDEX_DIR}. Use --rebuild to force.", file=sys.stderr)
        return 0

    chunks_hash = _sha256_file(CHUNKS_PATH)
    chunks = _load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH.name}", file=sys.stderr)

    chunk_ids = [c["chunk_id"] for c in chunks]
    meta = {
        c["chunk_id"]: {
            "chunk_id": c["chunk_id"],
            "source_short": c.get("source_short"),
            "article_id": c.get("article_id"),
            "page": c.get("page"),
            "section": c.get("section"),
            "text": c.get("text", ""),
        }
        for c in chunks
    }
    CHUNK_IDS_PATH.write_text(json.dumps(chunk_ids, ensure_ascii=False), encoding="utf-8")
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    if dry_run:
        # Stop short of loading any model. Write a manifest stub so downstream
        # debugging can confirm what *would* have been built.
        _write_manifest(chunks_hash, len(chunks), embedding_dim=0, extra={"dry_run": True})
        print(f"Dry run done; wrote chunk_ids ({len(chunk_ids)}) + manifest stub.", file=sys.stderr)
        return 0

    normalised = _normalize_chunks(chunks)

    print("Building BM25 (lemmatised)…", file=sys.stderr)
    bm25, _tokenised = _build_bm25(normalised)
    with BM25_PATH.open("wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  wrote {BM25_PATH.name}", file=sys.stderr)

    print(f"Embedding with {BGE_M3_MODEL}@{BGE_M3_REVISION[:7]}…", file=sys.stderr)
    embs = _build_dense(normalised, cache_dir=cache_dir)
    import numpy as np
    np.save(EMBEDDINGS_PATH, embs)
    print(f"  wrote {EMBEDDINGS_PATH.name}  shape={embs.shape}", file=sys.stderr)

    _write_manifest(chunks_hash, len(chunks), embedding_dim=int(embs.shape[1]))
    print(f"Index built at {INDEX_DIR}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild even if index is fresh")
    parser.add_argument("--dry-run", action="store_true", help="Skip model loads; write manifest stub")
    parser.add_argument("--cache-dir", type=str, default=None, help="HuggingFace cache override")
    args = parser.parse_args()
    return build_index(
        rebuild=args.rebuild,
        dry_run=args.dry_run,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
