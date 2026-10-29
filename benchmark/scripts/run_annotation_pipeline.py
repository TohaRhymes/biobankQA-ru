"""End-to-end runner for the post-annotation analysis pipeline.

Once the medical team returns their filled xlsx files, point this at
them and it runs all five analysis scripts in order:

  1. load_annotations    — xlsx → tidy JSONL
  2. inter_annotator_agreement — κ, ICC, hotspots
  3. auto_vs_human_correlation — consortium vs human per axis
  4. compare_iterations  — iter1 vs iter2 vs iter3 (auto-judge only)
  5. final_benchmark     — definitive scoreboard with both auto and human

Outputs land under benchmark/outputs/{annotations,iterations_comparison,
final}/.

Usage:
    python -m benchmark.scripts.run_annotation_pipeline \\
        --annotator path/to/annotator1.xlsx \\
        --annotator path/to/annotator2.xlsx

Or to smoke-test on synthetic fixtures:

    python -m benchmark.scripts.run_annotation_pipeline --synthetic
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = REPO_ROOT / "data" / "processed" / "expert_review" / "synthetic"


def _run(label: str, cmd: list[str]) -> int:
    print(f"\n{'=' * 60}\n[step] {label}\n{'=' * 60}", file=sys.stderr)
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"[step FAILED: {label}] (exit {proc.returncode})", file=sys.stderr)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotator", action="append", default=None,
        help="Path to a filled annotator xlsx. Repeat for each annotator.",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic fixtures instead of real annotator xlsx.",
    )
    parser.add_argument(
        "--skip-compare-iters", action="store_true",
        help="Skip the iter1/2/3 comparison step (already in repo).",
    )
    args = parser.parse_args()

    if args.synthetic:
        annotator_paths = [
            SYNTHETIC_DIR / "annotator_a_filled.xlsx",
            SYNTHETIC_DIR / "annotator_b_filled.xlsx",
        ]
        for p in annotator_paths:
            if not p.exists():
                print(f"[synthetic mode] missing {p}; regenerating", file=sys.stderr)
                _run(
                    "regenerate synthetic fixtures",
                    [sys.executable, "-m", "benchmark.scripts.make_synthetic_annotations"],
                )
                break
    elif args.annotator:
        annotator_paths = [Path(p) for p in args.annotator]
    else:
        parser.error("provide --annotator at least once, or use --synthetic")

    py = sys.executable

    # Step 1: load_annotations
    load_cmd = [py, "-m", "benchmark.scripts.load_annotations"]
    for p in annotator_paths:
        load_cmd.extend(["--annotator", str(p)])
    if _run("load_annotations", load_cmd) != 0:
        return 2

    # Step 2: inter_annotator_agreement
    if _run(
        "inter_annotator_agreement",
        [py, "-m", "benchmark.scripts.inter_annotator_agreement"],
    ) != 0:
        return 2

    # Step 3: auto_vs_human_correlation
    if _run(
        "auto_vs_human_correlation",
        [py, "-m", "benchmark.scripts.auto_vs_human_correlation"],
    ) != 0:
        return 2

    # Step 4: compare_iterations (optional — uses iter1/2/3, not annotations)
    if not args.skip_compare_iters:
        if _run(
            "compare_iterations",
            [py, "-m", "benchmark.scripts.compare_iterations"],
        ) != 0:
            return 2

    # Step 5: final_benchmark
    if _run(
        "final_benchmark",
        [py, "-m", "benchmark.scripts.final_benchmark"],
    ) != 0:
        return 2

    print("\n" + "=" * 60, file=sys.stderr)
    print("DONE. Key outputs:", file=sys.stderr)
    print("  benchmark/outputs/annotations/annotations.jsonl", file=sys.stderr)
    print("  benchmark/outputs/annotations/agreement_report.md", file=sys.stderr)
    print("  benchmark/outputs/annotations/auto_vs_human_report.md", file=sys.stderr)
    print("  benchmark/outputs/iterations_comparison/summary.md", file=sys.stderr)
    print("  benchmark/outputs/final/benchmark_v0.1.md", file=sys.stderr)
    print("  benchmark/outputs/final/rag_vs_direct_significance.csv", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
