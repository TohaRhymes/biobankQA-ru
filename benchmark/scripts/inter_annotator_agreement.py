"""Inter-annotator agreement on the annotation jsonl from load_annotations.

For each scoring axis, compute:
- Coverage matrix (which annotator filled which cells)
- Pairwise Cohen's kappa (quadratic-weighted for 0-5 scales,
  unweighted for binary use_in_practice)
- ICC(2,1) — intraclass correlation across all annotators
  (proxies "are we measuring the same construct")
- Mean absolute difference + count of severe disagreements (|Δ| ≥ 2 on
  a 0-5 axis, |Δ| ≥ 1 on hallucination_severity)
- Disagreement hotspot list — top cells that need adjudication

Writes:
- benchmark/outputs/annotations/agreement_report.md  (human-readable)
- benchmark/outputs/annotations/agreement_per_axis.csv (per axis row)
- benchmark/outputs/annotations/agreement_hotspots.csv (per cell)

Usage:
    python -m benchmark.scripts.inter_annotator_agreement
    # → reads benchmark/outputs/annotations/annotations.jsonl by default
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = REPO_ROOT / "benchmark" / "outputs" / "annotations" / "annotations.jsonl"
OUT_DIR = REPO_ROOT / "benchmark" / "outputs" / "annotations"

ORDINAL_AXES = (
    "completeness",
    "safety",
    "sop_applicability",
    "uncertainty_handling",
)
HALLUC_AXIS = "hallucination_severity"
BINARY_AXIS = "use_in_practice"


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pivot(records: list[dict]) -> dict[tuple, dict[str, dict[str, object]]]:
    """(item_id, letter) → axis → annotator_id → score (or comment)."""
    out: dict[tuple, dict[str, dict[str, object]]] = defaultdict(lambda: defaultdict(dict))
    for r in records:
        cell = (r["item_id"], r["letter"])
        if r["axis"] == "comment":
            out[cell]["comment"][r["annotator_id"]] = r.get("comment")
            continue
        score = r.get("score")
        if score is None:
            continue
        out[cell][r["axis"]][r["annotator_id"]] = score
    return out


def _cohen_kappa(a: list[float], b: list[float], *, max_score: int, weighted: bool) -> float | None:
    """Cohen's kappa. Linear- (unweighted) or quadratic-weighted.

    Both raters must have rated the same N cells (a[i] paired with b[i]).
    """
    if not a or len(a) != len(b):
        return None
    cats = list(range(max_score + 1))
    n = len(a)
    # Confusion matrix
    obs = {(i, j): 0 for i in cats for j in cats}
    for x, y in zip(a, b):
        xi, yi = int(round(x)), int(round(y))
        if xi not in cats or yi not in cats:
            return None
        obs[(xi, yi)] += 1
    # Marginals
    row = {i: sum(obs[(i, j)] for j in cats) for i in cats}
    col = {j: sum(obs[(i, j)] for i in cats) for j in cats}
    # Weights
    if weighted:
        w = {(i, j): 1 - ((i - j) ** 2) / (max_score ** 2) for i in cats for j in cats}
    else:
        w = {(i, j): 1.0 if i == j else 0.0 for i in cats for j in cats}
    p_o = sum(obs[k] * w[k] for k in obs) / n
    p_e = sum(row[i] * col[j] * w[(i, j)] for i in cats for j in cats) / (n * n)
    if abs(1 - p_e) < 1e-12:
        # DA #57 SS#2: when both raters give the same constant score
        # on every cell (common for Safety axis when everyone scores 5),
        # p_e ≈ 1 and κ is mathematically undefined. Previously returned
        # 1.0 in the lucky-p_o==1 case, masquerading an undefined stat
        # as perfect agreement. Now: always None, with a footnote in
        # the markdown report explaining why.
        return None
    return (p_o - p_e) / (1 - p_e)


def _icc_2_1(ratings: list[list[float]]) -> float | None:
    """ICC(2,1) — single-rater, two-way random effects, absolute agreement.

    Input: ratings[subject][rater]; all cells must be present.
    Reference: Shrout & Fleiss (1979).
    """
    if not ratings:
        return None
    n = len(ratings)
    k = len(ratings[0])
    if k < 2 or any(len(r) != k for r in ratings):
        return None
    flat = [v for row in ratings for v in row]
    grand_mean = sum(flat) / (n * k)
    row_means = [sum(r) / k for r in ratings]
    col_means = [sum(ratings[i][j] for i in range(n)) / n for j in range(k)]
    ss_between = k * sum((rm - grand_mean) ** 2 for rm in row_means)
    ss_within = sum((ratings[i][j] - row_means[i]) ** 2 for i in range(n) for j in range(k))
    ss_raters = n * sum((cm - grand_mean) ** 2 for cm in col_means)
    ss_error = ss_within - ss_raters
    df_between = n - 1
    df_raters = k - 1
    df_error = (n - 1) * (k - 1)
    if df_between == 0 or df_error == 0:
        return None
    ms_between = ss_between / df_between
    ms_raters = ss_raters / df_raters if df_raters else 0
    ms_error = ss_error / df_error
    denom = ms_between + (k - 1) * ms_error + (k * (ms_raters - ms_error) / n)
    if abs(denom) < 1e-12:
        return None
    return (ms_between - ms_error) / denom


def _binary_agreement(a: list[int], b: list[int]) -> tuple[float, float] | None:
    """Returns (% agreement, Cohen's kappa unweighted)."""
    if not a or len(a) != len(b):
        return None
    n = len(a)
    agree = sum(1 for x, y in zip(a, b) if x == y) / n
    k = _cohen_kappa(a, b, max_score=1, weighted=False)
    return agree, k or 0.0


def compute_agreement(records: list[dict]) -> dict:
    pivot = _pivot(records)
    annotators = sorted({r["annotator_id"] for r in records})

    # Coverage matrix per axis
    coverage: dict[str, dict[str, int]] = {}
    all_axes = ORDINAL_AXES + (HALLUC_AXIS, BINARY_AXIS)
    for axis in all_axes:
        coverage[axis] = {a: 0 for a in annotators}
    for cell, axes in pivot.items():
        for axis, by_ann in axes.items():
            if axis == "comment":
                continue
            for ann, score in by_ann.items():
                if score is not None and ann in coverage[axis]:
                    coverage[axis][ann] += 1

    # Per axis: kappa pairwise + ICC + MAE + hotspots
    per_axis: list[dict] = []
    hotspots: list[dict] = []
    for axis in all_axes:
        max_score = 5
        weighted = True
        if axis == HALLUC_AXIS:
            max_score = 2
        elif axis == BINARY_AXIS:
            max_score = 1
            weighted = False

        # Collect paired scores per cell across annotators
        paired_cells: list[tuple[tuple, dict[str, float]]] = []
        for cell, axes in pivot.items():
            by_ann = axes.get(axis, {})
            present = {a: v for a, v in by_ann.items() if v is not None}
            if len(present) >= 2:
                paired_cells.append((cell, present))

        # Pairwise kappa (between each pair of annotators on their overlap)
        pairs = list(combinations(annotators, 2))
        pair_kappas: list[float] = []
        pair_details = []
        for a1, a2 in pairs:
            xs, ys = [], []
            for _, present in paired_cells:
                if a1 in present and a2 in present:
                    xs.append(float(present[a1]))
                    ys.append(float(present[a2]))
            if not xs:
                pair_details.append({"pair": f"{a1} vs {a2}", "n": 0, "kappa": None})
                continue
            k = _cohen_kappa(xs, ys, max_score=max_score, weighted=weighted)
            pair_details.append({"pair": f"{a1} vs {a2}", "n": len(xs), "kappa": k})
            if k is not None:
                pair_kappas.append(k)

        # ICC needs all-annotator overlap; restrict to cells where every annotator scored
        full_overlap = [
            [float(present[a]) for a in annotators]
            for _, present in paired_cells
            if all(a in present for a in annotators)
        ]
        icc = _icc_2_1(full_overlap)

        # MAE + hotspots (only on pairs)
        diffs: list[float] = []
        for cell, present in paired_cells:
            vals = list(present.values())
            mx, mn = max(vals), min(vals)
            diffs.append(mx - mn)
            severe_threshold = 1 if axis == HALLUC_AXIS else 2
            if (mx - mn) >= severe_threshold:
                hotspots.append({
                    "item_id": cell[0],
                    "letter": cell[1],
                    "axis": axis,
                    "max_score": mx,
                    "min_score": mn,
                    "diff": mx - mn,
                    "annotator_values": dict(present),
                })

        per_axis.append({
            "axis": axis,
            "n_cells_with_pair": len(paired_cells),
            "n_full_overlap": len(full_overlap),
            "mean_pair_kappa": round(mean(pair_kappas), 3) if pair_kappas else None,
            "icc_2_1": round(icc, 3) if icc is not None else None,
            "mean_abs_diff": round(mean(diffs), 3) if diffs else None,
            "severe_disagreements": sum(1 for d in diffs if d >= (1 if axis == HALLUC_AXIS else 2)),
            "pairs": pair_details,
        })

    return {
        "annotators": annotators,
        "coverage": coverage,
        "per_axis": per_axis,
        "hotspots": hotspots,
    }


def write_outputs(report: dict, *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV: per-axis summary
    csv_path = out_dir / "agreement_per_axis.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["axis", "n_paired", "n_full_overlap", "mean_pair_kappa", "icc_2_1", "mean_abs_diff", "severe_disagreements"])
        for row in report["per_axis"]:
            w.writerow([
                row["axis"], row["n_cells_with_pair"], row["n_full_overlap"],
                row["mean_pair_kappa"], row["icc_2_1"],
                row["mean_abs_diff"], row["severe_disagreements"],
            ])

    # CSV: hotspots
    hot_path = out_dir / "agreement_hotspots.csv"
    with hot_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "letter", "axis", "min", "max", "diff", "annotator_values"])
        for h in report["hotspots"]:
            w.writerow([
                h["item_id"], h["letter"], h["axis"],
                h["min_score"], h["max_score"], h["diff"],
                json.dumps(h["annotator_values"], ensure_ascii=False),
            ])

    # Markdown report
    md_path = out_dir / "agreement_report.md"
    lines: list[str] = []
    lines.append("# Inter-annotator agreement report")
    lines.append("")
    lines.append(f"Annotators: {', '.join(report['annotators'])}")
    lines.append("")
    lines.append("## Coverage (cells filled per annotator per axis)")
    lines.append("")
    lines.append("| axis | " + " | ".join(report["annotators"]) + " |")
    lines.append("|" + "---|" * (len(report["annotators"]) + 1))
    for axis in ORDINAL_AXES + (HALLUC_AXIS, BINARY_AXIS):
        cov = report["coverage"][axis]
        lines.append("| " + axis + " | " + " | ".join(str(cov[a]) for a in report["annotators"]) + " |")
    lines.append("")
    lines.append("## Agreement per axis")
    lines.append("")
    lines.append("| axis | n_paired | n_full_overlap | mean pairwise κ | ICC(2,1) | mean |Δ| | severe disagreements |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in report["per_axis"]:
        lines.append(
            f"| {row['axis']} | {row['n_cells_with_pair']} | {row['n_full_overlap']} | "
            f"{row['mean_pair_kappa']} | {row['icc_2_1']} | {row['mean_abs_diff']} | "
            f"{row['severe_disagreements']} |"
        )
    lines.append("")
    lines.append("## Pairwise breakdown")
    lines.append("")
    for row in report["per_axis"]:
        lines.append(f"### {row['axis']}")
        for p in row["pairs"]:
            lines.append(f"- {p['pair']}: n={p['n']}, κ={p['kappa']}")
        lines.append("")

    lines.append("Footnote: κ = null means the statistic is undefined "
                 "(both raters gave a constant value on every overlapping cell — "
                 "p_expected ≈ 1, denominator collapses). Treat that axis as "
                 "trivially-agreed; report it as `n/a` in the paper rather than 1.0.")
    lines.append("")
    lines.append(f"## Hotspots ({len(report['hotspots'])} cells need adjudication)")
    lines.append("")
    lines.append("Severe = |Δ| ≥ 2 on 0-5 axes, |Δ| ≥ 1 on hallucination_severity.")
    lines.append(f"Full hotspot table: `{hot_path.name}`.")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {md_path}", file=sys.stderr)
    print(f"WROTE {csv_path}", file=sys.stderr)
    print(f"WROTE {hot_path} ({len(report['hotspots'])} rows)", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN,
                        help=f"Annotation JSONL (output of load_annotations). Default: {DEFAULT_IN.relative_to(REPO_ROOT)}")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if not args.input.exists():
        print(f"FATAL: input not found at {args.input}", file=sys.stderr)
        return 2
    records = _load(args.input)
    print(f"loaded {len(records)} annotation records from {args.input}", file=sys.stderr)
    report = compute_agreement(records)
    write_outputs(report, out_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
