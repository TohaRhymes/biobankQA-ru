"""Compare auto-judge scores across v0.1 iter1 / iter2 / iter3.

Reads iterations/iterN/scored.jsonl for N ∈ {1, 2, 3} and aggregates:
- Per-axis open-answer mean (consortium)
- Direct vs RAG split per iter (mode delta over time)
- Answer length distribution per mode per iter
- Refusal-phrase counts (mode-leakage signals over time):
  iter1 "в предоставленных" / iter2 "по общим принципам" / iter3 neither

Outputs:
- benchmark/outputs/iterations_comparison/summary.csv
- benchmark/outputs/iterations_comparison/summary.md
- plots/{axis_evolution.png, length_evolution.png, mode_gap_evolution.png}
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[2]
ITER_DIR_ROOT = REPO_ROOT / "data" / "processed" / "expert_review" / "iterations"
OUT_DIR = REPO_ROOT / "benchmark" / "outputs" / "iterations_comparison"
PLOT_DIR = OUT_DIR / "plots"

# Auto-judge writes "actionability"; human side calls it "sop_applicability".
# DA #57 R3: use the human-facing name uniformly so paper tables/plots agree.
HUMAN_AXES = [
    "completeness",
    "safety",
    "sop_applicability",
    "uncertainty_handling",
    "hallucination_severity",
]
AUTO_ALIAS = {"sop_applicability": "actionability"}
AXES = HUMAN_AXES  # backwards-compat name used below

# Mode-leakage refusal phrases per iter (post-hoc).
REFUSAL_PHRASES = [
    ("iter1_phrase", re.compile(r"в\s+предоставленных\s+источниках", re.IGNORECASE)),
    ("iter2_phrase", re.compile(r"по\s+общим\s+принципам", re.IGNORECASE)),
]


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _agg_per_iter(records: list[dict]) -> dict:
    """For one iter's scored.jsonl, aggregate per-axis means + length stats
    + refusal-phrase counts split by mode."""
    out = {
        "axes_overall": {ax: [] for ax in AXES},
        "axes_by_mode": {mode: {ax: [] for ax in AXES} for mode in ("direct", "rag")},
        "lengths_by_mode": {"direct": [], "rag": []},
        "refusal_counts_by_mode": {
            mode: {key: 0 for key, _ in REFUSAL_PHRASES} for mode in ("direct", "rag")
        },
        "n_open_records": 0,
    }
    for r in records:
        if r.get("item_type") != "open":
            continue
        mode = r.get("mode", "direct")
        out["n_open_records"] += 1
        # Length
        resp = (r.get("response") or "")
        if resp:
            out["lengths_by_mode"][mode].append(len(resp))
        # Refusal-phrase markers
        for key, pat in REFUSAL_PHRASES:
            if pat.search(resp):
                out["refusal_counts_by_mode"][mode][key] += 1
        # Axes — translate auto-judge alias to human-facing name on read.
        scores = r.get("open_scores") or {}
        for ax in AXES:
            val = scores.get(AUTO_ALIAS.get(ax, ax))
            if val is None:
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            out["axes_overall"][ax].append(v)
            if mode in out["axes_by_mode"]:
                out["axes_by_mode"][mode][ax].append(v)
    return out


def _mean(vals):
    return round(mean(vals), 3) if vals else None


def build_report(per_iter: dict[str, dict]) -> dict:
    iters = sorted(per_iter.keys())  # iter1, iter2, iter3
    rows: list[dict] = []
    for ax in AXES:
        row = {"axis": ax}
        for it in iters:
            row[f"{it}_overall"] = _mean(per_iter[it]["axes_overall"][ax])
            for mode in ("direct", "rag"):
                row[f"{it}_{mode}"] = _mean(per_iter[it]["axes_by_mode"][mode][ax])
            d = per_iter[it]["axes_by_mode"]["direct"]
            r = per_iter[it]["axes_by_mode"]["rag"]
            d_mean = _mean(d[ax])
            r_mean = _mean(r[ax])
            row[f"{it}_rag_minus_direct"] = (
                round(r_mean - d_mean, 3) if (d_mean is not None and r_mean is not None) else None
            )
        rows.append(row)

    length_rows = []
    for it in iters:
        for mode in ("direct", "rag"):
            lens = per_iter[it]["lengths_by_mode"][mode]
            if not lens:
                continue
            length_rows.append({
                "iter": it,
                "mode": mode,
                "n": len(lens),
                "mean": int(sum(lens) / len(lens)),
                "median": int(sorted(lens)[len(lens) // 2]),
                "max": max(lens),
            })

    refusal_rows = []
    for it in iters:
        for mode in ("direct", "rag"):
            for key, _ in REFUSAL_PHRASES:
                refusal_rows.append({
                    "iter": it,
                    "mode": mode,
                    "phrase": key,
                    "count": per_iter[it]["refusal_counts_by_mode"][mode][key],
                })

    return {
        "iters": iters,
        "axes": rows,
        "lengths": length_rows,
        "refusals": refusal_rows,
    }


def write_outputs(report: dict, *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    iters = report["iters"]

    # CSV: axes
    csv_path = out_dir / "summary_axes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        fields = ["axis"]
        for it in iters:
            fields += [f"{it}_overall", f"{it}_direct", f"{it}_rag", f"{it}_rag_minus_direct"]
        w.writerow(fields)
        for row in report["axes"]:
            w.writerow([row.get(c) for c in fields])

    # CSV: lengths
    len_path = out_dir / "summary_lengths.csv"
    with len_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iter", "mode", "n", "mean", "median", "max"])
        for r in report["lengths"]:
            w.writerow([r["iter"], r["mode"], r["n"], r["mean"], r["median"], r["max"]])

    # CSV: refusal phrases
    ref_path = out_dir / "summary_refusals.csv"
    with ref_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iter", "mode", "phrase", "count"])
        for r in report["refusals"]:
            w.writerow([r["iter"], r["mode"], r["phrase"], r["count"]])

    # Markdown
    md_path = out_dir / "summary.md"
    lines: list[str] = []
    lines.append("# v0.1 — iteration comparison (auto-judge consortium scores)")
    lines.append("")
    lines.append(f"Iterations compared: {', '.join(iters)}")
    lines.append("")
    lines.append("## Per-axis means")
    lines.append("")
    header = "| axis |"
    sep = "|---|"
    for it in iters:
        header += f" {it} overall | {it} direct | {it} rag | {it} Δ rag−direct |"
        sep += "---|" * 4
    lines.append(header)
    lines.append(sep)
    for row in report["axes"]:
        cells = [row["axis"]]
        for it in iters:
            cells += [
                str(row.get(f"{it}_overall")),
                str(row.get(f"{it}_direct")),
                str(row.get(f"{it}_rag")),
                str(row.get(f"{it}_rag_minus_direct")),
            ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Answer length distribution")
    lines.append("")
    lines.append("| iter | mode | n | mean | median | max |")
    lines.append("|---|---|---|---|---|---|")
    for r in report["lengths"]:
        lines.append(f"| {r['iter']} | {r['mode']} | {r['n']} | {r['mean']} | {r['median']} | {r['max']} |")
    lines.append("")
    lines.append("## Refusal-phrase mode-leakage signal")
    lines.append("")
    lines.append("| iter | mode | phrase | count |")
    lines.append("|---|---|---|---|")
    for r in report["refusals"]:
        lines.append(f"| {r['iter']} | {r['mode']} | {r['phrase']} | {r['count']} |")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    for fname in ("axis_evolution.png", "length_evolution.png", "mode_gap_evolution.png"):
        lines.append(f"![{fname}](plots/{fname})")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {md_path}", file=sys.stderr)
    print(f"WROTE {csv_path}", file=sys.stderr)
    print(f"WROTE {len_path}", file=sys.stderr)
    print(f"WROTE {ref_path}", file=sys.stderr)


def _plot(report: dict, per_iter: dict[str, dict] | None = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"WARN: matplotlib unavailable ({e}), skipping plots", file=sys.stderr)
        return
    from benchmark.scripts._plot_style import (
        PALETTE_QUALITATIVE,
        set_publication_style,
    )
    set_publication_style()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    iters = report["iters"]

    AXIS_SHORT = {
        "completeness": "полнота",
        "safety": "мед.\nбезопасн.",
        "sop_applicability": "СОП",
        "uncertainty_handling": "неопред.",
        "hallucination_severity": "галлюц.",
    }
    axis_labels = [AXIS_SHORT.get(a, a.replace("_", "\n")) for a in AXES]

    def _iter_label(it: str) -> str:
        # iter1/2/3 → «Вариант A/B/C» (three parallel prompt variants).
        mapping = {"iter1": "Вариант А", "iter2": "Вариант Б", "iter3": "Вариант В"}
        return mapping.get(it, it)

    def _style_violin(parts, colour, edge_alpha=1.0):
        for b in parts["bodies"]:
            b.set_facecolor(colour)
            b.set_edgecolor("black")
            b.set_alpha(0.75)
            b.set_linewidth(1.1)
        for key in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
            if key in parts and parts[key] is not None:
                parts[key].set_edgecolor("black")
                parts[key].set_color("black")
                parts[key].set_alpha(edge_alpha)
                parts[key].set_linewidth(1.1)

    # 1. Axis evolution as VIOLINS: distribution of individual auto scores
    #    per axis × variant. Direct + RAG pooled (mode-agnostic evolution).
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    n_iters = max(len(iters), 1)
    width_group = 0.8
    width_slot = width_group / n_iters
    x = list(range(len(AXES)))
    for i, it in enumerate(iters):
        offset = (i - (n_iters - 1) / 2) * width_slot
        for j, axis in enumerate(AXES):
            vals = (per_iter[it]["axes_overall"][axis] if per_iter else [])
            if not vals or len(vals) < 2:
                continue
            parts = ax.violinplot(
                vals,
                positions=[j + offset],
                widths=width_slot * 0.9,
                showmeans=True,
                showextrema=True,
            )
            _style_violin(parts, PALETTE_QUALITATIVE[i % len(PALETTE_QUALITATIVE)])
    # Legend proxies
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(facecolor=PALETTE_QUALITATIVE[i], edgecolor="black",
                       alpha=0.65, label=_iter_label(iters[i]))
        for i in range(len(iters))
    ]
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.28), ncol=len(iters))
    ax.set_xticks(x)
    ax.set_xticklabels(axis_labels, rotation=0, ha="center")
    ax.set_ylabel("Оценка коллегии судей (шкала критерия)")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    fig.savefig(PLOT_DIR / "axis_evolution.png", dpi=180)
    plt.close(fig)

    # 2. Length evolution as VIOLINS: distribution of answer lengths per
    #    variant × mode. Shows both means and spread → whether variant C
    #    truly closes the length gap or just moves the median.
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    positions = []
    labels_full: list[str] = []
    for i, it in enumerate(iters):
        for j, (mode, colour) in enumerate((("direct", PALETTE_QUALITATIVE[0]),
                                             ("rag", PALETTE_QUALITATIVE[1]))):
            pos = i * 1.4 + j * 0.6
            positions.append((pos, mode, colour, it))
    for pos, mode, colour, it in positions:
        vals = (per_iter[it]["lengths_by_mode"][mode] if per_iter else [])
        if not vals or len(vals) < 2:
            continue
        parts = ax.violinplot(
            vals,
            positions=[pos],
            widths=0.55,
            showmeans=True,
            showextrema=True,
        )
        _style_violin(parts, colour)
    # X-tick centered under each variant group
    group_centers = [i * 1.4 + 0.3 for i in range(len(iters))]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([_iter_label(it) for it in iters])
    ax.set_ylabel("Длина ответа, симв.")
    ax.set_xlabel("Вариант инструкции")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    legend_handles = [
        mpatches.Patch(facecolor=PALETTE_QUALITATIVE[0], edgecolor="black",
                       alpha=0.65, label="closed-book"),
        mpatches.Patch(facecolor=PALETTE_QUALITATIVE[1], edgecolor="black",
                       alpha=0.65, label="RAG"),
    ]
    ax.legend(handles=legend_handles, loc="best")
    fig.savefig(PLOT_DIR / "length_evolution.png", dpi=180)
    plt.close(fig)

    # 3. Mode gap (RAG − Direct) per axis per iter
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    width = 0.8 / n_iters
    for i, it in enumerate(iters):
        ys = []
        for axis in AXES:
            row = next(r for r in report["axes"] if r["axis"] == axis)
            ys.append(row.get(f"{it}_rag_minus_direct") or 0)
        offset = (i - (n_iters - 1) / 2) * width
        ax.bar(
            [xi + offset for xi in x],
            ys,
            width,
            label=it,
            color=PALETTE_QUALITATIVE[i],
            edgecolor="black",
            linewidth=0.7,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(axis_labels, rotation=0, ha="center")
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_ylabel(r"$\Delta$ RAG $-$ closed-book (среднее по критерию)")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=n_iters)
    fig.savefig(PLOT_DIR / "mode_gap_evolution.png", dpi=180)
    plt.close(fig)
    print(f"WROTE {PLOT_DIR}/*.png", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter-dir", type=Path, default=ITER_DIR_ROOT,
                        help="Root iterations/ directory")
    parser.add_argument("--iters", nargs="+", default=["iter1", "iter2", "iter3"])
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--all-models", action="store_true",
                        help="Compare every model present in each iteration. Off by "
                             "default: iter1/iter2 ran a 6-model subset of iter3, so "
                             "pooling them makes the per-variant numbers reflect a "
                             "different model mix rather than the prompt change.")
    args = parser.parse_args()

    raw: dict[str, list[dict]] = {}
    for it in args.iters:
        scored = args.iter_dir / it / "scored.jsonl"
        if not scored.exists():
            print(f"WARN: {scored} not found — skipping {it}", file=sys.stderr)
            continue
        raw[it] = _load(scored)

    # Restrict to models present in every iteration, so a variant-to-variant
    # difference can only come from the prompt, not from the model mix.
    if not args.all_models and len(raw) > 1:
        common: set[str] | None = None
        for records in raw.values():
            models = {r["model"] for r in records if r.get("item_type") == "open"}
            common = models if common is None else (common & models)
        dropped = sorted(
            {r["model"] for records in raw.values() for r in records
             if r.get("item_type") == "open"} - (common or set()))
        if dropped:
            print(f"  restricting to {len(common)} models common to all iterations; "
                  f"dropped: {', '.join(dropped)}", file=sys.stderr)
        raw = {it: [r for r in recs if r["model"] in (common or set())]
               for it, recs in raw.items()}

    per_iter: dict[str, dict] = {}
    for it, records in raw.items():
        per_iter[it] = _agg_per_iter(records)
        print(f"  {it}: {per_iter[it]['n_open_records']} open records", file=sys.stderr)

    if not per_iter:
        print("FATAL: no iterations loaded", file=sys.stderr)
        return 2

    report = build_report(per_iter)
    write_outputs(report, out_dir=args.output_dir)
    _plot(report, per_iter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
