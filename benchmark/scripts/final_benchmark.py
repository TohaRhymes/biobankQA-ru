"""Master v0.1 scoreboard — the definitive table for the paper.

Per (provider, model, mode):
- MCQ: F1 and strict accuracy (from iter3/scored.jsonl, MCQ records)
- Open: per-axis mean (consortium scores from iter3/scored.jsonl, open
        records) + optional per-axis human-annotator mean (from
        annotations.jsonl when present; "n/a" otherwise)
- For each (model, axis) pair: paired sign test for Direct vs RAG
  significance (RAG > Direct? more often than chance?)

Outputs:
- benchmark/outputs/final/benchmark_v0.1.csv         (long form)
- benchmark/outputs/final/benchmark_v0.1.md          (human-readable)
- benchmark/outputs/final/plots/per_model_axes.png   (ranked bar)
- benchmark/outputs/final/plots/mode_delta.png       (RAG − Direct)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from itertools import combinations
from math import comb
from pathlib import Path
from statistics import mean, pstdev


# Bootstrap settings (DA #57 / GH #20). Single seed → reproducible.
BOOTSTRAP_N = 1000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_SEED = 42


def _bh_fdr(pvalues: list[float | None]) -> list[float | None]:
    """Benjamini–Hochberg FDR adjusted p-values.

    Returns adjusted p-values aligned 1:1 with input. None inputs stay
    None and are skipped from the adjustment family.
    """
    indexed = [(i, p) for i, p in enumerate(pvalues) if p is not None]
    if not indexed:
        return list(pvalues)
    indexed.sort(key=lambda x: x[1])
    n = len(indexed)
    adjusted = [None] * len(pvalues)
    running_min = 1.0
    for k in range(n - 1, -1, -1):
        orig_idx, p = indexed[k]
        # BH formula: p * n / (k+1), then enforce monotone non-increasing
        # from the largest rank downwards.
        candidate = p * n / (k + 1)
        running_min = min(running_min, candidate)
        adjusted[orig_idx] = min(1.0, running_min)
    return adjusted


def _bootstrap_ci(
    values: list[float],
    *,
    paired_pairs: list[tuple[float, float]] | None = None,
    agg: str = "mean",
    n_resamples: int = BOOTSTRAP_N,
    alpha: float = BOOTSTRAP_ALPHA,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap 95% CI on an aggregate of `values`.

    Modes:
    - agg='mean' on `values` (unpaired): resample indices of values with
      replacement, recompute mean.
    - agg='mean_diff' with paired_pairs=[(a,b),...]: resample pair-indices,
      compute (b-a) per pair, average.

    Returns (low, high). Returns (None, None) if input too small (n<2).
    """
    rng = random.Random(seed)
    if agg == "mean_diff":
        if not paired_pairs or len(paired_pairs) < 2:
            return None, None
        n = len(paired_pairs)
        boots = []
        for _ in range(n_resamples):
            sample = [paired_pairs[rng.randrange(n)] for _ in range(n)]
            boots.append(sum(b - a for a, b in sample) / n)
    else:
        if not values or len(values) < 2:
            return None, None
        n = len(values)
        boots = []
        for _ in range(n_resamples):
            s = [values[rng.randrange(n)] for _ in range(n)]
            boots.append(sum(s) / n)
    boots.sort()
    lo_idx = max(0, int(n_resamples * alpha / 2))
    hi_idx = min(n_resamples - 1, int(n_resamples * (1 - alpha / 2)))
    return round(boots[lo_idx], 3), round(boots[hi_idx], 3)


REPO_ROOT = Path(__file__).resolve().parents[2]
ITER3_SCORED = REPO_ROOT / "data" / "processed" / "expert_review" / "iterations" / "iter3" / "scored_4judge.jsonl"
ANN_JSONL = REPO_ROOT / "benchmark" / "outputs" / "annotations" / "annotations.jsonl"
OUT_DIR = REPO_ROOT / "benchmark" / "outputs" / "final"
PLOT_DIR = OUT_DIR / "plots"

# Human annotations use v2 axis names (uncertainty_handling); v3 judge output
# uses no_uncertainty. Semantics per the rubric comment (prompts.py:100) are
# the same: higher = model recognises when to defer. AUTO_ALIAS maps
# human-side names → auto-side keys when we look up an auto score.
HUMAN_AXES = [
    "completeness",
    "safety",
    "sop_applicability",
    "uncertainty_handling",
    "hallucination_severity",
]
AUTO_ALIAS = {"uncertainty_handling": "no_uncertainty"}

# (provider, candidate_model) → judge display name in open_scores_individual.
# When candidate == judge, that judge is excluded from the consortium mean to
# neutralise self-preference.
JUDGE_DISPLAY_NAMES = {
    ("gigachat", "GigaChat-2-Max"): "gigachat:GigaChat-2-Max",
    ("yandex", "yandexgpt/latest"): "yandex:yandexgpt/latest",
    ("openrouter", "deepseek/deepseek-v4-pro"): "openrouter:deepseek/deepseek-v4-pro",
    ("openrouter", "qwen/qwen3.7-max"): "openrouter:qwen/qwen3.7-max",
}


def _consortium_scores_excluding_self(rec: dict) -> dict:
    """Compute per-axis consortium mean from individual judge scores, dropping
    the judge that matches the candidate (self-preference control).

    Returns {axis_name: mean_score, ...} with axis names as emitted by the
    judges (v3: no_uncertainty, sop_applicability, use_in_work). Empty dict
    if no individuals or all judges were the self-judge.
    """
    ind = rec.get("open_scores_individual") or {}
    self_judge = JUDGE_DISPLAY_NAMES.get((rec.get("provider"), rec.get("model")))
    per_axis: dict[str, list[float]] = defaultdict(list)
    for judge_name, scores in ind.items():
        if not scores or judge_name == self_judge:
            continue
        for axis, val in scores.items():
            if isinstance(val, (int, float)):
                per_axis[axis].append(float(val))
    return {ax: sum(vs) / len(vs) for ax, vs in per_axis.items() if vs}


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sign_test_pvalue(diffs: list[float]) -> float | None:
    """Two-sided sign test p-value for whether the median diff != 0.

    Ignores zero diffs. Returns None if fewer than 6 non-zero diffs
    (sample too small to be meaningful).
    """
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n < 6:
        return None
    k = min(pos, neg)
    # Two-sided binomial test against H0: P(>0) == 0.5
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def aggregate(iter3_records: list[dict], ann_records: list[dict] | None) -> dict:
    # ---- MCQ: per (provider, model, mode) ----
    mcq_by_key: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in iter3_records:
        if r.get("item_type") == "mcq":
            mcq_by_key[(r["provider"], r["model"], r["mode"])].append(r)

    mcq_summary: dict[tuple[str, str, str], dict] = {}
    for key, recs in mcq_by_key.items():
        scored = [x for x in recs if x.get("mcq_score") is not None]
        f1_vals = [x["mcq_score"] for x in scored]
        f1_lo, f1_hi = _bootstrap_ci(f1_vals)
        # strict accuracy CI via bootstrap on the 0/1 outcomes
        strict_vals = [1.0 if x.get("mcq_strict_correct") else 0.0 for x in scored]
        sa_lo, sa_hi = _bootstrap_ci(strict_vals)
        mcq_summary[key] = {
            "n_total": len(recs),
            "n_scored": len(scored),
            "f1": round(mean(f1_vals), 3) if f1_vals else None,
            "f1_ci_low": f1_lo,
            "f1_ci_high": f1_hi,
            "strict_acc": round(sum(strict_vals) / len(strict_vals), 3) if strict_vals else None,
            "strict_acc_ci_low": sa_lo,
            "strict_acc_ci_high": sa_hi,
            "invalid_format": sum(1 for x in recs if x.get("mcq_invalid_format")),
            "errors": sum(1 for x in recs if x.get("error") is not None),
        }

    # ---- Open: per (provider, model, mode, axis) — self-cell excluded ----
    open_auto: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in iter3_records:
        if r.get("item_type") != "open":
            continue
        scores = _consortium_scores_excluding_self(r)
        key = (r["provider"], r["model"], r["mode"])
        for human_axis in HUMAN_AXES:
            auto_axis = AUTO_ALIAS.get(human_axis, human_axis)
            val = scores.get(auto_axis)
            if val is not None:
                open_auto[key][human_axis].append(float(val))

    open_auto_summary: dict[tuple[str, str, str], dict[str, dict]] = {}
    for key, by_axis in open_auto.items():
        per_axis = {}
        for ax, vs in by_axis.items():
            lo, hi = _bootstrap_ci(vs)
            per_axis[ax] = {
                "mean": round(mean(vs), 3),
                "ci_low": lo,
                "ci_high": hi,
                "n": len(vs),
            }
        open_auto_summary[key] = per_axis

    # ---- Human aggregation (mean across annotators per cell) ----
    human_by_cell: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    if ann_records:
        for r in ann_records:
            if r["axis"] not in HUMAN_AXES:
                continue
            score = r.get("score")
            if score is None:
                continue
            cell = (r["provider"], r["model"], r["mode"], r["item_id"])
            human_by_cell[cell][r["axis"]].append(float(score))
        # Average annotators per cell, then average cells per (model, mode)
    human_summary: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    if human_by_cell:
        per_key_axis: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for (prov, model, mode, item_id), axes in human_by_cell.items():
            for axis, vals in axes.items():
                per_key_axis[(prov, model, mode)][axis].append(mean(vals))
        for key, by_axis in per_key_axis.items():
            for axis, cell_means in by_axis.items():
                lo, hi = _bootstrap_ci(cell_means)
                human_summary[key][axis] = {
                    "mean": round(mean(cell_means), 3),
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": len(cell_means),
                }

    # ---- Direct vs RAG paired tests per (provider, model, axis) ----
    # Pair on item_id so the test is on paired cells. Self-cell excluded.
    open_per_item: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in iter3_records:
        if r.get("item_type") != "open":
            continue
        scores = _consortium_scores_excluding_self(r)
        key = (r["provider"], r["model"])
        for human_axis in HUMAN_AXES:
            auto_axis = AUTO_ALIAS.get(human_axis, human_axis)
            val = scores.get(auto_axis)
            if val is None:
                continue
            open_per_item[key][r["item_id"]].setdefault(human_axis, {})
            open_per_item[key][r["item_id"]][human_axis][r["mode"]] = float(val)

    sig_rows = []
    for (prov, model), items in open_per_item.items():
        for axis in HUMAN_AXES:
            pairs: list[tuple[float, float]] = []  # (direct, rag) for paired bootstrap
            for item_id, by_axis in items.items():
                pair = by_axis.get(axis, {})
                if "direct" in pair and "rag" in pair:
                    pairs.append((pair["direct"], pair["rag"]))
            if len(pairs) < 6:
                continue
            diffs = [b - a for a, b in pairs]
            ci_lo, ci_hi = _bootstrap_ci(diffs, paired_pairs=pairs, agg="mean_diff")
            sig_rows.append({
                "provider": prov,
                "model": model,
                "axis": axis,
                "n_pairs": len(diffs),
                "mean_diff": round(mean(diffs), 3),
                "mean_diff_ci_low": ci_lo,
                "mean_diff_ci_high": ci_hi,
                "std_diff": round(pstdev(diffs), 3),
                "wins_rag": sum(1 for d in diffs if d > 0),
                "wins_direct": sum(1 for d in diffs if d < 0),
                "ties": sum(1 for d in diffs if d == 0),
                "sign_pvalue": _sign_test_pvalue(diffs),
            })

    # Apply BH-FDR correction globally across all sign-test p-values
    # (DA #57 / GH #20). Use the same correction family for all
    # 12 models × 5 axes = up to 60 tests so reviewers don't accuse us
    # of cherry-picking the correction scope.
    pvals = [r["sign_pvalue"] for r in sig_rows]
    pvals_adj = _bh_fdr(pvals)
    for row, p_adj in zip(sig_rows, pvals_adj):
        row["sign_pvalue_bh"] = round(p_adj, 4) if p_adj is not None else None

    return {
        "mcq": mcq_summary,
        "open_auto": open_auto_summary,
        "open_human": human_summary,
        "sig_tests": sig_rows,
    }


def write_outputs(report: dict, *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted(set(report["mcq"]) | set(report["open_auto"]) | set(report["open_human"]))

    # Long-form CSV. Columns include 95% bootstrap CI bands (GH #20).
    csv_path = out_dir / "benchmark_v0.1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = [
            "provider", "model", "mode",
            "mcq_f1", "mcq_f1_ci_low", "mcq_f1_ci_high",
            "mcq_strict_acc", "mcq_strict_acc_ci_low", "mcq_strict_acc_ci_high",
            "mcq_invalid_format", "mcq_errors",
        ]
        for ax in HUMAN_AXES:
            header.extend([
                f"auto_{ax}", f"auto_{ax}_ci_low", f"auto_{ax}_ci_high",
                f"human_{ax}", f"human_{ax}_ci_low", f"human_{ax}_ci_high",
            ])
        w.writerow(header)
        for key in keys:
            prov, model, mode = key
            mcq = report["mcq"].get(key, {})
            row = [
                prov, model, mode,
                mcq.get("f1"), mcq.get("f1_ci_low"), mcq.get("f1_ci_high"),
                mcq.get("strict_acc"), mcq.get("strict_acc_ci_low"), mcq.get("strict_acc_ci_high"),
                mcq.get("invalid_format"), mcq.get("errors"),
            ]
            for ax in HUMAN_AXES:
                a = report["open_auto"].get(key, {}).get(ax) or {}
                h = report["open_human"].get(key, {}).get(ax) or {}
                row.extend([
                    a.get("mean"), a.get("ci_low"), a.get("ci_high"),
                    h.get("mean"), h.get("ci_low"), h.get("ci_high"),
                ])
            w.writerow(row)

    # Significance CSV (paired bootstrap CI on mean_diff + BH-FDR on sign-test p)
    sig_path = out_dir / "rag_vs_direct_significance.csv"
    with sig_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["provider", "model", "axis", "n_pairs", "mean_diff",
                    "mean_diff_ci_low", "mean_diff_ci_high", "std_diff",
                    "wins_rag", "wins_direct", "ties",
                    "sign_pvalue", "sign_pvalue_bh"])
        for row in report["sig_tests"]:
            w.writerow([row[c] for c in (
                "provider", "model", "axis", "n_pairs", "mean_diff",
                "mean_diff_ci_low", "mean_diff_ci_high", "std_diff",
                "wins_rag", "wins_direct", "ties", "sign_pvalue", "sign_pvalue_bh",
            )])

    # Markdown
    md_path = out_dir / "benchmark_v0.1.md"
    lines: list[str] = []
    lines.append("# biobankQA-RU v0.1 — final scoreboard")
    lines.append("")
    has_human = any(report["open_human"].get(k) for k in keys)
    lines.append(f"Human annotator scores: {'PRESENT' if has_human else 'not loaded — auto only'}.")
    lines.append("")
    def _fmt_ci(val, lo, hi):
        if val is None:
            return "—"
        if lo is None or hi is None:
            return str(val)
        return f"{val} [{lo}–{hi}]"

    lines.append("## MCQ (47 closed items)")
    lines.append("")
    lines.append("All values: point estimate [95 % bootstrap CI, 1000 resamples, seed=42].")
    lines.append("")
    lines.append("| provider | model | mode | F1 | strict acc | invalid fmt | errors |")
    lines.append("|---|---|---|---|---|---|---|")
    for key in sorted(report["mcq"]):
        m = report["mcq"][key]
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | "
            f"{_fmt_ci(m.get('f1'), m.get('f1_ci_low'), m.get('f1_ci_high'))} | "
            f"{_fmt_ci(m.get('strict_acc'), m.get('strict_acc_ci_low'), m.get('strict_acc_ci_high'))} | "
            f"{m.get('invalid_format')} | {m.get('errors')} |"
        )
    lines.append("")
    lines.append("## Open (14 situational, axis means [95 % bootstrap CI])")
    lines.append("")
    head = "| provider | model | mode |"
    sep = "|---|---|---|"
    for ax in HUMAN_AXES:
        head += f" auto {ax} | human {ax} |"
        sep += "---|---|"
    lines.append(head)
    lines.append(sep)
    for key in sorted(set(report["open_auto"]) | set(report["open_human"])):
        prov, model, mode = key
        cells = [prov, model, mode]
        for ax in HUMAN_AXES:
            a = report["open_auto"].get(key, {}).get(ax) or {}
            h = report["open_human"].get(key, {}).get(ax) or {}
            cells.append(_fmt_ci(a.get("mean"), a.get("ci_low"), a.get("ci_high")) if a else "—")
            cells.append(_fmt_ci(h.get("mean"), h.get("ci_low"), h.get("ci_high")) if h else "n/a")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## RAG vs Direct significance (paired sign test + BH-FDR + paired bootstrap CI)")
    lines.append("")
    lines.append("- mean_diff: positive = RAG higher than Direct.")
    lines.append("- p_BH: sign-test p-value adjusted via Benjamini–Hochberg FDR across all "
                 "60 (model × axis) tests, so significance threshold is the same as "
                 "for raw p (p_BH < 0.05).")
    lines.append("- CI: 95 % paired bootstrap on the mean difference (1000 resamples, seed=42).")
    lines.append("")
    lines.append("| provider | model | axis | n | mean Δ [95 % CI] | RAG wins | Direct wins | p (sign) | p_BH |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in sorted(report["sig_tests"], key=lambda r: (r["model"], r["axis"])):
        p = row["sign_pvalue"]
        p_str = f"{p:.4f}" if p is not None else "n/a"
        p_bh = row.get("sign_pvalue_bh")
        p_bh_str = f"{p_bh:.4f}" if p_bh is not None else "n/a"
        md_ci = _fmt_ci(row["mean_diff"], row.get("mean_diff_ci_low"), row.get("mean_diff_ci_high"))
        lines.append(
            f"| {row['provider']} | {row['model']} | {row['axis']} | {row['n_pairs']} | "
            f"{md_ci} | {row['wins_rag']} | {row['wins_direct']} | {p_str} | {p_bh_str} |"
        )
    lines.append("")
    lines.append("Plots: see `plots/`.")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {md_path}", file=sys.stderr)
    print(f"WROTE {csv_path}", file=sys.stderr)
    print(f"WROTE {sig_path}", file=sys.stderr)


def _plot(report: dict, *, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"WARN: matplotlib unavailable ({e}), skipping plots", file=sys.stderr)
        return
    from benchmark.scripts._plot_style import (
        PALETTE_QUALITATIVE,
        set_publication_style,
    )
    set_publication_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Short, model-only label. openrouter model names are themselves
    # "vendor/slug" — keep just the slug when it's informative; for
    # generic slugs like "latest" (yandexgpt/latest etc.) keep the
    # full name so we don't collapse 3 models to "latest".
    GENERIC_SLUGS = {"latest", "stable", "default"}
    def _short(provider: str, model: str) -> str:
        if "/" in model:
            vendor, slug = model.split("/", 1)
            if slug in GENERIC_SLUGS:
                return model  # keep "yandexgpt/latest"
            return slug
        return model

    # Tableau-10-inspired saturated colors for crisp print rendering.
    COLOR_DIRECT = "#4C72B0"  # saturated blue
    COLOR_RAG = "#DD8452"     # saturated orange (NOT yellow)
    COLOR_POS = "#55A868"     # green (RAG helps)
    COLOR_NEG = "#C44E52"     # red (RAG hurts)

    # 1. MCQ F1 ranked with CI whiskers + numeric labels (horizontal bars).
    # Replaces the standalone F1 table in the paper — every value and CI
    # bound now visible on the figure itself.
    per_row: dict[tuple[str, str], dict] = {}
    for key, m in report["mcq"].items():
        if m.get("f1") is not None:
            per_row[(_short(key[0], key[1]), key[2])] = m
    if per_row:
        labels = sorted({lbl for (lbl, _) in per_row.keys()})
        rag_lookup = {lbl: per_row[(lbl, "rag")]["f1"]
                      for (lbl, m) in per_row.keys() if m == "rag"}
        labels.sort(key=lambda l: rag_lookup.get(l, 0), reverse=True)
        fig, ax = plt.subplots(figsize=(9.5, 7.5), constrained_layout=True)
        y = list(range(len(labels)))
        bar_h = 0.38
        for i, mode, color, label_ru in [
            (0, "direct", COLOR_DIRECT, "closed-book"),
            (1, "rag", COLOR_RAG, "RAG"),
        ]:
            vals, lo_err, hi_err = [], [], []
            for lbl in labels:
                m = per_row.get((lbl, mode), {})
                v = m.get("f1") or 0
                vals.append(v)
                lo = m.get("f1_ci_low")
                hi = m.get("f1_ci_high")
                lo_err.append((v - lo) if lo is not None else 0)
                hi_err.append((hi - v) if hi is not None else 0)
            positions = [yi + (i - 0.5) * bar_h for yi in y]
            ax.barh(positions, vals, bar_h, label=label_ru,
                    color=color, edgecolor="black", linewidth=0.7,
                    xerr=[lo_err, hi_err], error_kw=dict(
                        ecolor="black", elinewidth=0.9, capsize=2.5))
            # Numeric labels: "F1 [lo–hi]" printed just past the whisker tip.
            for pos, v, lo, hi in zip(positions, vals,
                                       [m.get("f1_ci_low") for lbl in labels
                                        for m in [per_row.get((lbl, mode), {})]],
                                       [m.get("f1_ci_high") for lbl in labels
                                        for m in [per_row.get((lbl, mode), {})]]):
                if lo is None or hi is None:
                    label_txt = f"{v:.2f}"
                else:
                    label_txt = f"{v:.2f} [{lo:.2f}–{hi:.2f}]"
                ax.text(hi + 0.01 if hi is not None else v + 0.01,
                        pos, label_txt, va="center", ha="left", fontsize=7.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("F1-мера на заданиях MSQ (упорядочено по RAG)")
        ax.set_xlim(0, 1.25)  # room for numeric labels
        ax.grid(True, axis="x", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  borderaxespad=0.0, frameon=False)
        fig.savefig(out_dir / "mcq_f1_ranked.png", dpi=180)
        plt.close(fig)

    # 2. Open per-axis mean: 5 axes per model, Direct vs RAG side by side
    axes_keys = HUMAN_AXES
    keys = sorted({(k[0], k[1]) for k in report["open_auto"]})
    if keys:
        n_ax = len(axes_keys)
        fig, axs = plt.subplots(
            n_ax,
            1,
            figsize=(10, 1.8 * n_ax + 1.4),
            constrained_layout=True,
            sharex=True,
        )
        if n_ax == 1:
            axs = [axs]
        labels = [_short(p, m) for p, m in keys]
        x = list(range(len(labels)))
        for ax_idx, axis in enumerate(axes_keys):
            ax = axs[ax_idx]
            d_vals = [(report["open_auto"].get((p, m, "direct"), {}).get(axis) or {}).get("mean", 0) for p, m in keys]
            r_vals = [(report["open_auto"].get((p, m, "rag"), {}).get(axis) or {}).get("mean", 0) for p, m in keys]
            ax.bar(
                [xi - 0.21 for xi in x],
                d_vals,
                0.4,
                label="closed-book",
                color=COLOR_DIRECT,
                edgecolor="black",
                linewidth=0.7,
            )
            ax.bar(
                [xi + 0.21 for xi in x],
                r_vals,
                0.4,
                label="RAG",
                color=COLOR_RAG,
                edgecolor="black",
                linewidth=0.7,
            )
            ax.set_xticks(x)
            # Only the bottom subplot shows x-tick labels (sharex).
            if ax_idx == n_ax - 1:
                ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel(axis.replace("_", "\n"))
            ax.set_ylim(0, 2 if axis == "hallucination_severity" else 5)
            ax.grid(True, axis="y", alpha=0.25, linestyle="--")
            ax.set_axisbelow(True)
        # Single shared legend at the very top of the figure.
        # Placing it above the top subplot avoids overlap with
        # rotated x-tick labels at the bottom.
        handles, labels_legend = axs[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels_legend,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            frameon=False,
        )
        fig.savefig(
            out_dir / "per_model_axes.png",
            dpi=180,
            bbox_inches="tight",
            pad_inches=0.15,
        )
        plt.close(fig)

    # 3. Mode delta (RAG − Direct) per axis per model
    if report["sig_tests"]:
        by_axis: dict[str, list[tuple[str, float, float | None]]] = defaultdict(list)
        for r in report["sig_tests"]:
            by_axis[r["axis"]].append((_short(r["provider"], r["model"]), r["mean_diff"], r["sign_pvalue"]))
        n_axes = len(by_axis)
        # Match the mcq_f1_ranked panel's (7, 7.5) aspect so the two
        # figures sit side by side in the paper without dead whitespace.
        fig, axs = plt.subplots(
            n_axes,
            1,
            figsize=(7, 7.5),
            constrained_layout=True,
            sharex=True,
        )
        if n_axes == 1:
            axs = [axs]
        # Use a fixed model ordering across subplots — alphabetical-by-name,
        # which matches per_model_axes above.
        # Short Russian labels for the y-axis (rotated horizontally, so
        # they fit even in the narrow (7, 7.5) panel).
        AXIS_LABEL_RU = {
            "completeness": r"$\Delta$ полнота",
            "safety": r"$\Delta$ мед. безопасн.",
            "sop_applicability": r"$\Delta$ СОП-примен.",
            "uncertainty_handling": r"$\Delta$ неопред.",
            "hallucination_severity": r"$\Delta$ галлюц.",
        }
        for i, (axis, rows) in enumerate(by_axis.items()):
            ax = axs[i]
            labels = [r[0] for r in rows]
            vals = [r[1] for r in rows]
            colors = [COLOR_POS if v > 0 else COLOR_NEG for v in vals]
            ax.bar(
                range(len(labels)),
                vals,
                color=colors,
                edgecolor="black",
                linewidth=0.6,
                width=0.7,
            )
            ax.axhline(0, color="#444", linewidth=0.8)
            ax.set_xticks(range(len(labels)))
            if i == n_axes - 1:
                ax.set_xticklabels(labels, rotation=30, ha="right")
            else:
                ax.set_xticklabels([])
            ax.set_ylabel(AXIS_LABEL_RU.get(axis, r"$\Delta$ " + axis), fontsize=9)
            # Polarity is ambiguous on hallucination_severity: a positive
            # delta (RAG > Direct) is BAD, not good — flag this on the
            # subplot itself so the green/red colour mapping isn't
            # misleading.
            if axis == "hallucination_severity":
                ax.set_title(
                    r"$\Delta$ галлюц. ($\uparrow$ — больше галлюцинаций)",
                    fontsize=9,
                    loc="left",
                    pad=2,
                )
            ax.grid(True, axis="y", alpha=0.25, linestyle="--")
            ax.set_axisbelow(True)
        fig.savefig(out_dir / "mode_delta.png", dpi=180)
        plt.close(fig)
    print(f"WROTE {out_dir}/*.png", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter3-scored", type=Path, default=ITER3_SCORED)
    parser.add_argument("--annotations", type=Path, default=ANN_JSONL)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if not args.iter3_scored.exists():
        print(f"FATAL: {args.iter3_scored} not found", file=sys.stderr)
        return 2
    iter3 = _load(args.iter3_scored)
    ann = None
    if args.annotations.exists():
        ann = _load(args.annotations)
        print(f"loaded {len(ann)} annotation records", file=sys.stderr)
    else:
        print(f"NOTE: {args.annotations} not present — human columns will be 'n/a'", file=sys.stderr)

    report = aggregate(iter3, ann)
    write_outputs(report, out_dir=args.output_dir)
    _plot(report, out_dir=PLOT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
