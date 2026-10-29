"""Audit the consortium auto-judge for self-preference bias and
ranking robustness under leave-one-out.

Two outputs answer two reviewer questions every NLP venue will ask
about LLM-as-judge:

1. SELF-PREFERENCE — does each judge rate its own answers higher
   than the other judges do?

   For each (judge × candidate-model × axis), we compute the gap
   between the judge's score and the mean of the other judges'
   scores on the same cell. A positive gap on a "self" pair
   (judge == candidate) is the textbook self-preference bias
   (Panickssery et al. 2024). We also track "family" pairs
   (same model lineage — e.g. GigaChat-Max judging GigaChat-2 —
   not technically self, but plausibly biased).

2. LEAVE-ONE-OUT RANKING SENSITIVITY — if we drop one judge from
   the consortium and re-rank the 12 candidates, does the ranking
   move? Spearman correlation between baseline and each LOO ranking
   per axis. ρ ≥ 0.9 → consortium robust; ρ < 0.7 → single-judge
   bias dominates.

Input: data/processed/expert_review/iterations/iter3/scored.jsonl
       (must contain `open_scores_individual` per record).

Outputs (in benchmark/outputs/judge_audit/):
- self_preference.csv         (judge × candidate × axis, with Δ to other judges)
- loo_rankings.csv            (baseline + 3 LOO rankings per axis)
- judge_audit_report.md       (human-readable, with interpretation)
- plots/self_preference_heatmap.png

No API calls: everything is from cached judge verdicts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = REPO_ROOT / "data" / "processed" / "expert_review" / "iterations" / "iter3" / "scored_4judge.jsonl"
OUT_DIR = REPO_ROOT / "benchmark" / "outputs" / "judge_audit"

# Axes as emitted by the v3 4-judge consortium.
AXES = ["completeness", "safety", "sop_applicability", "no_uncertainty", "hallucination_severity"]

# Model-family detection by model-name prefix (NOT provider — qwen and
# DeepSeek are both served on openrouter but are different families).
def _family_of(model: str) -> str:
    if model.startswith("GigaChat-"):
        return "gigachat"
    if model.startswith("yandexgpt"):
        return "yandexgpt"
    if model.startswith("gpt-oss"):
        return "gpt-oss"
    m = model.lower()
    if "deepseek" in m:
        return "deepseek"
    if "qwen" in m:
        return "qwen"
    return "unknown"


def _match_type(judge_key: str, candidate_provider: str, candidate_model: str) -> str:
    """`self` if judge == candidate; `family` if same model lineage; else `unrelated`."""
    cand_key = f"{candidate_provider}:{candidate_model}"
    if judge_key == cand_key:
        return "self"
    # Extract judge model name (strip the "provider:" prefix)
    judge_model = judge_key.split(":", 1)[1] if ":" in judge_key else judge_key
    if _family_of(judge_model) == _family_of(candidate_model):
        return "family"
    return "unrelated"


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _avg_rank(values: list[float]) -> list[float]:
    """Averaged ranks (1-indexed, higher value = lower rank number i.e.
    closer to 1 = best)."""
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: -values[i])  # desc: best first
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman_from_ranks(ra: list[float], rb: list[float]) -> float | None:
    if not ra or len(ra) != len(rb):
        return None
    n = len(ra)
    mra = sum(ra) / n
    mrb = sum(rb) / n
    num = sum((a - mra) * (b - mrb) for a, b in zip(ra, rb))
    dxa = sum((a - mra) ** 2 for a in ra)
    dxb = sum((b - mrb) ** 2 for b in rb)
    denom = (dxa * dxb) ** 0.5
    if denom < 1e-12:
        return None
    return num / denom


def compute_self_preference(records: list[dict]) -> tuple[list[dict], list[str], set[str]]:
    """For each (judge, candidate, mode, axis): mean score by judge +
    Δ vs other judges on the same cell.

    Returns (per_pair_rows, judges_sorted, candidate_keys).
    """
    # Gather: cell key = (provider, model, mode, item_id) → axis → judge → score
    bucket: dict[tuple, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    judges_set: set[str] = set()
    candidate_keys: set[str] = set()
    for r in records:
        if r.get("item_type") != "open":
            continue
        ind = r.get("open_scores_individual") or {}
        if not ind:
            continue
        cell = (r["provider"], r["model"], r["mode"], r["item_id"])
        candidate_keys.add(f"{r['provider']}:{r['model']}")
        for judge, axis_dict in ind.items():
            judges_set.add(judge)
            if not isinstance(axis_dict, dict):
                continue  # judge errored on this record → individual entry is null
            for axis in AXES:
                v = axis_dict.get(axis)
                if v is None:
                    continue
                try:
                    bucket[cell][axis][judge] = float(v)
                except (TypeError, ValueError):
                    continue
    judges = sorted(judges_set)

    # Aggregate per (judge, candidate_provider+candidate_model, axis)
    rows: list[dict] = []
    # group cells by (provider, model)
    by_candidate: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for cell in bucket:
        by_candidate[(cell[0], cell[1])].append(cell)

    for (cand_prov, cand_model), cells in by_candidate.items():
        for judge in judges:
            for axis in AXES:
                # Per-cell scores from this judge and from others
                this_scores: list[float] = []
                gaps: list[float] = []  # self - mean(others)
                n_with_others = 0
                for cell in cells:
                    axis_scores = bucket[cell].get(axis, {})
                    if judge not in axis_scores:
                        continue
                    this_scores.append(axis_scores[judge])
                    others = [v for j, v in axis_scores.items() if j != judge]
                    if others:
                        gaps.append(axis_scores[judge] - mean(others))
                        n_with_others += 1
                if not this_scores:
                    continue
                match = _match_type(judge, cand_prov, cand_model)
                rows.append({
                    "judge": judge,
                    "candidate_provider": cand_prov,
                    "candidate_model": cand_model,
                    "axis": axis,
                    "match_type": match,
                    "n_cells": len(this_scores),
                    "judge_mean_score": round(mean(this_scores), 3),
                    "n_with_others": n_with_others,
                    "mean_gap_vs_others": round(mean(gaps), 3) if gaps else None,
                })
    return rows, judges, candidate_keys


def compute_loo_rankings(records: list[dict], judges: list[str]) -> dict:
    """Per axis: baseline ranking (all judges) + LOO ranking (drop one).
    Returns {axis: {baseline: [...], loo: {judge: [...]}, spearman: {judge: ρ}}}.
    """
    # Per (provider, model, item_id, mode) → axis → {judge: score}
    bucket: dict[tuple, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    candidates_set: set[tuple[str, str]] = set()
    for r in records:
        if r.get("item_type") != "open":
            continue
        ind = r.get("open_scores_individual") or {}
        if not ind:
            continue
        cell = (r["provider"], r["model"], r["mode"], r["item_id"])
        candidates_set.add((r["provider"], r["model"]))
        for judge, axis_dict in ind.items():
            if not isinstance(axis_dict, dict):
                continue
            for axis in AXES:
                v = axis_dict.get(axis)
                if v is None:
                    continue
                try:
                    bucket[cell][axis][judge] = float(v)
                except (TypeError, ValueError):
                    continue
    candidates = sorted(candidates_set)

    def model_mean(cells_for_model, axis, allowed_judges):
        vals: list[float] = []
        for cell in cells_for_model:
            scores = {j: v for j, v in bucket[cell].get(axis, {}).items() if j in allowed_judges}
            if scores:
                vals.append(mean(scores.values()))
        return mean(vals) if vals else None, len(vals)

    # Group cells by candidate
    cells_by_cand: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for cell in bucket:
        cells_by_cand[(cell[0], cell[1])].append(cell)

    result: dict[str, dict] = {}
    for axis in AXES:
        baseline_vals = []
        baseline_n = []
        for cand in candidates:
            v, n = model_mean(cells_by_cand[cand], axis, set(judges))
            baseline_vals.append(v if v is not None else float("nan"))
            baseline_n.append(n)

        # Per-LOO compute vals first; ranking happens below on the intersection
        # of "candidates with a number in BOTH baseline and this LOO" so that
        # missing candidates don't get rank=N and silently distort Spearman ρ
        # (DA #58 SS).
        loo_raw: dict[str, dict] = {}
        for drop in judges:
            allowed = set(judges) - {drop}
            loo_vals = []
            loo_n = []
            for cand in candidates:
                v, n = model_mean(cells_by_cand[cand], axis, allowed)
                loo_vals.append(v if v is not None else float("nan"))
                loo_n.append(n)
            loo_raw[drop] = {"vals": loo_vals, "n": loo_n}

        baseline_ranks_aligned: dict[str, list[float]] = {}
        for drop in judges:
            both_indices = [
                i for i in range(len(candidates))
                if baseline_vals[i] == baseline_vals[i]
                and loo_raw[drop]["vals"][i] == loo_raw[drop]["vals"][i]
            ]
            base_subset = [baseline_vals[i] for i in both_indices]
            loo_subset = [loo_raw[drop]["vals"][i] for i in both_indices]
            base_ranks = _avg_rank(base_subset) if base_subset else []
            loo_ranks_subset = _avg_rank(loo_subset) if loo_subset else []
            rho = _spearman_from_ranks(base_ranks, loo_ranks_subset)
            full_loo_ranks = [float("nan")] * len(candidates)
            for sub_idx, orig_idx in enumerate(both_indices):
                full_loo_ranks[orig_idx] = loo_ranks_subset[sub_idx]
            loo_raw[drop].update({
                "ranks": full_loo_ranks,
                "spearman": round(rho, 3) if rho is not None else None,
                "n_used_for_rho": len(both_indices),
                "n_dropped_from_rho": len(candidates) - len(both_indices),
            })

        # Baseline rank shown in the CSV is on the full set (purely for the
        # CSV / report; Spearman is computed on the intersection above).
        full_baseline_ranks = _avg_rank(
            [baseline_vals[i] for i in range(len(candidates))
             if baseline_vals[i] == baseline_vals[i]]
        )
        full_baseline_ranks_aligned: list[float] = []
        rank_iter = iter(full_baseline_ranks)
        for v in baseline_vals:
            full_baseline_ranks_aligned.append(next(rank_iter) if v == v else float("nan"))

        result[axis] = {
            "candidates": [f"{p}:{m}" for p, m in candidates],
            "baseline_vals": baseline_vals,
            "baseline_n": baseline_n,
            "baseline_ranks": full_baseline_ranks_aligned,
            "loo": loo_raw,
        }
    return result


def write_outputs(
    self_pref_rows: list[dict],
    loo_results: dict,
    judges: list[str],
    *,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. self_preference.csv
    csv_path = out_dir / "self_preference.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["judge", "candidate_provider", "candidate_model", "axis",
                    "match_type", "n_cells", "judge_mean_score",
                    "n_with_others", "mean_gap_vs_others"])
        for r in self_pref_rows:
            w.writerow([r["judge"], r["candidate_provider"], r["candidate_model"],
                        r["axis"], r["match_type"], r["n_cells"], r["judge_mean_score"],
                        r["n_with_others"], r["mean_gap_vs_others"]])
    print(f"WROTE {csv_path}", file=sys.stderr)

    # 2. loo_rankings.csv
    loo_csv = out_dir / "loo_rankings.csv"
    with loo_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["axis", "candidate", "baseline_mean", "baseline_rank", "baseline_n"]
        for j in judges:
            header.extend([f"loo_{j}_mean", f"loo_{j}_rank", f"loo_{j}_n"])
        w.writerow(header)
        for axis in AXES:
            data = loo_results[axis]
            for idx, cand in enumerate(data["candidates"]):
                row = [axis, cand,
                       round(data["baseline_vals"][idx], 3) if data["baseline_vals"][idx] == data["baseline_vals"][idx] else None,
                       data["baseline_ranks"][idx],
                       data["baseline_n"][idx]]
                for j in judges:
                    lo = data["loo"][j]
                    val = lo["vals"][idx]
                    row.extend([
                        round(val, 3) if val == val else None,
                        lo["ranks"][idx],
                        lo["n"][idx],
                    ])
                w.writerow(row)
    print(f"WROTE {loo_csv}", file=sys.stderr)

    # 3. Markdown report
    md = out_dir / "judge_audit_report.md"
    lines = ["# Judge audit: self-preference + leave-one-out", ""]
    lines.append(f"Judges in consortium: {', '.join(judges)}")
    lines.append("")

    lines.append("## Self-preference summary")
    lines.append("")
    lines.append("Cells where the judge IS one of the candidates "
                 "(`match_type=self`). Positive `mean_gap_vs_others` = judge "
                 "rates this candidate HIGHER than the other judges do on "
                 "the same answer (classic self-preference bias).")
    lines.append("")
    lines.append("| judge | axis | n_cells | judge_mean | mean Δ vs others |")
    lines.append("|---|---|---|---|---|")
    for r in sorted([x for x in self_pref_rows if x["match_type"] == "self"],
                    key=lambda x: (x["judge"], x["axis"])):
        lines.append(f"| {r['judge']} | {r['axis']} | {r['n_cells']} | "
                     f"{r['judge_mean_score']} | {r['mean_gap_vs_others']} |")
    lines.append("")

    lines.append("## Family-preference summary")
    lines.append("")
    lines.append("Cells where judge and candidate are from the same model "
                 "family (`match_type=family` — e.g. GigaChat-Max judging "
                 "GigaChat-2). Plausibly biased even when not technically self.")
    lines.append("")
    lines.append("| judge | candidate | axis | n_cells | judge_mean | mean Δ vs others |")
    lines.append("|---|---|---|---|---|---|")
    for r in sorted([x for x in self_pref_rows if x["match_type"] == "family"],
                    key=lambda x: (x["judge"], x["candidate_model"], x["axis"])):
        lines.append(f"| {r['judge']} | {r['candidate_model']} | {r['axis']} | "
                     f"{r['n_cells']} | {r['judge_mean_score']} | {r['mean_gap_vs_others']} |")
    lines.append("")

    lines.append("## Unrelated-pair baseline")
    lines.append("")
    lines.append("Sanity: when judge and candidate are unrelated, |Δ vs others| "
                 "should be small (judge agrees with the others on average). "
                 "Big mean Δ here would mean the judge has a systematic offset "
                 "even on outside models.")
    lines.append("")
    lines.append("| judge | axis | n_pairs | mean Δ vs others |")
    lines.append("|---|---|---|---|")
    unrelated_by_axis: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in self_pref_rows:
        if r["match_type"] == "unrelated" and r["mean_gap_vs_others"] is not None:
            unrelated_by_axis[(r["judge"], r["axis"])].append(r["mean_gap_vs_others"])
    for (judge, axis), gaps in sorted(unrelated_by_axis.items()):
        lines.append(f"| {judge} | {axis} | {len(gaps)} | {round(mean(gaps), 3)} |")
    lines.append("")

    lines.append("## Leave-one-out ranking robustness")
    lines.append("")
    lines.append("For each axis, the 12 candidates are ranked by consortium "
                 "mean score (lower rank number = better). Spearman ρ measures "
                 "how much the ranking moves when one judge is dropped. ρ ≥ 0.9 → "
                 "consortium ranking is robust; ρ < 0.7 → that judge dominates.")
    lines.append("")
    lines.append("| axis | LOO " + " | LOO ".join(judges) + " |")
    lines.append("|---|" + "---|" * len(judges))
    for axis in AXES:
        cells = [axis]
        for j in judges:
            rho = loo_results[axis]["loo"][j]["spearman"]
            cells.append(str(rho))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- `self` mean Δ > 0.3 → judge over-rates own answers, "
                 "consortium score for this model is suspect.")
    lines.append("- `family` mean Δ > 0.3 → systemic provider bias.")
    lines.append("- `unrelated` mean Δ |·| > 0.5 → judge has a global scale "
                 "offset; consortium-mean still OK because offset washes out.")
    lines.append("- LOO ρ < 0.7 on an axis → don't report that axis as a "
                 "ranking; report only the dropped-judge-mean.")
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {md}", file=sys.stderr)


def _plot_heatmap(self_pref_rows: list[dict], *, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except Exception as e:
        print(f"WARN: matplotlib/numpy unavailable ({e}), skipping plot", file=sys.stderr)
        return
    from benchmark.scripts._plot_style import set_publication_style
    set_publication_style()
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Heatmap: judges (rows) × candidates (cols), one heatmap per axis.
    judges = sorted({r["judge"] for r in self_pref_rows})
    candidates = sorted({(r["candidate_provider"], r["candidate_model"]) for r in self_pref_rows})
    # Strip provider prefix from candidate labels — provider info is
    # redundant when the model name already identifies the family.
    # For openrouter models the name is itself "vendor/slug" (e.g.
    # "deepseek/deepseek-v4-flash"); keep just the slug, but for generic
    # slugs ("latest") keep the full name so we don't collapse 3
    # yandexgpt variants to identical labels.
    GENERIC_SLUGS = {"latest", "stable", "default"}
    def _clean_model(m: str) -> str:
        if "/" in m:
            _, slug = m.split("/", 1)
            return m if slug in GENERIC_SLUGS else slug
        return m
    cand_labels = [_clean_model(m) for _, m in candidates]
    judge_labels = [_clean_model(j.split(":", 1)[1] if ":" in j else j) for j in judges]

    for axis in AXES:
        matrix = np.full((len(judges), len(candidates)), np.nan)
        match_overlay = np.full((len(judges), len(candidates)), "", dtype=object)
        for r in self_pref_rows:
            if r["axis"] != axis or r["mean_gap_vs_others"] is None:
                continue
            i = judges.index(r["judge"])
            j = candidates.index((r["candidate_provider"], r["candidate_model"]))
            matrix[i, j] = r["mean_gap_vs_others"]
            match_overlay[i, j] = r["match_type"]  # 'self' | 'family' | 'unrelated'

        fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
        # Default scale [-1, +1] so most axes are visually comparable
        # (Δ on a 0-5 axis usually sits inside ±1). Auto-expand only
        # when a per-axis cell exceeds that range, otherwise extreme
        # values (e.g. GigaChat-Max self-pref on uncertainty ≈ +1.8)
        # get silently clipped.
        finite_vals = matrix[np.isfinite(matrix)]
        max_abs = float(np.max(np.abs(finite_vals))) if finite_vals.size else 0.0
        if max_abs > 1.0:
            # Round up to nearest 0.1 so the colourbar ticks stay tidy.
            vlim = np.ceil(max_abs * 10) / 10
        else:
            vlim = 1.0
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
        # Crisp black border around every cell — prevents the heatmap from
        # looking like a smeared gradient at print resolution.
        for i in range(len(judges)):
            for j in range(len(candidates)):
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="black", linewidth=0.5,
                ))
        ax.set_xticks(range(len(cand_labels)))
        ax.set_xticklabels(cand_labels, rotation=35, ha="right")
        ax.set_yticks(range(len(judge_labels)))
        ax.set_yticklabels(judge_labels)
        ax.set_xlabel("Оцениваемая модель")
        ax.set_ylabel("Модель-судья")
        # Remove spines for a cleaner heatmap look.
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(r"$\Delta$ (судья) $-$ среднее(остальные)")
        cbar.outline.set_visible(False)

        # Annotate each cell with the numeric Δ (or '·' for NaN).
        for i in range(len(judges)):
            for j in range(len(candidates)):
                v = matrix[i, j]
                if np.isnan(v):
                    ax.text(j, i, "·", ha="center", va="center",
                            fontsize=12, color="#888")
                else:
                    # Pick text color for contrast. Threshold scales with
                    # the colorbar range so wider axes (vlim > 1) still
                    # flip text colour around the same visual intensity.
                    text_color = "white" if abs(v) > 0.6 * vlim else "black"
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=9, color=text_color)
                # Bold black border around self/family cells.
                tag = match_overlay[i, j]
                if tag in ("self", "family"):
                    edge = "black" if tag == "self" else "#222"
                    lw = 2.0 if tag == "self" else 1.4
                    rect = mpatches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor=edge, linewidth=lw,
                    )
                    ax.add_patch(rect)

        AXIS_TITLE_RU = {
            "completeness": "полнота",
            "safety": "мед. безопасность рекомендации",
            "sop_applicability": "применимость в формате СОП",
            "no_uncertainty": "признание неопределенности",
            "hallucination_severity": "тяжесть галлюцинаций",
        }
        ax.set_title(
            "Предпочтение собственных ответов (self-preference bias)\n"
            f"критерий: {AXIS_TITLE_RU.get(axis, axis)}",
            pad=8,
        )
        fig.savefig(plot_dir / f"self_pref_{axis}.png", dpi=180)
        plt.close(fig)
    print(f"WROTE {plot_dir}/self_pref_*.png", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if not args.input.exists():
        print(f"FATAL: {args.input} not found", file=sys.stderr)
        return 2
    records = _load(args.input)
    n_with_ind = sum(1 for r in records
                     if r.get("item_type") == "open" and r.get("open_scores_individual"))
    if n_with_ind < 50:
        print(f"FATAL: only {n_with_ind} records have open_scores_individual — "
              "can't audit judges", file=sys.stderr)
        return 2
    print(f"loaded {n_with_ind} open records with individual judge scores",
          file=sys.stderr)
    rows, judges, _ = compute_self_preference(records)
    loo = compute_loo_rankings(records, judges)
    write_outputs(rows, loo, judges, out_dir=args.output_dir)
    _plot_heatmap(rows, out_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
