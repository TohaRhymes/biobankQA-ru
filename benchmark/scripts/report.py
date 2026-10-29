"""Aggregate `scored.jsonl` into a markdown report + csv + plots.

Outputs in `benchmark/outputs/reports/`:
- report.md     — human-readable summary
- summary.csv   — per-model row: mcq_acc, avg_latency, hallucination_rate, etc.
- plots/        — matplotlib PNGs
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from benchmark.model_meta import lookup as _model_meta


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "data" / "processed" / "biobankqa_v0_executable.jsonl"
SCORED = REPO_ROOT / "benchmark" / "outputs" / "scored" / "scored.jsonl"
REPORT_DIR = REPO_ROOT / "benchmark" / "outputs" / "reports"
PLOTS_DIR = REPORT_DIR / "plots"


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    COLUMNS = [
        "provider",
        "model",
        "mode",
        "size",
        "context",
        "n_mcq",
        "mcq_f1",
        "mcq_strict_acc",
        "mcq_invalid_fmt",
        "mcq_errors",
        "n_open",
        "open_errors",
        "open_completeness",
        "open_safety",
        "open_actionability",
        "open_halluc_severity",
        "open_uncertainty",
        "avg_latency_ms",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = {i["id"]: i for i in _load_jsonl(DATASET)}
    scored = _load_jsonl(SCORED)
    print(f"Loaded {len(scored)} scored records.", file=sys.stderr)

    # ---- per-model aggregates ----
    # Group by (provider, model, mode) so direct and RAG appear as separate
    # rows. Records without a mode field (legacy) default to "direct".
    rows: list[dict] = []
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for s in scored:
        mode = s.get("mode") or "direct"
        grouped[(s["provider"], s["model"], mode)].append(s)

    for (provider, model, mode), recs in grouped.items():
        meta = _model_meta(model)
        mcq = [r for r in recs if r["item_type"] == "mcq"]
        open_ = [r for r in recs if r["item_type"] == "open"]

        # Errors
        mcq_errors = sum(1 for r in mcq if r["error"] is not None)
        open_errors = sum(1 for r in open_ if r["error"] is not None)

        # MCQ scoring among items with gold set.
        # mcq_score uses F1 (partial credit for multi-correct).
        # mcq_strict_acc requires exact set match (strict accuracy).
        mcq_judged = [r for r in mcq if r.get("mcq_score") is not None]
        mcq_f1 = (
            round(sum(r["mcq_score"] for r in mcq_judged) / len(mcq_judged), 3)
            if mcq_judged else None
        )
        mcq_strict_acc = (
            round(sum(1 for r in mcq_judged if r["mcq_strict_correct"]) / len(mcq_judged), 3)
            if mcq_judged else None
        )
        mcq_invalid_fmt = sum(1 for r in mcq if r.get("mcq_invalid_format"))

        # Open scores
        open_scored = [r for r in open_ if r.get("open_scores")]
        def _avg(key: str) -> float | None:
            vals = [r["open_scores"][key] for r in open_scored if r["open_scores"].get(key) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        # Latency
        lats = [r["latency_ms"] for r in recs if r.get("latency_ms")]
        avg_lat = round(sum(lats) / len(lats)) if lats else None

        rows.append({
            "provider": provider,
            "model": model,
            "mode": mode,
            "size": meta["size"],
            "context": meta["context"],
            "n_mcq": len(mcq),
            "mcq_f1": mcq_f1,
            "mcq_strict_acc": mcq_strict_acc,
            "mcq_invalid_fmt": mcq_invalid_fmt,
            "mcq_errors": mcq_errors,
            "n_open": len(open_),
            "open_errors": open_errors,
            "open_completeness": _avg("completeness"),
            "open_safety": _avg("safety"),
            "open_actionability": _avg("actionability"),
            "open_halluc_severity": _avg("hallucination_severity"),
            "open_uncertainty": _avg("uncertainty_handling"),
            "avg_latency_ms": avg_lat,
        })

    df = pd.DataFrame(rows, columns=COLUMNS).sort_values(
        by=["mcq_f1", "open_safety"], ascending=False, na_position="last"
    )
    csv_path = REPORT_DIR / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"WROTE {csv_path}", file=sys.stderr)

    # ---- plot: MCQ accuracy bar ----
    plot_df = df.dropna(subset=["mcq_f1"]).copy()
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(plot_df))))
        labels = (plot_df["provider"] + "/" + plot_df["model"]).tolist()
        ax.barh(labels, plot_df["mcq_f1"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("MCQ mean F1 (multi-correct partial credit)")
        ax.set_title("BiobankQA-RU v0 — MCQ F1 by model (direct mode, shuffled options)")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "mcq_accuracy.png", dpi=150)
        plt.close(fig)

    # ---- plot: latency boxplot ----
    lat_by_model: dict[str, list[int]] = defaultdict(list)
    for s in scored:
        if s.get("latency_ms"):
            lat_by_model[f"{s['provider']}/{s['model']}"].append(s["latency_ms"])
    if lat_by_model:
        labels = list(lat_by_model.keys())
        data = [lat_by_model[k] for k in labels]
        fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(labels))))
        ax.boxplot(data, vert=False, tick_labels=labels, showfliers=False)
        ax.set_xlabel("latency (ms)")
        ax.set_title("Per-request latency by model")
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "latency.png", dpi=150)
        plt.close(fig)

    # ---- markdown report ----
    md_lines = [
        "# BiobankQA-RU v0 — Direct-mode results",
        "",
        f"- Dataset items: **{len(dataset)}** ({sum(1 for i in dataset.values() if i['type']=='mcq')} MCQ, {sum(1 for i in dataset.values() if i['type']=='open')} open)",
        f"- Models evaluated: **{len(df)}**",
        "- Mode: direct (no extra context, no RAG)",
        "",
        "## Per-model summary",
        "",
        _df_to_md(df),
        "",
        "## Notes",
        "",
        "- `mcq_f1` is mean F1 over MCQ with a gold set (red-marked or manual override). For single-correct items F1 equals exact accuracy; for multi-correct items it gives partial credit (precision × recall harmonic mean).",
        "- `mcq_strict_acc` is the share of items where the model's answer set exactly equals the gold set.",
        "- `mcq_invalid_fmt` counts cases where the model didn't emit any A-E letter.",
        "- Options are shuffled deterministically per item (seed=42) to mitigate positional bias; the model's letter is decoded back to the canonical layout before comparison with red-marked gold.",
        "- Open-ended axes are scored on 0..5 by GPT-5 judge (hallucination_severity is 0/1/2).",
        "- size column: parameter count where the model is open-weights, otherwise tier label (closed APIs don't disclose exact size).",
        "",
        "## Plots",
        "",
        "![MCQ accuracy](plots/mcq_accuracy.png)",
        "",
        "![Latency](plots/latency.png)",
        "",
    ]
    (REPORT_DIR / "report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"WROTE {REPORT_DIR / 'report.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
