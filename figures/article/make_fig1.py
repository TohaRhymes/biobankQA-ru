"""Fig. 1 (F1 on MSQ, closed-book vs RAG): vector export (SVG/PDF) from values printed on the submitted figure.
Run: ../../../../../.venv/bin/python make_fig1.py   (from this directory)"""
import csv, sys, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
D = pathlib.Path(__file__).parent
sys.path.insert(0, str(D.parents[2] / "benchmark/scripts"))
from _plot_style import set_publication_style
set_publication_style()
plt.rcParams["svg.fonttype"] = "none"   # keep text as text (editable)
plt.rcParams["pdf.fonttype"] = 42
raw = list(csv.DictReader((D / "Fig_1_data.csv").open()))
by = {(r["model"], r["mode"]): r for r in raw}
models = sorted({r["model"] for r in raw}, key=lambda m: -float(by[(m, "rag")]["mean"]))
rows = [{"model": m, **{k + suf: by[(m, md)][c] for md, k in (("direct", "cb"), ("rag", "rag")) for suf, c in (("", "mean"), ("_lo", "ci_low"), ("_hi", "ci_high"))}} for m in models]
fig, ax = plt.subplots(figsize=(9.5, 8.0), layout="constrained")
y = np.arange(len(models)); h = 0.38
for i, (mode, label, color) in enumerate([("cb", "closed-book", "#5471aa"), ("rag", "RAG", "#d0885c")]):
    v = np.array([float(r[mode]) for r in rows]); lo = np.array([float(r[mode + "_lo"]) for r in rows]); hi = np.array([float(r[mode + "_hi"]) for r in rows])
    pos = y + (i - 0.5) * h
    ax.barh(pos, v, h, label=label, color=color, edgecolor="black", linewidth=0.7,
            xerr=[v - lo, hi - v], error_kw=dict(ecolor="black", elinewidth=0.9, capsize=2.5))
    for p, m, l, u in zip(pos, v, lo, hi):
        ax.text(u + 0.01, p, f"{m:.2f} [{l:.2f}–{u:.2f}]", va="center", fontsize=7.5)
ax.set_yticks(y, models); ax.invert_yaxis(); ax.set_xlim(0, 1.25)
ax.set_xlabel("F1-мера на заданиях MSQ (упорядочено по RAG)")
ax.grid(True, axis="x", alpha=0.25, linestyle="--"); ax.set_axisbelow(True)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=False)
for ext in ("svg",):
    fig.savefig(D / f"Fig_1.{ext}", dpi=300)
