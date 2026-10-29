"""Conference-quality matplotlib defaults — call set_publication_style()
once at the top of any plotting function (after the matplotlib backend
is set with matplotlib.use("Agg")).

Palette: Wong 2011 / seaborn 'colorblind' — colorblind-safe qualitative
colors that also reproduce decently in grayscale. Diverging cmap is
RdBu_r for signed heatmaps (self-preference, mode-delta) so 0 = white.
"""
from __future__ import annotations

import matplotlib as mpl


PALETTE_QUALITATIVE = [
    "#0173b2",  # blue
    "#de8f05",  # orange
    "#029e73",  # green
    "#cc78bc",  # pink
    "#ca9161",  # tan
    "#fbafe4",  # light pink
    "#949494",  # grey
    "#ece133",  # yellow
    "#56b4e9",  # light blue
    "#b07aa1",  # mauve
]
PALETTE_DIVERGING = "RdBu_r"


def set_publication_style() -> None:
    """Apply NeurIPS/ICML/CVPR-style matplotlib defaults.

    Idempotent — safe to call from every _plot* function.
    """
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "legend.fontsize": 13,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
    })
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=PALETTE_QUALITATIVE)
