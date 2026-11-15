"""Shared helpers for this study's experiments: paths, plot style, JSON output."""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = RESULTS
RESULTS.mkdir(exist_ok=True)

# One colour per platform family, used in every figure.
PALETTE = {"G0": "#7f8c8d", "G1": "#34495e", "A": "#2471a3", "S": "#c0392b", "F": "#8e44ad", "H": "#d68910"}


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "font.size": 9, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "legend.frameon": False, "savefig.bbox": "tight",
    })


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        try:
            x = x.item()
        except Exception:
            pass
    if isinstance(x, float):
        return float(f"{x:.6g}")  # six significant digits, so microsecond-scale constants survive
    return x


def save_json(obj, path: pathlib.Path) -> None:
    path.write_text(json.dumps(_clean(obj), indent=1, default=float) + "\n")
