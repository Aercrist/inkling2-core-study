"""Shared helpers for experiments: paths, plot style, JSON output."""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "dsv41"
FIG = RESULTS
RESULTS.mkdir(parents=True, exist_ok=True)

PALETTE = {"gpu": "#7f8c8d", "gpu_ideal": "#95a5a6", "asic": "#c0392b", "fpga": "#2471a3", "sram": "#8e44ad"}


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
    if isinstance(x, float):
        return round(x, 6) if abs(x) < 1e6 else float(f"{x:.6e}")
    return x


def save_json(obj, path: pathlib.Path) -> None:
    path.write_text(json.dumps(_clean(obj), indent=1, default=float))
