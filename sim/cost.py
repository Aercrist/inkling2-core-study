"""Cost model: $/hour per TP group and $ per million output tokens.

GPU: either market rental ($/GPU-hour) or owned TCO (purchase price amortized + power).
ASIC: die cost from wafer price, dies per wafer and a Murphy yield model, plus HBM, package,
board/system share and NRE amortized over a production volume, plus power. All inputs are in
configs/hw/*.yaml `cost:` with sources in docs/sources.md; the uncertain ones are swept in
experiments/tornado.py.
"""
from __future__ import annotations

import math

from .energy import asic_area
from .hardware import Hardware

HOURS_PER_YEAR = 8766.0


def dies_per_wafer(area_mm2: float, wafer_mm: float = 300.0) -> float:
    d = wafer_mm
    return math.pi * (d / 2) ** 2 / area_mm2 - math.pi * d / math.sqrt(2 * area_mm2)


def murphy_yield(area_mm2: float, d0_per_cm2: float) -> float:
    ad = area_mm2 / 100.0 * d0_per_cm2
    return ((1 - math.exp(-ad)) / ad) ** 2 if ad > 0 else 1.0


def asic_unit_cost(hw: Hardware) -> dict:
    c = hw.cost
    ar = asic_area(hw)
    area = ar["die_mm2"]
    dpw = dies_per_wafer(area)
    y = murphy_yield(area, c["d0_per_cm2"])
    die = ar["dies"] * c["wafer_usd"] / (dpw * y)
    hbm_gb = hw.hbm_cap / 1e9
    parts = {
        "die": die,
        "hbm": hbm_gb * c["hbm_usd_per_gb"],
        "package_test": c["package_usd"],
        "host_dram": c.get("lpddr_gb", 0.0) * c.get("lpddr_usd_per_gb", 0.0),
        "system_share": c["system_usd_per_chip"],
        "nre_share": c["nre_usd"] / c["volume_units"],
    }
    parts["total"] = sum(parts.values())
    parts["area_mm2"] = ar["total"]
    parts["die_mm2"] = area
    parts["dies_per_wafer"] = dpw
    parts["yield"] = y
    return parts


def dollars_per_hour(hw: Hardware, watts: float, mode: str = "rental") -> float:
    """$/hour for the whole TP group (n_chips)."""
    c = hw.cost
    n = hw.n_chips
    elec = watts / 1000.0 * c.get("pue", 1.3) * c.get("usd_per_kwh", 0.08)
    if hw.kind == "gpu":
        if mode == "rental":
            return c["rental_usd_per_gpu_hour"] * n
        capex = c["purchase_usd_per_gpu"] * n
        return capex / (c["lifetime_years"] * HOURS_PER_YEAR) + elec
    unit = asic_unit_cost(hw)["total"] if hw.kind == "asic" else c["unit_usd"]
    capex = unit * n
    return capex / (c["lifetime_years"] * HOURS_PER_YEAR) + elec


def usd_per_mtok(tokens_per_s: float, usd_per_hour: float) -> float:
    return usd_per_hour / (tokens_per_s * 3600.0) * 1e6
