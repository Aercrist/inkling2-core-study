"""Energy, power and area models.

ASIC energy is bottom-up: bytes and FLOPs from the simulated step times per-unit energies
(pJ/bit for HBM, SRAM, links, LPDDR; pJ/FLOP by precision), plus static power. GPU energy is
top-down: board power = `load_fraction` x TDP, since the GPUs' internal energy split is not
public; `load_fraction` is a swept assumption (configs/hw/*.yaml).

ASIC area is bottom-up from per-block densities (configs/hw/asic_*.yaml `area:`); see
docs/sources.md for every constant.
"""
from __future__ import annotations

from dataclasses import dataclass

from .engine import StepResult
from .hardware import Hardware
from .ops import Op


@dataclass
class EnergyResult:
    joules_per_step: float  # whole TP group
    watts: float  # whole TP group, average over the step
    tokens_per_joule: float
    parts: dict


def step_energy(ops: list[Op], res: StepResult, hw: Hardware, tokens_per_step: float, hit_rate: float = 0.0) -> EnergyResult:
    n = hw.n_chips
    p = hw.power
    if hw.kind == "gpu" or p.get("model", "") == "tdp":
        watts = p.get("load_fraction", 0.7) * hw.tdp_w * n
        j = watts * res.step_time
        return EnergyResult(j, watts, tokens_per_step / j, {"board": j})
    pj = 1e-12
    hbm_bits = res.hbm_bytes * 8 * n
    routed = sum(o.w_dyn for o in ops)
    sram_bits = 8 * (routed * hit_rate + res.hbm_bytes * n)  # every streamed byte lands in SRAM once
    flops = {k: 0.0 for k in ("fp4", "fp8", "bf16", "fp32")}
    for o in ops:
        for k, v in o.flops.items():
            flops[k] += v
        for k, v in o.flops_rep.items():
            flops[k] += v * n
    coll_bits = 8 * sum(o.coll * 2 * (n - 1) / n + o.gather * (n - 1) for o in ops) * n
    host_bits = 8 * sum(o.host for o in ops)
    parts = {
        "hbm": hbm_bits * p["hbm_pj_per_bit"] * pj,
        "sram": sram_bits * p["sram_pj_per_bit"] * pj,
        "compute": sum(flops[k] * p["pj_per_flop"][k] for k in flops) * pj,
        "links": coll_bits * p["link_pj_per_bit"] * pj,
        "host_dram": host_bits * p.get("lpddr_pj_per_bit", 0.0) * pj,
        "noc": hbm_bits * p.get("noc_pj_per_bit", 0.0) * pj,
        "static": p["static_w"] * n * res.step_time,
    }
    vr = p.get("vr_efficiency", 1.0)
    parts["power_delivery_loss"] = sum(parts.values()) * (1.0 / vr - 1.0)
    j = sum(parts.values())
    return EnergyResult(j, j / res.step_time, tokens_per_step / j, parts)


def asic_area(hw: Hardware) -> dict:
    """Die area by block (mm^2) from configs/hw/asic_*.yaml `area:` densities."""
    a = hw.area
    peak_fp8 = hw.peak["fp8"]
    macs = peak_fp8 / 2 / a["clock_hz"]
    blocks = {
        "hbm_phy_ctrl": a["hbm_stacks"] * a["mm2_per_hbm_phy"],
        "sram": hw.sram_bytes / 2**20 * a["mm2_per_mib_sram"],
        "compute": macs * a["um2_per_fp8_mac"] * 1e-6,
        "serdes": a["serdes_lanes"] * a["mm2_per_serdes_lane"],
        "lpddr_phy": a.get("lpddr_channels", 0) * a.get("mm2_per_lpddr_channel", 0.0),
    }
    dies = a.get("dies", 1)
    blocks["d2d_phy"] = (a.get("d2d_mm2_per_die", 0.0) * dies) if dies > 1 else 0.0
    core = sum(blocks.values())
    blocks["noc_ctrl_misc"] = core * a["misc_fraction"]
    logic = sum(blocks.values())
    # HBM PHYs sit on the die edge, half of each die's stacks along each long side
    per_die_stacks = a["hbm_stacks"] / dies
    long_side = per_die_stacks / 2 * a.get("mm_shoreline_per_hbm", 0.0) + a.get("die_margin_mm", 0.0)
    shore = long_side * a.get("min_short_side_mm", 0.0)
    blocks["shoreline_floor_per_die"] = shore
    blocks["die_mm2"] = max(logic / dies, shore)
    blocks["dies"] = dies
    blocks["total"] = blocks["die_mm2"] * dies
    blocks["whitespace"] = blocks["total"] - logic
    return blocks
