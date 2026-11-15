"""Carry the synthesized NVFP4 block (results/silicon.json, ASAP7, placed and routed) into the
constants the system model uses for S and F.

ASAP7 is a predictive 7 nm kit, so three factors take its numbers to an N4P-class process. The
area factor is derived from published cell pitches; the speed and energy factors use only the one
published step (N5 to N4P) and treat ASAP7 as N5-class otherwise, which is conservative. All three
are swept in experiments/uncertainty.py. Writes results/silicon_scaled.json.
"""
from __future__ import annotations

import json

from experiments.common import RESULTS, save_json
from sim.hardware import load_yaml

AREA_ASAP7_TO_N4P = 0.69   # cell height x gate pitch: N5 2-fin (210 x 51 nm) / ASAP7 7.5-track (270 x 54 nm) = 0.73, / 1.06 for N4P; src: asap7-scaling
SPEED_ASAP7_TO_N4P = 1.11  # N4P is 11% faster than N5 (published); ASAP7 treated as N5-class; src: n4p
ENERGY_ASAP7_TO_N4P = 0.82  # N4P is 22% more power-efficient than N5 (published), taken as 1/1.22; src: n4p
UTILIZATION = 0.65         # standard-cell area / placed footprint for a datapath block at this size (assumption; the block itself was placed at 0.5-0.6)


def main() -> dict:
    si = json.loads((RESULTS / "silicon.json").read_text())["silicon"]
    rel = si["headline"]["reloadable"]
    assumed = load_yaml("configs/hw/s_mds1.yaml")
    k1 = si["hardwired_row_length_sweep"]["blocks_per_row"]["1"]
    lane_cell = rel["area_per_mac_lane_um2"]
    lane_n4p = lane_cell * AREA_ASAP7_TO_N4P / UTILIZATION
    pj_flop = rel["energy_per_mac_pj"] / 2.0 * ENERGY_ASAP7_TO_N4P
    f_clk = rel["clock_mhz"] * 1e6 * SPEED_ASAP7_TO_N4P
    a0, e0 = assumed["area"]["um2_per_mac"], assumed["power"]["pj_per_flop"]
    k_area = lane_n4p / a0["fp4"]
    out = {
        "label": "placed and routed on ASAP7 (measured in an open flow), scaled to N4P-class with stated factors",
        "asap7": {"area_per_mac_lane_um2": lane_cell, "energy_per_mac_pj": rel["energy_per_mac_pj"], "clock_mhz": rel["clock_mhz"],
                  "timing_closed": rel["timing_closed"], "best_min_period_ps": si["fmax"]["rel"]["best_min_period_ps"]},
        "factors": {"area": AREA_ASAP7_TO_N4P, "speed": SPEED_ASAP7_TO_N4P, "energy": ENERGY_ASAP7_TO_N4P, "utilization": UTILIZATION},
        "n4p": {"um2_per_mac_fp4": lane_n4p, "pj_per_flop_fp4": pj_flop, "f_clk": f_clk},
        "assumed_before": {"um2_per_mac_fp4": a0["fp4"], "pj_per_flop_fp4": e0["fp4"], "f_clk": assumed["f_clk"]},
        "measured_over_assumed": {"area": k_area, "energy": pj_flop / e0["fp4"], "clock": f_clk / assumed["f_clk"]},
        # Only the NVFP4 lane was built. The MXFP8 and BF16 lanes keep their assumed energy and take the same area correction.
        # the MXFP8 and BF16 lanes were not built: they take the NVFP4 lane's measured/assumed ratios for area and for energy
        "s": {"um2_per_mac": {"fp4": lane_n4p, "fp8": a0["fp8"] * k_area, "bf16": a0["bf16"] * k_area},
              "pj_per_flop": {**e0, "fp4": pj_flop, "fp8": e0["fp8"] * pj_flop / e0["fp4"], "bf16": e0["bf16"] * pj_flop / e0["fp4"]},
              "f_clk": f_clk},
        "reloadable_over_fixed": {
            "hardwired_lane_area": k1["area_ratio_reloadable_over_fixed"]["median"],
            "hardwired_lane_energy": k1["energy_ratio_reloadable_over_fixed"]["median"],
            "time_shared_row_area": si["ratios_reloadable_over_fixed"]["cell_area"]["median"],
            "time_shared_row_energy": si["ratios_reloadable_over_fixed"]["energy_per_block"]["median"],
            "time_shared_area_per_hardwired_weight_um2": si["headline"]["fixed"]["area_per_weight_um2"],
        },
    }
    # F: arithmetic energy of a lane whose weight never changes (the hard-wired case that was built).
    r = out["reloadable_over_fixed"]["hardwired_lane_energy"]
    out["f"] = {"pj_per_flop": {**out["s"]["pj_per_flop"], "fp4": pj_flop / r, "fp8": out["s"]["pj_per_flop"]["fp8"] / r}}
    save_json(out, RESULTS / "silicon_scaled.json")
    figure(si, out)
    print({k: round(v, 3) for k, v in out["measured_over_assumed"].items()}, {k: (round(v, 3) if v < 1e3 else v) for k, v in out["n4p"].items()})
    return out


def figure(si: dict, out: dict) -> None:
    """The synthesized block: reloadable against the two fixed variants, and assumed against synthesized constants."""
    from experiments.common import FIG, PALETTE, plt, style
    style()
    rel, fix = si["headline"]["reloadable"], si["headline"]["fixed"]
    k1 = si["hardwired_row_length_sweep"]["blocks_per_row"]["1"]
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4))
    names = ["reloadable\n(weights from memory)", "fixed, one weight\nper lane", "fixed, 4,608 weights\ntime-shared"]
    area = [rel["cell_area_um2"], k1["fixed_cell_area_um2"]["median"], fix["cell_area_um2"]]
    energy = [rel["energy_per_block_pj"], k1["fixed_energy_per_block_pj"]["median"], fix["energy_per_block_pj"]]
    colors = [PALETTE["S"], PALETTE["F"], PALETTE["F"]]
    for a, vals, yl, t in ((ax[0], area, "cell area of the block (um2)", "Area, placed and routed on ASAP7"),
                           (ax[1], energy, "pJ per 16-value block", "Energy on real tensors")):
        b = a.bar(names, vals, color=colors)
        b[2].set_alpha(0.55)
        for x, v in enumerate(vals):
            a.text(x, v, f"{v:,.0f}" if v > 50 else f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        a.set_ylabel(yl); a.set_title(t, fontsize=9); a.tick_params(axis="x", labelsize=6.5)
    m = out["measured_over_assumed"]
    labels = ["lane area", "energy per\noperation", "clock"]
    vals = [m["area"], m["energy"], m["clock"]]
    b = ax[2].bar(labels, vals, color=PALETTE["S"])
    ax[2].axhline(1.0, color="k", lw=0.6)
    for x, v in enumerate(vals):
        ax[2].text(x, v, f"{v:.2f}x", ha="center", va="bottom", fontsize=7)
    ax[2].set_ylabel("synthesized / assumed (N4P-class)"); ax[2].set_title("What synthesis did to the assumed constants", fontsize=9)
    fig.savefig(FIG / "fig_silicon.png"); plt.close(fig)


if __name__ == "__main__":
    main()
