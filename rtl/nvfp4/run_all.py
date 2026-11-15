#!/usr/bin/env python3
"""Regenerate everything behind results/silicon.json for the NVFP4 dot-product block.

    python rtl/nvfp4/run_all.py                 # RTL, 1.5 M-vector verification, place and route, power
    python rtl/nvfp4/run_all.py --skip-pnr      # reuse finished place-and-route runs, never start one
    python rtl/nvfp4/run_all.py --skip-verify   # reuse rtl/nvfp4/build/verify.json
    python rtl/nvfp4/run_all.py --quick         # small vector set; for a smoke test only

Steps: (1) rewrite rtl/nvfp4/generated/ from the config and the fixture; (2) verify.py: formats,
bit-exact vectors, Icarus cross-check, batch invariance, PyTorch; (3) flow/: OpenROAD-flow-scripts
on ASAP7 in Docker for every design in PLAN, a post-route gate-level check and OpenSTA power from
VCDs of real tensors; (4) write results/silicon.json. Verification takes about 5 minutes; a
place-and-route run takes 4 to 25 minutes and finished runs are reused. Needs Verilator, Icarus,
torch, and for step 3 Docker with the free image openroad/orfs.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import statistics
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "flow"))

import simlib as S          # noqa: E402
import verify as V          # noqa: E402
import orfs                 # noqa: E402
import pnr_jobs             # noqa: E402
import power                # noqa: E402
import silicon              # noqa: E402

PSTAR = 1800                                    # ps: the common clock constraint for comparisons
SWEEP = [1600, 1800]                            # ps: clock constraints tried for Fmax
PLAN = ([f"rel:{p}" for p in SWEEP] + [f"const_r0:{p}" for p in SWEEP]
        + [f"const_r{r}:{PSTAR}" for r in silicon.POWER_ROWS[1:]]
        + [f"const_t1:{PSTAR}", f"const_t1_b100:{PSTAR}"]                      # 288-block rows, one repeated block
        + [f"const_k1_b{b}:{PSTAR}" if b else f"const_k1:{PSTAR}" for b in silicon.K1_BLOCKS])   # rows of ONE block
# planned at first, then dropped to finish on time; never started (recorded in the JSON)
NOT_RUN = {"rel:1700": "a third clock point", "const_r0:1700": "a third clock point",
           "const_t1_b200:1800": "a third repeated-block design",
           "const_t4:1800, const_t16:1800, const_t72:1800": "weight tables of 4, 16 and 72 distinct blocks",
           "rel_x8:1800, const_x8:1800 (place and route)": "8-row slices were synthesized only"}
WIDE_SYNTH = ["rel_x8", "const_x8"]             # eight rows sharing activations: synthesis only
PUBLISHED_28NM = {"source": "arXiv 2603.28765 (as quoted to us; not re-read here)", "node": "28 nm",
                  "energy_per_16_element_block_pj": 21.5, "clock_mhz": 524, "area_mm2": 0.024}


def spread(vals):
    vals = [v for v in vals if v is not None]
    return {"n": len(vals), "min": min(vals), "median": statistics.median(vals), "max": max(vals)} if vals else None


def container_versions() -> dict:
    if not orfs.docker_ok():
        return {"openroad": "ORFS image not available", "opensta": "ORFS image not available", "orfs_yosys": "n/a"}
    r = subprocess.run(["docker", "run", "--rm", orfs.IMAGE, "bash", "-lc",
                        "cd /OpenROAD-flow-scripts && source ./env.sh >/dev/null 2>&1; yosys -V; "
                        "echo 'puts \"sta [sta::version]\"' > /tmp/v.tcl; openroad -no_init -exit /tmp/v.tcl 2>&1 | grep '^sta'; "
                        "klayout -v"], capture_output=True, text=True).stdout.strip().splitlines()
    img = subprocess.run(["docker", "image", "inspect", orfs.IMAGE, "--format", "{{index .RepoDigests 0}} created {{.Created}}"],
                         capture_output=True, text=True).stdout.strip()
    return {"orfs_image": img, "orfs_yosys": next((x for x in r if x.startswith("Yosys")), "?"),
            "openroad": f"OpenROAD binary of {img.split(' created ')[0]} (reports its version as 'unknown')",
            "opensta": next((x.replace("sta ", "OpenSTA ") + " (inside OpenROAD)" for x in r if x.startswith("sta ")), "?"),
            "klayout": next((x for x in r if x.startswith("KLayout")), "?")}


def slim(m: dict) -> dict:
    keep = ("design", "top", "rows", "fixed_weights", "hardwired_weights", "clock_period_constraint_ps", "synth_cell_area_um2",
            "synth_cells", "cell_area_um2", "cells", "sequential_cells", "area_by_class_um2", "count_by_class", "core_area_um2",
            "die_area_um2", "utilization", "setup_wns_ps", "hold_wns_ps", "min_period_ps", "fmax_mhz", "timing_closed",
            "route_drc_violations", "distinct_blocks_in_table",
            "sta", "vectorless_power_w", "workloads", "post_route_netlist_bit_exact", "flow_seconds")
    return {k: m[k] for k in keep if k in m}


def headline(m: dict, wl: str) -> dict:
    w = m["workloads"][wl]
    lanes = 16 * m["rows"]
    return {"design": m["design"], "top": m["top"], "workload": wl, "workload_description": w["description"],
            "clock_period_ps": m["clock_period_constraint_ps"], "clock_mhz": round(1e6 / m["clock_period_constraint_ps"], 1),
            "timing_closed": m["timing_closed"], "setup_wns_ps": m["setup_wns_ps"], "hold_wns_ps": m["hold_wns_ps"],
            "cell_area_um2": m["cell_area_um2"], "cells": m["cells"], "sequential_cells": m["sequential_cells"],
            "synth_cell_area_um2": m["synth_cell_area_um2"], "core_area_um2": m["core_area_um2"],
            "power_w": w["power_w"], "energy_per_block_pj": w["energy_per_block_pj"],
            "dynamic_energy_per_block_pj": w["dynamic_energy_per_block_pj"],
            "leakage_energy_per_block_pj": w["leakage_energy_per_block_pj"],
            "energy_per_mac_pj": w["energy_per_block_pj"] / lanes,
            "area_per_mac_lane_um2": m["cell_area_um2"] / lanes,
            "weights_in_module": m["hardwired_weights"] or lanes,
            "area_per_weight_um2": m["cell_area_um2"] / (m["hardwired_weights"] or lanes),
            "post_route_netlist_bit_exact": m["post_route_netlist_bit_exact"]}


def pair(rel: dict, fix: dict, wl_rel: str, wl_fix: str) -> dict:
    a, b = headline(rel, wl_rel), headline(fix, wl_fix)
    return {"reloadable": a, "fixed": b,
            "note": ("area_per_weight is not like for like: the reloadable module has 16 weight positions and stores "
                     "nothing; the fixed module's area is divided by the distinct weights hard-wired in it"),
            "ratio_reloadable_over_fixed": {
                "cell_area": a["cell_area_um2"] / b["cell_area_um2"],
                "area_per_mac_lane": a["area_per_mac_lane_um2"] / b["area_per_mac_lane_um2"],
                "area_per_weight": a["area_per_weight_um2"] / b["area_per_weight_um2"],
                "total_power": a["power_w"]["total"] / b["power_w"]["total"],
                "dynamic_power": a["power_w"]["dynamic"] / b["power_w"]["dynamic"],
                "leakage_power": a["power_w"]["leakage"] / b["power_w"]["leakage"],
                "energy_per_block": a["energy_per_block_pj"] / b["energy_per_block_pj"],
                "energy_per_mac": a["energy_per_mac_pj"] / b["energy_per_mac_pj"]}}


def build_silicon(skip_pnr: bool) -> dict:
    have_docker = orfs.docker_ok(start=not skip_pnr)
    runs, failed = {}, {}
    for job in PLAN:
        name, period = job.split(":")
        tag = pnr_jobs.tag_of(name, float(period))
        done = (orfs.WORK / tag / "out" / "results" / "asap7" / tag / "base" / "6_final.v").exists()
        if not done and not skip_pnr and have_docker:
            _, status, _ = pnr_jobs.run_one(job, cores=4)
            done = status == "ok"
            if not done:
                failed[job] = status
        if done and ((orfs.WORK / tag / "power" / "measure.json").exists() or have_docker):
            try:
                runs[job] = {**silicon.measure(name, float(period)), **power.collect(tag)}
            except Exception as e:  # noqa: BLE001
                failed[job] = f"measurement failed: {str(e)[:300]}"
        elif job not in failed:
            failed[job] = "not run (no finished place-and-route result in the work directory)"
    failed.update({k: f"planned, not run: {v}" for k, v in NOT_RUN.items()})
    out = {
        "process": "ASAP7", "library": "asap7sc7p5t RVT standard cells, NLDM, TT corner (0.70 V, 25 C); ORFS CORNER=TC",
        "place_and_route_ran": bool(runs),
        "flow": ("OpenROAD-flow-scripts, platform asap7, in the Docker image openroad/orfs: Yosys/ABC synthesis (flattened), "
                 "floorplan, global and detailed placement, clock-tree synthesis, global and detailed routing, fill, "
                 "parasitic extraction (OpenRCX), final timing with OpenSTA. Power: OpenSTA report_power on the routed "
                 "netlist with SPEF parasitics and a VCD from a zero-delay gate-level simulation (Icarus) of real tensors."),
        "constraints": {"sdc": orfs.SDC.format(top="<top>", period="<clock_ps>"),
                        "core_utilization_percent": 50, "place_density": 0.60, "common_clock_period_ps": PSTAR,
                        "clock_periods_tried_ps": SWEEP},
        "not_done": failed,
        "runs": {job: slim(m) for job, m in runs.items()},
    }
    # Fmax
    fmax = {}
    for name in ("rel", "const_r0"):
        pts = [{"constraint_ps": float(p), "setup_wns_ps": runs[f"{name}:{p}"]["setup_wns_ps"],
                "hold_wns_ps": runs[f"{name}:{p}"]["hold_wns_ps"], "closed": runs[f"{name}:{p}"]["timing_closed"],
                "min_period_ps": runs[f"{name}:{p}"]["min_period_ps"],
                "cell_area_um2": runs[f"{name}:{p}"]["cell_area_um2"],
                "critical_path": runs[f"{name}:{p}"]["sta"].get("critical_path")} for p in SWEEP if f"{name}:{p}" in runs]
        closed = [x["constraint_ps"] for x in pts if x["closed"]]
        fmax[name] = {"points": pts, "closed_at_ps": min(closed) if closed else None,
                      "fmax_mhz_at_closure": round(1e6 / min(closed), 1) if closed else None,
                      "best_min_period_ps": min(x["min_period_ps"] for x in pts) if pts else None}
    out["fmax"] = fmax
    rel = runs.get(f"rel:{PSTAR}")
    if not rel:
        return out
    # reloadable vs fixed, row by row, same tokens, same clock
    rows = {}
    for r in silicon.POWER_ROWS:
        fx = runs.get(f"const_r{r}:{PSTAR}")
        if fx:
            rows[f"w13_row{r}"] = pair(rel, fx, f"w13_row{r}", f"w13_row{r}")
    out["full_row_comparison"] = rows
    if "w13_row0" in rows:
        out["headline"] = {"reloadable": rows["w13_row0"]["reloadable"], "fixed": rows["w13_row0"]["fixed"]}
    out["ratios_reloadable_over_fixed"] = {
        k: spread([v["ratio_reloadable_over_fixed"][k] for v in rows.values()])
        for k in ("cell_area", "area_per_mac_lane", "area_per_weight", "total_power", "dynamic_power", "leakage_power",
                  "energy_per_block", "energy_per_mac")}
    out["fixed_spread_over_rows"] = {
        "rows": list(rows), "cell_area_um2": spread([v["fixed"]["cell_area_um2"] for v in rows.values()]),
        "cells": spread([v["fixed"]["cells"] for v in rows.values()]),
        "total_power_w": spread([v["fixed"]["power_w"]["total"] for v in rows.values()]),
        "energy_per_block_pj": spread([v["fixed"]["energy_per_block_pj"] for v in rows.values()]),
        "area_per_weight_um2": spread([v["fixed"]["area_per_weight_um2"] for v in rows.values()]),
        "setup_wns_ps": spread([v["fixed"]["setup_wns_ps"] for v in rows.values()])}
    # How many different weights a lane sees. "1": 288-block rows whose 16 hard-wired weights are the same in
    # every block (multipliers fold, accumulator fully used). "288": real rows (a 288-entry weight table).
    def sweep_entry(prs):
        return {"designs": [p["fixed"]["design"] for p in prs],
                "fixed_cell_area_um2": spread([p["fixed"]["cell_area_um2"] for p in prs]),
                "fixed_cells": spread([p["fixed"]["cells"] for p in prs]),
                "fixed_flip_flops": spread([p["fixed"]["sequential_cells"] for p in prs]),
                "fixed_setup_wns_ps": spread([p["fixed"]["setup_wns_ps"] for p in prs]),
                "fixed_energy_per_block_pj": spread([p["fixed"]["energy_per_block_pj"] for p in prs]),
                "reloadable_energy_per_block_pj_same_workload": spread([p["reloadable"]["energy_per_block_pj"] for p in prs]),
                "area_ratio_reloadable_over_fixed": spread([p["ratio_reloadable_over_fixed"]["cell_area"] for p in prs]),
                "energy_ratio_reloadable_over_fixed": spread([p["ratio_reloadable_over_fixed"]["energy_per_block"] for p in prs]),
                "dynamic_power_ratio_reloadable_over_fixed": spread([p["ratio_reloadable_over_fixed"]["dynamic_power"] for p in prs]),
                "fixed_area_per_hardwired_weight_um2": spread([p["fixed"]["area_per_weight_um2"] for p in prs])}

    ks = {}
    t1 = [pair(rel, runs[f"{n}:{PSTAR}"], wl, wl) for n, wl in (("const_t1", "t1_b0"), ("const_t1_b100", "t1_b100"))
          if f"{n}:{PSTAR}" in runs]
    if t1:
        ks["1"] = sweep_entry(t1)
    if rows:
        ks["288"] = sweep_entry(list(rows.values()))
    out["hardwired_row_length_sweep"] = {
        "what": ("blocks_per_row is the number of DISTINCT hard-wired blocks a 288-block row cycles through, at one block "
                 "per clock. '1': the same 16 weights in every block, so every lane always multiplies by one constant "
                 "(multipliers fold; the binary32 accumulator is fully used). '288': a real row. The reloadable module "
                 "does the identical work with the same weights driven on its pins and the same 8 tokens."),
        "reloadable_cell_area_um2": rel["cell_area_um2"], "blocks_per_row": ks}
    if "1" in ks and "288" in ks:   # marginal cost of one more distinct hard-wired weight
        a1, a288 = ks["1"]["fixed_cell_area_um2"]["median"], ks["288"]["fixed_cell_area_um2"]["median"]
        out["hardwired_row_length_sweep"]["marginal_area_per_hardwired_weight_um2_1_to_288_blocks"] = (a288 - a1) / (287 * 16)
    k1 = [pair(rel, runs[f"{n}:{PSTAR}"], wl, wl) for n, wl in
          [("const_k1", "k1_b0")] + [(f"const_k1_b{b}", f"k1_b{b}") for b in silicon.K1_BLOCKS[1:]] if f"{n}:{PSTAR}" in runs]
    if k1:
        out["hardwired_row_length_sweep"]["rows_of_one_block_accumulator_optimized_away"] = {
            **sweep_entry(k1),
            "warning": ("rows that are ONE block long: the accumulator always adds to zero, so synthesis removes the binary32 "
                        "adder from the fixed module. Not like for like with the reloadable module, which keeps it; these "
                        "ratios overstate what folding constants saves. Use blocks_per_row['1'].")}
    # eight rows sharing one activation stream: Yosys/ABC synthesis only (not placed and routed)
    wide = {}
    for name in WIDE_SYNTH:
        tag = f"syn_{name}"
        stat = orfs.WORK / tag / "out" / "reports" / "asap7" / tag / "base" / "synth_stat.txt"
        if not stat.exists() and not skip_pnr and have_docker:
            top, kw, _, _ = pnr_jobs.design(name)
            try:
                orfs.run_flow(tag, top, S.gen_rtl(top, **kw), PSTAR, 4, synth_only=True)
            except Exception as e:  # noqa: BLE001
                failed[f"{name} (synthesis)"] = str(e)[:300]
        if stat.exists():
            wide[name] = orfs.synth_stat(tag)
    if len(wide) == 2 and rows:
        fx_single = statistics.median(v["fixed"]["synth_cell_area_um2"] for v in rows.values())
        out["wide_slice_8_rows_synthesis_only"] = {
            "rows": "w13 rows 16..23 of layer 0 expert 0, one activation stream",
            "reloadable_x8": wide["rel_x8"], "fixed_x8": wide["const_x8"],
            "area_ratio_reloadable_over_fixed": wide["rel_x8"]["synth_cell_area_um2"] / wide["const_x8"]["synth_cell_area_um2"],
            "reloadable_x8_area_over_8x_single_row": wide["rel_x8"]["synth_cell_area_um2"] / (8 * rel["synth_cell_area_um2"]),
            "fixed_x8_area_over_8x_median_single_row": wide["const_x8"]["synth_cell_area_um2"] / (8 * fx_single),
            "fixed_x8_area_per_hardwired_weight_um2": wide["const_x8"]["synth_cell_area_um2"] / (8 * 4608),
            "note": "post-synthesis cell area (before placement, clock tree and timing repair); verified in RTL simulation only"}
    if "headline" in out:
        e = out["headline"]["reloadable"]["energy_per_block_pj"]
        out["context_published_28nm"] = {**PUBLISHED_28NM, "this_work_asap7_reloadable_energy_per_block_pj": e,
                                         "this_work_asap7_reloadable_area_mm2": out["headline"]["reloadable"]["cell_area_um2"] * 1e-6,
                                         "note": "different node, library, design and method; context only, not a comparison"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--skip-pnr", action="store_true")
    a = ap.parse_args()
    S.BUILD.mkdir(parents=True, exist_ok=True)
    S.gen.write_committed()
    vpath = S.BUILD / ("verify_quick.json" if a.quick else "verify.json")
    if not (a.skip_verify and vpath.exists()):
        subprocess.run([sys.executable, str(HERE / "verify.py")] + (["--quick"] if a.quick else []), check=True)
    ver = json.loads(vpath.read_text())
    fm = ver["formats"]
    lay = fm["checkpoint_layer0_expert0"]["w13"]
    out = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "rtl/nvfp4/run_all.py" + (" --quick" if a.quick else ""),
        "model": {"repo": "thinkingmachines/Inkling-2-Small", "revision": "8361e7a31ae7e9966a49a91d3994fde618936c2b",
                  "block_size": S.P["B"], "blocks_per_row": S.P["nblk"], "derived_widths": {k: S.P[k] for k in ("DW", "AW", "PW", "MMW")},
                  "fixture": json.loads((ROOT / "data" / "fixtures" / "manifest.json").read_text())["files"]
                  and {"whole_expert": "layer 0, expert 0 (w13 and w2)",
                       "row_samples_layer_expert": json.loads((ROOT / "data" / "fixtures" / "manifest.json").read_text())["row_samples"],
                       "fixed_weight_rows_w13": silicon.POWER_ROWS, "wide_slice_rows_w13": pnr_jobs.WIDE_ROWS}},
        "layout": {
            "nibble_order": "low nibble first",
            "element_order": "byte k of a packed row holds element 2k in bits 3:0 and element 2k+1 in bits 7:4",
            "e2m1_code": "bit 3 sign, bits 2:0 index into (0, 0.5, 1, 1.5, 2, 3, 4, 6); 0x0 and 0x8 are +0 and -0",
            "e4m3": "float8_e4m3fn: e=0 -> m*2^-9; e>=1 -> (8+m)*2^(e-10); 0x7F and 0xFF NaN; no infinity; max 448",
            "weight_value": "code * block_scale * amax[expert] / outer_scale_denominator",
            "weight_outer_scale_denominator": lay["outer_scale_denominator"],
            "activation_value": "code * block_scale * amax[token] / 1536",
            "output_scale": "(amax_token * amax_expert) / (1536 * outer_scale_denominator), applied once per output",
            "w13_rows": "row 2i is the gate row and row 2i+1 the up row of intermediate neuron i",
            "evidence": fm},
        "verification": {k: ver[k] for k in ("vectors", "icarus", "batch", "torch")} | {"formats": {"pass": fm["pass"]}},
        "silicon": build_silicon(a.skip_pnr),
    }
    cv = container_versions()
    out["tools"] = {"verilator": S.tool_version(["verilator", "--version"]), "iverilog": S.tool_version(["iverilog", "-V"]),
                    "yosys": S.tool_version(["yosys", "-V"]) + " (host; lint only)", "python": sys.version.split()[0],
                    "numpy": __import__("numpy").__version__, "torch": ver["torch"]["torch"], **cv}
    path = ROOT / "results" / "silicon.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    s = out["silicon"]
    print(f"written {path}: {ver['vectors']['vectors_total']:,} vectors, {ver['vectors']['mismatches_total']} mismatches; "
          f"{len(s['runs'])} placed-and-routed designs measured, {len(s['not_done'])} not done")


if __name__ == "__main__":
    main()
