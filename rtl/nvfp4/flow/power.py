"""Post-route checks on an ORFS result: gate-level simulation with real tensors (Icarus), then
timing and activity-based power with OpenSTA (the copy inside OpenROAD, run in the ORFS container).

The gate-level run drives the routed netlist (ASAP7 cell models, zero delay) with the same
stimulus format as the RTL testbench, compares every output with the NumPy reference, and dumps a
VCD of every net. OpenSTA reads the routed netlist, the extracted parasitics (SPEF), the SDC and
that VCD, and reports power by group. Zero-delay simulation counts functional transitions only:
glitches are not in the VCD (stated in REPORT.md).
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import orfs    # noqa: E402
import simlib  # noqa: E402

ASAP7_V = pathlib.Path(os.environ.get(
    "NVFP4_ASAP7_VERILOG", "/home/user/scratch/rtl/ext/orfs/flow/platforms/asap7/verilog/stdcell"))
CELL_MODELS = ["asap7sc7p5t_AO_RVT_TT_201020.v", "asap7sc7p5t_INVBUF_RVT_TT_201020.v", "asap7sc7p5t_OA_RVT_TT_201020.v",
               "asap7sc7p5t_SIMPLE_RVT_TT_201020.v", "dff.v", "empty.v"]  # dff.v: ORFS's behavioural DFFHQN (the SEQ models need specify-block delays)

STA_TCL = """
set libdir /OpenROAD-flow-scripts/flow/platforms/asap7/lib/NLDM
foreach lib [list asap7sc7p5t_AO_RVT_TT_nldm_211120.lib.gz asap7sc7p5t_INVBUF_RVT_TT_nldm_220122.lib.gz \
                  asap7sc7p5t_OA_RVT_TT_nldm_211120.lib.gz asap7sc7p5t_SIMPLE_RVT_TT_nldm_211120.lib.gz \
                  asap7sc7p5t_SEQ_RVT_TT_nldm_220123.lib] {{ read_liberty $libdir/$lib }}
read_db /work/out/results/asap7/{tag}/base/6_final.odb
read_sdc /work/out/results/asap7/{tag}/base/6_final.sdc
read_spef /work/out/results/asap7/{tag}/base/6_final.spef
puts "@@wns [sta::worst_slack -max]"
puts "@@tns [sta::total_negative_slack -max]"
puts "@@hold_wns [sta::worst_slack -min]"
report_clock_min_period
report_checks -path_delay max -group_path_count 1 -format full
puts "@@vectorless"
report_power -digits 5
{vcd_cmds}
"""
VCD_CMDS = """read_vcd -scope tb/dut /work/power/{name}.vcd
puts "@@activity {name}"
report_activity_annotation
puts "@@power {name}"
report_power -digits 5
"""


def _route_violations(log: pathlib.Path):
    """Violations left after the last detailed-routing iteration (TritonRoute's own count)."""
    if not log.exists():
        return None
    m = re.findall(r"Number of violations = (\d+)", log.read_text())
    return int(m[-1]) if m else None


def collect(tag: str) -> dict:
    """Area, cell counts and timing of a finished ORFS run, from its own final report."""
    base = orfs.WORK / tag / "out"
    rep = json.loads((base / "logs" / "asap7" / tag / "base" / "6_report.json").read_text())
    g = lambda k: rep.get("finish__" + k)  # noqa: E731
    # ORFS counts tap cells as standard cells and fill cells separately; neither is logic
    logic = g("design__instance__count__stdcell") - (g("design__instance__count__class:tap_cell") or 0)
    logic_area = g("design__instance__area__stdcell") - (g("design__instance__area__class:tap_cell") or 0)
    synth = (base / "reports" / "asap7" / tag / "base" / "synth_stat.txt").read_text()
    m = re.search(r"Chip area for module.*?:\s*([\d.]+)", synth)
    mc = re.search(r"^\s*(\d+)\s+\S+\s+\d+\s+\S+\s+cells\s*$", synth, re.M)
    sdc = (orfs.WORK / tag / "constraint.sdc").read_text()
    period = float(re.search(r"set clk_period ([\d.]+)", sdc).group(1))
    return {
        "tag": tag, "clock_period_constraint_ps": period,
        "synth_cell_area_um2": float(m.group(1)) if m else None,
        "synth_cells": int(mc.group(1)) if mc else None,
        "cell_area_um2": round(logic_area, 3),                         # placed and routed, without fill and tap cells
        "cells": int(logic),
        "area_by_class_um2": {k.split("class:")[1]: v for k, v in rep.items()
                              if k.startswith("finish__design__instance__area__class:")},
        "count_by_class": {k.split("class:")[1]: v for k, v in rep.items()
                           if k.startswith("finish__design__instance__count__class:")},
        "sequential_cells": g("design__instance__count__class:sequential_cell"),
        "core_area_um2": g("design__core__area"), "die_area_um2": g("design__die__area"),
        "utilization": g("design__instance__utilization"),
        "setup_wns_ps": g("timing__setup__ws"), "setup_tns_ps": g("timing__setup__tns"),
        "hold_wns_ps": g("timing__hold__ws"),
        "min_period_ps": round(period - g("timing__setup__ws"), 2),
        "fmax_mhz": round(g("timing__fmax") / 1e6, 1) if g("timing__fmax") else None,
        "timing_closed": bool(g("timing__setup__ws") >= 0 and g("timing__hold__ws") >= 0),
        "route_drc_violations": _route_violations(base / "logs" / "asap7" / tag / "base" / "5_2_route.log"),
        "orfs_vectorless_power_w": g("power__total"),
        "flow_seconds": float((orfs.WORK / tag / "flow_seconds.txt").read_text()) if (orfs.WORK / tag / "flow_seconds.txt").exists() else None,
    }


def worst_setup_path(tag: str) -> dict:
    """Start point, end point and logic depth of the worst setup path, from the flow's final timing
    report (6_finish.rpt, the section "finish report_checks -path_delay max"). The depth counts
    cell instances between the launching flip-flop (or input port) and the capturing flip-flop:
    every instance once, whatever the number of report lines it has; the two flip-flops are not counted."""
    rpt = (orfs.WORK / tag / "out" / "reports" / "asap7" / tag / "base" / "6_finish.rpt").read_text()
    sec = rpt.split("finish report_checks -path_delay max\n", 1)[1].split("\n====", 1)[0]
    start = re.search(r"^Startpoint: (\S+)", sec, re.M).group(1)
    end = re.search(r"^Endpoint: (\S+)", sec, re.M).group(1)
    arrival = sec.split("data arrival time", 1)[0]                      # launch clock network and data path
    pins = re.findall(r"^.*?([\d.]+)\s+[\^v]\s+(\S+)/(\w+)\s+\((\w+)\)\s*$", arrival, re.M)   # (time, instance, pin, cell)
    names = [q[1] for q in pins]
    assert names[-1] == end, (names[-1], end)
    first = names.index(start) if start in names else -1               # an input port has no instance line
    path = pins[first + 1:] if first >= 0 else pins
    cells, seen = [], set()
    for _, inst, _, cell in path:
        if inst not in seen and inst not in (start, end):
            seen.add(inst)
            cells.append(cell)
    launch = float(pins[first][0]) if first >= 0 else 0.0              # time at the launching flip-flop's clock pin
    slack = float(re.search(r"([-\d.]+)\s+slack", sec).group(1))
    return {"report": f"reports/asap7/{tag}/base/6_finish.rpt", "startpoint": start, "endpoint": end,
            "cells_on_path": len(cells),
            "of_which_buffers_and_inverters": sum(c.startswith(("BUF", "INV", "HB", "CKINV")) for c in cells),
            "launch_flip_flop_clock_pin_to_endpoint_ps": round(float(pins[-1][0]) - launch, 2),
            "data_arrival_ps": float(pins[-1][0]), "slack_ps": slack}


def gate_sim(tag: str, top: str, rows: int, const: bool, words: list[int], period_ps: float, name: str):
    """Simulate the routed netlist; returns (results [n, rows], vcd path)."""
    work = orfs.WORK / tag
    (work / "power").mkdir(exist_ok=True)
    tb = work / "power" / f"tb_{top}.sv"
    tb.write_text(simlib.gen.tb_sv(simlib.P, top, rows, const))
    netlist = work / "out" / "results" / "asap7" / tag / "base" / "6_final.v"
    vvp = work / "power" / "gate.vvp"
    if not vvp.exists() or vvp.stat().st_mtime < netlist.stat().st_mtime:
        simlib.run(["iverilog", "-g2012", "-s", "tb", f"-Ptb.HALF={int(round(period_ps / 2))}", "-o", str(vvp),
                    str(tb), str(netlist), *[str(ASAP7_V / f) for f in CELL_MODELS]])
    stim, out, vcd = work / "power" / f"{name}.hex", work / "power" / f"{name}.out", work / "power" / f"{name}.vcd"
    stim.write_text("\n".join(f"{w:x}" for w in words) + "\n")
    simlib.run(["vvp", "-n", str(vvp), f"+stim={stim}", f"+out={out}", f"+n={len(words)}", f"+vcd={vcd}"])
    txt = out.read_text().split()
    res = np.array([[(int(x, 16) >> (32 * k)) & 0xFFFFFFFF if "x" not in x.lower() else 0xDEADBEEF for k in range(rows)]
                    for x in txt], np.uint64).reshape(-1, rows)
    return res, vcd


def _power_table(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        m = re.match(r"^(Sequential|Combinational|Clock|Macro|Pad|Total)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", line.strip())
        if m:
            out[m.group(1).lower()] = {"internal_w": float(m.group(2)), "switching_w": float(m.group(3)),
                                       "leakage_w": float(m.group(4)), "total_w": float(m.group(5))}
    return out


def sta(tag: str, vcd_names: list[str]) -> dict:
    """Timing and power of the routed design; one activity-based power figure per VCD."""
    work = orfs.WORK / tag
    (work / "power").mkdir(exist_ok=True)
    (work / "power" / "sta.tcl").write_text(STA_TCL.format(
        tag=tag, vcd_cmds="".join(VCD_CMDS.format(name=n) for n in vcd_names)))
    r = orfs.in_container(work, "openroad -no_init -exit /work/power/sta.tcl > /work/power/sta.log 2>&1; chmod -R a+rwX /work/power")
    log = (work / "power" / "sta.log").read_text()
    if "@@vectorless" not in log:
        raise RuntimeError(f"OpenSTA failed for {tag}: {work}/power/sta.log\n{log[-1500:]}\n{r.stderr[-500:]}")
    out = {"setup_wns_ps": float(re.search(r"@@wns (\S+)", log).group(1)),
           "hold_wns_ps": float(re.search(r"@@hold_wns (\S+)", log).group(1))}
    m = re.search(r"period_min = ([\d.]+) fmax = ([\d.]+)", log)
    if m:
        out["min_period_ps"], out["fmax_mhz"] = float(m.group(1)), float(m.group(2))
    sp = re.search(r"Startpoint: (\S+).*?Endpoint: (\S+)", log, re.S)
    if sp:
        out["critical_path"] = {"from": sp.group(1), "to": sp.group(2)}
    out["vectorless"] = _power_table(log.split("@@vectorless")[1].split("@@activity")[0])
    for n in vcd_names:
        seg = log.split(f"@@activity {n}")[1]
        act, pw = seg.split(f"@@power {n}")
        pw = pw.split("@@activity")[0]
        ann = dict(re.findall(r"^(\w+)\s+(\d+)\s*$", act, re.M))
        out[n] = {"power": _power_table(pw), "activity_annotation": {k: int(v) for k, v in ann.items()}}
    return out


if __name__ == "__main__":
    print(json.dumps(collect(sys.argv[1]), indent=1))
