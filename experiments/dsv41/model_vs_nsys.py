#!/usr/bin/env python3
"""Falsify the fixed-latency diagnosis with one profile.

Prints the calibrated model's predicted time per decode step in the same kernel groups that
deepseek-ultrafast's own `tools/nsys_breakdown.py` reports (each instant attributed to the
earliest-started running kernel, which is the critical-path view our simulator also uses).

    # on 4 x GB300, in the deepseek-ultrafast repo:
    nsys profile -o step --capture-range=cudaProfilerApi ... target/release/run ...
    python3 tools/nsys_breakdown.py step.sqlite          # their numbers
    python3 tools/model_vs_nsys.py --config gb300_tp4     # our prediction (this repo)

If the measured "all-reduce" + launch gaps + small-kernel groups do not add up to roughly the
predicted fixed share, the diagnosis in docs/01 section 6 is wrong.
"""
import argparse
import collections

from experiments.dsv41.platforms import platform, spec
from sim.engine import simulate_step
from sim.models.dsv41.workload import DecodeWorkload, decode_step

GROUPS = [  # deepseek-ultrafast tools/nsys_breakdown.py group names
    "MoE (gate + route + experts)", "all-reduce / gather", "attention",
    "dense projections (tcgen05: wqkv, wq_b, wo_a, wo_b)", "mHC side branch (dots, mix, prep)",
    "indexer / top-k / compressor / rope", "draft phase (Markov head, FP8 draft head)",
    "verify head + sampler + accept", "engram", "mma.sync GEMVs (drafter blocks, main_proj)",
]


def group(op) -> str:
    if op.phase == "draft":
        if op.name in ("markov_head", "draft_head"):
            return GROUPS[6]
        if op.category == "collective":
            return GROUPS[1]
        return GROUPS[9]
    if op.phase == "head":
        return GROUPS[7]
    return {"moe_routed": GROUPS[0], "moe_shared": GROUPS[0], "router": GROUPS[0], "collective": GROUPS[1],
            "attn_core": GROUPS[2], "attn_proj": GROUPS[3], "mhc": GROUPS[4], "indexer": GROUPS[5],
            "engram": GROUPS[8]}.get(op.category, "other")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["gb300_tp4", "b300_tp8"], default="gb300_tp4")
    a = ap.parse_args()
    key = "gpu_tp4" if a.config == "gb300_tp4" else "gpu_tp8"
    ops = decode_step(spec(), DecodeWorkload(verify_head="fp8" if key == "gpu_tp4" else "bf16",
                                             layout_tp=4 if key == "gpu_tp4" else 8))
    r = simulate_step(ops, platform(key))
    acc = collections.defaultdict(float)
    for i, t in r.segments.items():
        acc[group(ops[i])] += t
    print(f"model: {a.config}, step {r.step_time * 1e6:.0f} us (calibrated)")
    print("| group | predicted us / step | share |\n|---|---:|---:|")
    for g, v in sorted(acc.items(), key=lambda kv: -kv[1]):
        print(f"| {g} | {v * 1e6:.0f} | {100 * v / r.step_time:.0f}% |")
    print(f"\npredicted fixed (dependent-stage + collective latency) share: "
          f"{100 * (r.breakdown['overhead'] + r.breakdown['collective']) / r.step_time:.0f}%")
    print("note: the calibration pins the *sum* of fixed costs, not how they split between groups "
          "(collective latencies of 2.37-10 us fit equally well). Compare totals first.")


if __name__ == "__main__":
    main()
