#!/usr/bin/env python3
"""Write the numbers of the aercrist.com blog post as a TypeScript module, from results/*.json.

The website post (benblaine/aercrist, src/app/blog/inkling-2-small-on-purpose-built-silicon/) imports
every figure from the module this script writes, so no number on the page is typed by hand.

    python tools/export_site_post.py OUT.ts
"""
from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((R / name).read_text())


def x(v: float, d: int = 1) -> str:
    return f"{v:.{d}f}x"


def n(v: float, d: int = 0) -> str:
    if d < 0:
        return f"{round(v, d):,.0f}"
    return f"{v:,.{d}f}"


def main(out: str) -> None:
    I, U, CG, SI, SS, RV, PR, W = (load(f) for f in ("inkling.json", "uncertainty.json", "calibration_gpu.json", "silicon.json",
                                                    "silicon_scaled.json", "reviews.json", "predictions.json", "phase0_workload.json"))
    H, RS, MC = I["headline"]["rows"], I["headline"]["ratios"]["s/g0"], U["monte_carlo"]
    RG = I["headline"]["ratios"]["g1_x4/g0_x4"]
    E, TH = I["end_to_end"]["rows"], I["thresholds"]["rows"]
    s, g = H["s"], H["g0"]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    speed_range = f"{math.floor(MC['speed_batch1']['median'] + 0.5):.0f}–{math.floor(RS['speed_batch1'] + 0.5):.0f}x"
    d = {
        "commit": commit,
        "speedRange": speed_range,
        "tpj": f"{MC['tokens_per_joule_best']['median']:.0f}x",
        "sTokS": n(s["batch1"]["tok_s_user"], -1),
        "gTokS": n(g["batch1"]["tok_s_user"], -1),
        "gLayoutOneUser": g["layout_names"]["batch1"],
        "gLayoutBusy": g["layout_names"]["best_cost"],
        "fitPoints": CG["n_points"],
        "fitRms": f"{100 * CG['rms_log_error']:.0f}%",
        "arithmeticShare": f"{100 * g['time_by_cause_batch1_ms']['compute'] / g['batch1']['step_ms']:.0f}%",
        "checkpointGb": n(W["bytes"]["total_with_draft"] / 1e9),
        "tokenGb": f"{W['touched_per_token']['total'] / 1e9:.1f} GB",
        "stepGb": f"{W['dspark_step']['bytes_total'] / 1e9:.0f} GB",
        "softwareGain": x(RG["speed_batch1"]),
        "dies": I["designs"]["s"]["area"]["dies"],
        "fixedDesigns": I["designs"]["f"]["cost"]["distinct_die_designs"],
        "vectors": n(SI["verification"]["vectors"]["vectors_total"]),
        "vectorsM": f"{SI['verification']['vectors']['vectors_total'] / 1e6:.1f}M",
        "mismatches": SI["verification"]["vectors"]["mismatches_total"],
        "clockMhz": n(SI["silicon"]["headline"]["reloadable"]["clock_mhz"]),
        "laneSaving": x(SS["reloadable_over_fixed"]["hardwired_lane_area"], 2),
        "laneSavingPct": f"{100 * (SS['reloadable_over_fixed']['hardwired_lane_area'] - 1):.0f}%",
        "costMedian": x(MC["cost_best"]["median"], 1),
        "thresholdTop": n(TH[-1]["tok_s_user_min"]),
        "thresholdTopGain": x(TH[-1]["s"]["cost"], 1),
        "prefillShare": f"{100 * E['g0']['prefill_share_of_cost']:.0f}%",
        "e2eS": x(E["s"]["cost_gain"]),
        "e2eSA": x(E["s_decode_a_prefill"]["cost_gain"]),
        "e2eA": x(E["a_x4"]["cost_gain"]),
        "breakevenT": f"{I['demand']['breakeven_tokens_per_day']['s'] / 1e12:.1f}",
        "ladder8b": x(I["ladder"]["rung_8b"]["gain"]["S"]["speed_batch1"]),
        "predictions": len(PR["rows"]),
        "predictionsInside": PR["inside"],
        "costBefore": x(RV["headline_change"]["s/g0:cost_best"]["before"]),
        "costAfter": x(RV["headline_change"]["s/g0:cost_best"]["after"]),
        "speedBefore": x(RV["headline_change"]["s/g0:speed_batch1"]["before"]),
        "speedAfter": x(RV["headline_change"]["s/g0:speed_batch1"]["after"]),
        "reviewFindings": RV["n_findings"],
        "datasheet": [
            {"term": "Speed for one user", "value": f"{speed_range} faster",
             "line": f"{n(s['batch1']['tok_s_user'], -1)} against {n(g['batch1']['tok_s_user'], -1)} tokens a second · range {x(MC['speed_batch1']['p05'])} to {x(MC['speed_batch1']['p95'])}"},
            {"term": "Tokens per joule, busy and interactive", "value": f"About {MC['tokens_per_joule_best']['median']:.0f}x",
             "line": f"range {x(MC['tokens_per_joule_best']['p05'])} to {x(MC['tokens_per_joule_best']['p95'])}"},
            {"term": "Decode cost at 100 tokens a second per user", "value": "On par",
             "line": f"median {x(MC['cost_best']['median'], 2)} · range {x(MC['cost_best']['p05'], 2)} to {x(MC['cost_best']['p95'], 2)}"},
            {"term": f"Decode cost at {n(TH[-1]['tok_s_user_min'])} tokens a second per user", "value": f"About {TH[-1]['s']['cost']:.0f}x lower",
             "line": "where the GPU model is least certain"},
        ],
        "speedBars": [
            {"name": "B300 GPUs", "sub": "2x, today's engines", "value": H["g0_x2"]["batch1"]["tok_s_user"]},
            {"name": "B300 GPUs", "sub": "4x, today's engines", "value": H["g0_x4"]["batch1"]["tok_s_user"]},
            {"name": "B300 GPUs", "sub": "8x, today's engines", "value": H["g0_x8"]["batch1"]["tok_s_user"]},
            {"name": "B300 GPUs", "sub": "4x, compiled software", "value": H["g1_x4"]["batch1"]["tok_s_user"]},
            {"name": "Core, weights in HBM", "sub": "4 packages", "value": H["a_x4"]["batch1"]["tok_s_user"]},
            {"name": "Core, weights on-die", "sub": f"{I['designs']['s']['area']['dies']} dies, reloadable", "value": s["batch1"]["tok_s_user"], "hi": True},
        ],
        "costBars": [{"name": f"{n(r['tok_s_user_min'])} tok/s", "sub": "per user", "value": r["s"]["cost"], "hi": r is TH[-1]} for r in TH],
        "endToEnd": [
            {"term": "Weights on-die, doing its own prefill", "value": x(E["s"]["cost_gain"]),
             "line": f"lower cost per output token than GPUs · {x(E['s']['tokens_per_joule_gain'])} the tokens per joule"},
            {"term": "Weights on-die for decode, weights in HBM for prefill", "value": x(E["s_decode_a_prefill"]["cost_gain"]),
             "line": f"lower cost · {x(E['s_decode_a_prefill']['tokens_per_joule_gain'])} the tokens per joule"},
            {"term": "Weights in HBM, for both", "value": x(E["a_x4"]["cost_gain"]),
             "line": "lower cost · the cheapest whole system we modeled, and no faster for one user"},
        ],
    }
    for b in d["speedBars"]:
        b["label"] = n(b["value"], -1) + " tok/s"
    for b in d["costBars"]:
        b["label"] = x(b["value"])
    text = ("/* Generated by tools/export_site_post.py in Aercrist/inkling2-core-study at " + commit + ".\n"
            "   Every number on the post comes from here. Do not edit by hand: regenerate. */\n\n"
            "export const D = " + json.dumps(d, indent=2, ensure_ascii=False) + " as const;\n")
    pathlib.Path(out).write_text(text)
    print(f"wrote {out} from {commit}")


if __name__ == "__main__":
    main(sys.argv[1])
