"""Score the pre-registered predictions (data/preregistration.yaml, frozen before any simulation)
against what the experiments produced. Writes results/predictions.json.

Nothing here changes a prediction. A prediction whose experiment has not run is reported as pending.
"""
from __future__ import annotations

import json

import yaml

from experiments.common import RESULTS, save_json
from experiments.render_docs import ROOT


def _load(name: str) -> dict:
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else {}


def observed() -> dict:
    o, u, cg, cl, si, rv = (_load(f) for f in ("inkling.json", "uncertainty.json", "calibration_gpu.json", "calibration_llama.json",
                                               "silicon_scaled.json", "reviews.json"))
    h, R = o["headline"]["rows"], o["headline"]["ratios"]
    lad, ctx, pm = o["ladder"], o["context"], o["power_management"]
    i8, i1m = ctx["contexts"].index(8192), ctx["contexts"].index(1048576)
    g2 = h["g0_x2"]
    unmanaged = pm["unmanaged"]["tokens_per_joule"]
    f32 = o["frontier"]["by_context"]["32768"]
    ib = o["frontier"]["batches"].index(256)
    gpu_tpj_256 = max(f32[k][ib]["tokens_per_joule"] for k in ("g0_x4", "g0_x2", "g0_x8"))
    out = {
        "P01": (g2["batch1"]["tok_s_user"], ""),
        "P02": (R["g1_x2/g0_x2"]["speed_batch1"], ""),
        "P03": (R["g1_x4/g0_x4"]["speed_batch1"], ""),
        "P04": (R["a_x4/g1_x4"]["speed_batch1"], "four packages against four GPUs"),
        "P05": (R["a_x4/g1_x4"]["tokens_per_joule_batch1"], "four packages against four GPUs"),
        "P06": (R["a_x4/g1_x4"]["cost_batch1"], "four packages against four GPUs"),
        "P07": (o["designs"]["s"]["area"]["dies"], ""),
        "P08": (h["s"]["batch1"]["tok_s_user"], ""),
        "P09": (R["s/g0_x2"]["speed_batch1"], f"against the best B300 layout: {R['s/g0']['speed_batch1']:.1f}x"),
        "P10": (R["s/g0_x2"]["tokens_per_joule_batch1"], f"against the best B300 layout: {R['s/g0']['tokens_per_joule_batch1']:.1f}x"),
        "P11": (unmanaged / g2["batch1"]["tokens_per_joule"], f"against the best B300 layout: {pm['unmanaged']['over_g0_batch1']:.1f}x"),
        "P12": (unmanaged / g2["best_energy"]["tokens_per_joule"], f"against the best B300 layout, the study's lead baseline: {pm['unmanaged']['over_g0_best_batch']:.2f}x, below the range"),
        "P13": (R["s/g0_x2"]["tokens_per_joule_best"], f"against the best B300 layout: {R['s/g0']['tokens_per_joule_best']:.1f}x"),
        "P14": (R["s/g0_x2"]["cost_best"], f"against the best B300 layout: {R['s/g0']['cost_best']:.1f}x"),
        "P15": (si.get("reloadable_over_fixed", {}).get("hardwired_lane_area"),
                "ASAP7, placed and routed; a lane whose weight never changes. With a row's weights time-shared through one lane the fixed variant is larger"),
        "P16": (si.get("reloadable_over_fixed", {}).get("hardwired_lane_energy"), "same designs, gate-level activity on real tensors"),
        "P17": (o["designs"]["s_over_f"]["dies"], ""),
        "P18": (h["f"]["best_energy"]["tokens_per_joule"] / h["s"]["best_energy"]["tokens_per_joule"], ""),
        "P19": (o["designs"]["f"]["area"]["dies"], ""),
        "P20": (lad["inkling2_small"]["gain"]["S"]["speed_batch1"], f"best GPU layout at batch 1: {lad['inkling2_small']['platforms']['G0']['layout_names']['batch1']}"),
        "P21": (lad["rung_100b"]["gain"]["S"]["speed_batch1"], ""),
        "P22": (lad["rung_30b"]["gain"]["S"]["speed_batch1"], ""),
        "P23": (lad["rung_8b"]["gain"]["S"]["speed_batch1"], ""),
        "P24": (o["ladder_fit"]["largest_on_8_dies_params"] / 1e9, "Inkling-shaped rungs only"),
        # untestable: no rung fits one package (P24), so there is no step from multi-board to one package
        "P25": (None if o["ladder_fit"]["largest_on_8_dies_params"] <= 0 else
                lad["rung_8b"]["gain"]["S"]["tokens_per_joule_best"] / lad["rung_30b"]["gain"]["S"]["tokens_per_joule_best"],
                "untestable: no rung fits on eight dies" if o["ladder_fit"]["largest_on_8_dies_params"] <= 0 else "from the ~30B rung to the ~8B rung"),
        "P26": (ctx["s_over_a"]["at_8k"], ""),
        "P27": (ctx["s_over_a"]["at_1m"], "S's attention tile has 26 HBM stacks; A's four packages have their own"),
        "P28": (o["kv_memory"]["crossover_context_vs_a"], f"with an LPDDR pool of {o['kv_memory']['lpddr_pool_bytes_per_s'] / 1e12:.2f} TB/s, chosen after the prediction, which assumed about 0.5 TB/s"),
        "P29": (ctx["rows"]["g0_x2"][i1m]["tok_s_user"] / ctx["rows"]["g0_x2"][i8]["tok_s_user"], f"on 4x B300: {ctx['g0_1m_over_8k']:.2f}"),
        "P30": (o["prefill"]["s_over_g0_per_usd"], ""),
        "P31": (f32["s"][ib]["tokens_per_joule"] / gpu_tpj_256, "batch 256; GPU at the better of its two layouts"),
        "P32": (cl["n_within"], ""),
        "P33": (100 * cg["loo_median_abs_error"], f"largest {100 * cg['loo_max_abs_error']:.0f}%"),
        "P34": (rv.get("largest_headline_change") if rv.get("n_reviews", 0) >= 2 else None, "after both reviews"),
        "P35": (g2["batch1"]["tokens_per_joule"], "GPU power from the measured batch curve, not a flat 60%"),
        "P36": (g2["batch1"]["usd_per_mtok"], ""),
    }
    return out


def main() -> dict:
    pre = yaml.safe_load((ROOT / "data" / "preregistration.yaml").read_text())
    obs = observed()
    rows, hit, scored = [], 0, 0
    for p in pre["predictions"]:
        v, note = obs[p["id"]]
        if v is None:
            verdict = "untestable" if note.startswith("untestable") else "pending"
        else:
            inside = p["low"] <= v <= p["high"]
            verdict = "inside" if inside else ("above" if v > p["high"] else "below")
            scored += 1
            hit += inside
        rows.append({"id": p["id"], "group": p["group"], "claim": p["claim"], "unit": p["unit"], "central": p["central"],
                     "low": p["low"], "high": p["high"], "observed": v, "verdict": verdict, "note": note})
    by_group = {}
    for r in rows:
        g = by_group.setdefault(r["group"], {"inside": 0, "scored": 0})
        if r["verdict"] != "pending":
            g["scored"] += 1
            g["inside"] += r["verdict"] == "inside"
    out = {"label": "predictions frozen before simulation; observed values are modeled unless the row says otherwise",
           "frozen_on": str(pre["frozen_on"]), "rows": rows, "inside": hit, "scored": scored, "by_group": by_group}
    save_json(out, RESULTS / "predictions.json")
    for r in rows:
        ob = "pending" if r["observed"] is None else f"{r['observed']:.3g}"
        print(f"{r['id']} {r['verdict']:8s} predicted {r['central']:>7g} ({r['low']:g}-{r['high']:g}) observed {ob:>8s}  {r['claim'][:70]}")
    print(f"inside range: {hit} of {scored} scored")
    return out


if __name__ == "__main__":
    main()
