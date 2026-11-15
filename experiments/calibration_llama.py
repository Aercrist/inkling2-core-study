"""Llama 3.1 8B on four memory architectures: what the two engines predict from public specs.

This is the outside check on the engines before they are used on Inkling 2 Small. Each machine is
described by its published specs only. Where a spec the model needs is not public (chip-to-chip
latency on Groq, anything about HC1's internals), the row says so and gives the bound the public
specs do support. Writes results/calibration_llama.json.
"""
from __future__ import annotations

import yaml

from experiments.common import FIG, PALETTE, RESULTS, ROOT, plt, save_json, style
from experiments.platforms import gpu_cal
from sim import stationary as st
from sim.fast import chain_step_time
from sim.hardware import load_hw
from sim.models.base import Workload
from sim.models.dense import llama31_8b

WL = Workload(batch=1, context=1500, speculation=False)  # 1k in / 1k out: about 1.5k tokens of context on average
TOL = 0.30


def _stationary(model, **kw) -> float:
    """Per-user tok/s on a weight-stationary machine described by a few timing fields."""
    hw = st.load_stationary("configs/hw/s_mds1.yaml", **kw)
    ops = st.prepare(model.decode_step(WL), hw, model.hidden, 0)
    return 1.0 / st.simulate(ops, hw).step_time


def main() -> dict:
    pub = yaml.safe_load((ROOT / "data" / "published_llama31_8b.yaml").read_text())["architectures"]
    ref = {k: next(p for p in v["points"] if p.get("reference")) for k, v in pub.items()}
    cal = gpu_cal()
    rows = {}

    # ---- H200: the GPU engine with the constants fitted on Blackwell engines, then NVIDIA's own two points
    def h200(bits, prec, **over):
        hw = load_hw("configs/hw/gpu_h200.yaml", **{**cal, **over})
        m = llama31_8b(weight_bits=bits, precision=prec)
        t = chain_step_time(m.decode_step(WL), hw)
        return 1.0 / t["step_time"], t
    fp8, t8 = h200(8.0, "fp8")
    bf16, _ = h200(16.0, "bf16")
    itl8, itl16 = 4.24e-3, 7.11e-3  # NVIDIA's published inter-token latencies
    extra_bytes = llama31_8b(16.0).resident_bytes()["total"] - llama31_8b(8.0).resident_bytes()["total"]
    eff = extra_bytes / (itl16 - itl8) / 4.8e12
    fixed = itl8 - t8["hbm_bytes"] / (4.8e12 * eff)
    rows["h200"] = {
        "published": ref["h200"]["value"], "predicted": fp8, "error": fp8 / ref["h200"]["value"] - 1,
        "bf16_published": 140.14, "bf16_predicted": bf16, "bf16_over_fp8_published": 140.14 / 234.95, "bf16_over_fp8_predicted": bf16 / fp8,
        "launches": sum(o.n_kernels for o in llama31_8b().decode_step(WL)),
        "implied_hbm_eff": eff, "implied_fixed_ms": 1e3 * fixed, "model_fixed_ms": 1e3 * (t8["overhead"] + t8["collective"]),
        "how": "GPU engine; HBM efficiency and per-launch latency as fitted on Blackwell engines serving Inkling 2 Small",
    }

    # ---- SRAM machines: one pass over a machine's memory takes capacity / bandwidth
    m16 = llama31_8b(16.0, "bf16")
    ideal = dict(t_pipe=0.0, vector_latency=0.0, router_latency=0.0, hop_latency=0.0, d2d_latency=0.0, step_overhead=0.0,
                 kv_latency=0.0, kv_bw=1e18, kv_window_bw=1e18, attn_peak=1e20, lane_rows_dense=1, tile_frac=0.0)
    cb = _stationary(m16, **{**ideal, "t_scan_fixed": 44e9 / 21e15})
    rows["cerebras"] = {
        "published": ref["cerebras"]["value"], "predicted": cb, "error": cb / ref["cerebras"]["value"] - 1,
        "t_scan_us": 1e6 * 44e9 / 21e15, "published_over_bound": ref["cerebras"]["value"] / cb,
        "implied_us_per_layer": 1e6 / ref["cerebras"]["value"] / m16.n_layers, "bound_us_per_layer": 1e6 / cb / m16.n_layers,
        "how": "weight-stationary engine; one pass = 44 GB / 21 PB/s; nothing else on the clock, so this is an upper bound",
    }
    gq = _stationary(m16, **{**ideal, "t_scan_fixed": 230e6 / 80e12})
    rows["groq"] = {
        "published": ref["groq"]["value"], "predicted": gq, "error": gq / ref["groq"]["value"] - 1,
        "t_scan_us": 1e6 * 230e6 / 80e12, "published_over_bound": ref["groq"]["value"] / gq,
        "implied_us_per_layer": 1e6 / ref["groq"]["value"] / m16.n_layers, "bound_us_per_layer": 1e6 / gq / m16.n_layers,
        "how": "weight-stationary engine; one pass = 230 MB / 80 TB/s; chip-to-chip latency and chips per model are not published, so this is an upper bound",
    }

    # ---- fixed-weight die: our own design constants (configs/hw/f_mdf1.yaml), one die, no hops
    m3 = llama31_8b(4.5, "fp4")
    hc = _stationary(m3, hop_latency=0.0, d2d_latency=0.0, kv_bw=40e12, kv_latency=0.0, lane_rows_dense=1, tile_frac=0.0, attn_peak_full=0.0)
    rows["hc1"] = {
        "published": ref["hc1"]["value"], "predicted": hc, "error": hc / ref["hc1"]["value"] - 1,
        "measured_independently": 14150, "error_vs_measured": hc / 14150 - 1,
        "implied_us_per_layer": 1e6 / ref["hc1"]["value"] / m3.n_layers, "model_us_per_layer": 1e6 / hc / m3.n_layers,
        "how": "weight-stationary engine with this study's own timing constants (bank depth, clock, hand-off), one die, KV on-die; HC1's internals are not published",
    }
    # What the published speeds imply beyond the memory pass, per weight operation (5 per layer plus the head).
    n_ops = sum(1 for o in m16.decode_step(WL) if o.kind == "gemv")
    for k in ("cerebras", "groq"):
        rows[k]["implied_extra_us_per_op"] = 1e6 * (1.0 / rows[k]["published"] - 1.0 / rows[k]["predicted"]) / n_ops
    rows["hc1"]["implied_total_us_per_op"] = 1e6 / rows["hc1"]["published"] / n_ops
    rows["hc1"]["implied_total_us_per_op_measured"] = 1e6 / 14150 / n_ops
    for k, r in rows.items():
        r["label"] = pub[k]["label"]; r["memory"] = pub[k]["memory"]; r["src"] = ref[k]["src"]; r["who"] = ref[k]["who"]
        r["conditions"] = ref[k]["conditions"]; r["within_30pct"] = abs(r["error"]) <= TOL
        r["other_published"] = [p for p in pub[k]["points"] if not p.get("reference")]
    out = {"label": "modeled from public specs against published speeds", "tolerance": TOL, "rows": rows,
           "n_within": sum(r["within_30pct"] for r in rows.values()), "n": len(rows), "context": WL.context,
           "bounds_hold": all(rows[k]["published"] <= rows[k]["predicted"] for k in ("cerebras", "groq")),
           "weight_ops_per_token": n_ops,
           # the two latencies the designs take from these machines (experiments/platforms.py)
           "latency_for_s_us": rows["cerebras"]["implied_extra_us_per_op"],
           "latency_for_f_total_us": rows["hc1"]["implied_total_us_per_op_measured"],
           "latency_groq_us": rows["groq"]["implied_extra_us_per_op"],
           # the whole published time per token divided by the weight operations a token needs: no model
           # of the machine's memory enters, so nothing is subtracted on the strength of a marketing figure
           "op_total_s_us": 1e6 / (rows["cerebras"]["published"] * n_ops),
           "op_total_groq_us": 1e6 / (rows["groq"]["published"] * n_ops)}
    save_json(out, RESULTS / "calibration_llama.json")

    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    keys = ["h200", "groq", "cerebras", "hc1"]
    xs = range(len(keys))
    ax.bar([i - 0.2 for i in xs], [rows[k]["published"] for k in keys], 0.4, color="#bdc3c7", label="published")
    ax.bar([i + 0.2 for i in xs], [rows[k]["predicted"] for k in keys], 0.4, color=PALETTE["S"], label="model, from public specs")
    ax.set_yscale("log"); ax.set_xticks(list(xs))
    ax.set_xticklabels(["H200\n(HBM)", "Groq\n(SRAM, many chips)", "Cerebras\n(SRAM, wafer)", "HC1\n(fixed weights)"], fontsize=7.5)
    ax.set_ylabel("tok/s per user, Llama 3.1 8B"); ax.set_title("Four memory architectures, one model (published vs modeled)"); ax.legend(fontsize=7)
    fig.savefig(FIG / "fig_calibration_llama.png"); plt.close(fig)
    return out


if __name__ == "__main__":
    o = main()
    for k, r in o["rows"].items():
        extra = {x: round(r[x], 3) for x in r if x.startswith(("implied", "bound_us", "model_us", "t_scan", "published_over", "error_vs", "bf16_over"))}
        print(f"{k:9s} published {r['published']:8.0f} predicted {r['predicted']:8.0f} ({r['error']:+.0%}) within30={r['within_30pct']} {extra}")
    print("within 30%:", o["n_within"], "of", o["n"], "| SRAM bounds hold:", o["bounds_hold"])
