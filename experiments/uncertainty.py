"""Uncertainty on the headline comparison (S against G0 on 4x B300): ablation, tornado, joint Monte Carlo.

Every uncertain input has a low and a high value with a stated reason (PARAMS). The tornado
moves one at a time; the Monte Carlo draws all of them together. Writes results/uncertainty.json.
"""
from __future__ import annotations

import math
import random

from experiments.common import FIG, PALETTE, RESULTS, plt, save_json, style
from experiments.platforms import gpu_cal, hardware, llama_cal, run, shape
from sim.models.base import Workload

from experiments.inkling import BEST_BATCHES as BATCHES  # same grid as the headline table
SLA = 100.0

# name: (target, kind, low, high, why). kind "abs" sets the value, "mul" multiplies the base value.
PARAMS = {
    # ---- shared by every platform
    "acceptance length": ("wl", "abs:accept_len", 3.0, 4.0, "published 3.3-3.5; wider to cover other workloads"),
    "routing locality": ("wl", "abs:locality", 0.0, 0.5, "no Inkling routing traces; neighbouring tokens share experts in open models"),
    "routing skew (Zipf)": ("wl", "abs:skew", 0.0, 1.1, "uniform to 'top 20% of experts get 80%'"),
    # ---- GPU
    "GPU power, share of TDP": ("g", "mul:load_fraction", 0.75, 1.25, "measured on H100, carried to B300"),
    "GPU price": ("g", "abs:cost.purchase_usd_per_gpu", 45000.0, 76000.0, "reseller range per GPU with networking"),
    "GPU attention kernel efficiency": ("g", "mul:attn_eff", 0.5, 1.8, "0.5x matches the engine's own kernel timings; 1.8x covers the fit's alternative readings"),
    "GPU per-launch latency": ("g", "mul:kernel_overhead", 0.92, 1.05, "range across the fit's alternative readings"),
    "GPU HBM efficiency": ("g", "abs:hbm_eff", 0.66, 0.93, "range across the fit's alternative readings"),
    # ---- S: timing
    "S per-operation latency": ("s", "abs:op_total", 1.2e-6, 13.05e-6, "whole time per weight operation: S's own bank pass plus half again, to a published many-chip SRAM system's; central is a published wafer-scale machine's (3.9 us)"),
    "S lane clock": ("s", "mul:f_clk", 0.9, 1.3, "timing as closed on the open kit with no process speed-up, to the best period reached with a full 7-to-4 nm speed-up"),
    "S bank depth (rows per pass)": ("s", "abs:scan_rows", 256, 1024, "design choice: deeper banks are denser and slower"),
    "S hop between tiles": ("s", "abs:hop_latency", 0.1e-6, 2.0e-6, "no public latency for a link of this kind"),
    "S hop inside a tile": ("s", "abs:d2d_latency", 0.02e-6, 0.5e-6, "standard under 2 ns; buffering unknown"),
    "S KV memory bandwidth": ("s", "mul:kv_bw_per_stack", 0.5, 2.0, "half to twice MDA-1's HBM stack rate"),
    "S attention tile rate": ("s", "mul:attn_peak_full", 0.5, 2.0, "sized by assumption; central equals what four B300 reach in the fit"),
    # ---- S: area, energy, cost
    "S SRAM density": ("s", "abs:area.mm2_per_mib_sram", 0.3, 0.6, "macro-level to die-level published densities"),
    "S multiplier area": ("s", "mul:area.um2_per_mac", 0.5, 2.0, "synthesis scaled across nodes"),
    "S SRAM read energy": ("s", "abs:power.sram_read_pj_per_bit", 0.05, 0.5, "full-row scan estimate to MDA-1's value"),
    "S arithmetic energy": ("s", "mul:power.pj_per_flop", 0.2, 2.0, "measured 5 nm 4-bit chip is far below the assumed value"),
    "S SRAM leakage": ("s", "abs:power.leak_w_per_gb", 0.22, 4.4, "measured at 25 C, times 1 to 20 for temperature"),
    "S retention factor": ("s", "abs:power.retention_factor", 0.05, 0.31, "published range"),
    "S logic static power per die": ("s", "abs:power.logic_static_w_per_die", 1.0, 15.0, "assumption"),
    "S idle link power per tile": ("s", "abs:power.link_idle_w_per_tile", 0.0, 36.0, "no public figure"),
    "S wafer price": ("s", "abs:cost.wafer_usd", 16000.0, 21300.0, "analyst band"),
    "S package and test per die": ("s", "abs:cost.package_test_usd_per_die", 400.0, 1250.0, "area-scaled substrate only, to the HBM chip's own per-die rate"),
    "S boards, power, cooling per die": ("s", "abs:cost.system_usd_per_die", 150.0, 1000.0, "assumption"),
    "S lifetime (years)": ("s", "abs:cost.lifetime_years", 2.0, 5.0, "a chip laid out for one architecture can be retired before a GPU, or can serve its successors longer"),
    "S seller's gross margin": ("s", "abs:cost.gross_margin", 0.0, 0.5, "costed to its owner in the central case; GPUs are at street price"),
    "S design NRE": ("s", "abs:cost.nre_usd", 30e6, 436e6, "a fixed-weight start-up's total to the IBS 5 nm estimate"),
    "fleet demand (tokens per day)": ("demand", "abs:demand", 0.5e12, 20e12, "assumption: design cost is spread over the systems this demand needs"),
}


def _apply(base: dict, kind: str, value):
    """Return overrides for hardware() from one parameter setting."""
    mode, field = kind.split(":")
    if "." in field:
        outer, inner = field.split(".")
        cur = base[outer][inner]
        if mode == "mul":
            new = {k: v * value for k, v in cur.items()} if isinstance(cur, dict) else cur * value
        else:
            new = value
        return {outer: {inner: new}}
    return {field: (base[field] * value if mode == "mul" else value)}


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = {**out[k], **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def evaluate(settings: dict) -> dict:
    """Headline ratios (S over the GPUs' best of 2x, 4x and 8x B300 for each metric) for one setting of the uncertain inputs."""
    m = shape()
    wl_kw, g_over, s_over = {}, {}, {}
    cal = gpu_cal()
    gbase = {**cal, "load_fraction": 1.0, "cost": {"purchase_usd_per_gpu": 60000.0}}
    hs = hardware("s")
    sbase = {"f_clk": hs.f_clk, "t_pipe": hs.t_pipe, "scan_rows": hs.scan_rows, "hop_latency": hs.hop_latency, "d2d_latency": hs.d2d_latency,
             "kv_bw_per_stack": hs.kv_bw_per_stack, "attn_peak_full": hs.attn_peak_full, "area": hs.area, "power": hs.power, "cost": hs.cost, "op_total": hs.t_pipe + hs.t_scan}
    lf_scale = 1.0
    for name, value in settings.items():
        target, kind, *_ = PARAMS[name]
        if target == "demand":
            s_over["demand_tok_s"] = value / 86400.0
        elif target == "wl":
            wl_kw[kind.split(":")[1]] = value
        elif target == "g":
            if kind == "mul:load_fraction":
                lf_scale = value
            else:
                g_over = _merge(g_over, _apply(gbase, kind, value))
        else:
            s_over = _merge(s_over, _apply(sbase, kind, value))
    if "scan_rows" in s_over:
        s_over["scan_rows"] = int(s_over["scan_rows"])
    # S's whole time per weight operation is held fixed unless it is the input being moved; its own bank pass is
    # inside it, so bank depth and clock change speed only if the pass outgrows the whole (same rule in tornado and Monte Carlo)
    t_scan = s_over.get("scan_rows", hs.scan_rows) / s_over.get("f_clk", hs.f_clk)
    s_over["t_pipe"] = max(0.02e-6, s_over.pop("op_total", hs.t_pipe + hs.t_scan) - t_scan)

    def best(key, over, lf=None):
        from experiments.platforms import best_point, gpu_load_fraction
        def at(b):
            o = dict(over)
            if lf is not None:
                o["load_fraction"] = min(1.0, lf * gpu_load_fraction({"load_fraction_b1": 0.43, "load_fraction_b128": 0.86}, b))
            return run(key, m, Workload(batch=b, context=32768, **wl_kw), **o)
        bp = best_point(at, SLA)
        return bp["best_energy"]["tokens_per_joule"], bp["best_cost"]["usd_per_mtok"]

    # the GPUs get the best of 2, 4 and 8 for each metric
    g1s = [run("g0_x4", m, Workload(batch=1, context=8192, **wl_kw), **{**g_over, "n_chips": n, "load_fraction": min(1.0, lf_scale * 0.43)}) for n in (2, 4, 8)]
    s1 = run("s", m, Workload(batch=1, context=8192, **wl_kw), **s_over)
    gb = [best("g0_x4", {**g_over, "n_chips": n}, lf_scale) for n in (2, 4, 8)]
    ge, gc = max(x[0] for x in gb), min(x[1] for x in gb)
    g1 = max(g1s, key=lambda r: r["tok_s_user"])
    g1e = max(r["tokens_per_joule"] for r in g1s)
    se, sc = best("s", s_over)
    return {"speed_batch1": s1["tok_s_user"] / g1["tok_s_user"], "tokens_per_joule_batch1": s1["tokens_per_joule"] / g1e,
            "tokens_per_joule_best": se / ge, "cost_best": gc / sc, "s_tok_s_user": s1["tok_s_user"], "g0_tok_s_user": g1["tok_s_user"]}


METRICS = ("speed_batch1", "tokens_per_joule_batch1", "tokens_per_joule_best", "cost_best")


def tornado() -> dict:
    base = evaluate({})
    rows = []
    for name, (_t, _k, lo, hi, why) in PARAMS.items():
        a, b = evaluate({name: lo}), evaluate({name: hi})
        rows.append({"parameter": name, "low": lo, "high": hi, "why": why,
                     **{f"{mt}_low": a[mt] for mt in METRICS}, **{f"{mt}_high": b[mt] for mt in METRICS}})
    for mt in METRICS:
        rows_sorted = sorted(rows, key=lambda r: -abs(math.log(r[f"{mt}_high"] / r[f"{mt}_low"])))
        base[f"top_{mt}"] = [r["parameter"] for r in rows_sorted[:5]]
    return {"base": base, "rows": rows}


def monte_carlo(n: int = 400, seed: int = 1) -> dict:
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        st_ = {}
        for name, (_t, kind, lo, hi, _w) in PARAMS.items():
            if name == "S bank depth (rows per pass)":
                st_[name] = rng.choice([256, 512, 1024])
            elif lo > 0 and hi / lo >= 4:  # wide ranges are drawn log-uniformly
                st_[name] = math.exp(rng.uniform(math.log(lo), math.log(hi)))
            else:
                st_[name] = rng.uniform(lo, hi)
        draws.append(evaluate(st_))
    out = {"n": n, "seed": seed}
    for mt in METRICS:
        v = sorted(d[mt] for d in draws)
        q = lambda p: v[min(n - 1, int(p * n))]
        out[mt] = {"p05": q(0.05), "p25": q(0.25), "median": q(0.5), "p75": q(0.75), "p95": q(0.95), "min": v[0], "max": v[-1],
                   "share_above_1": sum(1 for x in v if x > 1) / n, "share_above_10": sum(1 for x in v if x > 10) / n}
        out[mt + "_samples"] = v[:: max(1, n // 200)]
    return out


def ablation() -> dict:
    """From today's GPUs to the fixed-weight chip, one change at a time; then the process node."""
    m = shape()
    steps = [("g0_x4", "today's engines on 4x B300", "baseline"),
             ("g1_x4", "+ dataflow-style software on the same GPUs", "software"),
             ("a_x4", "+ a decode chip on the same HBM (A)", "datapath"),
             ("s", "+ weights moved on-die into SRAM (S)", "memory physics"),
             ("f", "+ weights fixed in the masks (F)", "reloadability given up")]
    rows = []
    for key, name, lever in steps:
        b1 = run(key, m, Workload(batch=1, context=8192))
        from experiments.platforms import best_point
        bp = best_point(lambda b: run(key, m, Workload(batch=b, context=32768)), SLA)
        rows.append({"platform": key, "step": name, "lever": lever, "tok_s_user_batch1": b1["tok_s_user"],
                     "tokens_per_joule_batch1": b1["tokens_per_joule"], "tokens_per_joule_best": bp["best_energy"]["tokens_per_joule"],
                     "usd_per_mtok_best": bp["best_cost"]["usd_per_mtok"]})
    for i, r in enumerate(rows):
        p = rows[i - 1] if i else r
        r["speed_vs_previous"] = r["tok_s_user_batch1"] / p["tok_s_user_batch1"]
        r["speed_vs_g0"] = r["tok_s_user_batch1"] / rows[0]["tok_s_user_batch1"]
        r["energy_best_vs_g0"] = r["tokens_per_joule_best"] / rows[0]["tokens_per_joule_best"]
        r["cost_best_vs_g0"] = rows[0]["usd_per_mtok_best"] / r["usd_per_mtok_best"]
    # process node: N3-class for S only, as a sensitivity. Factors are unverified (foundry pages were unreachable).
    hs = hardware("s")
    n3 = {"area": {"mm2_per_mib_sram": hs.area["mm2_per_mib_sram"] / 1.1, "um2_per_mac": {k: v / 1.6 for k, v in hs.area["um2_per_mac"].items()}},
          "power": {"pj_per_flop": {k: v * 0.7 for k, v in hs.power["pj_per_flop"].items()}, "sram_read_pj_per_bit": hs.power["sram_read_pj_per_bit"] * 0.7},
          "cost": {"wafer_usd": 20000.0}}
    b1 = run("s", m, Workload(batch=1, context=8192), **n3)
    bp3 = best_point(lambda b: run("s", m, Workload(batch=b, context=32768), **n3), SLA)
    s_row = next(r for r in rows if r["platform"] == "s")
    node = {"assumed": "N3-class: SRAM 1.1x denser, logic 1.6x denser, 30% less energy, wafer $20,000 (unverified factors)",
            "speed": b1["tok_s_user"] / s_row["tok_s_user_batch1"],
            "tokens_per_joule_best": bp3["best_energy"]["tokens_per_joule"] / s_row["tokens_per_joule_best"],
            "cost_best": s_row["usd_per_mtok_best"] / bp3["best_cost"]["usd_per_mtok"]}
    return {"steps": rows, "process_node": node}


def main() -> dict:
    out = {"label": "modeled", "params": {k: {"low": v[2], "high": v[3], "why": v[4], "kind": v[1].split(":")[0], "field": v[1].split(":")[1]}
                      for k, v in PARAMS.items()},
           "ablation": ablation(), "tornado": tornado(), "monte_carlo": monte_carlo()}
    save_json(out, RESULTS / "uncertainty.json")
    figures(out)
    return out


def figures(o: dict) -> None:
    style()
    t = o["tornado"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 5.2))
    for a, mt, title in zip(ax, ("speed_batch1", "tokens_per_joule_best", "cost_best"),
                            ("per-user speed, batch 1", "tokens per joule, best interactive point", "$ per million tokens, best interactive point")):
        rows = sorted(t["rows"], key=lambda r: abs(math.log(r[f"{mt}_high"] / r[f"{mt}_low"])))[-12:]
        base = t["base"][mt]
        for i, r in enumerate(rows):
            lo, hi = sorted((r[f"{mt}_low"], r[f"{mt}_high"]))
            a.barh(i, hi - lo, left=lo, color=PALETTE["S"], alpha=0.75)
        a.axvline(base, color="k", lw=0.8)
        a.set_yticks(range(len(rows))); a.set_yticklabels([r["parameter"] for r in rows], fontsize=6.5)
        a.set_xlabel("S over G0 (4x B300)"); a.set_title(title, fontsize=8.5)
    fig.suptitle("Tornado: one input at a time, low to high (modeled)", fontsize=10, y=1.0)
    fig.savefig(FIG / "fig_tornado.png"); plt.close(fig)

    mc = o["monte_carlo"]
    fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.2))
    for a, mt, title in zip(ax, METRICS, ("speed, batch 1", "tokens/J, batch 1", "tokens/J, best point", "$ per M tokens, best point")):
        a.hist([math.log10(x) for x in mc[mt + "_samples"]], bins=24, color=PALETTE["S"], alpha=0.8)
        for q in ("p05", "median", "p95"):
            a.axvline(math.log10(mc[mt][q]), color="k", lw=0.8, ls="--" if q != "median" else "-")
        a.axvline(0, color="#7f8c8d", lw=0.6)
        ticks = [0.3, 1, 3, 10, 30]
        a.set_xticks([math.log10(x) for x in ticks]); a.set_xticklabels([str(x) for x in ticks])
        a.set_xlabel("S over G0 (4x B300)"); a.set_title(title, fontsize=8.5)
    fig.suptitle(f"Joint Monte Carlo over {len(o['params'])} inputs, {mc['n']} draws (modeled; lines: 5th, 50th, 95th percentile)", fontsize=9.5, y=1.04)
    fig.savefig(FIG / "fig_monte_carlo.png"); plt.close(fig)

    ab = o["ablation"]["steps"]
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
    names = ["G0", "G1", "A", "S", "F"]
    cols = [PALETTE[n] for n in names]
    for a, f_, yl in zip(ax, ("tok_s_user_batch1", "tokens_per_joule_best", "usd_per_mtok_best"),
                         ("tok/s per user, batch 1", "tokens per joule, best interactive point", "$ per million tokens, best interactive point")):
        a.bar(names, [r[f_] for r in ab], color=cols); a.set_yscale("log"); a.set_ylabel(yl)
    fig.suptitle("Ablation: software, then datapath, then memory physics, then fixed weights (modeled)", fontsize=9.5, y=1.02)
    fig.savefig(FIG / "fig_ablation.png"); plt.close(fig)


if __name__ == "__main__":
    o = main()
    for r in o["ablation"]["steps"]:
        print(f"{r['platform']:6s} {r['tok_s_user_batch1']:8.0f} tok/s  x{r['speed_vs_g0']:.1f}  best {r['tokens_per_joule_best']:.2f} tok/J x{r['energy_best_vs_g0']:.1f}  ${r['usd_per_mtok_best']:.3f}/M x{r['cost_best_vs_g0']:.1f}")
    print("node:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in o["ablation"]["process_node"].items()})
    b = o["tornado"]["base"]; print("base:", {k: round(b[k], 2) for k in METRICS}, "| top speed:", b["top_speed_batch1"][:3], "| top energy:", b["top_tokens_per_joule_best"][:3], "| top cost:", b["top_cost_best"][:3])
    for mt in METRICS:
        q = o["monte_carlo"][mt]; print(f"MC {mt:26s} p05 {q['p05']:.2f} median {q['median']:.2f} p95 {q['p95']:.2f} | >1: {q['share_above_1']:.0%} >10: {q['share_above_10']:.0%}")
