"""Decode experiments: calibration, baseline vs proposed, silicon vs software, scaling sweeps,
expert cache, tornado, joint Monte Carlo and ablation. Outputs: results/decode.json + PNGs.

Every number here is SIMULATED/PROJECTED except the GPU measurements quoted from
data/reported_baseline.yaml.
"""
from __future__ import annotations

import collections
import copy
import dataclasses

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from experiments.dsv41.common import FIG, PALETTE, RESULTS, save_json, style
from experiments.dsv41.platforms import cal, label, layout_tp, platform, spec
from sim.models.dsv41.calibrate import CONFIGS, calibrate
from sim.cost import asic_unit_cost, dollars_per_hour, usd_per_mtok
from sim.energy import asic_area, step_energy
from sim.engine import simulate_step
from sim.expert_cache import che_lru_hit, hit_rate, sticky_zipf_trace
from sim.hardware import load_yaml
from sim.models.dsv41.prefill import prefill_time
from sim.models.dsv41.workload import DecodeWorkload, decode_step

REP = load_yaml("data/dsv41/reported_baseline.yaml")
ACCEPT = REP["configs"]["gb300_tp4"]["accept_len"]
OUT_TOKENS = REP["workload"]["output_tokens_example"]
SW = "#5d6d7e"  # software projections
MAIN = ("gpu_tp4", "gpu_mk_tp4", "gpu_df_tp4", "asic")


def run(hw, wl: DecodeWorkload | None = None, hit: float = 0.0, cost_mode: str = "rental") -> dict:
    wl = wl or DecodeWorkload()
    if wl.layout_tp == 0 and layout_tp(hw):
        wl = dataclasses.replace(wl, layout_tp=layout_tp(hw))
    ops = decode_step(spec(), wl)
    r = simulate_step(ops, hw, hit_rate=hit)
    tok_step = wl.accept_len * wl.batch
    tps = tok_step / r.step_time
    e = step_energy(ops, r, hw, tok_step, hit_rate=hit)
    usd_h = dollars_per_hour(hw, e.watts, mode=cost_mode)
    return {
        "platform": hw.name, "n_chips": hw.n_chips, "batch": wl.batch, "context": wl.context,
        "step_ms": r.step_time * 1e3, "tok_s_user": wl.accept_len / r.step_time, "tok_s_system": tps,
        "tok_s_per_chip": tps / hw.n_chips, "hbm_busy": r.hbm_busy, "hbm_gb_per_chip": r.hbm_bytes / 1e9,
        "watts": e.watts, "tok_per_joule": e.tokens_per_joule, "usd_per_hour": usd_h,
        "usd_per_mtok": usd_per_mtok(tps, usd_h),
        "breakdown_ms": {k: v * 1e3 for k, v in r.breakdown.items()},
        "by_category_ms": {k: v * 1e3 for k, v in r.by_category.items()},
        "energy_parts_j": e.parts,
    }


def _owned(hw, wl=None) -> float:
    return run(hw, wl, cost_mode="owned")["usd_per_mtok"] if hw.kind == "gpu" else run(hw, wl)["usd_per_mtok"]


def _fixed_share(c, meas_tp4: float) -> float:
    hw = platform("gpu_tp4", hbm_eff=c.hbm_eff, kernel_overhead=c.kernel_overhead)
    r = simulate_step(decode_step(spec(), DecodeWorkload(verify_head="fp8", layout_tp=4)), hw)
    return (r.breakdown["overhead"] + r.breakdown["collective"]) * 1e3 / meas_tp4


# ---------------------------------------------------------------------------------------------
def calibration_report() -> dict:
    c = cal()
    out = {"hbm_eff": c.hbm_eff, "kernel_overhead_us": c.kernel_overhead * 1e6, "eff_prior": c.eff_prior,
           "predicted_ms": c.predicted_ms, "measured_ms": c.measured_ms, "loo": c.loo}
    decomp = {}
    for k in CONFIGS:
        hw = platform("gpu_tp4" if k == "gb300_tp4" else "gpu_tp8")
        decomp[k] = run(hw, DecodeWorkload(verify_head=REP["configs"][k]["verify_head"]))
    out["decomposition"] = {k: v["breakdown_ms"] for k, v in decomp.items()}
    out["hbm_busy"] = {k: v["hbm_busy"] for k, v in decomp.items()}
    out["gb_per_gpu"] = {k: v["hbm_gb_per_chip"] for k, v in decomp.items()}
    ops4 = decode_step(spec(), DecodeWorkload(verify_head="fp8", layout_tp=4))
    out["n_kernels_per_step"] = sum(o.n_kernels for o in ops4)
    out["n_collectives_per_step"] = sum(1 for o in ops4 if o.coll > 0)
    out["stages_per_block"] = (out["n_kernels_per_step"]) / (spec().n_layers + spec().n_mtp_layers)
    # prediction in deepseek-ultrafast's own nsys_breakdown.py groups (tools/model_vs_nsys.py)
    from experiments.dsv41.model_vs_nsys import group
    r4 = simulate_step(ops4, platform("gpu_tp4"))
    acc = collections.defaultdict(float)
    for i, tseg in r4.segments.items():
        acc[group(ops4[i])] += tseg
    out["nsys_groups_tp4"] = [{"group": g, "us": v * 1e6, "share": v / r4.step_time}
                              for g, v in sorted(acc.items(), key=lambda kv: -kv[1])]
    # leave-one-out error vs the bandwidth-efficiency prior
    m = {k: REP["configs"][k]["step_ms"] for k in CONFIGS}
    loo_curve = []
    for eff in (0.65, 0.70, 0.75, 0.774, 0.80, 0.85, 0.90, 0.935):
        hw4 = platform("gpu_tp4", hbm_eff=eff)
        hw8 = platform("gpu_tp8", hbm_eff=eff)
        o8 = decode_step(spec(), DecodeWorkload(verify_head="bf16", layout_tp=8))
        ko = brentq(lambda x: simulate_step(ops4, hw4.replace(kernel_overhead=x * 1e-6)).step_time * 1e3 - m["gb300_tp4"], 0, 80) * 1e-6
        p8 = simulate_step(o8, hw8.replace(kernel_overhead=ko)).step_time * 1e3
        loo_curve.append({"eff": eff, "kernel_overhead_us": ko * 1e6, "pred_tp8_ms": p8, "err_pct": 100 * (p8 - m["b300_tp8"]) / m["b300_tp8"]})
    out["loo_vs_eff_prior"] = loo_curve
    # refits under alternative assumptions and at the README's run-to-run extremes
    r4lo, r4hi = REP["configs"]["gb300_tp4"]["step_ms_range"]
    r8lo, r8hi = REP["configs"]["b300_tp8"]["step_ms_range"]
    cases = [("baseline", {}, {}, None),
             ("all-reduce 5 us (typical)", {"coll_latency": 5e-6}, {}, None),
             ("all-reduce 10 us", {"coll_latency": 10e-6}, {}, None),
             ("+0.5 us all-reduce per extra peer", {"coll_latency_per_peer": 0.5e-6}, {}, None),
             ("routing locality 0.5", {}, {"locality": 0.5}, None),
             ("context 32K", {}, {"context": 32768}, None),
             (f"run-to-run: TP4 {r4lo}, TP8 {r8hi}", {}, {}, {"gb300_tp4": r4lo, "b300_tp8": r8hi}),
             (f"run-to-run: TP4 {r4hi}, TP8 {r8lo}", {}, {}, {"gb300_tp4": r4hi, "b300_tp8": r8lo}),
             (f"run-to-run: TP4 {r4lo}, TP8 {r8lo}", {}, {}, {"gb300_tp4": r4lo, "b300_tp8": r8lo}),
             (f"run-to-run: TP4 {r4hi}, TP8 {r8hi}", {}, {}, {"gb300_tp4": r4hi, "b300_tp8": r8hi})]
    sens = []
    for name, hw_kw, wl_kw, meas in cases:
        cc = calibrate(spec(), measured=meas, hw_overrides=hw_kw, **wl_kw)
        hw = platform("gpu_tp4", hbm_eff=cc.hbm_eff, kernel_overhead=cc.kernel_overhead, **hw_kw)
        r = simulate_step(decode_step(spec(), DecodeWorkload(verify_head="fp8", layout_tp=4, **wl_kw)), hw)
        tp4 = (meas or m)["gb300_tp4"]
        sens.append({"case": name, "hbm_eff": cc.hbm_eff, "kernel_overhead_us": cc.kernel_overhead * 1e6,
                     "fixed_ms_tp4": (r.breakdown["overhead"] + r.breakdown["collective"]) * 1e3,
                     "fixed_share_tp4": (r.breakdown["overhead"] + r.breakdown["collective"]) * 1e3 / tp4,
                     "at_bound": cc.hbm_eff > 0.999})
    out["fit_sensitivity"] = sens
    shares = [s["fixed_share_tp4"] for s in sens]
    out["fixed_share_range"] = [min(shares), max(shares)]
    return out


# ---------------------------------------------------------------------------------------------
def _prefill_usd_per_mtok() -> dict:
    """GPU prefill-pool cost per output token, for an agent turn (200K cached + 4K new tokens)
    and the tetris answer length; identical for every decode platform (the GPU baseline spends
    the same GPU-seconds prefilling on its own GPUs)."""
    g = platform("gpu_tp4")
    model1m = prefill_time(spec(), g, 1_000_000, mfu=0.5).time
    k = REP["configs"]["gb300_tp4"]["prefill_1m_s"] / model1m
    pf_s = k * prefill_time(spec(), g, 4096, prefix=200_000, mfu=0.5).time
    owned_usd_h = dollars_per_hour(g, run(g)["watts"], mode="owned")
    return {"prefill_s": pf_s, "usd_per_mtok": pf_s * owned_usd_h / 3600 / OUT_TOKENS * 1e6, "output_tokens": OUT_TOKENS}


def baseline_vs_proposed() -> dict:
    rows = []
    for key in ("gpu_tp4", "gpu_tp8", "gpu_mk_tp4", "gpu_df_tp4", "asic", "fpga", "rubin_tp4", "rubin_df_tp4", "asic_hbm4"):
        hw = platform(key)
        wl = DecodeWorkload(verify_head="bf16" if key == "gpu_tp8" else "fp8")
        r = run(hw, wl)
        r["key"] = key
        r["usd_per_mtok_owned"] = _owned(hw, wl)
        rows.append(r)
    C = {r["key"]: r for r in rows}
    asic = platform("asic")
    pf = _prefill_usd_per_mtok()
    # GPU-side assumption sweeps: power (fraction of TDP) and purchase price
    g = platform("gpu_tp4")
    power_sweep = []
    for lf in (0.4, 0.6, 0.8, 1.0):
        rg = run(g.replace(power={**g.power, "load_fraction": lf}))
        rd = run(platform("gpu_df_tp4").replace(power={**g.power, "load_fraction": lf}))
        power_sweep.append({"load_fraction": lf, "gpu_tok_per_joule": rg["tok_per_joule"], "gpu_df_tok_per_joule": rd["tok_per_joule"],
                            "asic_ratio_vs_gpu": C["asic"]["tok_per_joule"] / rg["tok_per_joule"],
                            "asic_ratio_vs_gpu_df": C["asic"]["tok_per_joule"] / rd["tok_per_joule"]})
    price_sweep = []
    for price in (25000, 45000, 65000):
        rg = _owned(g.replace(cost={**g.cost, "purchase_usd_per_gpu": price}))
        rd = _owned(platform("gpu_df_tp4").replace(cost={**g.cost, "purchase_usd_per_gpu": price}))
        price_sweep.append({"purchase_usd_per_gpu": price, "gpu_usd_per_mtok": rg, "gpu_df_usd_per_mtok": rd,
                            "asic_ratio_vs_gpu": rg / C["asic"]["usd_per_mtok"], "asic_ratio_vs_gpu_df": rd / C["asic"]["usd_per_mtok"]})
    # break-even production volume vs GPUs owned at $45K
    def asic_usd(volume, nre):
        return run(asic.replace(cost=dict(asic.cost, volume_units=volume, nre_usd=nre)))["usd_per_mtok"]
    breakeven = {}
    for gk in ("gpu_tp4", "gpu_df_tp4"):
        for nre in (150e6, 300e6, 600e6):
            try:
                breakeven[f"{gk}@nre{int(nre / 1e6)}M"] = brentq(lambda v: asic_usd(v, nre) - C[gk]["usd_per_mtok_owned"], 100, 1e8)
            except ValueError:
                breakeven[f"{gk}@nre{int(nre / 1e6)}M"] = None
    out = {"rows": rows, "asic_area_mm2": asic_area(asic), "asic_unit_cost": asic_unit_cost(asic),
           "asic_hbm4_unit_cost": asic_unit_cost(platform("asic_hbm4")), "prefill_adder": pf,
           "gpu_power_sweep": power_sweep, "gpu_price_sweep": price_sweep, "breakeven_volume": breakeven,
           "measured": {k: {"tok_s": v["decode_tok_s"], "step_ms": v["step_ms"]} for k, v in REP["configs"].items()},
           "roofline_tok_s_asic": ACCEPT / (C["asic"]["hbm_gb_per_chip"] * 1e9 / (asic.hbm_bw * cal().hbm_eff))}
    style()
    sel = [C[k] for k in MAIN]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    parts = [("memory", "#2471a3", "weight/KV streaming (HBM)"), ("compute", "#16a085", "compute-bound"),
             ("collective", "#e67e22", "collective latency"), ("overhead", "#c0392b", "per-stage fixed latency")]
    y = np.arange(len(sel))
    left = np.zeros(len(sel))
    for k, col, lab in parts:
        v = np.array([r["breakdown_ms"][k] for r in sel])
        ax.barh(y, v, left=left, color=col, label=lab, height=0.6)
        left += v
    for i, r in enumerate(sel):
        ax.text(left[i] + 0.04, i, f'{r["step_ms"]:.2f} ms -> {r["tok_s_user"]:,.0f} tok/s/user', va="center", fontsize=8)
    ax.set_yticks(y, [label(r["key"]) for r in sel], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(left) * 1.6)
    ax.set_xlabel("decode step time (ms); batch 1, 4 chips x 8 TB/s HBM (calibrated / simulated)")
    ax.set_title("Where a DeepSeek-V4.1-Flash decode step goes")
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(FIG / "fig_step_breakdown.png")
    plt.close(fig)
    return out


def strong_scaling() -> dict:
    out = {}
    chips = [2, 4, 8, 16]
    for key in ("gpu_tp4", "gpu_df_tp4", "asic"):
        out[key] = [{"n_chips": n, **{k: v for k, v in run(platform(key, n_chips=n, coll_latency_per_peer=0.1e-6)).items()
                                      if k in ("tok_s_user", "step_ms")}} for n in chips]
    style()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    names = {"gpu_tp4": "GPU, deepseek-ultrafast overheads (calibrated)", "gpu_df_tp4": "GPU, dataflow-style software (projected)",
             "asic": "MDA-1 (proposed; >4 needs a fabric not designed here)"}
    for key, col, m in (("gpu_tp4", PALETTE["gpu"], "o"), ("gpu_df_tp4", SW, "s"), ("asic", PALETTE["asic"], "D")):
        ax.plot(chips, [p["tok_s_user"] for p in out[key]], marker=m, color=col, label=names[key])
    for k, v in REP["configs"].items():
        n = 4 if k == "gb300_tp4" else 8
        ax.scatter([n], [v["decode_tok_s"]], marker="*", s=140, color="black", zorder=5, label="measured (deepseek-ultrafast README)" if n == 4 else None)
    ax.set_xscale("log", base=2)
    ax.set_xticks(chips, [str(c) for c in chips])
    ax.set_xlabel("chips per sequence (tensor-parallel width); +0.1 us all-reduce per extra peer, all platforms")
    ax.set_ylabel("tokens / s / user (batch 1)")
    ax.set_title("Per-stage fixed latency caps strong scaling (simulated)")
    ax.legend(fontsize=6.5)
    fig.savefig(FIG / "fig_strong_scaling.png")
    plt.close(fig)
    return out


def batch_sweep() -> dict:
    out = {}
    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for key in ("gpu_tp4", "gpu_df_tp4", "asic"):
        pts = []
        for B in batches:
            hw = platform(key)
            r = run(hw, DecodeWorkload(batch=B))
            pts.append({"batch": B, "tok_s_user": r["tok_s_user"], "tok_s_system": r["tok_s_system"], "step_ms": r["step_ms"],
                        "usd_per_mtok_owned": _owned(hw, DecodeWorkload(batch=B)), "tok_per_joule": r["tok_per_joule"],
                        "hbm_busy": r["hbm_busy"]})
        out[key] = pts
    style()
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for key, col, m in (("gpu_tp4", PALETTE["gpu"], "o"), ("gpu_df_tp4", SW, "s"), ("asic", PALETTE["asic"], "D")):
        p = out[key]
        axs[0].plot([q["tok_s_system"] / 4 for q in p], [q["tok_s_user"] for q in p], marker=m, color=col, label=label(key))
        for q in p:
            if q["batch"] in (1, 16, 256) and key == "asic":
                axs[0].annotate(f'B={q["batch"]}', (q["tok_s_system"] / 4, q["tok_s_user"]), fontsize=6, xytext=(4, 4), textcoords="offset points")
        axs[1].plot([q["tok_s_user"] for q in p], [q["usd_per_mtok_owned"] for q in p], marker=m, color=col, label=label(key))
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("throughput per chip (tok/s)")
    axs[0].set_ylabel("interactivity (tok/s/user)")
    axs[0].set_title("Throughput-interactivity frontier, 4 chips (simulated)")
    axs[1].set_xscale("log"); axs[1].set_yscale("log")
    axs[1].set_xlabel("interactivity (tok/s/user)")
    axs[1].set_ylabel("$ / M output tokens (owned basis)")
    axs[1].set_title("Cost vs interactivity (GPU price $45K is an assumption)")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_batch_frontier.png")
    plt.close(fig)
    return out


def context_sweep() -> dict:
    out = {}
    ctxs = [4096, 16384, 65536, 262144, 1 << 20]
    for key in ("gpu_tp4", "asic"):
        for B in (1, 16):
            out[f"{key}_b{B}"] = [{"context": c, **{k: v for k, v in run(platform(key), DecodeWorkload(context=c, batch=B)).items()
                                                    if k in ("tok_s_user", "tok_s_system", "step_ms", "hbm_gb_per_chip")}} for c in ctxs]
    style()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for key, col in (("gpu_tp4", PALETTE["gpu"]), ("asic", PALETTE["asic"])):
        for B, ls in ((1, "-"), (16, "--")):
            p = out[f"{key}_b{B}"]
            ax.plot(ctxs, [q["tok_s_user"] for q in p], ls, marker="o", color=col, label=f"{label(key)}, batch {B}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("tok/s/user")
    ax.set_title("Decode speed vs context (CSA2 keeps it nearly flat; simulated)")
    ax.legend(fontsize=6)
    fig.savefig(FIG / "fig_context_sweep.png")
    plt.close(fig)
    return out


def hbm_sweep() -> dict:
    out = {}
    bws = [4e12, 6e12, 8e12, 12e12, 16e12, 19.2e12]
    for key in ("gpu_tp4", "gpu_df_tp4", "asic"):
        out[key] = [{"hbm_tb_s": b / 1e12, "tok_s_user": run(platform(key, hbm_bw=b))["tok_s_user"]} for b in bws]
    style()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for key, col, m in (("gpu_tp4", PALETTE["gpu"], "o"), ("gpu_df_tp4", SW, "s"), ("asic", PALETTE["asic"], "D")):
        ax.plot([p["hbm_tb_s"] for p in out[key]], [p["tok_s_user"] for p in out[key]], marker=m, color=col, label=label(key))
    for x_, t_ in ((8, "GB300"), (19.2, "Rubin")):
        ax.axvline(x_, color="0.6", lw=0.8, ls=":")
        ax.text(x_ + 0.2, 1100, t_, fontsize=7, color="0.4")
    ax.set_xlabel("HBM bandwidth per chip (TB/s)")
    ax.set_ylabel("tok/s/user (batch 1, 4 chips)")
    ax.set_title("Who converts memory bandwidth into speed (simulated)")
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(FIG / "fig_hbm_sweep.png")
    plt.close(fig)
    return out


def sram_sweep() -> dict:
    out = []
    for mb in (0, 4, 8, 16, 32, 64, 128, 256):
        hw = platform("asic", stage_bytes=mb * 1e6)
        r = run(hw)
        out.append({"stage_mb": mb, "tok_s_user": r["tok_s_user"], "hbm_busy": r["hbm_busy"]})
    style()
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot([max(p["stage_mb"], 1) for p in out], [p["tok_s_user"] for p in out], "D-", color=PALETTE["asic"])
    ax.set_xscale("log", base=2)
    ax.set_xticks([max(p["stage_mb"], 1) for p in out], ["0"] + [str(p["stage_mb"]) for p in out[1:]])
    ax.set_xlabel("weight staging buffer per chip (MB)")
    ax.set_ylabel("tok/s/user (batch 1)")
    ax.set_title("MDA-1: prefetch window needed to hide dependency bubbles (simulated)")
    fig.savefig(FIG / "fig_sram_sweep.png")
    plt.close(fig)
    return {"points": out}


def expert_cache_study() -> dict:
    """Hit rate needed vs SRAM it costs. Synthetic traces calibrated to a published LRU miss rate."""
    s = spec()
    best = None
    for stay in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        for zs in (0.0, 0.5, 0.8, 1.0, 1.2):
            h = hit_rate(sticky_zipf_trace(400, 64, 6, 1, stay, zs, seed=1), 32, "lru")
            if best is None or abs(h - 0.72) < abs(best[2] - 0.72):
                best = (stay, zs, h)
    stay, zs, h_cal = best
    caps = [6, 12, 24, 48, 96, 192]
    tr = sticky_zipf_trace(300, s.n_routed_experts, s.n_activated_experts, 6, stay, zs, seed=2)
    curves = {pol: [{"slots": c, "hit": hit_rate(tr, c, pol)} for c in caps] for pol in ("lru", "lfu", "opt")}
    per_slot_mb = s.expert_bytes / 4 / 1e6
    for pol in curves:
        for p in curves[pol]:
            p["sram_gb_per_chip"] = p["slots"] * s.n_layers * per_slot_mb / 1e3
    che = [{"slots": c, "hit": che_lru_hit(np.ones(s.n_routed_experts), c)} for c in caps]
    gains = [{"hit": h, "tok_s_user": run(platform("asic"), hit=h)["tok_s_user"]} for h in (0.0, 0.1, 0.2, 0.4, 0.6, 0.8)]
    sram_mb = platform("asic").sram_bytes / 2**20
    slots_fit = (sram_mb * 2**20 / 1e6 * 0.5) / (s.n_layers * per_slot_mb)
    style()
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for pol, col in (("lru", "#c0392b"), ("lfu", "#e67e22"), ("opt", "#2471a3")):
        axs[0].plot([p["sram_gb_per_chip"] for p in curves[pol]], [p["hit"] * 100 for p in curves[pol]], "o-", color=col,
                    label=pol.upper() if pol != "opt" else "Belady OPT (bound)")
    axs[0].plot([c["slots"] * s.n_layers * per_slot_mb / 1e3 for c in che], [c["hit"] * 100 for c in che], ":", color="0.5", label="Che approx., no locality")
    axs[0].axvline(sram_mb / 1024 / 2, color="0.3", lw=0.8, ls="--")
    axs[0].text(sram_mb / 1024 / 2 * 1.1, 5, "half of MDA-1 SRAM", fontsize=7)
    axs[0].set_xscale("log")
    axs[0].set_xlabel("expert-cache capacity per chip (GB)")
    axs[0].set_ylabel("hit rate (%)")
    axs[0].set_title("Synthetic traces, locality fit to DeepSeek-V2-Lite LRU data")
    axs[0].legend(fontsize=7)
    axs[1].plot([g["hit"] * 100 for g in gains], [g["tok_s_user"] for g in gains], "D-", color=PALETTE["asic"])
    axs[1].set_xlabel("expert-cache hit rate (%)")
    axs[1].set_ylabel("MDA-1 tok/s/user")
    axs[1].set_title("What a hit rate would buy")
    fig.tight_layout()
    fig.savefig(FIG / "fig_expert_cache.png")
    plt.close(fig)
    return {"locality_fit": {"stay": stay, "zipf_s": zs, "lru_hit_dsv2lite": h_cal, "target": 0.72},
            "curves": curves, "che_uniform": che, "gains": gains, "slots_per_layer_in_half_sram": slots_fit, "per_slot_mb": per_slot_mb}


# ---------------------------------------------------------------------------------------------
TORNADO = [
    ("ASIC HBM efficiency (0.65-0.935)", "hw", "hbm_eff", 0.65, 0.935),
    ("ASIC collective latency (1.4-5 us)", "hw", "coll_latency", 5.0e-6, 1.4e-6),
    ("ASIC per-op hand-off (0.1-0.9 us)", "hw", "op_overhead", 0.9e-6, 0.1e-6),
    ("ASIC dependent-fetch latency (0.2-2 us)", "hw", "hbm_latency", 2.0e-6, 0.2e-6),
    ("ASIC router latency (0.03-0.33 us)", "hw", "router_latency", 0.33e-6, 0.03e-6),
    ("ASIC staging buffer (16-256 MB)", "hw", "stage_bytes", 16e6, 256e6),
    ("routing locality (0-0.6)", "wl", "locality", 0.0, 0.6),
    ("acceptance length (5.1-6.0)", "wl", "accept_len", 5.1, 6.0),
    ("context length (2.6K-128K)", "wl", "context", 131072, 2600),
    ("NRE ($100M-$600M)", "cost", "nre_usd", 600e6, 100e6),
    ("production volume (5K-100K chips)", "cost", "volume_units", 5000, 100000),
    ("HBM price ($10-20/GB)", "cost", "hbm_usd_per_gb", 20.0, 10.0),
    ("system share per chip ($4K-16K)", "cost", "system_usd_per_chip", 16000, 4000),
    ("SRAM density (0.3-0.8 mm2/MiB)", "area", "mm2_per_mib_sram", 0.8, 0.3),
    ("HBM shoreline per stack (9-13 mm)", "area", "mm_shoreline_per_hbm", 13.0, 9.0),
    ("HBM pJ/bit (3-6)", "power", "hbm_pj_per_bit", 6.0, 3.0),
    ("on-die NoC pJ/bit (0.2-1.5)", "power", "noc_pj_per_bit", 1.5, 0.2),
    ("ASIC static power (75-300 W)", "power", "static_w", 300.0, 75.0),
]


def _asic_with(kind, key, val):
    hw = platform("asic")
    wl = DecodeWorkload()
    if kind == "hw":
        hw = hw.replace(**{key: val})
    elif kind == "wl":
        wl = dataclasses.replace(wl, **{key: val})
    else:
        d = copy.deepcopy({"cost": hw.cost, "area": hw.area, "power": hw.power}[kind])
        d[key] = val
        hw = hw.replace(**{kind: d})
    return hw, wl


def tornado() -> dict:
    base = run(platform("asic"))
    base_gpu = run(platform("gpu_tp4"))["tok_s_user"]
    base["speedup"] = base["tok_s_user"] / base_gpu
    rows = []
    for lab, kind, key, lo, hi in TORNADO:
        res = {}
        for tag, v in (("low", lo), ("high", hi)):
            hw, wl = _asic_with(kind, key, v)
            r = run(hw, wl)
            g = run(platform("gpu_tp4"), wl)["tok_s_user"] if kind == "wl" else base_gpu
            res[tag] = {"value": v, "tok_s_user": r["tok_s_user"], "speedup": r["tok_s_user"] / g,
                        "usd_per_mtok": r["usd_per_mtok"], "tok_per_joule": r["tok_per_joule"]}
        rows.append({"param": lab, **res})
    out = {"base": {k: base[k] for k in ("tok_s_user", "speedup", "usd_per_mtok", "tok_per_joule")}, "rows": rows}
    style()
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 5.6), sharey=True)
    metrics = (("speedup", "MDA-1 speedup vs calibrated 4x GB300 (same workload)"), ("usd_per_mtok", "MDA-1 $/M tokens (TCO)"),
               ("tok_per_joule", "MDA-1 tokens/J"))
    order = sorted(range(len(rows)), key=lambda i: -abs(rows[i]["high"]["speedup"] - rows[i]["low"]["speedup"])
                   - 1e-3 * abs(rows[i]["high"]["usd_per_mtok"] - rows[i]["low"]["usd_per_mtok"]))
    y = np.arange(len(rows))
    for ax, (m, title) in zip(axs, metrics):
        b = base[m]
        for yi, i in enumerate(order):
            ax.barh(yi, rows[i]["low"][m] - b, left=b, color="#95a5a6", height=0.6)
            ax.barh(yi, rows[i]["high"][m] - b, left=b, color=PALETTE["asic"], height=0.6)
        ax.axvline(b, color="black", lw=0.8)
        ax.set_title(title, fontsize=9)
    axs[0].set_yticks(y, [rows[i]["param"].replace("$", r"\$") for i in order], fontsize=7)
    axs[0].invert_yaxis()
    axs[0].axvline(1.0, color=PALETTE["gpu"], ls="--", lw=1)
    fig.suptitle("Tornado: one-at-a-time sensitivity (red = favourable end; simulated). Joint intervals: fig_monte_carlo.png", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_tornado.png")
    plt.close(fig)
    return out


def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def monte_carlo(n: int = 300, seed: int = 7) -> dict:
    """Joint uncertainty. (1) MDA-1 vs the calibrated GPU with all MDA-1 assumptions drawn jointly.
    (2) MDA-1 vs dataflow-style GPU software with the *shared physics* (HBM efficiency, DRAM
    first-byte latency, collective latency, staging) drawn once and applied to both, and only the
    mechanism-specific constants (hand-off, router) drawn separately."""
    rng = np.random.default_rng(seed)
    g = run(platform("gpu_tp4"))["tok_s_user"]
    a0, d0 = platform("asic"), platform("gpu_df_tp4")
    vs_gpu, vs_sw, corner = [], [], None
    for _ in range(n):
        # every range is log-centred on the platform's base value (factor 3 either way)
        shared = {"hbm_eff": float(rng.uniform(0.65, 0.935)), "hbm_latency": _logu(rng, 0.5e-6 / 3, 0.5e-6 * 3),
                  "coll_latency": _logu(rng, 1.4e-6, 4.0e-6), "stage_bytes": _logu(rng, 16e6, 256e6)}
        a = run(a0.replace(**shared, op_overhead=_logu(rng, a0.op_overhead / 3, a0.op_overhead * 3),
                           router_latency=_logu(rng, a0.router_latency / 3, a0.router_latency * 3)))["tok_s_user"]
        d = run(d0.replace(**shared, op_overhead=_logu(rng, d0.op_overhead / 3, d0.op_overhead * 3),
                           router_latency=_logu(rng, d0.router_latency / 3, d0.router_latency * 3)))["tok_s_user"]
        vs_gpu.append(a / g)
        vs_sw.append(a / d)
    corner = run(a0.replace(hbm_eff=0.65, hbm_latency=1.5e-6, coll_latency=4e-6, stage_bytes=16e6,
                            op_overhead=a0.op_overhead * 3, router_latency=a0.router_latency * 3))["tok_s_user"] / g
    q = lambda v: {"p5": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
                   "min": float(np.min(v)), "max": float(np.max(v))}
    out = {"n": n, "vs_gpu": q(vs_gpu), "vs_sw": q(vs_sw), "worst_corner_vs_gpu": corner,
           "vs_gpu_measured": {k: v * g / REP["configs"]["gb300_tp4"]["decode_tok_s"] for k, v in q(vs_gpu).items()}}
    style()
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    axs[0].hist(vs_gpu, bins=30, color=PALETTE["asic"], alpha=0.8)
    axs[0].axvline(corner, color="black", ls="--", lw=1)
    axs[0].text(corner, axs[0].get_ylim()[1] * 0.9, " joint worst corner", fontsize=7)
    axs[0].set_xlabel("MDA-1 speedup vs calibrated 4x GB300 (batch 1)")
    axs[0].set_title(f"All MDA-1 assumptions drawn jointly (n={n})")
    axs[1].hist(vs_sw, bins=30, color=SW, alpha=0.8)
    axs[1].axvline(1.0, color="black", ls="--", lw=1)
    axs[1].set_xlabel("MDA-1 / dataflow-style GPU software (same HBM, shared physics)")
    axs[1].set_title("Silicon vs software at iso-bandwidth")
    fig.tight_layout()
    fig.savefig(FIG / "fig_monte_carlo.png")
    plt.close(fig)
    return out


def ablation() -> dict:
    """Walk from the calibrated GPU to MDA-1 on the same HBM, one change at a time. Steps A1-A5
    are software on the same GPUs; only A6 needs new silicon."""
    g = platform("gpu_tp4")
    d = platform("gpu_df_tp4")
    a = platform("asic")

    def disk(hw):  # stop streaming the engine's padded layout
        hw = hw.replace()
        hw.notes = "ablation"
        return hw

    df = dict(prefetch=True, op_overhead=d.op_overhead, router_latency=d.router_latency, hbm_latency=d.hbm_latency)
    steps = [
        ("A0", "today", "4x GB300, deepseek-ultrafast (calibrated)", g, "engine"),
        ("A1", "software", "+ unpadded expert layout", disk(g), "disk"),
        ("A2", "software", "+ one persistent program, barrier per stage (1 us; optimistic)", disk(g.replace(kernel_overhead=1e-6)), "disk"),
        ("A3", "software", "+ flag hand-offs instead of barriers (0.5 us), no prefetch", disk(g.replace(stage_bytes=0.0, **df)), "disk"),
        ("A4", "software", "+ cross-layer weight prefetch into L2 (64 MB) = dataflow-style GPU software",
         disk(g.replace(stage_bytes=d.stage_bytes, coll_latency=d.coll_latency, **df)), "disk"),
        ("A5", "silicon", "MDA-1: on-chip hand-off (0.3 us), HW router (0.1 us), Engram off-HBM", a, "disk"),
    ]
    out = []
    for sid, kind, lab, hw, _lay in steps:
        r = run(hw)
        out.append({"id": sid, "kind": kind, "step": lab, "tok_s_user": r["tok_s_user"], "step_ms": r["step_ms"],
                    "hbm_busy": r["hbm_busy"], "breakdown_ms": r["breakdown_ms"]})
    extra = run(a, DecodeWorkload(shard_replicated=True))
    style()
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    vals = [o["tok_s_user"] for o in out]
    cols = [PALETTE["gpu"] if o["kind"] == "today" else ("#85929e" if o["kind"] == "software" else PALETTE["asic"]) for o in out]
    ax.bar(range(len(vals)), vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 40, f"{v:,.0f}", ha="center", fontsize=7)
    short = ["GPU today\n(calibrated)", "unpadded\nlayout", "persistent\n+ barriers", "flag\nhand-offs", "+ L2\nprefetch",
             "MDA-1\n(silicon)"]
    ax.set_xticks(range(len(vals)), [f'{o["id"]}\n{s}' for o, s in zip(out, short)], fontsize=7)
    ax.set_ylabel("tok/s/user (batch 1, 4 chips x 8 TB/s)")
    ax.set_title("Ablation: grey = software on the same GPUs, red = new silicon (simulated)")
    fig.savefig(FIG / "fig_ablation.png")
    plt.close(fig)
    return {"steps": out, "shard_replicated_on_mda1": {"tok_s_user": extra["tok_s_user"]}}


def main() -> dict:
    out = {
        "calibration": calibration_report(),
        "comparison": baseline_vs_proposed(),
        "strong_scaling": strong_scaling(),
        "batch_sweep": batch_sweep(),
        "context_sweep": context_sweep(),
        "hbm_sweep": hbm_sweep(),
        "sram_sweep": sram_sweep(),
        "expert_cache": expert_cache_study(),
        "tornado": tornado(),
        "monte_carlo": monte_carlo(),
        "ablation": ablation(),
    }
    save_json(out, RESULTS / "decode.json")
    return out


if __name__ == "__main__":
    r = main()
    for row in r["comparison"]["rows"]:
        print(f'{row["key"]:14s} {row["tok_s_user"]:7.0f} tok/s  {row["tok_per_joule"]:.2f} tok/J  owned ${row["usd_per_mtok_owned"]:.3f}/M')
    print("fixed share range", r["calibration"]["fixed_share_range"])
    print("MC", r["monte_carlo"])
    for s in r["ablation"]["steps"]:
        print(s["id"], s["step"][:60], round(s["tok_s_user"]))
