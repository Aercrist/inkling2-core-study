"""Prefill / TTFT and the platform trade study (FPGA vs ASIC vs SRAM-only vs wafer-scale).

Prefill is NOT calibrated: the measured 1M-token prefill runs far below our roofline model,
so we report the gap, fit a single efficiency multiplier on TP4 and show how badly it
predicts TP8. TTFT for the proposed system keeps prefill on GPUs (disaggregated) and adds
the KV hand-off. SIMULATED/PROJECTED throughout.
"""
from __future__ import annotations

import math

from experiments.dsv41.common import RESULTS, save_json
from experiments.dsv41.decode_experiments import REP, run
from experiments.dsv41.platforms import platform, spec
from sim.hardware import Hardware
from sim.models.dsv41.prefill import prefill_time
from sim.models.dsv41.workload import DecodeWorkload

KV_LINK_BW = 50e9  # 400 Gb/s RDMA NIC per decode group (assumption)
KV_LINK_LAT = 20e-6


def prefill_report() -> dict:
    s = spec()
    out = {"model_vs_measured": []}
    for key, cfg in (("gpu_tp4", "gb300_tp4"), ("gpu_tp8", "b300_tp8")):
        hw = platform(key)
        for mfu in (0.3, 0.5, 0.7):
            r = prefill_time(s, hw, 1_000_000, mfu=mfu)
            out["model_vs_measured"].append({"config": cfg, "mfu": mfu, "model_s": r.time, "measured_s": REP["configs"][cfg]["prefill_1m_s"],
                                             "parts_s": r.parts, "pflop": r.flops / 1e15})
    # one-parameter fit: measured = k * model(mfu=0.5) on TP4, predict TP8
    m4 = prefill_time(s, platform("gpu_tp4"), 1_000_000, mfu=0.5).time
    m8 = prefill_time(s, platform("gpu_tp8"), 1_000_000, mfu=0.5).time
    k = REP["configs"]["gb300_tp4"]["prefill_1m_s"] / m4
    out["fit_k_tp4"] = k
    out["pred_tp8_s"] = k * m8
    out["pred_tp8_err_pct"] = 100 * (k * m8 - REP["configs"]["b300_tp8"]["prefill_1m_s"]) / REP["configs"]["b300_tp8"]["prefill_1m_s"]
    out["measured_tp_speedup"] = REP["configs"]["gb300_tp4"]["prefill_1m_s"] / REP["configs"]["b300_tp8"]["prefill_1m_s"]
    out["model_tp_speedup"] = m4 / m8
    return out, k


def ttft(k: float) -> dict:
    """TTFT scenarios. GPU prefill time = k x roofline model (k fitted on the 1M TP4 point)."""
    s = spec()
    scen = [("short prompt (60 tokens, cold)", 60, 0), ("agent turn: 200K cached + 4K new", 4096, 200_000),
            ("cold 128K prompt", 131072, 0)]
    out = []
    g = platform("gpu_tp4")
    a = platform("asic")
    first_gpu = run(g)["step_ms"] / 1e3
    first_asic = run(a)["step_ms"] / 1e3
    for name, new, prefix in scen:
        pf = k * prefill_time(s, g, new, prefix=prefix, mfu=0.5).time
        kv = new * s.global_kv_bytes_per_token + s.window_size * (s.n_layers // 2) * s.kv_entry_bytes["swa"]
        hand = KV_LINK_LAT + kv / KV_LINK_BW
        pa = prefill_time(s, a, new, prefix=prefix, mfu=a.mm_eff).time * k  # same software-efficiency gap assumed
        out.append({"scenario": name, "gpu_ttft_s": pf + first_gpu,
                    "hybrid_ttft_s": pf + hand + first_asic, "kv_handoff_ms": hand * 1e3, "kv_mb": kv / 1e6,
                    "asic_standalone_ttft_s": pa + first_asic})
    return {"scenarios": out, "kv_link_gb_s": KV_LINK_BW / 1e9}


def trade_study() -> dict:
    s = spec()
    # weight bytes that must be resident for decode (backbone + drafter + heads; Engram excluded)
    resident = sum(t.total_bytes for k, t in s.tensors.items()
                   if (k.startswith("layers.") or k.startswith("mtp.") or k in ("head.weight", "embed.weight"))
                   and ".engram.embed." not in k)
    rows = []
    for key in ("gpu_tp4", "gpu_df_tp4", "fpga", "asic"):
        r = run(platform(key))
        hw = platform(key)
        from experiments.dsv41.decode_experiments import _owned
        rows.append({"option": hw.name, "chips": hw.n_chips, "tok_s_user": r["tok_s_user"], "kw": r["watts"] / 1e3,
                     "tok_per_joule": r["tok_per_joule"], "usd_per_mtok": _owned(hw),
                     "hbm_or_sram_gb": hw.n_chips * hw.hbm_cap / 1e9})
    # SRAM-only dataflow (GroqChip-class: 230 MB, 80 TB/s on-die, 185 W average; src: groq)
    per_layer = resident / (s.n_layers + s.n_mtp_layers)
    g = math.ceil(per_layer / (0.9 * 230e6))
    n_sram = g * (s.n_layers + s.n_mtp_layers) + math.ceil(2 * s.tensors["head.weight"].bytes_each / (0.9 * 230e6))
    hw = Hardware("SRAM-only dataflow (GroqChip-class)", "sram", n_chips=g, hbm_bw=80e12, hbm_eff=0.9, hbm_cap=230e6,
                  prefetch=True, stage_bytes=1e12, op_overhead=0.1e-6, router_latency=0.1e-6, hbm_latency=0.1e-6,
                  coll_latency=1.4e-6, coll_latency_per_peer=0.05e-6, link_bw=100e9, engram_in_hbm=False,
                  peak={"fp4": 0.75e15, "fp8": 0.75e15, "bf16": 0.188e15, "fp32": 0.05e15}, mm_eff=0.5,
                  step_overhead=(s.n_layers + s.n_mtp_layers) * 1.0e-6, tdp_w=185,
                  power={"model": "tdp", "load_fraction": 1.0}, cost={"unit_usd": 0, "lifetime_years": 4})
    r = run(hw)
    rows.append({"option": hw.name + f" ({n_sram} chips, {g} per layer)", "chips": n_sram, "tok_s_user": r["tok_s_user"],
                 "kw": n_sram * 185 / 1e3, "tok_per_joule": r["tok_s_user"] / (n_sram * 185),
                 "usd_per_mtok": None, "hbm_or_sram_gb": n_sram * 0.23})
    # wafer-scale (WSE-3: 44 GB SRAM, 21 PB/s; CS-3 23 kW peak; src: cerebras)
    wafers = math.ceil(resident / (0.9 * 44e9))
    # one wafer holds ~6 layers; every op traverses the wafer fabric (~1 us, assumption) and the
    # two per-layer reductions span the wafer (~2 us each, assumption); 5 us per wafer hop
    n_blocks = s.n_layers + s.n_mtp_layers
    hw = Hardware("Wafer-scale (WSE-3-class)", "sram", n_chips=1, hbm_bw=21e15 / wafers, hbm_eff=0.9, hbm_cap=44e9,
                  prefetch=True, stage_bytes=1e12, op_overhead=1.0e-6, router_latency=0.5e-6, hbm_latency=0.1e-6,
                  coll_latency=0.0, engram_in_hbm=False, peak={"fp4": 125e15 / wafers, "fp8": 125e15 / wafers, "bf16": 62e15 / wafers, "fp32": 10e15 / wafers},
                  mm_eff=0.5, step_overhead=wafers * 5e-6 + n_blocks * 2 * 2e-6, tdp_w=23000, power={"model": "tdp", "load_fraction": 1.0},
                  cost={"unit_usd": 0, "lifetime_years": 4})
    r = run(hw)
    rows.append({"option": f"Wafer-scale (WSE-3-class, {wafers} wafers)", "chips": wafers, "tok_s_user": r["tok_s_user"],
                 "kw": wafers * 23.0, "tok_per_joule": r["tok_s_user"] / (wafers * 23000), "usd_per_mtok": None,
                 "hbm_or_sram_gb": wafers * 44})
    return {"resident_weight_gb": resident / 1e9, "rows": rows,
            "notes": "SRAM-only and wafer-scale rows are idealized first-order bounds, not predictions: per-layer chip "
                     "groups (or wafers) with assumed fabric latencies; public demos of large MoEs on such systems are "
                     "far slower. No public chip prices, so no $/token. They are here for capacity and power."}


def main() -> dict:
    pr, k = prefill_report()
    out = {"prefill": pr, "ttft": ttft(k), "trade_study": trade_study()}
    save_json(out, RESULTS / "prefill_tradestudy.json")
    return out


if __name__ == "__main__":
    import json
    o = main()
    print(json.dumps({"k": o["prefill"]["fit_k_tp4"], "tp8err": o["prefill"]["pred_tp8_err_pct"]}, indent=1))
    for r in o["ttft"]["scenarios"]:
        print(r)
    for r in o["trade_study"]["rows"]:
        print(r)
