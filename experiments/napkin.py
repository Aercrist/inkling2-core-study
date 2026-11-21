"""Hand arithmetic for each headline, from a few inputs, set beside the simulator's figure.

Each napkin is a short list of steps. The inputs come from the checkpoint arithmetic
(results/phase0_workload.json), the hardware files and the GPU fit; the arithmetic is done here so
that no number in docs/napkin.md is typed by hand. A napkin that disagrees with the simulator by
more than about 20% carries the reason. Writes results/napkin.json.
"""
from __future__ import annotations

import json
import math

from experiments.common import RESULTS, save_json
from experiments.platforms import hardware, shape
from sim.hardware import load_yaml
from sim.models.base import Workload

HOURS = 4 * 8766.0


def _fl(o) -> float:
    return sum(o.flops.values()) if isinstance(o.flops, dict) else o.flops


def main() -> dict:
    W = json.loads((RESULTS / "phase0_workload.json").read_text())
    I = json.loads((RESULTS / "inkling.json").read_text())
    CG = json.loads((RESULTS / "calibration_gpu.json").read_text())
    CL = json.loads((RESULTS / "calibration_llama.json").read_text())
    H = I["headline"]["rows"]
    g, s, f = H["g0_x4"], H["s"], H["f"]
    G, S, F = load_yaml("configs/hw/gpu_b300.yaml"), load_yaml("configs/hw/s_mds1.yaml"), load_yaml("configs/hw/f_mdf1.yaml")
    hs, hf = hardware("s"), hardware("f")  # with the synthesized block's constants applied
    S.update(f_clk=hs.f_clk, area=hs.area, power=hs.power)
    F.update(area=hf.area, power=hf.power)
    m = shape()
    eta, kappa = CG["params"]["hbm_eff"], CG["params"]["kernel_overhead"]
    acc = 3.5
    ops8 = m.decode_step(Workload(batch=1, context=8192))
    ops32 = m.decode_step(Workload(batch=1, context=32768))
    kernels = sum(o.n_kernels for o in ops8)
    out = {}

    # ---- 1. GPU speed at batch 1
    step_bytes = W["dspark_step"]["bytes_total"]
    bw = 4 * G["hbm_bw"] * eta
    t_mem = step_bytes / bw
    t_launch = kernels * kappa
    nap = acc / (t_mem + t_launch)
    out["gpu_batch1"] = {
        "title": "GPUs, one request: 4x B300 with the draft model on",
        "steps": [
            f"A speculative step verifies 8 rows and reads {step_bytes / 1e9:.1f} GB of weights (checkpoint arithmetic, docs/01).",
            f"Four B300 deliver 4 x {G['hbm_bw'] / 1e12:.1f} TB/s x {eta:.2f} (fitted) = {bw / 1e12:.1f} TB/s, so the read takes {1e3 * t_mem:.2f} ms.",
            f"The step issues {kernels} dependent kernel launches at {1e6 * kappa:.2f} us (fitted): {1e3 * t_launch:.2f} ms.",
            f"Step = {1e3 * (t_mem + t_launch):.2f} ms; {acc} tokens accepted per step gives {nap:,.0f} tok/s per user.",
        ],
        "napkin": nap, "simulator": g["batch1"]["tok_s_user"], "unit": "tok/s per user",
        "published": "476 tok/s on 4x B300 at 94K context (tokenspeed-blog); about 630 on 4x GB200 (vllm-blog)",
        "gap_reason": "The napkin does not include collectives between GPUs and the uneven split of experts across the four GPUs. "
                      f"These are {g['time_by_cause_batch1_ms']['collective']:.2f} ms and about {g['time_by_cause_batch1_ms']['memory'] - 1e3 * t_mem:.2f} ms in the simulator.",
    }

    # ---- 2. S speed at batch 1
    n_layers, n_draft = m.n_layers, m.draft.layers
    serial = 5 * n_layers + 5 * n_draft + 4
    t_op = hs.t_scan + hs.t_pipe  # together: the whole time per weight operation of a published SRAM wafer machine
    att = [o for o in ops8 if o.kind == "attn"]
    full = [o for o in att if o.category != "attn_sliding"]
    t_att_lat = len(att) * hs.vector_latency  # attention is inside the borrowed per-operation time; it pays only a hand-off
    t_att_math = sum(_fl(o) for o in full) / S["attn_peak_full"] + sum(_fl(o) for o in att if o.category == "attn_sliding") / S["attn_peak"]
    stacks = I["designs"]["s"]["area"]["kv_stacks"]
    t_att_kv = len(full) * S["kv_latency"] + sum(o.kv_tp + o.kv_rep for o in full) / (stacks * S["kv_bw_per_stack"] * eta)
    t_step = serial * t_op + t_att_lat + t_att_math + t_att_kv
    nap = acc / t_step
    out["s_batch1"] = {
        "title": "Reloadable weights on the chip (S), one request",
        "steps": [
            f"Each layer has about 5 weight operations in series (query/key/value projection, output projection, router, expert up, expert down). "
            f"The total is 5 x {n_layers} layers + 5 x {n_draft} draft layers + 4 for embedding and heads = {serial}.",
            f"Each operation takes {1e6 * t_op:.2f} us. That is the whole time per weight operation that a published wafer-scale SRAM machine shows on Llama 3.1 8B "
            f"(docs/06). The bank pass of S, {S['scan_rows']} cycles at {S['f_clk'] / 1e6:.0f} MHz = {1e6 * hs.t_scan:.2f} us, is inside that figure.",
            f"Weight operations: {serial} x {1e6 * t_op:.2f} us = {1e3 * serial * t_op:.3f} ms.",
            f"Attention: {len(att)} attention operations each pay a hand-off ({1e3 * t_att_lat:.3f} ms). "
            f"Their arithmetic takes {1e3 * t_att_math:.3f} ms and the KV fetch takes {1e3 * t_att_kv:.3f} ms.",
            f"Step = {1e3 * t_step:.3f} ms; {acc} tokens per step gives {nap:,.0f} tok/s per user.",
        ],
        "napkin": nap, "simulator": s["batch1"]["tok_s_user"], "unit": "tok/s per user", "published": "",
        "gap_reason": "",
    }
    out["speed_ratio"] = {"napkin": out["s_batch1"]["napkin"] / out["gpu_batch1"]["napkin"],
                          "simulator": s["batch1"]["tok_s_user"] / g["batch1"]["tok_s_user"]}

    # ---- 3. tokens per joule at the best interactive point
    bg = g["best_energy"]
    lf = G["power"]["load_fraction_b1"] + (G["power"]["load_fraction_b128"] - G["power"]["load_fraction_b1"]) * min(1.0, math.log2(bg["batch"]) / 7.0)
    gw = 4 * G["tdp_w"] * lf + 500.0
    t_all = W["bytes"]["total_with_draft"] / bw
    att32 = [o for o in ops32 if o.kind == "attn" and o.category != "attn_sliding"]
    att_flops_tok = sum(_fl(o) for o in att32) / acc
    kv_bytes_tok = sum(o.kv_tp + o.kv_rep for o in att32) / acc
    cap_math = S["attn_peak_full"] / att_flops_tok
    cap_kv = stacks * S["kv_bw_per_stack"] * eta / kv_bytes_tok
    s_tok_s = min(cap_math, cap_kv)
    ds = I["designs"]["s"]
    static = ds["static_w_managed"]["total"]
    step_bits = 8.0 * step_bytes
    e_read = step_bits * S["power"]["sram_read_pj_per_bit"] * 1e-12 / acc
    wf = {k: sum(o.flops.get(k, 0.0) for o in ops32 if (o.w_tp + o.w_rep + o.w_dyn) > 0) for k in ("fp4", "fp8", "bf16")}
    e_math = sum(wf[k] * S["power"]["pj_per_flop"][k] for k in wf) * 1e-12 / acc
    e_att = att_flops_tok * S["power"]["pj_per_flop"]["bf16"] * 1e-12 + 8.0 * kv_bytes_tok * S["power"]["kv_pj_per_bit"] * 1e-12
    e_dyn = (e_read + e_math + e_att) / S["power"]["vr_efficiency"]
    sw = static + s_tok_s * e_dyn
    nap_s, nap_g = s_tok_s / sw, bg["tok_s"] / gw
    out["energy_best"] = {
        "title": "Tokens per joule at the best interactive point (32K context, at least 100 tok/s per user)",
        "steps": [
            f"GPUs: at batch {bg['batch']}, every step touches every expert. Thus the step streams the whole {W['bytes']['total_with_draft'] / 1e9:.0f} GB, which takes "
            f"{1e3 * t_all:.1f} ms before attention and launches. The step of the simulator is {bg['step_ms']:.1f} ms, which gives {bg['tok_s']:,.0f} tok/s in total.",
            f"GPU power: 4 x {G['tdp_w']:.0f} W x {lf:.2f} (measured share of TDP at this batch, carried from H100) + 500 W host = {gw:,.0f} W. "
            f"{bg['tok_s']:,.0f} / {gw:,.0f} = {nap_g:.2f} tokens/J.",
            f"S: the shared attention tile limits throughput. A token needs {att_flops_tok / 1e9:.1f} GFLOP of full attention at 32K, "
            f"so {S['attn_peak_full'] / 1e15:.0f} PFLOP/s gives {cap_math:,.0f} tok/s. A token reads {kv_bytes_tok / 1e9:.2f} GB of KV, "
            f"so {stacks} HBM stacks at {eta:.2f} give {cap_kv:,.0f} tok/s. The lower value is {s_tok_s:,.0f} tok/s.",
            f"S energy per token: {1e3 * e_read:.1f} mJ to read weights ({step_bytes / acc / 1e9:.1f} GB at {S['power']['sram_read_pj_per_bit']} pJ/bit), "
            f"{1e3 * e_math:.1f} mJ for weight arithmetic, {1e3 * e_att:.1f} mJ for attention and KV. With power-delivery loss, the total is {1e3 * e_dyn:.1f} mJ.",
            f"S power: {static:,.0f} W static + {s_tok_s:,.0f} tok/s x {1e3 * e_dyn:.1f} mJ = {sw:,.0f} W. {nap_s:.1f} tokens/J.",
            f"Ratio: {nap_s / nap_g:.1f}x.",
        ],
        "napkin": nap_s / nap_g, "simulator": s["best_energy"]["tokens_per_joule"] / bg["tokens_per_joule"], "unit": "x",
        "napkin_s": nap_s, "simulator_s": s["best_energy"]["tokens_per_joule"], "napkin_g": nap_g, "simulator_g": bg["tokens_per_joule"], "published": "",
        "gap_reason": "The napkin uses the limit of the attention tile as the throughput and ignores groups that wait for the tile. "
                      "The simulator also lets one bank pass serve several rows of a group.",
    }

    # ---- 4. cost per token at the best interactive point
    bc, sc = g["best_cost"], s["best_cost"]
    g_cap = 4 * G["cost"]["purchase_usd_per_gpu"]
    g_h = g_cap / HOURS + gw / 1000.0 * G["cost"]["pue"] * G["cost"]["usd_per_kwh"]
    g_usd = g_h / (bc["tok_s"] * 3600 / 1e6)
    s_cap = ds["cost"]["total"]
    s_h = s_cap / HOURS + sw / 1000.0 * S["cost"]["pue"] * S["cost"]["usd_per_kwh"]
    s_usd = s_h / (s_tok_s * 3600 / 1e6)
    out["cost_best"] = {
        "title": "Cost per million tokens at the best interactive point",
        "steps": [
            f"GPUs: 4 x ${G['cost']['purchase_usd_per_gpu']:,.0f} = ${g_cap:,.0f} over 4 years = ${g_cap / HOURS:.2f}/h, plus {gw / 1000:.1f} kW x "
            f"{G['cost']['pue']} x ${G['cost']['usd_per_kwh']}/kWh = ${g_h:.2f}/h. At {bc['tok_s']:,.0f} tok/s that is ${g_usd:.3f} per million tokens.",
            f"S: {ds['area']['dies']} dies. The build cost is ${ds['cost']['manufacturing']:,.0f}. The share of design cost is ${ds['cost']['nre_share']:,.0f} "
            f"(${S['cost']['nre_usd'] / 1e6:,.0f}M over the {ds['cost']['systems']:,.0f} systems that {I['fleet']['demand_tokens_per_day'] / 1e12:.0f} trillion tokens a day need). The total is ${s_cap:,.0f}. "
            f"That is ${s_cap / HOURS:.2f}/h, plus {sw / 1000:.1f} kW of power = ${s_h:.2f}/h. At {s_tok_s:,.0f} tok/s (napkin 3) that is ${s_usd:.4f} per million tokens.",
            f"Ratio: {g_usd / s_usd:.1f}x.",
        ],
        "napkin": g_usd / s_usd, "simulator": bc["usd_per_mtok"] / sc["usd_per_mtok"], "unit": "x", "published": "",
        "gap_reason": "The throughput cap is the same as in napkin 3. The napkin lets the attention tile run at its limit. The simulator gets "
                      f"{sc['tok_s']:,.0f} tok/s.",
    }

    # ---- 5. dies: reloadable against fixed
    mib = W["bytes"]["total_with_draft"] / 2 ** 20
    a = S["area"]
    usable = a["die_mm2"] * a["die_fill"] - a["mm2_d2d_phy_per_die"] - a["mm2_ctrl_per_die"]
    s_store, s_lanes = mib * a["mm2_per_mib_sram"], ds["area"]["mac_lanes"]
    f_store = mib * F["area"]["mm2_per_mib_rom"]
    win = ds["area"]["kv_window_sram"]
    s_d, f_d = (s_store + s_lanes + win) / usable, (f_store + win) / usable
    df = I["designs"]["f"]
    out["dies"] = {
        "title": "What reloadability costs in silicon",
        "steps": [
            f"The checkpoint is {W['bytes']['total_with_draft'] / 1e9:.1f} GB = {mib:,.0f} MiB.",
            f"The two designs also keep the sliding-window KV cache of {ds['area']['kv_window_sessions']} sessions in SRAM: {win:,.0f} mm2.",
            f"S: weights in SRAM at {a['mm2_per_mib_sram']} mm2/MiB = {s_store:,.0f} mm2, plus {s_lanes:,.0f} mm2 of multiplier lanes and the window cache. "
            f"A die gives {usable:.0f} mm2 after pads, links and control: {s_d:.0f} dies.",
            f"F: mask-programmed storage with its arithmetic at {F['area']['mm2_per_mib_rom']} mm2/MiB (from one shipped chip, scaled) = {f_store:,.0f} mm2: {f_d:.0f} dies.",
            f"Ratio: {s_d / f_d:.1f}x.",
        ],
        "napkin": s_d / f_d, "simulator": I["designs"]["s_over_f"]["dies"], "unit": "x",
        "napkin_s": s_d, "simulator_s": ds["area"]["dies"], "napkin_f": f_d, "simulator_f": df["area"]["dies"], "published": "",
        "gap_reason": "The simulator adds the attention engines, the HBM PHYs and the links between tiles to both designs.",
    }
    for v in out.values():
        if "simulator" in v and "napkin" in v:
            v["gap"] = v["napkin"] / v["simulator"] - 1.0
    out["s_op_time"] = t_op
    out["kernels_per_step"] = kernels
    out["label"] = "napkin arithmetic beside the simulator"
    save_json(out, RESULTS / "napkin.json")
    for k, v in out.items():
        if isinstance(v, dict) and "gap" in v:
            print(f"{k:14s} napkin {v['napkin']:10.3f} simulator {v['simulator']:10.3f} gap {100 * v['gap']:+.0f}%")
    return out


if __name__ == "__main__":
    main()
