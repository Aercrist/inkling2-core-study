"""Experiments on Inkling 2 Small: headline table, frontier, context sweep, energy breakdown, ladder.

Every platform is run through `experiments.platforms.run`, so all of them see the same op
graph. Results go to results/inkling.json; figures to results/fig_*.png. All numbers here are
modeled; the GPU rows are calibrated to published speeds (results/calibration_gpu.json).
"""
from __future__ import annotations

import math

from experiments.common import FIG, PALETTE, RESULTS, plt, save_json, style
from experiments.platforms import PLATFORMS, best_point, fleet, hardware, label, run, shape, system_cost
from sim import stationary as st
from sim.models.base import Workload
from sim.models.inkling2 import ladder as ladder_shapes

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
# finer grid for finding each platform's best interactive point, so no platform is penalized by the grid
BEST_BATCHES = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
CONTEXTS = [1024, 4096, 8192, 32768, 131072, 262144, 524288, 1048576]
ALL = ["g0_x4", "g0_x8", "g0_x2", "g1_x4", "g1_x2", "a_x4", "a_x2", "s", "f"]
GPU_LAYOUTS = ["g0_x2", "g0_x4", "g0_x8"]  # the GPUs get whichever layout is best for each metric
LEAD = ["g0_x4", "g1_x4", "a_x4", "s", "f"]
SLA = 100.0  # tok/s per user that defines an interactive operating point (vLLM's convention)


def _keep(r: dict) -> dict:
    return {k: r[k] for k in ("platform", "family", "batch", "context", "tok_s_user", "tok_s", "step_ms", "watts",
                              "tokens_per_joule", "usd_per_mtok", "system_usd", "chips")}


def headline(m) -> dict:
    """Batch 1 at 8K context, and each platform's best interactive point at 32K."""
    rows = {}
    for key in ALL:
        b1 = run(key, m, Workload(batch=1, context=8192))
        bp = best_point(lambda b: run(key, m, Workload(batch=b, context=32768)), SLA)
        best_e, best_c = bp["best_energy"], bp["best_cost"]
        rows[key] = {"label": label(key), "batch1": _keep(b1), "best_energy": _keep(best_e), "best_cost": _keep(best_c),
                     "batch_max_at_sla": bp["batch_max"],
                     "energy_parts_batch1": b1["energy_parts_j_per_token"], "time_by_cause_batch1_ms": b1["time_by_cause_ms"],
                     "energy_parts_best": best_e["energy_parts_j_per_token"]}
    rows["g0"] = gpu_best({k: rows[k] for k in GPU_LAYOUTS})
    ratios = {}
    for base in ("g0", "g0_x4", "g0_x2", "g1_x4"):
        for key in ALL:
            if key == base or (base == "g0" and key in GPU_LAYOUTS):
                continue
            a, b = rows[key], rows[base]
            ratios[f"{key}/{base}"] = {
                "speed_batch1": a["batch1"]["tok_s_user"] / b["batch1"]["tok_s_user"],
                "tokens_per_joule_batch1": a["batch1"]["tokens_per_joule"] / b["batch1"]["tokens_per_joule"],
                "cost_batch1": b["batch1"]["usd_per_mtok"] / a["batch1"]["usd_per_mtok"],
                "tokens_per_joule_best": a["best_energy"]["tokens_per_joule"] / b["best_energy"]["tokens_per_joule"],
                "cost_best": b["best_cost"]["usd_per_mtok"] / a["best_cost"]["usd_per_mtok"],
            }
    pub = 630.0  # best published single-user figure on any GPU layout: vLLM on 4x GB200 (measured; vllm-blog)
    return {"rows": rows, "ratios": ratios, "sla_tok_s_user": SLA, "context_batch1": 8192, "context_best": 32768,
            "best_published_single_user": {"tok_s": pub, "src": "vllm-blog", "hardware": "4x GB200",
                                           "s_over_published": rows["s"]["batch1"]["tok_s_user"] / pub,
                                           "f_over_published": rows["f"]["batch1"]["tok_s_user"] / pub}}


def gpu_best(rows: dict) -> dict:
    """One GPU baseline built from several layouts: for each metric, the layout that does best on it."""
    pick = {
        "batch1": max(rows, key=lambda k: rows[k]["batch1"]["tok_s_user"]),
        "batch1_energy": max(rows, key=lambda k: rows[k]["batch1"]["tokens_per_joule"]),
        "batch1_cost": min(rows, key=lambda k: rows[k]["batch1"]["usd_per_mtok"]),
        "best_energy": max(rows, key=lambda k: rows[k]["best_energy"]["tokens_per_joule"]),
        "best_cost": min(rows, key=lambda k: rows[k]["best_cost"]["usd_per_mtok"]),
    }
    b1 = dict(rows[pick["batch1"]]["batch1"])
    b1["tokens_per_joule"] = rows[pick["batch1_energy"]]["batch1"]["tokens_per_joule"]
    b1["usd_per_mtok"] = rows[pick["batch1_cost"]]["batch1"]["usd_per_mtok"]
    names = {k: rows[v].get("label", v).split(",")[0].replace("G0 · ", "") for k, v in pick.items()}
    return {"label": "G0 · best B300 layout for each metric, today's engines (calibrated)", "layout": pick, "layout_names": names,
            "batch1": b1, "best_energy": rows[pick["best_energy"]]["best_energy"], "best_cost": rows[pick["best_cost"]]["best_cost"],
            "batch_max_at_sla": rows[pick["best_cost"]].get("batch_max_at_sla"),
            "energy_parts_batch1": rows[pick["batch1"]].get("energy_parts_batch1"),
            "time_by_cause_batch1_ms": rows[pick["batch1"]].get("time_by_cause_batch1_ms"),
            "energy_parts_best": rows[pick["best_energy"]].get("energy_parts_best")}


def frontier(m) -> dict:
    out = {}
    for ctx in (8192, 32768):
        out[str(ctx)] = {key: [_keep(run(key, m, Workload(batch=b, context=ctx))) for b in BATCHES] for key in ALL}
    return {"batches": BATCHES, "by_context": out}


def context_sweep(m) -> dict:
    rows = {key: [_keep(run(key, m, Workload(batch=1, context=c))) for c in CONTEXTS] for key in ALL}
    kv = []
    for c in CONTEXTS:
        wl = Workload(batch=1, context=c)
        ops = m.decode_step(wl)
        local = sum(o.kv_tp for o in ops if o.category == "attn_sliding")
        full = sum(o.kv_tp for o in ops if o.category == "attn_full")
        draft = sum(o.kv_tp for o in ops if o.category == "draft_attn")
        weights = sum(o.w_tp + o.w_rep + o.w_dyn for o in ops)
        kv.append({"context": c, "kv_sliding_bytes": local, "kv_full_bytes": full, "kv_draft_bytes": draft, "weight_bytes": weights,
                   "kv_share_of_bytes": (local + full + draft) / (local + full + draft + weights)})
    def ratio(a, b, i):
        return rows[a][i]["tok_s_user"] / rows[b][i]["tok_s_user"]
    i8, i1m = CONTEXTS.index(8192), CONTEXTS.index(1048576)
    return {"contexts": CONTEXTS, "rows": rows, "kv": kv,
            "s_over_a": {"at_8k": ratio("s", "a_x4", i8), "at_1m": ratio("s", "a_x4", i1m)},
            "s_over_g0": {"at_8k": ratio("s", "g0_x4", i8), "at_1m": ratio("s", "g0_x4", i1m)},
            "g0_1m_over_8k": rows["g0_x4"][i1m]["tok_s_user"] / rows["g0_x4"][i8]["tok_s_user"]}


def ladder(base) -> dict:
    """The same comparison down hypothetical rungs with Inkling 2 Small's ratios."""
    out = {}
    for key, m in ladder_shapes(base).items():
        rb = m.resident_bytes()["total"]
        n_gpu = max(1, math.ceil(rb * 1.25 / 288e9)) if key != "inkling2_small" else 4
        wl = Workload(batch=1, context=8192)
        row = {"name": m.name, "hypothetical": m.hypothetical, "params": m.total_params(), "active_params": m.active_params(),
               "resident_gb": rb / 1e9, "n_gpu": n_gpu, "platforms": {}}
        g_rows = {}
        for n in (n_gpu, 2 * n_gpu):
            g_rows[f"x{n}"] = {**_best("g0_x4", m, hardware("g0_x4", n_chips=n)), "label": f"{n}x B300"}
        row["platforms"]["G0"] = gpu_best(g_rows)
        for fam, pk in (("G1", "g1_x4"), ("A", "a_x4")):
            hw = hardware(pk, n_chips=n_gpu)
            row["platforms"][fam] = _best(pk, m, hw)
        for fam, pk in (("S", "s"), ("F", "f")):
            hw = hardware(pk)
            ar = st.area(hw, m)
            row["platforms"][fam] = {**_best(pk, m, hw), "dies": ar["dies"], "tiles": ar["tiles"], "silicon_mm2": ar["silicon_mm2"]}
        g = row["platforms"]["G0"]
        row["gain"] = {fam: {"speed_batch1": p["batch1"]["tok_s_user"] / g["batch1"]["tok_s_user"],
                             "tokens_per_joule_batch1": p["batch1"]["tokens_per_joule"] / g["batch1"]["tokens_per_joule"],
                             "tokens_per_joule_best": p["best_energy"]["tokens_per_joule"] / g["best_energy"]["tokens_per_joule"],
                             "cost_best": g["best_cost"]["usd_per_mtok"] / p["best_cost"]["usd_per_mtok"]}
                       for fam, p in row["platforms"].items() if fam != "G0"}
        out[key] = row
    return out


def ladder_fit(base) -> dict:
    """Largest Inkling-shaped model whose reloadable design fits on eight dies (one tile, no serial links)."""
    rows = []
    for hidden, layers in [(768, n) for n in range(12, 49, 6)] + [(1152, n) for n in range(12, 25, 6)]:
        m = base.scaled(f"scan {hidden}x{layers}", hidden, layers)
        ar = st.area(hardware("s"), m)
        rows.append({"hidden": hidden, "layers": layers, "params": m.total_params(), "dies": ar["dies"], "tiles": ar["tiles"]})
    rows.sort(key=lambda r: r["params"])
    fit = [r for r in rows if r["dies"] <= 8]
    return {"rows": rows, "largest_on_8_dies_params": max(r["params"] for r in fit) if fit else 0.0}


def dense_8b() -> dict:
    """The one rung with published speeds on four memory architectures: Llama 3.1 8B (dense, no draft).

    Published ratios, not modeled ones: the simulator did not reproduce these within 30% from
    public specifications (results/calibration_llama.json), so the published figures stand.
    """
    import json
    rows = json.loads((RESULTS / "calibration_llama.json").read_text())["rows"]
    gpu = rows["h200"]["published"]
    return {"label": "measured by others (published)", "gpu_tok_s": gpu,
            "rows": {k: {"label": r["label"], "published": r["published"], "over_gpu": r["published"] / gpu, "src": r["src"],
                         "who": r["who"], "simulator_error": r["error"]} for k, r in rows.items()}}


# LPDDR5X: 8,533 Mb/s per pin, 16 data pins per channel, 4 channels per package = 68 GB/s per package
# (standard figures); 16 packages on the attention tile is an assumption.
LPDDR_POOL_BW = 16 * 4 * 16 * 8533e6 / 8


def kv_memory_options(m) -> dict:
    """S with its KV cache in LPDDR instead of HBM: the context above which A (HBM) is faster per user."""
    hs = hardware("s")
    stacks = st.area(hs, m)["kv_stacks"]
    hl = hardware("s", kv_bw_per_stack=LPDDR_POOL_BW / stacks)
    ctxs = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    rows = []
    for c in ctxs:
        wl = Workload(batch=1, context=c)
        rows.append({"context": c, "s_hbm": run("s", m, wl)["tok_s_user"], "s_lpddr": run("s", m, wl, hw=hl)["tok_s_user"],
                     "a_x4": run("a_x4", m, wl)["tok_s_user"], "g0_x4": run("g0_x4", m, wl)["tok_s_user"]})
    def crossover(other):
        for lo, hi in zip(rows, rows[1:]):
            d0, d1 = math.log(lo["s_lpddr"] / lo[other]), math.log(hi["s_lpddr"] / hi[other])
            if d0 > 0 >= d1:
                f = d0 / (d0 - d1)
                return math.exp(math.log(lo["context"]) + f * (math.log(hi["context"]) - math.log(lo["context"])))
        return None
    return {"lpddr_pool_bytes_per_s": LPDDR_POOL_BW, "hbm_pool_bytes_per_s": stacks * hs.kv_bw_per_stack, "rows": rows,
            "crossover_context_vs_a": crossover("a_x4"), "crossover_context_vs_g0": crossover("g0_x4")}


def synthesis_effect(m) -> dict:
    """What replacing the assumed lane constants with the synthesized block's did to S."""
    from experiments.platforms import silicon
    if silicon() is None:
        return {"available": False}
    g = gpu_best({k: {**_best(k, m, hardware(k)), "label": label(k)} for k in ("g0_x4", "g0_x8")})
    out = {"available": True}
    for name, syn in (("assumed", False), ("synthesized", True)):
        hw = hardware("s", synthesized=syn)
        ar = st.area(hw, m)
        b = _best("s", m, hw)
        out[name] = {"um2_per_mac_fp4": hw.area["um2_per_mac"]["fp4"], "pj_per_flop_fp4": hw.power["pj_per_flop"]["fp4"], "f_clk": hw.f_clk,
                     "t_scan_us": 1e6 * hw.t_scan, "dies": ar["dies"], "mac_lanes_mm2": ar["mac_lanes"], "system_usd": system_cost("s", m, hw)["total"],
                     "tok_s_user_batch1": b["batch1"]["tok_s_user"],
                     "speed_batch1": b["batch1"]["tok_s_user"] / g["batch1"]["tok_s_user"],
                     "tokens_per_joule_batch1": b["batch1"]["tokens_per_joule"] / g["batch1"]["tokens_per_joule"],
                     "tokens_per_joule_best": b["best_energy"]["tokens_per_joule"] / g["best_energy"]["tokens_per_joule"],
                     "cost_best": g["best_cost"]["usd_per_mtok"] / b["best_cost"]["usd_per_mtok"]}
    return out


def end_to_end(m, head: dict, pre: dict) -> dict:
    """Decode and prefill together on the published agentic workload: about 3 new prompt tokens per output token.

    Each platform's decode cost and energy at its best interactive point, plus its own prefill at full
    use; and S's decode with prefill done on GPUs or on HBM chips instead.
    """
    k = pre["prefill_tokens_per_output_token"]
    rows = {}
    for key in ["g0"] + LEAD[1:]:
        d, pf = head["rows"][key], pre["rows"][key]
        rows[key] = {"label": d["label"], "decode_usd_per_mtok": d["best_cost"]["usd_per_mtok"], "prefill_usd_per_mtok_output": k * pf["usd_per_mtok_prefill"],
                     "decode_j_per_token": 1.0 / d["best_energy"]["tokens_per_joule"], "prefill_j_per_output_token": k * pf["j_per_prefill_token"]}
    for name, dec, prf in (("s_decode_gpu_prefill", "s", "g0"), ("s_decode_a_prefill", "s", "a_x4")):
        rows[name] = {"label": f"{label(dec).split(',')[0]} for decode, {(head['rows'][prf]['label'] if prf == 'g0' else label(prf)).split(',')[0]} for prefill",
                      "decode_usd_per_mtok": rows[dec]["decode_usd_per_mtok"], "prefill_usd_per_mtok_output": rows[prf]["prefill_usd_per_mtok_output"],
                      "decode_j_per_token": rows[dec]["decode_j_per_token"], "prefill_j_per_output_token": rows[prf]["prefill_j_per_output_token"]}
    for r in rows.values():
        r["usd_per_mtok_output"] = r["decode_usd_per_mtok"] + r["prefill_usd_per_mtok_output"]
        r["tokens_per_joule_output"] = 1.0 / (r["decode_j_per_token"] + r["prefill_j_per_output_token"])
        r["prefill_share_of_cost"] = r["prefill_usd_per_mtok_output"] / r["usd_per_mtok_output"]
    g = rows["g0"]
    for r in rows.values():
        r["cost_gain"] = g["usd_per_mtok_output"] / r["usd_per_mtok_output"]
        r["tokens_per_joule_gain"] = r["tokens_per_joule_output"] / g["tokens_per_joule_output"]
    return {"prefill_tokens_per_output_token": k, "rows": rows}


def thresholds(m) -> dict:
    """How the choice of 'interactive' moves the comparison: gains of A, S and F over G0 at several per-user speeds."""
    out = []
    for sla in (25.0, 50.0, 100.0, 150.0, 200.0):
        pts = {key: best_point(lambda b, key=key: run(key, m, Workload(batch=b, context=32768)), sla) for key in ("g0_x4", "g0_x8", "a_x4", "s", "f")}
        ge = max(("g0_x4", "g0_x8"), key=lambda k: pts[k]["best_energy"]["tokens_per_joule"])
        gc = min(("g0_x4", "g0_x8"), key=lambda k: pts[k]["best_cost"]["usd_per_mtok"])
        g = {"best_energy": pts[ge]["best_energy"], "best_cost": pts[gc]["best_cost"], "batch_max": pts[gc]["batch_max"],
             "meets_sla": pts[gc]["meets_sla"], "layout": gc}
        row = {"tok_s_user_min": sla, "gpu_batch": g["batch_max"], "gpu_meets": g["meets_sla"], "gpu_layout": label(g["layout"]).split(",")[0].replace("G0 · ", "")}
        for key in ("a_x4", "s", "f"):
            row[key] = {"batch": pts[key]["batch_max"], "meets": pts[key]["meets_sla"],
                        "tokens_per_joule": pts[key]["best_energy"]["tokens_per_joule"] / g["best_energy"]["tokens_per_joule"],
                        "cost": g["best_cost"]["usd_per_mtok"] / pts[key]["best_cost"]["usd_per_mtok"]}
        out.append(row)
    return {"rows": out}


def demand_sweep(m, head: dict) -> dict:
    """Decode cost gain over G0 against the token demand the chip fleet serves, and the break-even demand."""
    g = head["rows"]["g0"]["best_cost"]["usd_per_mtok"]
    days = [0.25e12, 0.5e12, 1e12, 2e12, 5e12, 10e12, 20e12, 50e12]
    rows = []
    for d in days:
        gl = head["rows"]["g0"]
        row = {"tokens_per_day": d, "b300_equivalent": d / 86400.0 / (gl["best_cost"]["tok_s"] / gl["best_cost"]["chips"])}
        for key in ("a_x4", "s", "f"):
            b = head["rows"][key]["best_cost"]["batch"]
            r = run(key, m, Workload(batch=b, context=32768), demand_tok_s=d / 86400.0)
            row[key] = {"usd_per_mtok": r["usd_per_mtok"], "cost_gain": g / r["usd_per_mtok"], "systems": r.get("systems")}
        rows.append(row)
    def breakeven(key):
        for lo, hi in zip(rows, rows[1:]):
            a, b = math.log(lo[key]["cost_gain"]), math.log(hi[key]["cost_gain"])
            if a < 0 <= b:
                f = -a / (b - a)
                return math.exp(math.log(lo["tokens_per_day"]) + f * (math.log(hi["tokens_per_day"]) - math.log(lo["tokens_per_day"])))
        return None if rows[0][key]["cost_gain"] < 1 else rows[0]["tokens_per_day"]
    hs = hardware("s")
    nre_per_mtok = hs.cost["nre_usd"] / (fleet()["demand_tok_s"] * hs.cost["lifetime_years"] * 8766 * 3600 / 1e6)
    return {"rows": rows, "central_tokens_per_day": fleet()["demand_tokens_per_day"],
            "breakeven_tokens_per_day": {key: breakeven(key) for key in ("a_x4", "s", "f")},
            "nre_usd_per_mtok_at_central_demand": nre_per_mtok}


def attention_tile_sweep(m, head: dict) -> dict:
    """S with a larger shared attention tile (more HBM stacks, a faster pooled engine, more sessions' window cache).

    The central design's tile is sized by assumption and caps the system's throughput; this shows what that choice is worth.
    """
    g = head["rows"]["g0"]
    base = hardware("s")
    rows = []
    for k in (1, 2, 4, 8):
        over = {"attn_peak_full": base.attn_peak_full * k,
                "area": {"kv_stacks_per_layer": base.area["kv_stacks_per_layer"] * k, "mm2_attn_tile": base.area["mm2_attn_tile"] * k,
                         "kv_window_sessions": base.area["kv_window_sessions"] * k}}
        hw = hardware("s", **over)
        ar = st.area(hw, m)
        bp = best_point(lambda b: run("s", m, Workload(batch=b, context=32768), **over), SLA)
        rows.append({"scale": k, "kv_stacks": ar["kv_stacks"], "dies": ar["dies"], "sessions": ar["kv_window_sessions"], "system_usd": bp["best_cost"]["system_usd"],
                     "batch": bp["best_cost"]["batch"], "tok_s": bp["best_cost"]["tok_s"], "usd_per_mtok": bp["best_cost"]["usd_per_mtok"],
                     "tokens_per_joule": bp["best_energy"]["tokens_per_joule"],
                     "cost_gain": g["best_cost"]["usd_per_mtok"] / bp["best_cost"]["usd_per_mtok"],
                     "tokens_per_joule_gain": bp["best_energy"]["tokens_per_joule"] / g["best_energy"]["tokens_per_joule"]})
    return {"rows": rows}


def margin(m, head: dict) -> dict:
    """The chips are costed to their owner; GPUs are at street price. What a seller's gross margin on S does."""
    g = head["rows"]["g0"]["best_cost"]["usd_per_mtok"]
    b = head["rows"]["s"]["best_cost"]["batch"]
    rows = []
    for gm in (0.0, 0.3, 0.5):
        r = run("s", m, Workload(batch=b, context=32768), cost={"gross_margin": gm})
        rows.append({"gross_margin": gm, "system_usd": r["system_usd"], "usd_per_mtok": r["usd_per_mtok"], "cost_gain": g / r["usd_per_mtok"]})
    return {"rows": rows}


def _best(pk: str, m, hw) -> dict:
    b1 = run(pk, m, Workload(batch=1, context=8192), hw=hw)
    import copy
    bp = best_point(lambda b: run(pk, m, Workload(batch=b, context=32768), hw=copy.deepcopy(hw)), SLA)
    return {"batch1": _keep(b1), "best_energy": _keep(bp["best_energy"]), "best_cost": _keep(bp["best_cost"])}


def designs(m) -> dict:
    """Area, power and cost of the two weight-stationary designs."""
    out = {}
    for key in ("s", "f"):
        hw = hardware(key)
        ar = st.area(hw, m)
        out[key] = {
            "label": label(key), "area": ar, "cost": system_cost(key, m, hw),
            "static_w_managed": st.static_power(hw.replace(managed=True), m, ar, 0.02),
            "static_w_unmanaged": st.static_power(hw.replace(managed=False), m, ar, 0.02),
            "t_scan_us": 1e6 * hw.t_scan, "scan_rows": hw.scan_rows, "f_clk": hw.f_clk,
        }
        # cost against the token demand the fleet serves (design cost is spread over the systems it needs)
        vols = []
        for per_day in (0.5e12, 1e12, 2e12, 5e12, 10e12, 20e12):
            cv = system_cost(key, m, hw, demand_tok_s=per_day / 86400.0)
            vols.append({"tokens_per_day": per_day, "systems": cv["systems"], "total_usd": cv["total"], "nre_share_usd": cv["nre_share"],
                         "unique_mask_usd_per_system": cv["unique_mask_usd"] / cv["systems"]})
        out[key]["cost_vs_volume"] = vols
        busy = run(key, m, Workload(batch=256, context=32768))
        out[key]["busy"] = {"batch": 256, "context": 32768, "watts": busy["watts"], "w_per_die": busy["watts"] / ar["dies"],
                            "w_per_cm2": busy["watts"] / (ar["silicon_mm2"] / 100.0)}
        out[key]["reload_seconds_at_10_gb_s"] = ar["resident_gb"] / 10.0
    s, f = out["s"], out["f"]
    out["s_over_f"] = {"dies": s["area"]["dies"] / f["area"]["dies"], "silicon": s["area"]["silicon_mm2"] / f["area"]["silicon_mm2"],
                       "manufacturing_usd": s["cost"]["manufacturing"] / f["cost"]["manufacturing"]}
    return out


def power_management(m) -> dict:
    """H2: what idle-bank power management is worth at low batch."""
    wl = Workload(batch=1, context=8192)
    g0 = headline_cache["rows"]["g0"]["batch1"]
    g0_best = headline_cache["rows"]["g0"]["best_energy"]
    out = {}
    for name, managed in (("managed", True), ("unmanaged", False)):
        r = run("s", m, wl, managed=managed)
        out[name] = {"tok_s_user": r["tok_s_user"], "watts": r["watts"], "tokens_per_joule": r["tokens_per_joule"],
                     "over_g0_batch1": r["tokens_per_joule"] / g0["tokens_per_joule"],
                     "over_g0_best_batch": r["tokens_per_joule"] / g0_best["tokens_per_joule"]}
    return out


def prefill(m) -> dict:
    """One agentic turn's prefill: 3K new tokens over a 32K cached prefix, each platform at the number of
    concurrent prompts that makes its prefill cheapest (a pipelined design needs several prompts in flight
    to use its layers). Modeled; only the GPU row is anchored (two prefill forward times in the calibration)."""
    rows = {}
    for key in LEAD + ["g0_x8"]:
        tries = {b: run(key, m, Workload(batch=b, context=32768, prefill_tokens=3072)) for b in (1, 4, 16, 48)}
        tries = {b: r for b, r in tries.items() if r.get("feasible", True)}
        b, r = min(tries.items(), key=lambda kv: kv[1]["usd_per_mtok"])
        one = tries[1]
        rows[key] = {"label": label(key), "prompts_in_flight": b, "ms_one_prompt": one["step_ms"], "tok_s": r["tok_s"],
                     "tok_s_one_prompt": one["tok_s"], "system_usd": r["system_usd"],
                     "tok_s_per_kusd": r["tok_s"] / (r["system_usd"] / 1e3), "usd_per_mtok_prefill": r["usd_per_mtok"],
                     "usd_per_mtok_prefill_one_prompt": one["usd_per_mtok"], "j_per_prefill_token": r["watts"] / r["tok_s"]}
    gk = min(("g0_x4", "g0_x8"), key=lambda k: rows[k]["usd_per_mtok_prefill"])
    rows["g0"] = {**rows[gk], "label": "G0 · best B300 layout for prefill", "layout": gk}
    g = rows["g0"]
    per_output = 3072 / 1024  # vLLM's agentic workload: about 3K new tokens per 1,024 generated
    return {"rows": rows, "new_tokens": 3072, "context": 32768, "prefill_tokens_per_output_token": per_output,
            "gpu_pool_adder_usd_per_mtok_output": g["usd_per_mtok_prefill"] * per_output,
            "s_over_g0_per_usd": rows["s"]["tok_s_per_kusd"] / g["tok_s_per_kusd"],
            "f_over_g0_per_usd": rows["f"]["tok_s_per_kusd"] / g["tok_s_per_kusd"],
            "a_over_g0_per_usd": rows["a_x4"]["tok_s_per_kusd"] / g["tok_s_per_kusd"],
            "s_over_g0_cost": g["usd_per_mtok_prefill"] / rows["s"]["usd_per_mtok_prefill"],
            "a_over_g0_cost": g["usd_per_mtok_prefill"] / rows["a_x4"]["usd_per_mtok_prefill"]}


def _skew_for_share(target: float) -> float:
    """Zipf exponent at which the most popular 20% of experts receive `target` of the routing probability."""
    from sim.routing import inclusion_probs
    lo, hi = 0.0, 4.0
    for _ in range(40):
        mid = (lo + hi) / 2
        p = sorted(inclusion_probs(8, 384, mid), reverse=True)
        share = sum(p[:77]) / 8.0
        lo, hi = (mid, hi) if share < target else (lo, mid)
    return (lo + hi) / 2


def routing(m) -> dict:
    """No Inkling routing traces exist: sweep from uniform to strongly skewed, bracketed by published statistics."""
    cases = [("uniform, independent rows", 0.20, 0.0), ("measured-mild: top 20% of experts get 47%", 0.47, 0.0),
             ("measured-mild plus neighbouring-token locality 0.3", 0.47, 0.3),
             ("measured-strong: top 20% get 80%", 0.80, 0.0), ("measured-strong plus locality 0.3", 0.80, 0.3)]
    out = []
    for name, share, loc in cases:
        skew = 0.0 if share <= 0.2 else _skew_for_share(share)
        row = {"case": name, "top20_share": share, "zipf_skew": skew, "locality": loc, "platforms": {}}
        wl = Workload(batch=1, context=8192, skew=skew, locality=loc)
        row["step_gb"] = m.step_bytes(wl)["total"] / 1e9
        for key in LEAD + ["g0_x8"]:
            b1 = run(key, m, wl)
            bp = best_point(lambda b: run(key, m, Workload(batch=b, context=32768, skew=skew, locality=loc)), SLA)
            row["platforms"][key] = {"tok_s_user_batch1": b1["tok_s_user"], "tokens_per_joule_best": bp["best_energy"]["tokens_per_joule"]}
        g = {"tok_s_user_batch1": max(row["platforms"][k]["tok_s_user_batch1"] for k in ("g0_x4", "g0_x8")),
             "tokens_per_joule_best": max(row["platforms"][k]["tokens_per_joule_best"] for k in ("g0_x4", "g0_x8"))}
        row["platforms"]["g0"] = g
        row["s_over_g0"] = {"speed_batch1": row["platforms"]["s"]["tok_s_user_batch1"] / g["tok_s_user_batch1"],
                            "tokens_per_joule_best": row["platforms"]["s"]["tokens_per_joule_best"] / g["tokens_per_joule_best"]}
        out.append(row)
    return {"cases": out, "note": "S places one expert's bank per expert, so skew does not change its speed at batch 1; "
            "at high batch the hottest experts need more passes."}


def lora(m) -> dict:
    """Multi-LoRA serving: adapter sizes from Tinker's own counts, and what an adapter costs on each platform."""
    import yaml
    from experiments.common import ROOT
    d = yaml.safe_load((ROOT / "data" / "lora_sizes.yaml").read_text())
    pr, rank = d["per_rank"], d["default_rank"]
    hw = hardware("s")
    ar = st.area(hw, m)
    usable = hw.area["die_mm2"] * hw.area["die_fill"]
    step = m.step_bytes(Workload(batch=1, context=8192))["total"]
    out = {"rank": rank, "per_rank": pr, "variants": {}}
    for name, parts in (("attention only", ("attn",)), ("attention and output head", ("attn", "unembed")),
                        ("attention, output head and experts", ("attn", "unembed", "mlp"))):
        params = rank * sum(pr[k] for k in parts)
        nbytes = params * 2.0
        mm2 = nbytes / 2**20 * hw.area["mm2_per_mib_sram"]
        out["variants"][name] = {"params": params, "bytes": nbytes, "share_of_model_params": params / m.total_params(),
                                 "sram_mm2": mm2, "adapters_per_die": usable / mm2, "dies_per_adapter": mm2 / usable,
                                 "gpu_extra_bytes_per_step_share": nbytes / step}
    out["servable_on_quantized_checkpoint"] = "attention only"
    out["s_dies"] = ar["dies"]
    a = out["variants"]["attention only"]
    out["attention_only_100_adapters_extra_dies"] = 100 * a["dies_per_adapter"]
    return out


headline_cache: dict = {}


def main() -> dict:
    m = shape()
    global headline_cache
    head = headline_cache = headline(m)
    pre = prefill(m)
    out = {"label": "modeled; GPU rows calibrated to published speeds", "headline": head, "frontier": frontier(m),
           "context": context_sweep(m), "designs": designs(m), "power_management": power_management(m), "ladder": ladder(m),
           "prefill": pre, "end_to_end": end_to_end(m, head, pre), "thresholds": thresholds(m), "demand": demand_sweep(m, head),
           "attention_tile": attention_tile_sweep(m, head), "margin": margin(m, head),
           "routing": routing(m), "lora": lora(m), "ladder_fit": ladder_fit(m), "dense_8b": dense_8b(),
           "kv_memory": kv_memory_options(m), "synthesis_effect": synthesis_effect(m),
           "fleet": {"demand_tokens_per_day": fleet()["demand_tokens_per_day"], "demand_tok_s": fleet()["demand_tok_s"],
                     "capacity_point": fleet()["capacity_point"]}}
    save_json(out, RESULTS / "inkling.json")
    figures(out)
    return out


# ------------------------------------------------------------------------------------------- figures
def figures(o: dict) -> None:
    style()
    fam = {k: PLATFORMS[k]["family"] for k in PLATFORMS}
    short = {"g0_x4": "G0 (4x B300, today)", "g1_x4": "G1 (4x B300, dataflow software)", "a_x4": "A (4x HBM chip)",
             "s": "S (weights on-die, reloadable)", "f": "F (fixed weights)"}

    # frontier: per-user speed against tokens/J and $/M, batch 1 -> 1024, 32K context
    fr = o["frontier"]["by_context"]["32768"]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for key in LEAD:
        xs = [r["tok_s_user"] for r in fr[key]]
        ax[0].plot(xs, [r["tokens_per_joule"] for r in fr[key]], "o-", ms=3, color=PALETTE[fam[key]], label=short[key])
        ax[1].plot(xs, [r["usd_per_mtok"] for r in fr[key]], "o-", ms=3, color=PALETTE[fam[key]], label=short[key])
        if key in ("g0_x4", "s", "f"):
            ax[0].annotate("one user", (xs[0], fr[key][0]["tokens_per_joule"]), fontsize=6, xytext=(-34 if key == "f" else 4, -9), textcoords="offset points")
    for a, yl in zip(ax, ("tokens per joule", "$ per million tokens")):
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("tok/s per user"); a.set_ylabel(yl)
        a.axvline(o["headline"]["sla_tok_s_user"], color="k", lw=0.5, ls=":")
    ax[0].set_title("Speed against energy, batch 1 to 1024 (modeled, 32K context)")
    ax[1].set_title("Speed against cost (modeled, 32K context)")
    ax[0].text(o["headline"]["sla_tok_s_user"], ax[0].get_ylim()[0], " 100 tok/s per user", fontsize=6, va="bottom")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.savefig(FIG / "fig_frontier.png"); plt.close(fig)

    # context sweep
    c = o["context"]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for key in LEAD:
        ax[0].plot(c["contexts"], [r["tok_s_user"] for r in c["rows"][key]], "o-", ms=3, color=PALETTE[fam[key]], label=short[key])
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("context (tokens)"); ax[0].set_ylabel("tok/s per user, batch 1")
    ax[0].set_title("Per-user speed against context (modeled)"); ax[0].legend(fontsize=6.5)
    xs = c["contexts"]
    ax[1].stackplot(xs, [k["weight_bytes"] / 1e9 for k in c["kv"]], [k["kv_sliding_bytes"] / 1e9 for k in c["kv"]],
                    [k["kv_full_bytes"] / 1e9 for k in c["kv"]], [k["kv_draft_bytes"] / 1e9 for k in c["kv"]],
                    labels=["weights", "KV, sliding (local) layers", "KV, full (global) layers", "KV, draft"],
                    colors=["#bdc3c7", "#a9cce3", "#2471a3", "#d68910"])
    ax[1].set_xscale("log"); ax[1].set_xlabel("context (tokens)"); ax[1].set_ylabel("GB read per step, batch 1")
    ax[1].set_title("What one step reads (derived from the checkpoint)"); ax[1].legend(fontsize=6.5, loc="upper left")
    fig.savefig(FIG / "fig_context.png"); plt.close(fig)

    # where the energy goes: joules per token at batch 1 and at the best interactive point
    rows = o["headline"]["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for a, field, title in ((ax[0], "energy_parts_batch1", "batch 1, 8K context"), (ax[1], "energy_parts_best", "best interactive point, 32K context")):
        keys = sorted({p for k in LEAD for p in rows[k][field]})
        bottom = [0.0] * len(LEAD)
        cmap = plt.get_cmap("tab20")
        for i, part in enumerate(keys):
            vals = [1e3 * rows[k][field].get(part, 0.0) for k in LEAD]
            a.bar(range(len(LEAD)), vals, bottom=bottom, label=part.replace("_", " "), color=cmap(i % 20))
            bottom = [x + y for x, y in zip(bottom, vals)]
        a.set_xticks(range(len(LEAD))); a.set_xticklabels([fam[k] for k in LEAD]); a.set_yscale("log")
        a.set_ylabel("millijoules per token"); a.set_title(f"Where the energy goes: {title} (modeled)")
    ax[1].legend(fontsize=5.5, ncol=2)
    fig.savefig(FIG / "fig_energy.png"); plt.close(fig)

    # ladder
    lad = o["ladder"]
    keys = list(lad)
    xs = [lad[k]["params"] / 1e9 for k in keys]
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))
    for f_ in ("A", "S", "F"):
        ax[0].plot(xs, [lad[k]["gain"][f_]["speed_batch1"] for k in keys], "o-", color=PALETTE[f_], label=f_)
        ax[1].plot(xs, [lad[k]["gain"][f_]["tokens_per_joule_best"] for k in keys], "o-", color=PALETTE[f_], label=f_)
        ax[2].plot(xs, [lad[k]["gain"][f_]["cost_best"] for k in keys], "o-", color=PALETTE[f_], label=f_)
    for a, t in zip(ax, ("per-user speed at batch 1", "tokens per joule, best interactive point", "$ per million tokens, best interactive point")):
        a.set_xscale("log"); a.set_yscale("log"); a.invert_xaxis(); a.axhline(1, color="k", lw=0.5)
        a.set_xlabel("model size (billion parameters)"); a.set_title(t, fontsize=8.5)
    ax[0].set_ylabel("gain over GPUs running today's engines")
    names = {"A": "A (HBM chip)", "S": "S (weights on-die, reloadable)", "F": "F (fixed weights)"}
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, [names[x] for x in l], fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle("The ladder: gain against model size (modeled; smaller rungs are hypothetical)", fontsize=9.5, y=1.02)
    fig.savefig(FIG / "fig_ladder.png"); plt.close(fig)


if __name__ == "__main__":
    o = main()
    for k, r in o["headline"]["rows"].items():
        b, e = r["batch1"], r["best_energy"]
        print(f"{k:6s} batch1 {b['tok_s_user']:8.0f} tok/s/user {b['tokens_per_joule']:6.2f} tok/J ${b['usd_per_mtok']:6.3f}/M | best B={e['batch']:4d} {e['tok_s_user']:7.0f} tok/s/user {e['tokens_per_joule']:6.2f} tok/J ${r['best_cost']['usd_per_mtok']:6.3f}/M")
