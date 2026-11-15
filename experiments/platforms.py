"""The platforms compared in every experiment, with the GPU calibration applied.

  G0  GPUs running today's best published engine: all four fitted parameters.
  G1  the same GPUs running dataflow-style software: the fitted HBM efficiency and attention
      kernel efficiency (same memory, same kernels), flag hand-offs instead of kernel launches.
  A   MDA-1, the HBM decode chip, re-targeted: the fitted HBM efficiency (same HBM).
  S   MDS-1, reloadable weight-stationary: the fitted HBM efficiency for its KV memory.
  F   MDF-1, fixed-weight: as S.

`run()` returns the same fields for every platform so experiments never branch on its kind.
"""
from __future__ import annotations

import json
from dataclasses import replace
from functools import lru_cache

from experiments.common import RESULTS
from sim import stationary as st
from sim.cost import asic_unit_cost, dollars_per_hour, usd_per_mtok
from sim.energy import step_energy
from sim.engine import simulate_step
from sim.fast import chain_step_time
from sim.hardware import load_hw
from sim.models.base import Workload
from sim.models.inkling2 import InklingShape, apply_expert_parallel

HOST_W = 500.0  # host server and fans, counted once per system on every platform (assumption, swept)


def gpu_load_fraction(power: dict, batch: int) -> float:
    """GPU power as a share of TDP: measured 43% at one request rising to 86% at 128 (log-linear between)."""
    import math
    lo, hi = power["load_fraction_b1"], power["load_fraction_b128"]
    return lo + (hi - lo) * min(1.0, math.log2(max(1, batch)) / 7.0)


PLATFORMS = {
    "g0_x4": dict(family="G0", engine="hbm", path="configs/hw/gpu_b300.yaml", n=4, cal="full",
                  label="G0 · 4x B300, today's engines (calibrated)"),
    "g0_x8": dict(family="G0", engine="hbm", path="configs/hw/gpu_b300.yaml", n=8, cal="full",
                  label="G0 · 8x B300, today's engines (calibrated)"),
    "g0_x2": dict(family="G0", engine="hbm", path="configs/hw/gpu_b300.yaml", n=2, cal="full",
                  label="G0 · 2x B300, today's engines (calibrated)"),
    "g1_x4": dict(family="G1", engine="hbm", path="configs/hw/gpu_b300_dataflow.yaml", n=4, cal="software",
                  label="G1 · 4x B300, dataflow-style software (projected)"),
    "g1_x2": dict(family="G1", engine="hbm", path="configs/hw/gpu_b300_dataflow.yaml", n=2, cal="software",
                  label="G1 · 2x B300, dataflow-style software (projected)"),
    "a_x4": dict(family="A", engine="hbm", path="configs/hw/a_mda1.yaml", n=4, cal="eff",
                 label="A · 4x MDA-1, HBM decode chip (proposed)"),
    "a_x2": dict(family="A", engine="hbm", path="configs/hw/a_mda1.yaml", n=2, cal="eff",
                 label="A · 2x MDA-1, HBM decode chip (proposed)"),
    "s": dict(family="S", engine="stationary", path="configs/hw/s_mds1.yaml", n=1, cal="kv",
              label="S · MDS-1, reloadable weights on-die (proposed)"),
    "f": dict(family="F", engine="stationary", path="configs/hw/f_mdf1.yaml", n=1, cal="kv",
              label="F · MDF-1, fixed weights (comparison point)"),
}


@lru_cache(maxsize=1)
def shape() -> InklingShape:
    return InklingShape.from_checkpoint()


@lru_cache(maxsize=1)
def gpu_cal() -> dict:
    return json.loads((RESULTS / "calibration_gpu.json").read_text())["params"]


@lru_cache(maxsize=1)
def llama_cal() -> dict:
    return json.loads((RESULTS / "calibration_llama.json").read_text())


@lru_cache(maxsize=1)
def silicon() -> dict | None:
    """Constants from the synthesized block, scaled to N4P-class (results/silicon_scaled.json), if present."""
    f = RESULTS / "silicon_scaled.json"
    return json.loads(f.read_text()) if f.exists() else None


def s_latency(synthesized: bool = True) -> dict:
    """Per-operation time of S. The total is what a published SRAM wafer machine takes per weight
    operation, all in; S's own bank pass is part of it, and the rest is latency beyond the pass."""
    sil = silicon() if synthesized else None
    base = st.load_stationary(PLATFORMS["s"]["path"], **({"f_clk": sil["s"]["f_clk"]} if sil else {}))
    total = 1e-6 * llama_cal()["op_total_s_us"]
    return {"total": total, "t_scan": base.t_scan, "t_pipe": max(0.02e-6, total - base.t_scan)}


def overrides(key: str, synthesized: bool = True) -> dict:
    p, cal = gpu_cal(), PLATFORMS[key]["cal"]
    if cal == "full":
        return dict(p)
    if cal == "software":  # same HBM and attention kernels; per-row cost shrinks with the hand-off
        return {"hbm_eff": p["hbm_eff"], "attn_eff": p["attn_eff"], "kernel_overhead": 0.0,
                "row_overhead": p["row_overhead"] * 0.5e-6 / p["kernel_overhead"]}
    if cal == "eff":
        # A is a dataflow chip like S, so beyond its memory time it pays the same per-operation
        # latency as S does beyond its bank pass (same physics, same constant).
        return {"hbm_eff": p["hbm_eff"], "op_overhead": s_latency(synthesized)["t_pipe"]}
    if cal == "kv":
        lat = llama_cal()
        fam = PLATFORMS[key]["family"]
        sil = silicon() if synthesized else None
        if fam == "S":
            # reloadable SRAM: per-operation time of a published SRAM wafer machine; lane area, NVFP4
            # arithmetic energy and clock from the synthesized block
            o = {"kv_eff": p["hbm_eff"], "t_pipe": s_latency(synthesized)["t_pipe"]}
            if sil:
                o.update({"f_clk": sil["s"]["f_clk"], "wake_latency": 15.0 / sil["s"]["f_clk"], "area": {"um2_per_mac": sil["s"]["um2_per_mac"]},
                          "power": {"pj_per_flop": sil["s"]["pj_per_flop"]}})
            return o
        # fixed weights: the per-operation total a shipped fixed-weight die shows, split as this design's
        # memory pass at its own clock plus the remainder. The synthesized clock is not applied: F's speed
        # is the shipped chip's. Its arithmetic energy is the hard-wired lane that was synthesized.
        t_scan = st.load_stationary(PLATFORMS[key]["path"]).t_scan
        o = {"kv_eff": p["hbm_eff"], "t_pipe": max(0.02e-6, 1e-6 * lat["latency_for_f_total_us"] - t_scan)}
        if sil:
            o["power"] = {"pj_per_flop": sil["f"]["pj_per_flop"]}
        return o
    return {}


def hardware(key: str, synthesized: bool = True, **over):
    """`synthesized=False` builds S and F with the constants assumed before the block was synthesized."""
    from sim.hardware import _deep_merge
    spec = PLATFORMS[key]
    o = _deep_merge(overrides(key, synthesized), over)  # deep: an override of one area or power field keeps the others
    if spec["engine"] == "stationary":
        hw = st.load_stationary(spec["path"], **o)
    else:
        o.pop("load_fraction", None)
        hw = load_hw(spec["path"], n_chips=o.pop("n_chips", spec["n"]), **o)
    hw.notes = key
    return hw


@lru_cache(maxsize=1)
def fleet() -> dict:
    from sim.hardware import load_yaml
    d = load_yaml("configs/fleet.yaml")
    d["demand_tok_s"] = d["demand_tokens_per_day"] / 86400.0
    return d


_CAP: dict = {}


def _capacity(key: str, model, hw) -> float:
    """Tokens per second one system of a chip platform delivers at the fleet's capacity point."""
    import dataclasses
    cp = fleet()["capacity_point"]
    wl = Workload(batch=int(cp["batch"]), context=int(cp["context"]))
    if not hasattr(model, "expert_bytes"):
        wl = dataclasses.replace(wl, speculation=False)
    sig = (key, getattr(model, "name", ""), json.dumps({k: v for k, v in vars(hw).items() if k != "cost"}, sort_keys=True, default=str))
    if sig not in _CAP:
        if PLATFORMS[key]["engine"] == "stationary":
            # a weight-stationary system is sized for a number of sessions (its window cache); that is its capacity point
            sessions = st.area(hw, model)["kv_window_sessions"]
            if sessions:
                wl = dataclasses.replace(wl, batch=int(sessions))
            _CAP[sig] = st.serve(model, wl, hw).tok_s
        else:
            ops = model.decode_step(wl)
            if hasattr(model, "expert_bytes") and wl.expert_parallel:
                ops = apply_expert_parallel(ops, model, hw.n_chips)
            _CAP[sig] = wl.tokens_per_step * wl.batch / chain_step_time(ops, hw)["step_time"] if not hw.prefetch else \
                wl.tokens_per_step * wl.batch / simulate_step(ops, hw, chunk=64 << 20).step_time
    return _CAP[sig]


def _feasible(model, wl, hbm_capacity: float | None = None, kv_capacity: float | None = None, sessions: int = 0) -> bool:
    """Does the KV cache of `wl.batch` sessions at `wl.context` fit the memory installed for it?"""
    if not hasattr(model, "kv_bytes_per_token"):
        return True
    kv = model.kv_bytes_per_token(wl)
    grow = wl.context * (kv.get("target_full_per_token", 0.0) + kv.get("draft_stored_per_token", 0.0))
    window = kv.get("target_sliding_per_request", 0.0)
    if hbm_capacity is not None:  # weights and every session's KV cache share the HBM
        return model.resident_bytes()["total"] + wl.batch * (grow + window) <= hbm_capacity
    return wl.batch * grow <= kv_capacity and (not sessions or wl.batch <= sessions)


def run(key: str, model, wl, hw=None, **over) -> dict:
    """Serve `wl` on a platform. Returns speed, power, energy and cost in common fields."""
    spec = PLATFORMS[key]
    over = dict(over)
    demand = over.pop("demand_tok_s", fleet()["demand_tok_s"])
    hw = hw or hardware(key, **over)
    tokens = wl.tokens_per_step
    if spec["engine"] == "stationary":
        r = st.serve(model, wl, hw)
        ar = st.area(hw, model)
        cap = _capacity(key, model, hw)
        c = st.cost(hw, model, ar, volume_systems=max(1.0, demand / cap))
        usd_h = st.usd_per_hour(hw, c["total"], r.watts)
        return {"platform": key, "family": spec["family"], "batch": wl.batch, "context": wl.context,
                "tok_s_user": r.tok_s_user, "tok_s": r.tok_s, "step_ms": 1e3 * r.step_time, "watts": r.watts,
                "tokens_per_joule": r.tokens_per_joule, "usd_per_hour": usd_h, "usd_per_mtok": usd_per_mtok(r.tok_s, usd_h),
                "energy_parts_j_per_token": r.energy_parts, "time_by_cause_ms": {k: 1e3 * v for k, v in r.step.by_cause.items()},
                "group": r.group, "groups": r.groups, "system_usd": c["total"], "chips": ar["dies"], "systems": c["systems"],
                "feasible": _feasible(model, wl, kv_capacity=ar["kv_stacks"] * hw.cost["kv_gb_per_stack"] * 1e9,
                                      sessions=ar["kv_window_sessions"])}
    n = hw.n_chips
    ops = model.decode_step(wl)
    if hasattr(model, "expert_bytes") and wl.expert_parallel:
        ops = apply_expert_parallel(ops, model, n)
    if not hw.prefetch:
        t = chain_step_time(ops, hw)
        step, breakdown = t["step_time"], {k: t[k] for k in ("overhead", "memory", "compute", "collective")}
        res = _Res(step, t["hbm_bytes"])
    else:
        nbytes = sum(o.w_tp + o.w_dyn + o.kv_tp for o in ops) / n + sum(o.w_rep + o.kv_rep for o in ops)
        res = simulate_step(ops, hw, chunk=max(4 << 20, nbytes / 2500))
        step, breakdown = res.step_time, res.breakdown
    if hw.kind == "gpu" and "load_fraction_b1" in hw.power:
        hw.power["load_fraction"] = over.get("load_fraction", gpu_load_fraction(hw.power, wl.batch))
    e = step_energy(ops, res, hw, tokens * wl.batch)
    tok_s = tokens * wl.batch / step
    e.watts += HOST_W
    e.parts["host"] = HOST_W * step
    e.tokens_per_joule = tok_s / e.watts
    if hw.kind == "gpu":
        usd_h = dollars_per_hour(hw, e.watts, "owned")
        system = hw.cost["purchase_usd_per_gpu"] * n
    else:
        # the same token demand as every other chip sets how many packages share the design cost
        hw.cost["volume_units"] = max(1.0, demand / _capacity(key, model, hw)) * n
        usd_h = dollars_per_hour(hw, e.watts)
        system = asic_unit_cost(hw)["total"] * n
    return {"platform": key, "family": spec["family"], "batch": wl.batch, "context": wl.context,
            "tok_s_user": tokens / step, "tok_s": tok_s, "step_ms": 1e3 * step, "watts": e.watts,
            "tokens_per_joule": e.tokens_per_joule, "usd_per_hour": usd_h, "usd_per_mtok": usd_per_mtok(tok_s, usd_h),
            "energy_parts_j_per_token": {k: v / (tokens * wl.batch) for k, v in e.parts.items()},
            "time_by_cause_ms": {k: 1e3 * v for k, v in breakdown.items()}, "system_usd": system, "chips": n,
            "feasible": _feasible(model, wl, hbm_capacity=n * hw.hbm_cap)}


class _Res:
    """The two fields of the engine's StepResult that the energy model reads."""

    def __init__(self, step_time: float, hbm_bytes: float):
        self.step_time, self.hbm_bytes = step_time, hbm_bytes


def label(key: str) -> str:
    return PLATFORMS[key]["label"]


def system_cost(key: str, model, hw=None, demand_tok_s: float | None = None) -> dict:
    """Cost of one S or F system, with design cost spread over the systems the fleet's token demand needs."""
    hw = hw or hardware(key)
    ar = st.area(hw, model)
    demand = demand_tok_s if demand_tok_s is not None else fleet()["demand_tok_s"]
    cap = _capacity(key, model, hw)
    c = st.cost(hw, model, ar, volume_systems=max(1.0, demand / cap))
    c["capacity_tok_s"] = cap
    c["demand_tok_s"] = demand
    return c


def best_point(fn, sla: float, b_max: int = 4096) -> dict:
    """Best interactive operating point of a platform: `fn(batch)` returns a `run` result.

    Per-user speed falls as batch grows, so the largest batch that still gives every user `sla`
    tokens per second (and whose KV cache fits) is found by bisection on integer batches, not on a
    grid. Energy and cost are then taken at their best among that batch and a few smaller ones.
    """
    def ok(r):
        return r["tok_s_user"] >= sla and r.get("feasible", True)
    cache = {}
    def at(b):
        if b not in cache:
            cache[b] = fn(b)
        return cache[b]
    if not ok(at(1)):
        r = at(1)
        return {"batch_max": 1, "meets_sla": False, "best_energy": r, "best_cost": r}
    lo, hi = 1, 2
    while hi <= b_max and ok(at(hi)):
        lo, hi = hi, hi * 2
    hi = min(hi, b_max + 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(at(mid)):
            lo = mid
        else:
            hi = mid
    cands = [at(b) for b in sorted({max(1, round(lo * f)) for f in (1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25)})]
    return {"batch_max": lo, "meets_sla": True, "best_energy": max(cands, key=lambda r: r["tokens_per_joule"]),
            "best_cost": min(cands, key=lambda r: r["usd_per_mtok"])}
