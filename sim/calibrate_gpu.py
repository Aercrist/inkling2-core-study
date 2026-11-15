"""Fit the GPU model to the published Inkling 2 Small speeds.

Four parameters describe today's engines on Blackwell GPUs:

  hbm_eff          achieved / peak HBM bandwidth
  kernel_overhead  exposed latency per dependent kernel
  attn_eff         attention kernels' share of peak tensor-core throughput
  row_overhead     extra latency per activation row in attention and MoE ops

Nine published points are fitted (data/calibration_points.yaml): three single-user speeds on
two GPU types; vLLM's four throughput points at 32K-256K context, where the published quantity
is the concurrency at which each user gets 100 tok/s while every session also prefills about 3K
new prompt tokens per turn on the same GPUs; and two prefill forward times quoted in
TokenSpeed's source. The loss is the squared log of predicted over published speed. Leave-one-out
refits without each point in turn and predicts it. With nine points and four parameters this is
a modest test, and several model inputs are not published; `sensitivity()` refits under the
alternatives. A kernel-level attention timing from the same source is held out as a check.
"""
from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass, replace

import numpy as np
import yaml
from scipy.optimize import least_squares

from .fast import chain_step_time
from .hardware import Hardware, load_hw
from .models.base import Workload
from .models.inkling2 import InklingShape, apply_expert_parallel

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARAMS = ("hbm_eff", "kernel_overhead", "attn_eff", "row_overhead")
X0 = np.array([0.8375, 4.2e-6, 0.05, 0.3e-6])
LO = np.array([0.40, 0.3e-6, 1e-4, 0.0])
HI = np.array([1.00, 20e-6, 1.0, 5e-6])
SCALE = np.array([1.0, 1e-6, 1.0, 1e-6])


@dataclass
class Point:
    id: str
    hw: str
    n_gpus: int
    batch: int
    context: int
    speculation: bool
    kv_format: str
    target: float
    accept_len: float = 3.5
    fit: bool = True
    prefill_share: float = 0.0  # share of GPU time spent on prefill (alternative assumption)
    prefill_tokens: int = 0  # > 0: the point is one prefill forward of this many tokens
    target_ms: float = 0.0  # published forward time, for prefill points
    turn_new_tokens: int = 0  # > 0: each session prefills this many new prompt tokens per turn on the same GPUs
    turn_output_tokens: int = 0  # tokens generated per turn

    def workload(self) -> Workload:
        return Workload(batch=self.batch, context=self.context, speculation=self.speculation,
                        accept_len=self.accept_len, kv_format=self.kv_format, prefill_tokens=self.prefill_tokens)

    @property
    def published(self) -> float:
        """The published quantity in the unit `predict` returns as `value`."""
        return self.target_ms if self.prefill_tokens else self.target


def load_points(path: pathlib.Path | None = None) -> tuple[list[Point], list[dict]]:
    d = yaml.safe_load((path or ROOT / "data" / "calibration_points.yaml").read_text())
    pts = [Point(**{k: v for k, v in p.items() if k != "assumed"}) for p in d["points"]]
    return pts, d.get("checks", [])


def gpu(point_hw: str, n: int, x) -> Hardware:
    return load_hw(point_hw, n_chips=n, **dict(zip(PARAMS, (float(v) for v in x))))


def predict(shape: InklingShape, p: Point, x) -> dict:
    hw = gpu(p.hw, p.n_gpus, x)
    wl = p.workload()
    ops = apply_expert_parallel(shape.decode_step(wl), shape, p.n_gpus)
    t = chain_step_time(ops, hw)
    speed = wl.tokens_per_step / t["step_time"] * (1.0 - p.prefill_share)
    if p.turn_new_tokens and not p.prefill_tokens:
        # A turn: each of the `batch` sessions prefills its new prompt tokens once (one forward each, on the
        # same GPUs) and all of them decode `turn_output_tokens` together. Per-user speed over the turn.
        pre = replace(wl, batch=1, prefill_tokens=p.turn_new_tokens)
        t_pre = chain_step_time(apply_expert_parallel(shape.decode_step(pre), shape, p.n_gpus), hw)["step_time"]
        steps = p.turn_output_tokens / wl.tokens_per_step
        t_turn = steps * t["step_time"] + p.batch * t_pre
        speed = p.turn_output_tokens / t_turn
        t = {**t, "prefill_share": p.batch * t_pre / t_turn}
    value = 1e3 * t["step_time"] if p.prefill_tokens else speed
    return {"tok_s_user": speed, "value": value, **t}


def fit(shape: InklingShape, points: list[Point], x0=X0, kernel_checks: list[dict] | None = None) -> dict:
    """Least squares on log errors. `kernel_checks` (attention kernel timings) join the loss when given."""
    pts = [p for p in points if p.fit]
    kcs = kernel_checks or []

    def resid(z):
        x = z * SCALE
        r = [math.log(predict(shape, p, x)["value"] / p.published) for p in pts]
        for c in kcs:
            r += [math.log(1.0 + k["error"]) for k in attention_kernel_check(shape, c, x)]
        return r

    best = None
    for start in (x0, x0 * np.array([0.8, 2.0, 0.2, 3.0]), x0 * np.array([1.1, 0.5, 5.0, 0.3])):
        s = np.clip(start, LO, HI) / SCALE
        r = least_squares(resid, s, bounds=(LO / SCALE, HI / SCALE), x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    x = best.x * SCALE
    res = [math.exp(v) - 1.0 for v in resid(best.x)[:len(pts)]]
    return {"x": x, "params": dict(zip(PARAMS, (float(v) for v in x))),
            "residuals": {p.id: r for p, r in zip(pts, res)},
            "rms_log_error": float(np.sqrt(np.mean(np.square(resid(best.x))))),
            "at_bound": {n: bool(abs(v - lo) < 1e-9 * max(1, abs(lo)) or abs(v - hi) < 1e-9 * abs(hi))
                         for n, v, lo, hi in zip(PARAMS, x, LO, HI)}}


def leave_one_out(shape: InklingShape, points: list[Point], x0) -> list[dict]:
    out = []
    pts = [p for p in points if p.fit]
    for held in pts:
        f = fit(shape, [p for p in pts if p.id != held.id], x0)
        pred = predict(shape, held, f["x"])["value"]
        out.append({"id": held.id, "published": held.published, "predicted": pred, "error": pred / held.published - 1.0,
                    "params": f["params"]})
    return out


def sensitivity(shape: InklingShape, points: list[Point], x0) -> list[dict]:
    """Refit under alternative readings of what the publications leave out."""
    alts = [
        ("as fitted", lambda p: p),
        ("TokenSpeed acceptance 3.0", lambda p: replace(p, accept_len=3.0) if p.id == "tokenspeed_b300_c1_dspark" else p),
        ("TokenSpeed acceptance 4.0", lambda p: replace(p, accept_len=4.0) if p.id == "tokenspeed_b300_c1_dspark" else p),
        ("vLLM single-user context 1K", lambda p: replace(p, context=1024) if p.id == "vllm_gb200_tp4_single_user" else p),
        ("vLLM single-user context 64K", lambda p: replace(p, context=65536) if p.id == "vllm_gb200_tp4_single_user" else p),
        ("without the two prefill points", lambda p: replace(p, fit=p.fit and not p.prefill_tokens)),
        ("prefill forwards carried a 32K cached prefix", lambda p: replace(p, context=32768) if p.prefill_tokens else p),
        ("prefill takes none of the GPUs' time at the throughput points (our first reading)", lambda p: replace(p, turn_new_tokens=0)),
        ("prefill is a fixed 30% of GPU time at the throughput points", lambda p: replace(p, turn_new_tokens=0, prefill_share=0.30) if p.batch > 1 else p),
        ("2K new prompt tokens per turn", lambda p: replace(p, turn_new_tokens=2000) if p.turn_new_tokens else p),
        ("4K new prompt tokens per turn", lambda p: replace(p, turn_new_tokens=4000) if p.turn_new_tokens else p),
        ("throughput sessions 80% in flight", lambda p: replace(p, batch=max(1, round(0.8 * p.batch))) if p.batch > 1 else p),
        ("single-user decode points only", lambda p: replace(p, fit=p.batch == 1 and not p.prefill_tokens)),
        ("vLLM points only", lambda p: replace(p, fit=p.id.startswith("vllm"))),
    ]
    out = []
    for name, f in alts:
        pts = [f(p) for p in points]
        r = fit(shape, pts, x0)
        out.append({"alternative": name, "params": r["params"], "rms_log_error": r["rms_log_error"],
                    "n_points": sum(1 for p in pts if p.fit), "at_bound": r["at_bound"]})
    return out


def attention_kernel_check(shape: InklingShape, check: dict, x) -> list[dict]:
    """Model time of one full-attention layer's attention op against a published kernel timing."""
    from .engine import _op_costs
    out = []
    for ctx, target in zip(check["contexts"], check["target_us"]):
        hw = gpu(check["hw"], check["n_gpus"], x)
        ops = shape.decode_step(Workload(batch=1, context=ctx, kv_format=check["kv_format"]))
        op = next(o for o in ops if o.category == "attn_full")
        static, dyn, _s, t_comp, _c, ovh = _op_costs(op, hw, 0.0)
        t_mem = (static + dyn) / (hw.hbm_bw * hw.hbm_eff)
        one_launch = hw.kernel_overhead + hw.row_overhead * op.rows
        t = one_launch + max(t_comp, t_mem)
        out.append({"context": ctx, "published_us": target, "model_us": 1e6 * t, "error": 1e6 * t / target - 1.0,
                    "arithmetic_us": 1e6 * t_comp, "kv_stream_us": 1e6 * t_mem, "launch_us": 1e6 * one_launch})
    return out
