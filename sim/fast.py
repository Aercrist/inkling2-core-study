"""Closed form of `sim.engine.simulate_step` for the case calibration needs thousands of times.

For a chain of ops under the per-kernel fetch policy (`prefetch=False`: GPUs running today's
engines), each op's bytes start streaming when the op starts, nothing overlaps, and the step is

    sum over ops of  overhead + max(compute, bytes / bandwidth) + collective.

`tests/test_calibration.py` checks this against the event-driven engine.
"""
from __future__ import annotations

from .engine import _op_costs
from .hardware import Hardware
from .ops import Op


def chain_step_time(ops: list[Op], hw: Hardware) -> dict:
    assert not hw.prefetch, "closed form holds for the per-kernel fetch policy only"
    bw = hw.hbm_bw * hw.hbm_eff
    t = {"overhead": hw.step_overhead, "memory": 0.0, "compute": 0.0, "collective": 0.0}
    hbm_bytes = 0.0
    for o in ops:
        static, dyn, sram_t, t_comp, t_coll, ovh = _op_costs(o, hw, 0.0)
        t_mem = (static + dyn) / bw
        t["overhead"] += ovh
        t["collective"] += t_coll
        t["compute"] += min(t_comp, max(t_comp, t_mem))
        t["memory"] += max(0.0, t_mem - t_comp)
        hbm_bytes += static + dyn
    t["step_time"] = sum(t.values())
    t["hbm_bytes"] = hbm_bytes
    return t
