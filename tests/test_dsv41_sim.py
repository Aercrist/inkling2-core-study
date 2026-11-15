"""Invariants for the model description, workload graph, scheduler and calibration."""
import math

import numpy as np
import pytest

from sim.models.dsv41.calibrate import calibrate
from sim.engine import simulate_step
from sim.expert_cache import hit_rate, sticky_zipf_trace
from sim.hardware import Hardware, load_hw
from sim.models.dsv41.modelspec import ModelSpec
from sim.routing import distinct_experts
from sim.models.dsv41.workload import DecodeWorkload, decode_step, summarize


@pytest.fixture(scope="module")
def spec():
    return ModelSpec.load()


def test_paper_numbers(spec):
    pc = spec.param_counts
    assert abs(pc["backbone_total"] / 552e9 - 1) < 0.01
    assert abs(pc["engram_tables"] / 196e9 - 1) < 0.01
    assert abs(pc["activated_decode"] / 16e9 - 1) < 0.02
    assert abs(pc["activated_prefill"] / 8e9 - 1) < 0.03
    assert spec.global_kv_bytes_per_token == pytest.approx(890.0)


def test_expert_bytes_match_checkpoint(spec):
    # 3 matrices of 5120 x 2304 FP4 (0.5 B) + E8M0 scale per 32
    n = 3 * 5120 * 2304
    assert spec.expert_bytes == pytest.approx(n * 0.5 + n / 32)


def test_distinct_experts():
    assert distinct_experts(1, 6, 384) == pytest.approx(6.0)
    assert distinct_experts(6, 6, 384) < 36
    assert distinct_experts(10_000, 6, 384) == pytest.approx(384, rel=1e-3)
    assert distinct_experts(6, 6, 384, locality=1.0) == pytest.approx(6.0)


def test_unique_bytes_independent_of_tp(spec):
    ops = decode_step(spec, DecodeWorkload())
    s4, s8 = summarize(ops, 4), summarize(ops, 8)
    rep = sum(o.w_rep for o in ops) + sum(o.kv_rep for o in ops)
    uniq4 = s4["hbm_total"] * 4 - rep * 3
    uniq8 = s8["hbm_total"] * 8 - rep * 7
    assert uniq4 == pytest.approx(uniq8, rel=1e-9)


def test_microbatch_shares_weights(spec):
    one = decode_step(spec, DecodeWorkload(batch=2, microbatches=1))
    two = decode_step(spec, DecodeWorkload(batch=2, microbatches=2))
    w1 = sum(o.w_tp + o.w_rep for o in one)
    w2 = sum(o.w_tp + o.w_rep for o in two)
    assert w1 == pytest.approx(w2, rel=1e-9)
    r1 = sum(o.w_dyn for o in one if o.phase == "verify")
    r2 = sum(o.w_dyn for o in two if o.phase == "verify")
    assert r1 == pytest.approx(r2, rel=1e-9)


def test_gpu_chain_matches_closed_form(spec):
    """At batch 1 with GPU policy the scheduler must equal sum(ovh + max(comp, mem) + coll)."""
    hw = load_hw("configs/hw/dsv41/gb300_tp4.yaml")
    ops = decode_step(spec, DecodeWorkload())
    r = simulate_step(ops, hw)
    from sim.engine import _op_costs
    t = 0.0
    for o in ops:
        st, dy, sram_t, comp, coll, ovh = _op_costs(o, hw, 0.0)
        mem = (st + dy) / (hw.hbm_bw * hw.hbm_eff)
        t += ovh + max(comp, mem) + coll
    assert r.step_time == pytest.approx(t, rel=1e-6)
    assert sum(r.breakdown.values()) == pytest.approx(r.step_time, rel=1e-6)


@pytest.mark.parametrize("path", ["configs/hw/dsv41/gb300_tp4.yaml", "configs/hw/dsv41/asic_mda1.yaml"])
def test_monotonic(spec, path):
    hw = load_hw(path)
    ops = decode_step(spec, DecodeWorkload())
    base = simulate_step(ops, hw).step_time
    assert simulate_step(ops, hw.replace(hbm_bw=hw.hbm_bw * 1.5)).step_time <= base + 1e-12
    assert simulate_step(ops, hw.replace(coll_latency=hw.coll_latency * 2)).step_time >= base - 1e-12
    assert simulate_step(ops, hw.replace(hbm_eff=hw.hbm_eff * 0.8)).step_time >= base - 1e-12


def test_prefetch_never_hurts(spec):
    hw = load_hw("configs/hw/dsv41/asic_mda1.yaml")
    ops = decode_step(spec, DecodeWorkload())
    t0 = simulate_step(ops, hw.replace(stage_bytes=0.0)).step_time
    t1 = simulate_step(ops, hw.replace(stage_bytes=32e6)).step_time
    t2 = simulate_step(ops, hw.replace(stage_bytes=256e6)).step_time
    assert t0 >= t1 >= t2 - 1e-12


def test_calibration_reproduces_measurements(spec):
    c = calibrate(spec)
    for k, m in c.measured_ms.items():
        assert c.predicted_ms[k] == pytest.approx(m, rel=5e-3)
    assert 0.6 < c.hbm_eff <= 1.0
    assert 0.5e-6 < c.kernel_overhead < 10e-6
    for v in c.loo.values():
        assert abs(v["error_pct"]) < 5.0  # weak test by design; see docs/04 calibration


def test_cache_policies_ordered():
    tr = sticky_zipf_trace(200, 64, 6, 1, 0.5, 0.5, seed=3)
    for cap in (8, 16, 32):
        assert hit_rate(tr, cap, "opt") >= hit_rate(tr, cap, "lru") - 1e-12
        assert hit_rate(tr, cap, "opt") >= hit_rate(tr, cap, "lfu") - 1e-12


def test_area_floor():
    from sim.energy import asic_area
    a = asic_area(load_hw("configs/hw/dsv41/asic_mda1.yaml"))
    assert a["die_mm2"] >= a["shoreline_floor_per_die"] - 1e-9
    assert a["total"] >= sum(a[k] for k in ("hbm_phy_ctrl", "sram", "compute", "serdes", "lpddr_phy", "d2d_phy", "noc_ctrl_misc")) - 1e-9
    assert a["die_mm2"] < 858  # reticle limit


def test_yaml_numeric_coercion():
    hw = load_hw("configs/hw/dsv41/b300_tp8.yaml")
    assert isinstance(hw.hbm_bw, float) and hw.hbm_bw == pytest.approx(7.7e12)
    assert all(isinstance(v, float) for v in hw.peak.values())
