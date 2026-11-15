"""Calibration and engine invariants for the Inkling platforms."""
import json
import pathlib

import pytest

from sim.calibrate_gpu import load_points, predict
from sim.engine import simulate_step
from sim.fast import chain_step_time
from sim.hardware import load_hw
from sim.models.base import Workload
from sim.models.inkling2 import InklingShape, apply_expert_parallel

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAL = json.loads((ROOT / "results" / "calibration_gpu.json").read_text())


@pytest.fixture(scope="module")
def shape():
    return InklingShape.from_checkpoint()


@pytest.mark.parametrize("batch,context,n", [(1, 8192, 4), (1, 94000, 4), (8, 32768, 8)])
def test_closed_form_equals_engine(shape, batch, context, n):
    hw = load_hw("configs/hw/gpu_b300.yaml", n_chips=n, row_overhead=0.2e-6)
    ops = apply_expert_parallel(shape.decode_step(Workload(batch=batch, context=context)), shape, n)
    assert chain_step_time(ops, hw)["step_time"] == pytest.approx(simulate_step(ops, hw).step_time, rel=1e-9)


def test_fit_reproduces_published_speeds(shape):
    pts, _ = load_points()
    x = [CAL["params"][k] for k in ("hbm_eff", "kernel_overhead", "attn_eff", "row_overhead")]
    for p in pts:
        assert predict(shape, p, x)["value"] == pytest.approx(p.published, rel=0.2), p.id
    assert CAL["n_points"] == 9 and CAL["rms_log_error"] < 0.1
    assert not any(CAL["at_bound"].values())
    # leave-one-out: every point but the only no-speculation one is predicted within 30% when held out
    loo = {r["id"]: abs(r["error"]) for r in CAL["loo"]}
    assert all(v < 0.3 for k, v in loo.items() if k != "tokenspeed_b300_c1_target_only"), loo


def test_fitted_hbm_efficiency_agrees_with_the_deepseek_study():
    """HBM efficiency was fitted independently on another model and engine; the two should be close.

    The per-kernel latency is not compared: this study counts kernel launches from source, that
    study counted dependent stages, so the units differ.
    """
    assert CAL["params"]["hbm_eff"] == pytest.approx(CAL["dsv41_params"]["hbm_eff"], rel=0.15)


def test_kernel_counts_match_the_source_reading(shape):
    """About 977 launches per speculative step for one request and about 931 without speculation."""
    a = sum(o.n_kernels for o in shape.decode_step(Workload()))
    c = sum(o.n_kernels for o in shape.decode_step(Workload(speculation=False)))
    assert a == pytest.approx(977, rel=0.06) and c == pytest.approx(931, rel=0.06)
    layers = [o for o in shape.decode_step(Workload()) if o.phase == "verify" and o.layer >= 0]
    assert sum(o.n_kernels for o in layers if o.layer == 0) == 17
    assert sum(o.n_kernels for o in layers if o.layer == 47) == 16


def test_hbm_chip_matches_dataflow_software_at_equal_bandwidth(shape):
    """H1's premise: with the same HBM, speed converges on the bandwidth ceiling."""
    from experiments.platforms import run
    wl = Workload(batch=1, context=8192)
    g1, a = run("g1_x4", shape, wl), run("a_x4", shape, wl)
    assert a["tok_s_user"] / g1["tok_s_user"] == pytest.approx(1.0, abs=0.25)


# ---------------------------------------------------------------- later phases: experiments and write-up
def _load(name):
    import json
    import pathlib
    return json.loads((pathlib.Path(__file__).resolve().parents[1] / "results" / name).read_text())


def test_napkins_agree_with_the_simulator():
    """Hand arithmetic for each headline stays within 25% of the simulator."""
    nk = _load("napkin.json")
    for key in ("gpu_batch1", "s_batch1", "speed_ratio", "energy_best", "cost_best", "dies"):
        assert abs(nk[key]["gap"]) <= 0.25, (key, nk[key]["gap"])


def test_every_prediction_is_scored_or_pending():
    pr = _load("predictions.json")
    assert len(pr["rows"]) == 36
    assert all(r["verdict"] in ("inside", "above", "below", "pending", "untestable") for r in pr["rows"])
    assert pr["inside"] <= pr["scored"] <= 36


def test_weight_stationary_design_is_not_slower_than_hbm_at_any_context():
    """The pooled attention tile keeps S ahead of A and the GPUs at every context length swept."""
    ctx = _load("inkling.json")["context"]["rows"]
    for s, a, g in zip(ctx["s"], ctx["a_x4"], ctx["g0_x4"]):
        assert s["tok_s_user"] > a["tok_s_user"] > 0 and s["tok_s_user"] > g["tok_s_user"]


def test_same_physics_same_constants():
    """A and S pay the same per-operation latency; KV memory on S runs at the GPUs' fitted HBM efficiency."""
    from experiments.platforms import gpu_cal, hardware, llama_cal, s_latency
    a, s = hardware("a_x4"), hardware("s")
    lat = s_latency()
    assert abs(a.op_overhead - lat["t_pipe"]) < 1e-12 and abs(s.t_pipe - lat["t_pipe"]) < 1e-12
    # S's whole time per weight operation is the published SRAM wafer machine's
    assert abs(s.t_scan + s.t_pipe - 1e-6 * llama_cal()["op_total_s_us"]) < 1e-12
    assert abs(s.kv_eff - gpu_cal()["hbm_eff"]) < 1e-9 and abs(a.hbm_eff - gpu_cal()["hbm_eff"]) < 1e-9


def test_fixed_weight_design_is_a_comparison_point_with_unique_masks():
    d = _load("inkling.json")["designs"]
    # S: three die designs (weight, hub, attention tile) and no masks tied to a model version;
    # F: every weight die is unique to one model version
    assert d["s"]["cost"]["distinct_die_designs"] == 3 and d["s"]["cost"]["unique_mask_usd"] == 0
    assert d["f"]["cost"]["distinct_die_designs"] >= d["f"]["area"]["dies"] and d["f"]["cost"]["unique_mask_usd"] > 0
