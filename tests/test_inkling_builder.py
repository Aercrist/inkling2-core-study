"""The Inkling 2 builder's closed-form counts must equal the checkpoint, tensor family by family."""
import json
import pathlib

import pytest

from sim.models.base import Workload
from sim.models.inkling2 import InklingShape, apply_expert_parallel, ladder
from sim.routing import distinct_experts, distinct_experts_skewed, ep_imbalance, inclusion_probs

ROOT = pathlib.Path(__file__).resolve().parents[1]
W = json.loads((ROOT / "results" / "phase0_workload.json").read_text())


@pytest.fixture(scope="module")
def shape():
    return InklingShape.from_checkpoint()


def _manifest_sum(shape, pred, key="elements"):
    return sum(r[key] for r in shape.overrides["manifest"] if pred(r))


def test_parameter_counts_match_checkpoint(shape):
    assert shape.total_params() == W["params"]["total"] == 594_050_641_920
    assert shape.active_params() == W["active"]["decoder_layers"]
    assert shape.draft.params(shape.vocab) == _manifest_sum(shape, lambda r: r["file"] == "mtp.safetensors")
    assert shape.attn_res_params() == _manifest_sum(shape, lambda r: ".attn_res." in r["tensor"])


def test_every_layer_matches_checkpoint(shape):
    for l in range(shape.n_layers):
        ref = W["layers"][l]
        assert shape.dense_layer_params(l) == ref["dense_params"]
        assert shape.dense_layer_bytes(l) == ref["dense_bytes"]
        assert shape.expert_bytes(l) == pytest.approx(ref["expert_bytes_each"], rel=1e-12)
        assert shape.is_full(l) == (ref["attention"] == "full")


def test_resident_bytes_match_checkpoint(shape):
    assert shape.resident_bytes()["total"] == pytest.approx(_manifest_sum(shape, lambda r: True, "bytes"), abs=1.0)


def test_step_bytes_match_header_arithmetic(shape):
    nospec = shape.step_bytes(Workload(batch=1, context=0, speculation=False))
    assert nospec["total"] == pytest.approx(W["touched_per_token"]["total"], rel=1e-5)  # differs by two norm vectors
    spec = shape.step_bytes(Workload(batch=1, context=0))
    head = 2.0 * shape.vocab * shape.hidden  # the draft reads the output head a second time (found in source after Phase 0)
    assert spec["total"] - head == pytest.approx(W["dspark_step"]["bytes_total"], rel=1e-5)


def test_kv_matches_published_per_rank_bytes(shape):
    kv = shape.kv_bytes_per_token(Workload(kv_format="mxfp8"))
    assert kv["per_layer"] / 4 == 528  # TokenSpeed: 528 B per token per target layer, per rank at TP4
    assert shape.kv_bytes_per_token(Workload(kv_format="bf16"))["per_layer"] == 4096


def test_ops_form_a_chain_with_expected_structure(shape):
    ops = shape.decode_step(Workload())
    assert [o.deps for o in ops[:3]] == [[], [0], [1]]
    assert sum(1 for o in ops if o.name == "routed_experts") == shape.n_layers
    assert sum(1 for o in ops if o.category == "attn_full") == len(shape.full_layer_ids) == 8
    nospec = shape.decode_step(Workload(speculation=False))
    assert not any(o.phase == "draft" for o in nospec)


def test_batch_and_context_are_monotonic(shape):
    b = [shape.step_bytes(Workload(batch=n))["total"] for n in (1, 4, 16, 64, 256)]
    assert b == sorted(b)
    c = [shape.step_bytes(Workload(context=n))["kv"] for n in (1024, 8192, 65536, 1048576)]
    assert c == sorted(c)
    # experts saturate: at high batch a step reads every expert once, no more
    big = shape.step_bytes(Workload(batch=1024))
    rb = shape.resident_bytes()
    assert big["experts"] == pytest.approx(rb["experts_nvfp4"] + rb["experts_mxfp8"], rel=1e-3)


def test_routing_helpers():
    assert distinct_experts_skewed(8, 8, 384) == pytest.approx(distinct_experts(8, 8, 384))
    for skew in (0.0, 0.5, 1.0, 1.5):
        p = inclusion_probs(8, 384, skew)
        assert sum(p) == pytest.approx(8.0) and max(p) <= 1.0 + 1e-12
    d = [distinct_experts_skewed(64, 8, 384, 0.0, s) for s in (0.0, 0.5, 1.0, 1.5)]
    assert d == sorted(d, reverse=True)  # more skew, fewer distinct experts
    assert ep_imbalance(59.5, 384, 1) == 1.0
    assert 1.0 < ep_imbalance(59.5, 384, 2) < ep_imbalance(59.5, 384, 4) < ep_imbalance(59.5, 384, 8)
    assert ep_imbalance(384, 384, 4) == pytest.approx(1.0)


def test_expert_parallel_only_touches_experts(shape):
    wl = Workload()
    a, b = shape.decode_step(wl), apply_expert_parallel(shape.decode_step(wl), shape, 4)
    for x, y in zip(a, b):
        if x.name == "routed_experts":
            assert y.w_dyn > x.w_dyn
        else:
            assert (x.w_tp, x.w_rep, x.w_dyn, x.kv_tp) == (y.w_tp, y.w_rep, y.w_dyn, y.kv_tp)


def test_ladder_keeps_the_ratios(shape):
    rungs = ladder(shape)
    sizes = [r.total_params() for r in rungs.values()]
    assert sizes == sorted(sizes, reverse=True)
    for key, r in rungs.items():
        assert r.n_experts == 384 and r.top_k == 8 and r.inter == r.hidden // 2
        assert 0.025 < r.active_params() / r.total_params() < 0.035
        assert r.hypothetical == (key != "inkling2_small")
