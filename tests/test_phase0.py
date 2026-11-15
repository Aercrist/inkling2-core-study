"""Workload invariants. These guard the facts the pre-registered predictions rest on."""
import hashlib
import json
import pathlib
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
W = json.loads((ROOT / "results" / "phase0_workload.json").read_text())
HUB = json.loads((ROOT / "data" / "inkling2_small" / "hub_model_info.json").read_text())
PRE = yaml.safe_load((ROOT / "data" / "preregistration.yaml").read_text())


def test_594b_reproduced_exactly():
    assert W["params"]["total_hub_convention"] == HUB["safetensors_total"] == 594_050_672_720
    assert round(W["params"]["total"] / 1e9) == 594


def test_17b_active_reproduced():
    assert round(W["active"]["decoder_layers"] / 1e9) == 17


def test_vision_encoder_is_292m():
    assert round(W["params"]["vision_backbone"] / 1e6) == 292


def test_precision_split_matches_serving_docs():
    g = W["expert_groups"]
    assert g["nvfp4"]["layers"] == list(range(0, 40))
    assert g["mxfp8"]["layers"] == list(range(40, 48))
    assert abs(g["nvfp4"]["expert_bits_per_param"] - 4.5) < 1e-3     # 4 bits + one E4M3 scale per 16
    assert abs(g["mxfp8"]["expert_bits_per_param"] - 8.25) < 1e-9    # 8 bits + one scale per 32


def test_layer_pattern():
    c = W["config"]
    assert (c["layers"], c["experts"], c["top_k"], c["shared_experts"]) == (48, 384, 8, 0)
    assert c["full_layer_ids"] == [5, 11, 17, 23, 29, 35, 41, 47]
    assert c["max_context"] == 1_048_576


def test_bytes_add_up():
    b = W["bytes"]
    assert sum(b["by_class_main"].values()) + b["draft"] == b["total_with_draft"]
    assert b["total_with_draft"] <= W["source"]["file_bytes"]          # files also hold headers
    assert W["source"]["file_bytes"] - b["total_with_draft"] < 1_000_000


def test_touched_bytes_add_up():
    t = W["touched_per_token"]
    assert abs(sum(t["bytes"].values()) - t["total"]) < 1
    assert abs(sum(t["share"].values()) - 1) < 1e-12


def test_kv_cross_checks_against_published_figures():
    x = W["kv"]["cross_checks"]
    assert x["mxfp8_bytes_per_rank_tp4"]["ours"] == x["mxfp8_bytes_per_rank_tp4"]["published"] == 528
    assert abs(x["mxfp8_over_bf16_with_draft"]["ours"] - x["mxfp8_over_bf16_with_draft"]["published"]) < 0.005
    # Known open discrepancy (docs/01_workload.md section 7). If this starts passing, the doc is stale.
    assert x["one_request_at_max_context_gib_bf16"]["ratio_published_over_ours"] > 1.15


def test_predictions_are_well_formed():
    ids = [p["id"] for p in PRE["predictions"]]
    assert ids == [f"P{i:02d}" for i in range(1, len(ids) + 1)]
    for p in PRE["predictions"]:
        assert p["low"] <= p["central"] <= p["high"], p["id"]
        assert p["basis"].strip() and p["claim"].strip()
    assert {p["group"] for p in PRE["predictions"]} == {"baseline", "H1", "H2", "H3", "H4", "H5", "limits", "method"}


def test_predictions_are_frozen():
    """The pre-registration must not change after the commit that froze it."""
    lock = ROOT / "data" / "preregistration.sha256"
    digest = hashlib.sha256((ROOT / "data" / "preregistration.yaml").read_bytes()).hexdigest()
    assert lock.read_text().split()[0] == digest, "data/preregistration.yaml was edited after it was frozen"


def test_docs_are_in_sync_with_results():
    """Re-rendering must not change any committed doc."""
    targets = ["README.md", "docs/00_hypotheses.md", "docs/01_workload.md", "docs/decisions.md"]
    before = {t: (ROOT / t).read_text() for t in targets}
    subprocess.run([sys.executable, "-m", "experiments.render_docs"], cwd=ROOT, check=True, capture_output=True)
    assert before == {t: (ROOT / t).read_text() for t in targets}
