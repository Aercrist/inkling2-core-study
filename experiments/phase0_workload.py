#!/usr/bin/env python3
"""Phase 0 workload summary for Inkling 2 Small, derived from the checkpoint headers.

Inputs (all committed, all read at source on 2026-11-14):
  data/inkling2_small/checkpoint_tensors.csv   every tensor's dtype, shape and bytes (safetensors headers)
  data/inkling2_small/config.json              model config
  data/inkling2_small/hf_quant_config.json     which tensors are NVFP4 / MXFP8 / BF16 / FP32
  data/inkling2_small/hub_model_info.json      the Hub's own parameter counts (cross-check)
  data/public_measurements.yaml                published serving figures (cross-check)
  data/phase0_constants.yaml                   GPU and SRAM constants for the napkin

Output: results/phase0_workload.json. Docs are rendered from that file; no number is typed by hand.

Labels used in the output:
  "derived"  exact arithmetic on the checkpoint headers and config
  "measured" quoted from a public source (with its key)
  "napkin"   a first-order estimate that the simulator must later reproduce or correct
This is not the simulator. It runs before any simulation, to support the pre-registration.
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
from collections import defaultdict

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "inkling2_small"
OUT = ROOT / "results" / "phase0_workload.json"

AUX_SUFFIXES = ("_scale", "_amax", "_outer_scale_denominator")
CLASS_OF = {"nvfp4": "nvfp4", "mxfp8": "mxfp8", "bfloat16": "bf16", "float32": "f32"}


def load_tensors() -> list[dict]:
    rows = []
    with open(DATA / "checkpoint_tensors.csv") as f:
        for r in csv.DictReader(f):
            r["elements"] = int(r["elements"])
            r["bytes"] = int(r["bytes"])
            r["shape"] = [int(x) for x in r["shape"].split("x")] if r["shape"] else []
            rows.append(r)
    return rows


def classify(tensors: list[dict], quant: dict) -> None:
    """Attach part, layer, precision class, aux flag and logical parameter count to every tensor."""
    cls_by_name = {}
    for part in ("model", "mtp"):
        for g in quant[part]["parameter_groups"]:
            for name in g["tensors"]:
                key = (part, name)
                assert key not in cls_by_name, f"{name} is in two quantization groups"
                cls_by_name[key] = CLASS_OF[g["compute_dtype"]]

    for t in tensors:
        part = "mtp" if t["file"] == "mtp.safetensors" else "model"
        name = t["tensor"]
        t["aux"] = name.endswith(AUX_SUFFIXES)
        parent = name
        if t["aux"]:
            for s in AUX_SUFFIXES:
                if name.endswith(s):
                    parent = name[: -len(s)]
        t["cls"] = cls_by_name[(part, parent)]  # KeyError = a tensor the quant config does not cover
        m = re.match(r"model\.llm\.layers\.(\d+)\.", name)
        t["layer"] = int(m.group(1)) if m else None
        if part == "mtp":
            t["part"] = "draft"
        elif name.startswith("model.visual."):
            t["part"] = "vision"
        else:
            t["part"] = "llm"
        t["expert"] = ".mlp.experts." in name
        if t["aux"]:
            t["params"] = 0
        elif t["cls"] == "nvfp4":
            assert t["dtype"] == "U8"
            t["params"] = 2 * t["elements"]  # two E2M1 codes per byte
        else:
            t["params"] = t["elements"]

    covered = {(("mtp" if t["file"] == "mtp.safetensors" else "model"), t["tensor"]) for t in tensors if not t["aux"]}
    assert covered == set(cls_by_name), "checkpoint tensors and quantization config disagree"


def distinct_experts(n_experts: int, top_k: int, rows: int) -> float:
    """Expected distinct experts hit by `rows` tokens that each pick top_k of n_experts uniformly and independently."""
    return n_experts * (1.0 - (1.0 - top_k / n_experts) ** rows)


def main() -> None:
    cfg = json.loads((DATA / "config.json").read_text())
    quant = json.loads((DATA / "hf_quant_config.json").read_text())
    hub = json.loads((DATA / "hub_model_info.json").read_text())
    src = json.loads((DATA / "checkpoint_source.json").read_text())
    const = yaml.safe_load((ROOT / "data" / "phase0_constants.yaml").read_text())
    pub = yaml.safe_load((ROOT / "data" / "public_measurements.yaml").read_text())
    meas = {m["id"]: m for m in pub["measurements"]}
    kvf = {m["id"]: m for m in pub["kv_facts"]}

    tc, mc = cfg["text_config"], cfg["mtp_config"]
    H, L = tc["hidden_size"], tc["num_hidden_layers"]
    E, K, I = tc["n_routed_experts"], tc["num_experts_per_tok"], tc["intermediate_size"]
    n_heads, n_kv, d_head = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
    window = tc["sliding_window_size"]
    local = sorted(tc["local_layer_ids"])
    full = [i for i in range(L) if i not in set(local)]

    tensors = load_tensors()
    classify(tensors, quant)

    # ---------------------------------------------------------------- 1. parameters and bytes
    main_t = [t for t in tensors if t["part"] != "draft"]
    params_by_cls = defaultdict(int)
    bytes_by_cls = defaultdict(int)
    for t in main_t:
        params_by_cls[t["cls"]] += t["params"]
        bytes_by_cls[t["cls"]] += t["bytes"]
    total_params = sum(params_by_cls.values())
    # The Hub's count follows a slightly different convention: it also counts the NVFP4 second-level
    # FP32 scales (one `_amax` per expert, one `_outer_scale_denominator` per tensor) as parameters,
    # and, like us, leaves out the per-block `_scale` tensors. Reproduce it exactly, then keep our own
    # weights-only count for everything else.
    second_level = sum(t["elements"] for t in main_t if t["aux"] and not t["tensor"].endswith("_scale"))
    hub_by = hub["safetensors_parameters_by_dtype"]
    ours_hub_convention = {"U8": params_by_cls["nvfp4"], "F8_E4M3": params_by_cls["mxfp8"],
                           "BF16": params_by_cls["bf16"], "F32": params_by_cls["f32"] + second_level}
    assert ours_hub_convention == hub_by, f"headers give {ours_hub_convention}, Hub reports {hub_by}"
    total_hub_convention = sum(ours_hub_convention.values())
    assert total_hub_convention == hub["safetensors_total"]

    draft_params = sum(t["params"] for t in tensors if t["part"] == "draft")
    draft_bytes = sum(t["bytes"] for t in tensors if t["part"] == "draft")
    vision_backbone = sum(t["params"] for t in tensors if t["tensor"].startswith("model.visual.backbone."))
    vision_adapter = sum(t["params"] for t in tensors if t["part"] == "vision") - vision_backbone
    total_bytes = sum(t["bytes"] for t in tensors)
    assert total_bytes == src["tensor_bytes"]

    expert_params = sum(t["params"] for t in main_t if t["expert"])
    per_expert_params = 2 * I * H + H * I
    assert expert_params == L * E * per_expert_params

    # ---------------------------------------------------------------- 2. per-layer table
    layers = []
    for i in range(L):
        lt = [t for t in tensors if t["layer"] == i]
        ex = [t for t in lt if t["expert"]]
        dn = [t for t in lt if not t["expert"]]
        ex_cls = {t["cls"] for t in ex}
        assert len(ex_cls) == 1
        ex_cls = ex_cls.pop()
        ex_bytes = sum(t["bytes"] for t in ex)
        dense_params = sum(t["params"] for t in dn)
        dense_bytes = sum(t["bytes"] for t in dn)
        router = sum(t["params"] for t in dn if ".mlp.gate." in t["tensor"])
        dense_bf16_bytes = sum(t["bytes"] for t in dn if t["cls"] == "bf16")
        layers.append({
            "layer": i,
            "attention": "sliding" if i in set(local) else "full",
            "expert_precision": ex_cls,
            "dense_params": dense_params,
            "dense_bytes": dense_bytes,
            "dense_bf16_bytes": dense_bf16_bytes,
            "router_params": router,
            "expert_params_each": per_expert_params,
            "expert_bytes_each": ex_bytes / E,
            "expert_bits_per_param": 8 * ex_bytes / (E * per_expert_params),
            "routed_params_per_token": K * per_expert_params,
            "routed_bytes_per_token": K * ex_bytes / E,
            # weight MACs per decoded token: every dense weight once, K of E experts
            "weight_macs_per_token": dense_params + K * per_expert_params,
            "resident_bytes": dense_bytes + ex_bytes,
        })
    by_group = {}
    for label, sel in (("nvfp4", [l for l in layers if l["expert_precision"] == "nvfp4"]),
                       ("mxfp8", [l for l in layers if l["expert_precision"] == "mxfp8"])):
        by_group[label] = {
            "layers": [l["layer"] for l in sel],
            "n_layers": len(sel),
            "expert_bytes_each": sel[0]["expert_bytes_each"],
            "expert_bits_per_param": sel[0]["expert_bits_per_param"],
            "resident_expert_bytes": sum(l["expert_bytes_each"] * E for l in sel),
        }

    # ---------------------------------------------------------------- 3. active parameters per token
    body_dense = sum(l["dense_params"] for l in layers)
    body_routed = sum(l["routed_params_per_token"] for l in layers)
    unembed = next(t for t in tensors if t["tensor"] == "model.llm.unembed.weight")
    embed = next(t for t in tensors if t["tensor"] == "model.llm.embed.weight")
    other_llm = sum(t["params"] for t in tensors
                    if t["part"] == "llm" and t["layer"] is None and t["tensor"] not in (unembed["tensor"], embed["tensor"]))
    active = {
        "decoder_layers": body_dense + body_routed,                       # attention, router, 8 experts, per layer
        "decoder_layers_plus_unembed": body_dense + body_routed + unembed["params"] + other_llm,
        "all_nonexpert_plus_routed": total_params - expert_params + body_routed,   # also embedding table and vision
        "published": 17e9,
        "routed_experts": body_routed,
        "dense_in_layers": body_dense,
        "unembed": unembed["params"],
        "embed_table": embed["params"],
    }

    # ---------------------------------------------------------------- 4. bytes touched per decoded token (batch 1, no speculation)
    dense_bytes_step = sum(l["dense_bytes"] for l in layers)
    routed_bytes_tok = sum(l["routed_bytes_per_token"] for l in layers)
    routed_nvfp4 = sum(l["routed_bytes_per_token"] for l in layers if l["expert_precision"] == "nvfp4")
    routed_mxfp8 = routed_bytes_tok - routed_nvfp4
    other_bytes = sum(t["bytes"] for t in tensors
                      if t["part"] == "llm" and t["layer"] is None and t["tensor"] not in (unembed["tensor"], embed["tensor"]))
    embed_row = embed["bytes"] / embed["shape"][0]
    touched = {
        "dense_bf16_attention_and_router": dense_bytes_step,
        "routed_experts_nvfp4": routed_nvfp4,
        "routed_experts_mxfp8": routed_mxfp8,
        "unembed_bf16": unembed["bytes"],
        "other": other_bytes + embed_row,
    }
    touched_total = sum(touched.values())
    touched_share = {k: v / touched_total for k, v in touched.items()}
    weight_macs_tok = sum(l["weight_macs_per_token"] for l in layers) + unembed["params"]

    # ---------------------------------------------------------------- 5. KV cache
    kv_elems = 2 * n_kv * d_head                      # K and V, per token, per layer
    fmt = {
        "bf16": 2.0,
        "mxfp8": 1.0 + 1.0 / const["kv_cache"]["mxfp8_block"],
        "nvfp4_style": 0.5 + 1.0 / const["kv_cache"]["nvfp4_block"],
    }
    per_layer = {k: kv_elems * v for k, v in fmt.items()}
    n_full, n_local = len(full), len(local)
    n_draft = const["kv_cache"]["draft_full_history_layers"]
    assert n_draft == mc["num_hidden_layers"]
    draft_kv_tok = n_draft * per_layer["bf16"]
    kv = {
        "elements_per_token_per_layer": kv_elems,
        "bytes_per_token_per_layer": per_layer,
        "full_attention_layers": full,
        "sliding_layers": n_local,
        "window": window,
        "target_full_bytes_per_token": {k: n_full * v for k, v in per_layer.items()},
        "draft_full_bytes_per_token_bf16": draft_kv_tok,
        "sliding_bytes_per_request_bf16": n_local * window * per_layer["bf16"],
        "max_context": tc["model_max_length"],
    }
    ctx = tc["model_max_length"]
    kv["at_max_context_bytes"] = {
        "target_bf16": ctx * kv["target_full_bytes_per_token"]["bf16"],
        "target_mxfp8": ctx * kv["target_full_bytes_per_token"]["mxfp8"],
        "target_nvfp4_style": ctx * kv["target_full_bytes_per_token"]["nvfp4_style"],
        "draft_bf16": ctx * draft_kv_tok,
    }
    # cross-checks against published KV facts
    rank_bytes = per_layer["mxfp8"] / 4
    ratio = (kv["target_full_bytes_per_token"]["mxfp8"] + draft_kv_tok) / (kv["target_full_bytes_per_token"]["bf16"] + draft_kv_tok)
    ours_1m_gib = (ctx * (kv["target_full_bytes_per_token"]["bf16"] + draft_kv_tok) + kv["sliding_bytes_per_request_bf16"]) / 2**30
    kv["cross_checks"] = {
        "mxfp8_bytes_per_rank_tp4": {"ours": rank_bytes, "published": kvf["tokenspeed_kv_bytes_per_rank"]["value"],
                                     "src": kvf["tokenspeed_kv_bytes_per_rank"]["src"]},
        "mxfp8_over_bf16_with_draft": {"ours": ratio, "published": kvf["tokenspeed_kv_ratio_mxfp8"]["value"],
                                       "src": kvf["tokenspeed_kv_ratio_mxfp8"]["src"]},
        "one_request_at_max_context_gib_bf16": {"ours": ours_1m_gib, "published": kvf["vllm_kv_1m_request"]["value"],
                                                "ratio_published_over_ours": kvf["vllm_kv_1m_request"]["value"] / ours_1m_gib,
                                                "src": kvf["vllm_kv_1m_request"]["src"]},
    }
    assert abs(rank_bytes - kvf["tokenspeed_kv_bytes_per_rank"]["value"]) < 1e-9

    # ---------------------------------------------------------------- 6. attention compute (MACs per row per layer)
    attn_macs_per_position = 2 * n_heads * d_head          # QK^T and PV for every query head
    attn = {
        "macs_per_row_per_layer_per_context_token": attn_macs_per_position,
        "sliding_macs_per_row": n_local * window * attn_macs_per_position,
        "full_macs_per_row_per_context_token": n_full * attn_macs_per_position,
        "context_where_full_attention_macs_equal_weight_macs": weight_macs_tok / (n_full * attn_macs_per_position),
    }

    # ---------------------------------------------------------------- 7. one speculative (DSpark) step, napkin
    rows = const["speculation"]["verify_rows"]
    assert rows == mc["block_size"]
    d = distinct_experts(E, K, rows)
    expert_bytes_step = sum(l["expert_bytes_each"] * d for l in layers)
    step = {
        "verify_rows": rows,
        "expert_picks_per_layer": rows * K,
        "distinct_experts_per_layer_uniform_independent": d,
        "bytes": {
            "dense_bf16_attention_and_router": dense_bytes_step,
            "experts": expert_bytes_step,
            "unembed_bf16": unembed["bytes"],
            "draft_bf16": draft_bytes,
            "other": other_bytes,
        },
    }
    step["bytes_total"] = sum(step["bytes"].values())
    acc = const["speculation"]["acceptance_napkin"]
    step["acceptance_assumed"] = acc
    step["bytes_per_accepted_token"] = step["bytes_total"] / acc
    step["bytes_per_token_vs_no_speculation"] = step["bytes_per_accepted_token"] / touched_total
    # context at which KV traffic per step equals weight traffic per step
    step["context_where_kv_equals_weights"] = {
        "dspark_mxfp8_target_bf16_draft": step["bytes_total"] / (kv["target_full_bytes_per_token"]["mxfp8"] + draft_kv_tok),
        "dspark_bf16": step["bytes_total"] / (kv["target_full_bytes_per_token"]["bf16"] + draft_kv_tok),
        "no_speculation_mxfp8": touched_total / kv["target_full_bytes_per_token"]["mxfp8"],
        "no_speculation_bf16": touched_total / kv["target_full_bytes_per_token"]["bf16"],
    }

    # ---------------------------------------------------------------- 8. GPU bandwidth ceilings vs published speeds, napkin
    g = const["gpu"]
    eta = g["hbm_eff"]

    def ceiling(n, bw, bytes_step, tokens):
        t = bytes_step / (n * bw * eta)
        return {"n_gpus": n, "stream_time_ms": 1e3 * t, "tok_s_ceiling": tokens / t}

    roof = {
        "hbm_eff": eta,
        "b300_x4_no_speculation": ceiling(4, g["b300"]["hbm_bw"], touched_total, 1.0),
        "b300_x4_dspark": ceiling(4, g["b300"]["hbm_bw"], step["bytes_total"], acc),
        "b300_x2_no_speculation": ceiling(2, g["b300"]["hbm_bw"], touched_total, 1.0),
        "b300_x2_dspark": ceiling(2, g["b300"]["hbm_bw"], step["bytes_total"], acc),
        "gb200_x4_dspark": ceiling(4, g["gb200"]["hbm_bw"], step["bytes_total"], acc),
    }

    def versus(meas_id, roof_key, tokens):
        m = meas[meas_id]
        step_ms = 1e3 * tokens / m["value"]
        r = roof[roof_key]
        return {"measured_tok_s": m["value"], "src": m["src"], "hardware": m["hardware"], "engine": m["engine"],
                "tokens_per_step": tokens, "implied_step_ms": step_ms, "stream_time_ms": r["stream_time_ms"],
                "share_of_step_that_is_weight_streaming": r["stream_time_ms"] / step_ms,
                "measured_over_ceiling": m["value"] / r["tok_s_ceiling"]}

    roof["vs_published"] = {
        "tokenspeed_b300_x4_no_speculation": versus("tokenspeed_b300_c1_target_only", "b300_x4_no_speculation", 1.0),
        "tokenspeed_b300_x4_dspark": versus("tokenspeed_b300_c1_dspark", "b300_x4_dspark", acc),
        "vllm_gb200_x4_dspark": versus("vllm_gb200_tp4_single_user", "gb200_x4_dspark", meas["vllm_gb200_tp4_single_user"]["acceptance_length"]),
    }
    load = g["load_fraction"]
    roof["gpu_power_napkin"] = {
        "load_fraction_assumed": load,
        "b300_x4_w": 4 * g["b300"]["tdp_w"] * load,
        "b300_x2_w": 2 * g["b300"]["tdp_w"] * load,
        "tokenspeed_b300_x4_dspark_tok_per_j": meas["tokenspeed_b300_c1_dspark"]["value"] / (4 * g["b300"]["tdp_w"] * load),
    }

    # ---------------------------------------------------------------- 9. weights-on-chip sizing, napkin
    s = const["sram_sizing"]
    mib = total_bytes / 2**20
    per_die = s["die_mm2"] * s["sram_fraction"]

    def size(mm2_per_mib):
        area = mib * mm2_per_mib
        return {"mm2_per_mib": mm2_per_mib, "sram_mm2": area, "dies": area / per_die, "gb_per_die": total_bytes / 1e9 / (area / per_die)}

    sizing = {
        "resident_bytes": total_bytes,
        "resident_bits": 8 * total_bytes,
        "die_mm2_assumed": s["die_mm2"],
        "sram_fraction_assumed": s["sram_fraction"],
        "with_periphery": size(s["mm2_per_mib"]),
        "array_only_bound": size(s["mm2_per_mib_array_only"]),
        "share_of_weight_bytes_used_per_token": touched_total / total_bytes,
        "share_of_expert_bytes_used_per_token": K / E,
    }

    # ---------------------------------------------------------------- 10. our starting anchors vs the checkpoint
    all_nvfp4_bytes = total_params * 4.5 / 8
    anchors = {
        "all_nvfp4_gb_noted": 334.0,
        "all_nvfp4_gb_recomputed": all_nvfp4_bytes / 1e9,
        "checkpoint_gb": total_bytes / 1e9,
        "checkpoint_over_noted": total_bytes / 334e9,
        "touched_gb_noted": 9.6,
        "touched_gb": touched_total / 1e9,
        "touched_over_noted": touched_total / 9.6e9,
    }

    out = {
        "_note": "Phase 0 workload summary. 'derived' = exact arithmetic on checkpoint headers and config; "
                 "'measured' = quoted from a public source; 'napkin' = first-order estimate, not a simulator result.",
        "source": {"repo": src["repo"], "revision": src["revision"], "n_weight_files": src["n_weight_files"],
                   "n_tensors": src["n_tensors"], "file_bytes": src["file_bytes"]},
        "config": {
            "hidden_size": H, "layers": L, "experts": E, "top_k": K, "expert_intermediate": I,
            "shared_experts": tc["n_shared_experts"], "query_heads": n_heads, "kv_heads": n_kv, "head_dim": d_head,
            "sliding_window": window, "sliding_layers": n_local, "full_layers": n_full, "full_layer_ids": full,
            "max_context": tc["model_max_length"], "vocab": tc["vocab_size"],
            "attn_res_block": tc["n_layers_per_attn_res_block"], "sconv_kernel": tc["sconv_kernel_size"],
            "draft_layers": mc["num_hidden_layers"], "draft_block": mc["block_size"], "draft_mlp": mc["intermediate_size"],
            "vision_layers": cfg["vision_config"]["num_hidden_layers"], "vision_width": cfg["vision_config"]["hidden_size"],
        },
        "params": {                                                   # derived
            "label": "derived",
            "total": total_params, "total_hub_convention": total_hub_convention,
            "hub_total": hub["safetensors_total"], "second_level_scales": second_level, "published_total": 594e9,
            "by_class": dict(params_by_cls),
            "experts": expert_params, "per_expert": per_expert_params,
            "draft": draft_params, "vision_backbone": vision_backbone, "vision_adapter": vision_adapter,
            "published_vision": 292e6,
            "share_by_class": {k: v / total_params for k, v in params_by_cls.items()},
        },
        "bytes": {                                                    # derived
            "label": "derived",
            "total_with_draft": total_bytes, "draft": draft_bytes,
            "by_class_main": dict(bytes_by_cls),
            "bits_per_param": {k: 8 * bytes_by_cls[k] / params_by_cls[k] for k in params_by_cls},
            "share_by_class_main": {k: v / sum(bytes_by_cls.values()) for k, v in bytes_by_cls.items()},
        },
        "expert_groups": by_group,
        "layers": layers,
        "active": {"label": "derived", **active},
        "touched_per_token": {"label": "derived", "bytes": touched, "total": touched_total, "share": touched_share,
                              "weight_macs": weight_macs_tok, "weight_flops": 2 * weight_macs_tok},
        "kv": {"label": "derived; cross-checks are measured (source)", **kv},
        "attention": {"label": "derived", **attn},
        "dspark_step": {"label": "napkin", **step},
        "gpu_ceiling": {"label": "napkin", **roof},
        "weights_on_chip_sizing": {"label": "napkin", **sizing},
        "starting_anchors": {"label": "derived", **anchors},
        "quantization": {
            "label": "derived",
            "nvfp4_weights": next(g for g in quant["model"]["parameter_groups"] if g["compute_dtype"] == "nvfp4")
                             | {"tensors": None},
            "mxfp8_weights": next(g for g in quant["model"]["parameter_groups"] if g["compute_dtype"] == "mxfp8")
                             | {"tensors": None},
        },
    }
    for k in ("nvfp4_weights", "mxfp8_weights"):
        out["quantization"][k].pop("tensors")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"params {total_hub_convention:,} in the Hub's convention (Hub {hub['safetensors_total']:,}); bytes {total_bytes:,}; "
          f"touched/token {touched_total / 1e9:.2f} GB -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
