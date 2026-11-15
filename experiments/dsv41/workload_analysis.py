"""Phase 1: per-op FLOPs and bytes for DeepSeek-V4.1-Flash decode and prefill, plus a roofline
on the GPU baseline. Writes results/workload.json, results/fig_roofline.png and
results/fig_decode_bytes_vs_context.png.
"""
from __future__ import annotations

import collections
import json

import matplotlib.pyplot as plt
import numpy as np

from experiments.dsv41.common import FIG, RESULTS, save_json, style
from sim.hardware import load_hw
from sim.models.dsv41.modelspec import ModelSpec
from sim.models.dsv41.prefill import _layer
from sim.routing import distinct_experts
from sim.models.dsv41.workload import CATEGORY_LABELS, DecodeWorkload, decode_step


def op_table(ops, n_chips: int, tokens: float) -> list[dict]:
    """System-unique bytes and FLOPs per category, normalized per output token."""
    agg = collections.OrderedDict((k, {"flops": 0.0, "bytes": 0.0, "coll_bytes": 0.0}) for k in CATEGORY_LABELS)
    for o in ops:
        a = agg[o.category]
        a["flops"] += o.total_flops()
        a["bytes"] += o.w_tp + o.w_rep + o.w_dyn + o.kv_rep + o.kv_tp + o.host
        a["coll_bytes"] += (o.coll * 2 * (n_chips - 1) / n_chips + o.gather * (n_chips - 1)) * n_chips
    rows = []
    for k, a in agg.items():
        if a["flops"] == 0 and a["bytes"] == 0 and a["coll_bytes"] == 0:
            continue
        rows.append({"category": k, "label": CATEGORY_LABELS[k],
                     "flops_per_token": a["flops"] / tokens, "bytes_per_token": a["bytes"] / tokens,
                     "coll_bytes_per_token": a["coll_bytes"] / tokens,
                     "intensity": a["flops"] / a["bytes"] if a["bytes"] else float("inf")})
    return rows


def weighted_decode_flops(ops) -> float:
    """Paper Figure 2 convention: BF16 x1, FP8 x0.5, FP4 x0.25 (fp32 vector ops x1)."""
    w = {"bf16": 1.0, "fp8": 0.5, "fp4": 0.25, "fp32": 1.0}
    return sum(v * w[k] for o in ops for d in (o.flops, o.flops_rep) for k, v in d.items())


def prefill_table(s: ModelSpec, ctx: int, chunk: int = 8192) -> list[dict]:
    """Per prompt token at context position ctx (encoder layers + decoder KV projection)."""
    hw = load_hw("configs/hw/dsv41/b300_tp8.yaml", n_chips=1)
    D = s.dim
    attn_params = sum(s.params(f"layers.*.attn.{k}") for k in ("wq_a", "wkv", "wq_b", "wo_a", "wo_b"))
    L2 = s.n_layers // 2
    dexp = distinct_experts(chunk, s.n_activated_experts, s.n_routed_experts)
    rows = []
    attn_b = sum(s.attn_bytes().values())
    rows.append(("attn_proj", 2.0 * attn_params * L2, attn_b * L2 / chunk))
    n_att = sum(s.attended_entries(l, ctx) for l in range(L2))
    kv_b = sum(s.attended_entries(l, ctx) for l in range(L2)) * 0  # gathered from on-chip cache tiles
    rows.append(("attn_core", 4.0 * s.n_heads * s.head_dim * n_att, kv_b))
    scan = sum(s.index_scan_entries(l, ctx) for l in range(L2))
    rows.append(("indexer", 2.0 * s.index_n_heads * s.index_head_dim * scan + 2.0 * D * s.head_dim * 2,
                 scan * s.kv_entry_bytes["index"] / chunk))
    rows.append(("router", 2.0 * D * s.n_routed_experts * L2, s.tensors["layers.*.ffn.gate.weight"].bytes_each * L2 / chunk))
    rows.append(("moe_shared", 2.0 * 3 * D * s.moe_inter_dim * L2, s.shared_expert_bytes * L2 / chunk))
    rows.append(("moe_routed", 2.0 * s.n_activated_experts * 3 * D * s.moe_inter_dim * L2, dexp * s.expert_bytes * L2 / chunk))
    rows.append(("mhc", 2.0 * 2 * s.hc_mult * D * (2 + s.hc_mult) * s.hc_mult * L2,
                 2 * s.tensors["layers.*.hc_attn_fn"].bytes_each * L2 / chunk))
    n_cols = (s.engram_max_ngram_size - 1) * s.engram_n_heads
    rows.append(("engram", 2.0 * n_cols * s.engram_head_dim * D * (s.hc_mult + 1) * 2,
                 2 * s.wbytes("layers.*.engram.wkv") / chunk + 2 * n_cols * (s.engram_head_dim * 1.03125)))
    rows.append(("collective", 0.0, 0.0))
    out = []
    for k, f, b in rows:
        out.append({"category": k, "label": CATEGORY_LABELS[k], "flops_per_token": f, "bytes_per_token": b,
                    "coll_bytes_per_token": 2 * 2 * D * 2 * L2 if k == "collective" else 0.0,
                    "intensity": f / b if b else float("inf")})
    return out


def main() -> dict:
    style()
    s = ModelSpec.load()
    out: dict = {}
    # ---- model validation against the tech report -------------------------------------------
    pc = s.param_counts
    out["validation"] = [
        {"quantity": "backbone parameters", "paper": "552B", "ours": pc["backbone_total"]},
        {"quantity": "Engram parameters", "paper": "196B", "ours": pc["engram_tables"]},
        {"quantity": "activated / token, decode", "paper": "16B", "ours": pc["activated_decode"]},
        {"quantity": "activated / token, prefill", "paper": "8B", "ours": pc["activated_prefill"]},
        {"quantity": "global KV bytes / token", "paper": "890", "ours": s.global_kv_bytes_per_token},
    ]
    w4 = weighted_decode_flops(decode_step(s, DecodeWorkload(context=4096, verify_rows=1, draft_rows=1)))
    w1m = weighted_decode_flops(decode_step(s, DecodeWorkload(context=1 << 20, verify_rows=1, draft_rows=1)))
    out["validation"].append({"quantity": "weighted decode FLOPs, 1M vs 4K context", "paper": "+1/4", "ours": w1m / w4 - 1})
    out["param_counts"] = pc
    out["expert_bytes"] = s.expert_bytes
    out["shared_expert_bytes"] = s.shared_expert_bytes
    out["kv_entry_bytes"] = s.kv_entry_bytes

    # ---- decode: the measured workload (batch 1, 6 verify rows, 5 draft rows) ---------------
    wl = DecodeWorkload(context=2600)
    ops = decode_step(s, wl)
    out["decode_table"] = op_table(ops, 4, wl.accept_len)
    out["decode_step"] = {
        "accept_len": wl.accept_len, "verify_rows": wl.verify_rows, "draft_rows": wl.draft_rows,
        "distinct_experts_verify": distinct_experts(wl.verify_rows, s.n_activated_experts, s.n_routed_experts),
        "distinct_experts_draft": distinct_experts(wl.draft_rows, s.dspark_n_activated_experts, s.dspark_n_routed_experts),
        "unique_bytes": sum(o.w_tp + o.w_rep + o.w_dyn + o.kv_rep + o.kv_tp + o.host for o in ops),
        "flops": sum(o.total_flops() for o in ops),
    }
    for tp in (4, 8):  # as built: the engine's padded expert layout and its verify-head precision
        o_tp = decode_step(s, DecodeWorkload(context=2600, layout_tp=tp, verify_head="fp8" if tp == 4 else "bf16"))
        per = {"weights_tp": sum(o.w_tp for o in o_tp) / tp, "weights_replicated": sum(o.w_rep for o in o_tp),
               "experts_routed": sum(o.w_dyn for o in o_tp) / tp, "kv": sum(o.kv_rep for o in o_tp) + sum(o.kv_tp for o in o_tp) / tp}
        per["total"] = sum(per.values())
        out[f"decode_per_gpu_tp{tp}"] = per
    out["engram"] = {
        "lookup_bytes_per_step": sum(o.host for o in ops),
        "table_bytes": pc["engram_tables"] * (1 + 1 / 32),
        "wkv_bytes_per_step": sum(o.w_tp for o in ops if o.name == "engram_wkv"),
    }
    out["kernels_per_step"] = sum(o.n_kernels for o in ops)
    out["layout_factor"] = {"tp4": s.expert_layout_factor(4), "tp8": s.expert_layout_factor(8)}
    out["collectives_per_step"] = sum(1 for o in ops if o.coll > 0)
    out["ops_per_step"] = len(ops)
    # what an expert-parallel (all-to-all) layout would move instead of the two all-reduces
    rows = wl.verify_rows
    out["ep_alltoall_bytes_per_layer"] = 2 * rows * s.n_activated_experts * s.dim * 2  # dispatch + combine, bf16
    out["tp_allreduce_bytes_per_layer"] = 2 * rows * s.dim * 2

    # ---- prefill per token at two context positions -------------------------------------------
    out["prefill_table_64k"] = prefill_table(s, 65536)
    out["prefill_table_1m"] = prefill_table(s, 1 << 20)

    # ---- roofline on B300 ----------------------------------------------------------------------
    hw = load_hw("configs/hw/dsv41/b300_tp8.yaml", n_chips=1)
    out["ridge"] = {p: hw.peak[p] / hw.hbm_bw for p in ("fp4", "fp8", "bf16")}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ai = np.logspace(-1, 4.5, 200)
    for p, ls in (("fp4", "-"), ("fp8", "--"), ("bf16", ":")):
        ax.plot(ai, np.minimum(hw.peak[p], ai * hw.hbm_bw) / 1e12, ls, color="0.35", lw=1.2, label=f"B300 {p.upper()} roof")
    pts = []
    for label, wlx in (("decode, batch 1 (measured workload)", DecodeWorkload(context=2600)),
                       ("decode, batch 64", DecodeWorkload(context=2600, batch=64))):
        o2 = decode_step(s, wlx)
        for cat in ("attn_proj", "moe_routed", "moe_shared", "attn_core", "indexer", "head"):
            sel = [o for o in o2 if o.category == cat]
            f = sum(o.total_flops() for o in sel)
            b = sum(o.w_tp + o.w_rep + o.w_dyn + o.kv_rep + o.kv_tp for o in sel)
            if b > 0 and f > 0:
                pts.append((label, cat, f / b))
    for row in out["prefill_table_64k"]:
        if row["bytes_per_token"] > 0 and row["flops_per_token"] > 0 and row["category"] in ("attn_proj", "moe_routed", "moe_shared", "indexer"):
            pts.append(("prefill, 8K chunk @64K ctx", row["category"], row["intensity"]))
    markers = {"decode, batch 1 (measured workload)": ("o", "#c0392b"), "decode, batch 64": ("s", "#e67e22"),
               "prefill, 8K chunk @64K ctx": ("^", "#2471a3")}
    seen = set()
    prec = {"attn_core": "bf16", "indexer": "fp4"}
    short = {"attn_proj": "attn proj", "moe_routed": "routed experts", "moe_shared": "shared expert", "attn_core": "attn core",
             "indexer": "indexer", "head": "LM heads"}
    offs = {"attn_proj": (6, -12), "moe_shared": (6, 4), "head": (-48, 6), "moe_routed": (6, -6), "attn_core": (6, -10), "indexer": (6, -6)}
    for label, cat, x in pts:
        y = min(hw.peak[prec.get(cat, "fp8")], x * hw.hbm_bw) / 1e12
        m, c = markers[label]
        ax.scatter([x], [y], marker=m, color=c, s=36, zorder=3, label=None if label in seen else label)
        seen.add(label)
        if label.startswith("decode, batch 1") or label.startswith("prefill"):
            ax.annotate(short[cat], (x, y), textcoords="offset points", xytext=offs.get(cat, (4, -10)), fontsize=6.5, color=c)
    out["roofline_points"] = [{"series": a, "category": b, "intensity": c} for a, b, c in pts]
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte of HBM traffic)")
    ax.set_ylabel("attainable TFLOP/s per GPU")
    ax.set_title("DeepSeek-V4.1-Flash ops on the HGX B300 roofline (per GPU, modeled)")
    for p_, x_ in (("fp8", hw.peak["fp8"] / hw.hbm_bw),):
        ax.axvline(x_, color="0.75", lw=0.8, ls=":")
        ax.text(x_ * 1.1, 40, f"FP8 ridge\n{x_:.0f} FLOP/B", fontsize=6.5, color="0.4")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_roofline.png", dpi=160)
    plt.close(fig)

    # ---- decode bytes per step vs context (where KV starts to matter) -------------------------
    ctxs = [4096, 16384, 65536, 262144, 1 << 20]
    series = collections.defaultdict(list)
    for B in (1, 32):
        for c in ctxs:
            o2 = decode_step(s, DecodeWorkload(context=c, batch=B))
            w = sum(x.w_tp + x.w_rep + x.w_dyn for x in o2)
            kv = sum(x.kv_rep + x.kv_tp for x in o2)
            series[B].append((c, w, kv))
    out["bytes_vs_context"] = {str(B): [{"context": c, "weights": w, "kv": kv} for c, w, kv in v] for B, v in series.items()}
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for B, col in ((1, "#c0392b"), (32, "#2471a3")):
        c, w, kv = zip(*series[B])
        ax.plot(c, np.array(kv) / (np.array(w) + np.array(kv)) * 100, "o-", color=col, label=f"batch {B}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("KV + indexer share of step bytes (%)")
    ax.set_title("Decode traffic stays weight-dominated until long context x large batch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig_decode_bytes_vs_context.png", dpi=160)
    plt.close(fig)

    save_json(out, RESULTS / "workload.json")
    return out


if __name__ == "__main__":
    r = main()
    for v in r["validation"]:
        print(v)
    print(json.dumps(r["decode_step"], indent=1))
