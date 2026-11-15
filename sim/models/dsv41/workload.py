"""Op graph for one DeepSeek-V4.1-Flash decode step, as executed by deepseek-ultrafast.

A step (crates/engine/src/model.rs: capture_spec_blk / dspark_draft / verify_prep) is

  1. DSpark draft: main_proj, then 3 drafter blocks (SWA-only attention + 128-expert top-3
     MoE) over `draft_rows` = block size 5 rows, the FP8 draft head and the Markov head.
  2. Verify: the 40-layer backbone over `verify_rows` = block + 1 = 6 rows.
  3. Verify head + sampler + accept.

Every op carries the bytes it must read and the FLOPs it performs, split by how tensor
parallelism (TP) distributes them, so the same graph can be costed on GPUs, an ASIC or an
FPGA. Byte counts are on-disk checkpoint bytes (sim/modelspec.py).

Sharding (weights.rs `layer()`): wq_b by heads, wo_a by groups, wo_b by input rows, routed
and shared experts by intermediate dim (2304/TP), head/markov head by vocab, engram wkv by
output, main_proj by output. wq_a/wkv (fused wqkv), the indexer wq_b, compressor, gate and
mHC projections are replicated on every rank. The latent KV (one 512-wide KV head shared by
all 64 query heads) is read by every rank.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from ...ops import PRECISIONS, Op  # noqa: F401  (re-exported)
from ...routing import distinct_experts
from .modelspec import ModelSpec

@dataclass
class DecodeWorkload:
    verify_rows: int = 6  # DSpark block 5 + 1 (capture_spec_blk: verify_prep(blk + 1))
    draft_rows: int = 5  # dspark_block_size
    accept_len: float = 5.6  # tokens emitted per step (reported)
    batch: int = 1  # concurrent sequences per step (deepseek-ultrafast serves 1)
    context: int = 2600  # mean context during decode
    locality: float = 0.0  # expert-routing locality across a step's rows (routing.py)
    verify_head: str = "fp8"  # 'fp8' on GB300 TP4 (ops::verify_head_fp8), 'bf16' otherwise
    draft_head: str = "fp8"
    act_bytes: int = 2  # bf16 activations on the wire
    # Shard the projections the GPU engine replicates (wqkv, indexer wq_b, compressor, gate,
    # mHC, idx weights) and all-gather their small outputs instead. Trades ~0.74 GB/chip/step
    # of replicated weight reads for one extra small collective per replicated op.
    shard_replicated: bool = False
    # Split the batch into independent micro-batches whose ops interleave, so one micro-batch's
    # attention/compute overlaps the other's expert streaming (DeepSeek-style two-batch
    # overlap). None = 2 when batch >= 2, else 1. Weights are fetched once and shared.
    microbatches: int | None = None
    # 0: stream on-disk expert bytes. N: deepseek-ultrafast's padded in-memory layout at TP N
    # (sgl layout; drafter experts use it only at TP8, `ops::sgl_mtp_on`).
    layout_tp: int = 0

    @property
    def rv(self) -> int:
        return self.verify_rows * self.batch

    @property
    def rd(self) -> int:
        return self.draft_rows * self.batch


def _gemv(name, phase, layer, din, dout, rows, prec, *, w_tp=0.0, w_rep=0.0, kernels=1, cat="dense"):
    f = 2.0 * rows * din * dout
    op = Op(name, phase, layer, "gemv", w_tp=w_tp, w_rep=w_rep, rows=rows, n_kernels=kernels, category=cat)
    if w_rep and not w_tp:
        op.flops_rep = {prec: f}
    else:
        op.flops = {prec: f}
    return op


def _block(s: ModelSpec, wl: DecodeWorkload, phase: str, l: int, rows: int, mtp: bool) -> list[Op]:
    """One transformer block (attention + MoE) at `rows` activation rows."""
    pre = "mtp.*" if mtp else "layers.*"
    D, A = s.dim, wl.act_bytes
    ops: list[Op] = []
    ab = s.attn_bytes(pre)
    hc_b = s.tensors[f"{pre}.hc_attn_fn"].bytes_each
    hc_f = 2.0 * rows * s.hc_mult * D * (2 + s.hc_mult) * s.hc_mult
    sink = 20.0 * rows * s.hc_mult * s.hc_mult * 4  # Sinkhorn on the 4x4 comb matrix
    ctx = wl.context
    role = "swa" if mtp else s.layer_role(l)

    if not mtp and l in s.engram_layer_ids:
        n_cols = (s.engram_max_ngram_size - 1) * s.engram_n_heads
        row_b = s.engram_head_dim + s.engram_head_dim / 32  # FP8 row + E8M0 scales
        ops.append(Op("engram_lookup", phase, l, "lookup", host=rows * n_cols * row_b, rows=rows, n_kernels=1, category="engram"))
        ops.append(_gemv("engram_wkv", phase, l, n_cols * s.engram_head_dim, D * (s.hc_mult + 1), rows, "fp8",
                         w_tp=s.wbytes("layers.*.engram.wkv"), kernels=2, cat="engram"))

    ops.append(Op("hc_attn", phase, l, "vector", w_rep=hc_b, rows=rows, flops_rep={"fp32": hc_f + sink}, n_kernels=0, category="mhc",
                  out_w=(2 + s.hc_mult) * s.hc_mult * 2))
    op = _gemv("wqkv", phase, l, D, s.q_lora_rank + s.head_dim, rows, "fp8", w_rep=ab["wq_a"] + ab["wkv"], cat="attn_proj")
    op.out_w = s.q_lora_rank + s.head_dim
    ops.append(op)

    is_index = role in ("full", "reindex")
    nq_i = s.index_n_heads * s.index_head_dim
    op = _gemv("wq_b", phase, l, s.q_lora_rank, s.n_heads * s.head_dim, rows, "fp8", w_tp=ab["wq_b"], cat="attn_proj")
    if is_index:
        op.w_rep = s.wbytes("layers.*.attn.indexer.wq_b")
        op.flops_rep = {"fp8": 2.0 * rows * s.q_lora_rank * nq_i}
        op.out_w = nq_i
    ops.append(op)

    if role == "full":  # compressor + index-key projection on the new rows
        r = s.compress_ratios[l]
        cb = s.tensors["layers.*.attn.compressor.wkv.weight"].bytes_each * (2 if r > 1 else 1)
        cb += s.tensors["layers.*.attn.indexer.wk.weight"].bytes_each
        f = 2.0 * rows * D * s.head_dim * (2 if r > 1 else 1) + 2.0 * rows * s.head_dim * s.index_head_dim
        ops.append(Op("compressor", phase, l, "gemv", w_rep=cb, rows=rows, flops_rep={"bf16": f}, n_kernels=0, category="indexer",
                      out_w=s.head_dim * (2 if r > 1 else 1) + s.index_head_dim))

    if is_index:
        n_scan = s.index_scan_entries(l, ctx)
        e_idx = s.kv_entry_bytes["index"]
        wp = s.tensors["layers.*.attn.indexer.weights_proj.weight"].bytes_each
        ops.append(Op("idx_weights", phase, l, "gemv", w_rep=wp, rows=rows, flops_rep={"bf16": 2.0 * rows * D * s.index_n_heads},
                      n_kernels=0, category="indexer", out_w=s.index_n_heads))
        # scores: every query row against the shared index K; rows of one sequence share the K read
        # decode (<= 16 rows) runs the scan and top-k replicated on every rank (model.rs::idx_rows
        # returns None), so there is no gather; the scan bytes are per chip
        ops.append(Op("idx_score", phase, l, "indexer", kv_rep=wl.batch * n_scan * e_idx, rows=rows,
                      flops_rep={"fp4": 2.0 * rows * nq_i * n_scan, "fp32": 2.0 * rows * s.index_n_heads * n_scan},
                      n_kernels=2, category="indexer"))
        topk_elems = rows * n_scan
        if l == s.candidate_source_layer:
            topk_elems += rows * n_scan / s.candidate_block_size
        ops.append(Op("idx_topk", phase, l, "topk", rows=rows, flops_rep={"fp32": 4.0 * topk_elems},
                      n_kernels=2, category="indexer"))

    n_att = s.window_size + (0 if role == "swa" else min(s.index_topk, ctx // max(1, s.compress_ratios[l])))
    if mtp:
        n_att = min(s.window_size, ctx) + rows // wl.batch
    e = s.kv_entry_bytes
    kv = wl.batch * min(s.window_size, ctx) * e["swa"]  # the window ring is shared by a sequence's rows
    if role != "swa":
        kv += rows * min(s.index_topk, ctx) * e["main"]  # each row has its own top-k (upper bound)
    ops.append(Op("sparse_attn", phase, l, "attn", kv_rep=kv, rows=rows,
                  flops={"bf16": 4.0 * rows * s.n_heads * s.head_dim * n_att}, n_kernels=1, category="attn_core"))
    ops.append(_gemv("wo_a", phase, l, s.n_heads * s.head_dim // s.o_groups, s.o_lora_rank * s.o_groups, rows, "fp8",
                     w_tp=ab["wo_a"], cat="attn_proj"))
    ops.append(_gemv("wo_b", phase, l, s.o_groups * s.o_lora_rank, D, rows, "fp8", w_tp=ab["wo_b"], cat="attn_proj"))
    # all-reduces are pushed from the producer kernel and fused with the mHC post/next-layer prep
    # (NVLS multimem + Lamport buffers: comm_dec_attn_pp, comm_lp_sgl): latency, no extra kernel
    ops.append(Op("ar_attn", phase, l, "collective", coll=rows * D * A, rows=rows, n_kernels=0, category="collective"))

    ops.append(Op("hc_ffn", phase, l, "vector", w_rep=hc_b, rows=rows, flops_rep={"fp32": hc_f + sink}, n_kernels=0, category="mhc",
                  out_w=(2 + s.hc_mult) * s.hc_mult * 2))
    n_exp = s.dspark_n_routed_experts if mtp else s.n_routed_experts
    k = s.dspark_n_activated_experts if mtp else s.n_activated_experts
    gb = s.tensors[f"{pre}.ffn.gate.weight"].bytes_each
    ops.append(Op("gate", phase, l, "router", w_rep=gb, rows=rows, flops_rep={"bf16": 2.0 * rows * D * n_exp, "fp32": 8.0 * rows * n_exp},
                  n_kernels=1, category="router", out_w=n_exp * 2))
    inter = s.moe_inter_dim
    pad = s.expert_layout_factor(wl.layout_tp) if (not mtp or wl.layout_tp == 8) else 1.0
    spad = s.shared_layout_factor(wl.layout_tp) if (not mtp or wl.layout_tp == 8) else 1.0
    ops.append(Op("shared_expert", phase, l, "moe", w_tp=s.shared_expert_bytes * spad, rows=rows,
                  flops={"fp8": 2.0 * rows * 3 * D * inter}, n_kernels=0, category="moe_shared"))
    d = distinct_experts(rows, k, n_exp, wl.locality)
    eb = s.mtp_expert_bytes if mtp else s.expert_bytes
    ops.append(Op("routed_experts", phase, l, "moe", w_dyn=d * eb * pad, rows=rows, rows_eff=rows * k / d,
                  flops={"fp4": 2.0 * rows * k * 3 * D * inter}, n_kernels=2, category="moe_routed"))
    ops.append(Op("ar_ffn", phase, l, "collective", coll=rows * D * A, rows=rows, n_kernels=0, category="collective"))
    return ops


def decode_step(s: ModelSpec, wl: DecodeWorkload) -> list[Op]:
    mbs = wl.microbatches if wl.microbatches is not None else (2 if wl.batch >= 2 else 1)
    if mbs == 1:
        ops = _step_ops(s, wl)
        for i, o in enumerate(ops):
            o.deps = [i - 1] if i else []
    else:
        if mbs != 2 or wl.batch % 2:
            raise ValueError("micro-batching supports 2 micro-batches of equal size")
        half = replace(wl, batch=wl.batch // 2, microbatches=1)
        a, b = _step_ops(s, half), _step_ops(s, half)
        assert len(a) == len(b)
        ops = []
        for x, y in zip(a, b):
            ia = len(ops)
            x.mb, y.mb = 0, 1
            x.deps = [ia - 2] if ia else []
            y.deps = [ia - 1] if ia else []
            y.twin = ia
            if y.name == "routed_experts":  # B fetches only the experts A did not
                n_exp = s.dspark_n_routed_experts if y.phase == "draft" else s.n_routed_experts
                k = s.dspark_n_activated_experts if y.phase == "draft" else s.n_activated_experts
                eb = s.mtp_expert_bytes if y.phase == "draft" else s.expert_bytes
                d1 = distinct_experts(y.rows, k, n_exp, wl.locality)
                d2 = distinct_experts(2 * y.rows, k, n_exp, wl.locality)
                pad = s.expert_layout_factor(wl.layout_tp) if (y.phase != "draft" or wl.layout_tp == 8) else 1.0
                y.w_dyn = (d2 - d1) * eb * pad
            elif y.kind != "lookup":  # weights are shared; embedding rows are per token
                y.w_tp = y.w_rep = 0.0
            ops.extend([x, y])
        for i, o in enumerate(ops):
            o.deps = [d for d in o.deps if d >= 0]
    if wl.shard_replicated:
        for o in ops:
            if o.w_rep > 0 and o.out_w > 0:
                o.w_tp += o.w_rep
                o.w_rep = 0.0
                for k, v in o.flops_rep.items():
                    o.flops[k] = o.flops.get(k, 0.0) + v
                o.flops_rep = {}
                o.gather += o.rows * o.out_w * 4.0  # fp32 partials / outputs
    return ops


def _step_ops(s: ModelSpec, wl: DecodeWorkload) -> list[Op]:
    ops: list[Op] = []
    D, V = s.dim, s.vocab_size
    rv, rd, B = wl.rv, wl.rd, wl.batch
    # ---- draft ---------------------------------------------------------------------------
    mp = s.wbytes("mtp.*.main_proj")
    # main_proj runs on a side stream during the previous verify (model.rs prep_proj): no boundary
    ops.append(_gemv("main_proj", "draft", -1, D * len(s.dspark_target_layer_ids), D, B, "fp8", w_tp=mp, kernels=0, cat="draft"))
    for i in range(s.n_mtp_layers):
        for op in _block(s, wl, "draft", s.n_layers + i, rd, mtp=True):
            ops.append(op)
    head_fp8 = s.tensors["head.weight"].bytes_each / 2 + s.tensors["head.weight"].bytes_each / 2 / 1024  # 32x32 E8M0
    head_bf16 = s.tensors["head.weight"].bytes_each
    ops.append(_gemv("draft_head", "draft", -1, D, V, rd, "fp8", w_tp=head_fp8 if wl.draft_head == "fp8" else head_bf16, kernels=1, cat="head"))
    mk = s.tensors["mtp.*.markov_head.head.weight"].bytes_each
    # Markov head: draft_rows sequential rounds (each needs the token sampled in the round
    # before), each a 256 -> vocab projection; weights are read from HBM once and reused.
    ops.append(Op("markov_head", "draft", -1, "gemv", w_tp=mk, rows=B, flops={"bf16": 2.0 * rd * s.dspark_markov_rank * V},
                  n_kernels=1, serial_rounds=wl.draft_rows, category="draft"))
    # ---- verify --------------------------------------------------------------------------
    ops.append(Op("embed", "verify", -1, "lookup", w_tp=rv * D * 2, rows=rv, n_kernels=1, category="misc"))
    for l in range(s.n_layers):
        ops.extend(_block(s, wl, "verify", l, rv, mtp=False))
    ops.append(_gemv("verify_head", "head", -1, D, V, rv, "fp8" if wl.verify_head == "fp8" else "bf16",
                     w_tp=head_fp8 if wl.verify_head == "fp8" else head_bf16, kernels=1, cat="head"))
    ops.append(Op("sample_accept", "head", -1, "vector", rows=rv, flops={"fp32": 8.0 * rv * V}, gather=rv * 64.0,
                  n_kernels=2, category="head"))
    return ops


CATEGORY_LABELS = {
    "attn_proj": "CSA2 attention projections (wq_a, wkv, wq_b, wo_a, wo_b)",
    "attn_core": "sparse attention core (SWA window + top-512 latent KV)",
    "indexer": "indexer score/top-k + KV compressor",
    "router": "MoE gate + top-k routing",
    "moe_shared": "shared expert (FP8)",
    "moe_routed": "routed experts (FP4 x FP8)",
    "mhc": "mHC residual mixing (Sinkhorn)",
    "engram": "Engram n-gram memory",
    "collective": "TP all-reduce / gathers",
    "draft": "DSpark drafter glue (main_proj, Markov head)",
    "head": "LM heads (draft + verify) + sampling",
    "misc": "embedding",
}


def summarize(ops: list[Op], n_chips: int) -> dict:
    """Per-chip bytes and FLOPs for a step on n_chips-way TP."""
    w = sum(o.w_tp for o in ops) / n_chips + sum(o.w_rep for o in ops)
    wd = sum(o.w_dyn for o in ops) / n_chips
    kv = sum(o.kv_rep for o in ops) + sum(o.kv_tp for o in ops) / n_chips
    host = sum(o.host for o in ops) / n_chips
    fl = {p: sum(o.flops.get(p, 0.0) for o in ops) / n_chips + sum(o.flops_rep.get(p, 0.0) for o in ops) for p in PRECISIONS}
    return {"weights_static": w, "weights_routed": wd, "kv": kv, "host": host,
            "hbm_total": w + wd + kv, "flops": fl, "flops_total": sum(fl.values()),
            "n_collectives": sum(1 for o in ops if o.coll > 0),
            "n_kernels": sum(o.n_kernels for o in ops), "n_ops": len(ops)}
