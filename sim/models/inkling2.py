"""Inkling 2 Small, and scaled models of the same shape, as an op graph.

The shape comes from the released `config.json`; bytes come from closed-form counts that are
checked, tensor family by tensor family, against the safetensors headers of the released
checkpoint (`data/inkling2_small/checkpoint_tensors.csv`). Because the counts are closed-form,
the same builder also describes hypothetical smaller rungs with Inkling 2's ratios
(`InklingShape.scaled`), which the ladder experiment uses.

One decode step, as the published engines run it (docs/01_workload.md):

  1. Draft (DSpark): a separate dense model drafts 7 tokens: `draft_layers` blocks over
     `draft_rows` positions, then a low-rank recurrent head.
  2. Verify: the 48-layer model over `verify_rows` rows per request. Rows share the dense
     weights and route to experts separately.
  3. Output projection and sampling; `accept_len` tokens are emitted.

Per layer: attention-residual mix -> q/k/v/relative/gate/sink projections (BF16) -> attention
over the layer's KV cache (the last `window` tokens in sliding layers, the whole context in
full layers) -> output projection -> all-reduce -> router (FP32) -> 8 experts (NVFP4 or MXFP8,
activations quantized to match) -> all-reduce.

Layout on a multi-chip tensor-parallel group: attention projections and the output head are
sharded across chips; each expert lives whole on one chip (expert parallelism), because an
NVFP4 expert's activation scales must cover its full input width; the KV cache is sharded by
KV head. With expert parallelism the experts a step touches land unevenly on the chips and
the step waits for the busiest one (`sim.routing.ep_imbalance`).
"""
from __future__ import annotations

import csv
import json
import pathlib
from dataclasses import dataclass, field, replace

from ..ops import Op
from ..routing import distinct_experts_skewed, ep_imbalance
from .base import Workload, chain

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "inkling2_small"

# Stored bits per parameter, including block scales (docs/01_workload.md section 3).
BITS = {"fp4": 4.0 + 8.0 / 16, "fp8": 8.0 + 8.0 / 32, "bf16": 16.0, "fp32": 32.0}
# KV cache bytes per stored value.
KV_BYTES = {"bf16": 2.0, "mxfp8": 1.0 + 1.0 / 32, "nvfp4": 0.5 + 1.0 / 16}

# GPU kernel launches per op, counted in the TokenSpeed source at a439df4 (docs/01_workload.md).
# A layer is 17 launches with NVFP4 experts and 16 with MXFP8 when one request is verified with
# speculation: the ranks' partial sums then travel through a shared-memory mailbox and no
# all-reduce kernel runs. With more than 8 rows, or without speculation, each of the two
# sublayers adds a standalone all-reduce kernel (19 / 18 launches). Whole step: about 977
# launches with speculation and one request, about 931 without speculation.
KERNELS = {
    "qkvr_proj": 3,  # depth mix + norm, joint projection GEMM, relative prologue (convolutions, clips, KV store)
    "attn": 2,  # attention over the paged cache, split reduction with gate and sink
    "o_proj": 1,  # output projection, publishes this rank's partial
    "boundary": 1,  # short convolution that sums the ranks' partials
    "all_reduce": 1,  # standalone all-reduce kernel when the mailbox is not used
    "router": 3,  # depth mix + norm, router GEMM, top-k
    "experts_fp4": 6,  # route/tile/quantize, gate-up GEMM, SwiGLU, quantize, down GEMM, fold
    "experts_fp8": 5,  # SwiGLU and quantize are one launch
    "draft_proj": 3, "draft_attn": 1, "draft_o": 2, "draft_mlp": 5, "draft_head": 1, "markov_round": 3,
    "draft_prep": 19,  # block setup, anchor embedding and the launches not traced one by one
    "embed": 4, "head": 4, "sample": 10, "taps": 8,
}
MAILBOX_ROWS = 8  # rows up to which a speculative verify skips both all-reduces


@dataclass
class DraftShape:
    hidden: int
    layers: int
    inter: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    markov_rank: int
    block: int
    full_layer_ids: tuple[int, ...]  # draft layers that attend the whole context
    window: int

    def layer_params(self) -> int:
        h, q, kv = self.hidden, self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim
        return 3 * h * self.inter + 2 * h * q + 4 * h * kv + 3 * h + 3 * self.head_dim

    def head_params(self, vocab: int) -> int:
        h, r = self.hidden, self.markov_rank
        rnn_in = 2 * h + 2 * r  # hidden state, previous token features
        return vocab * r + 3 * r * rnn_in + 3 * r + h * r + h + h

    def other_params(self, n_targets: int) -> int:
        return (n_targets - 2) * 8 + 2 * self.hidden  # fuse weights, mask embedding, final norm

    def params(self, vocab: int, n_targets: int = 8) -> int:
        return self.layers * self.layer_params() + self.head_params(vocab) + self.other_params(n_targets)


@dataclass
class InklingShape:
    name: str
    hidden: int
    n_layers: int
    n_experts: int
    top_k: int
    inter: int  # expert intermediate width; one expert is hidden -> 2*inter -> hidden
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_rel: int
    rel_extent: int
    window: int
    full_layer_ids: tuple[int, ...]
    vocab: int
    sconv_kernel: int
    max_context: int
    expert_precision: tuple[str, ...]  # per layer: "fp4" (NVFP4) or "fp8" (MXFP8)
    draft: DraftShape | None
    vision_params: int = 0
    hypothetical: bool = False
    overrides: dict = field(default_factory=dict)  # exact byte counts from a checkpoint, when there is one

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_checkpoint(cls, data_dir: pathlib.Path = DATA) -> "InklingShape":
        cfg = json.loads((data_dir / "config.json").read_text())
        quant = json.loads((data_dir / "hf_quant_config.json").read_text())
        t, m = cfg["text_config"], cfg["mtp_config"]
        prec = ["fp4"] * t["num_hidden_layers"]
        for g in quant["model"]["parameter_groups"]:
            if g["compute_dtype"] == "mxfp8":
                for name in g["tensors"]:
                    prec[int(name.split(".layers.")[1].split(".")[0])] = "fp8"
        local = set(t["local_layer_ids"])
        draft = DraftShape(
            hidden=m["hidden_size"], layers=m["num_hidden_layers"], inter=m["intermediate_size"],
            n_heads=m["num_attention_heads"], n_kv_heads=m["num_key_value_heads"], head_dim=m["head_dim"],
            markov_rank=m["markov_rank"], block=m["block_size"],
            full_layer_ids=tuple(i for i in range(m["num_hidden_layers"]) if i not in set(m["local_layer_ids"])),
            window=m["sliding_window_size"])
        shape = cls(
            name="Inkling 2 Small", hidden=t["hidden_size"], n_layers=t["num_hidden_layers"],
            n_experts=t["n_routed_experts"], top_k=t["num_experts_per_tok"], inter=t["intermediate_size"],
            n_heads=t["num_attention_heads"], n_kv_heads=t["num_key_value_heads"], head_dim=t["head_dim"],
            d_rel=t["d_rel"], rel_extent=t["rel_extent"], window=t["sliding_window_size"],
            full_layer_ids=tuple(i for i in range(t["num_hidden_layers"]) if i not in local),
            vocab=t["vocab_size"], sconv_kernel=t["sconv_kernel_size"], max_context=t["model_max_length"],
            expert_precision=tuple(prec), draft=draft)
        man = load_manifest(data_dir / "checkpoint_tensors.csv")
        shape.vision_params = sum(r["elements"] for r in man if r["tensor"].startswith("model.visual."))
        shape.overrides = {"manifest": man}
        return shape

    def scaled(self, name: str, hidden: int, n_layers: int) -> "InklingShape":
        """A hypothetical model with this model's ratios at another width and depth.

        Kept: 384 experts with 8 routed, expert width = hidden/2, query width = 4/3 hidden with
        128-wide heads, 8 KV heads, a full-attention layer every 6th layer, NVFP4 experts in the
        first 5/6 of the layers and MXFP8 in the rest, the vocabulary, and a 5-layer dense draft
        of the same hidden size with a 4x MLP.
        """
        assert hidden % 384 == 0 and n_layers % 6 == 0
        n_fp8 = n_layers // 6
        d = self.draft
        draft = replace(d, hidden=hidden, inter=4 * hidden, n_heads=hidden // 96) if d else None
        return replace(
            self, name=name, hidden=hidden, n_layers=n_layers, inter=hidden // 2, n_heads=hidden // 96,
            full_layer_ids=tuple(range(5, n_layers, 6)),
            expert_precision=tuple(["fp4"] * (n_layers - n_fp8) + ["fp8"] * n_fp8),
            draft=draft, vision_params=0, hypothetical=True, overrides={})

    # ------------------------------------------------------------------ parameter counts
    @property
    def q_width(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.n_kv_heads * self.head_dim

    def is_full(self, layer: int) -> bool:
        return layer in self.full_layer_ids

    def expert_params(self) -> int:
        return 2 * self.inter * self.hidden + self.hidden * self.inter

    def attn_proj_params(self) -> dict:
        h = self.hidden
        return {"q": h * self.q_width, "gate": h * self.q_width, "o": self.q_width * h,
                "k": h * self.kv_width, "v": h * self.kv_width,
                "r": h * self.n_heads * self.d_rel, "sink": h * self.n_heads}

    def layer_small_params(self, layer: int) -> int:
        """Norms, clips, relative-logit table and short convolutions of one layer (BF16)."""
        h, k = self.hidden, self.sconv_kernel
        n = self.q_width + self.n_heads * self.d_rel + self.d_rel * self.rel_extent  # clips, relative table
        n += 2 * h * k + h  # post-attention and post-MLP convolutions, MLP norm
        if self.is_full(layer):
            n += 2 * (k * self.kv_width + self.n_kv_heads * k * h) + 3 * h  # input-dependent K/V taps, 3 norms
        else:
            n += 2 * self.kv_width * k + h
        return n

    def router_params(self) -> tuple[int, int]:
        return self.n_experts * self.hidden, self.n_experts  # BF16 weight, FP32 bias

    def dense_layer_params(self, layer: int) -> int:
        w, b = self.router_params()
        return sum(self.attn_proj_params().values()) + self.layer_small_params(layer) + w + b

    def attn_res_params(self) -> int:
        """Attention-residual queries (FP32): one per sublayer, three for a full layer's attention."""
        h = self.hidden
        n = 0
        for l in range(self.n_layers):
            if l > 0:
                n += 3 * h if self.is_full(l) else h
            n += h
        return n + h  # output query

    def total_params(self, with_draft: bool = False) -> int:
        n = sum(self.dense_layer_params(l) for l in range(self.n_layers))
        n += self.n_layers * self.n_experts * self.expert_params()
        n += 2 * self.vocab * self.hidden + 2 * self.hidden + self.attn_res_params() + self.vision_params
        if with_draft and self.draft:
            n += self.draft.params(self.vocab)
        return n

    def active_params(self) -> int:
        """Decoder layers only: attention, router and the routed experts (the published definition)."""
        return sum(self.dense_layer_params(l) for l in range(self.n_layers)) + self.n_layers * self.top_k * self.expert_params()

    # ------------------------------------------------------------------ bytes
    def expert_bytes(self, layer: int) -> float:
        b = self.expert_params() * BITS[self.expert_precision[layer]] / 8
        if self.expert_precision[layer] == "fp4":
            b += 8 + 8 / self.n_experts  # two FP32 second-level scales per expert, two shared denominators per layer
        return b

    def dense_layer_bytes(self, layer: int) -> float:
        _w, bias = self.router_params()
        return (self.dense_layer_params(layer) - bias) * 2 + bias * 4

    def resident_bytes(self) -> dict:
        ex = {"fp4": 0.0, "fp8": 0.0}
        for l in range(self.n_layers):
            ex[self.expert_precision[l]] += self.n_experts * self.expert_bytes(l)
        out = {
            "experts_nvfp4": ex["fp4"], "experts_mxfp8": ex["fp8"],
            "dense_layers_bf16": sum(self.dense_layer_bytes(l) for l in range(self.n_layers)),
            "embed_bf16": self.vocab * self.hidden * 2.0, "unembed_bf16": self.vocab * self.hidden * 2.0,
            "other": 2 * self.hidden * 2.0 + self.attn_res_params() * 4.0,
            "vision_bf16": self.vision_params * 2.0,
            "draft_bf16": self.draft.params(self.vocab) * 2.0 if self.draft else 0.0,
        }
        out["total"] = sum(out.values())
        return out

    def kv_bytes_per_token(self, wl: Workload) -> dict:
        per_layer = 2 * self.kv_width * KV_BYTES[wl.kv_format]
        d = self.draft
        draft_per_layer = 2 * d.n_kv_heads * d.head_dim * KV_BYTES["bf16"] if d else 0.0
        return {
            "target_full_per_token": len(self.full_layer_ids) * per_layer,
            "target_sliding_per_request": (self.n_layers - len(self.full_layer_ids)) * self.window * per_layer,
            "draft_stored_per_token": d.layers * draft_per_layer if d else 0.0,
            "per_layer": per_layer, "draft_per_layer": draft_per_layer,
        }

    # ------------------------------------------------------------------ the op graph
    def decode_step(self, wl: Workload) -> list[Op]:
        ops: list[Op] = []
        prefill = wl.prefill_tokens > 0
        spec = wl.speculation and self.draft is not None and not prefill
        B = wl.batch
        rv = wl.prefill_tokens * B if prefill else (wl.verify_rows if spec else 1) * B
        H, V, A = self.hidden, self.vocab, wl.act_bytes
        mailbox = spec and rv <= MAILBOX_ROWS
        if spec:
            ops += self._draft_ops(wl)
        ops.append(Op("embed", "verify", -1, "lookup", w_tp=rv * H * 2.0, rows=rv, n_kernels=KERNELS["embed"], category="misc"))
        ops.append(Op("embed_ar", "verify", -1, "collective", coll=rv * H * A, rows=rv, n_kernels=0, category="collective"))
        for l in range(self.n_layers):
            ops += self._layer_ops(wl, l, rv, mailbox)
        if spec:  # hidden states handed to the draft at 7 layer entries and the final norm
            ops.append(Op("taps", "head", -1, "vector", rows=rv, n_kernels=KERNELS["taps"], category="draft"))
        rh = B if prefill else rv  # a prefill chunk needs logits for its last token only
        ops.append(Op("verify_head", "head", -1, "gemv", w_tp=V * H * 2.0, rows=rh, flops={"bf16": 2.0 * rh * H * V},
                      gather=rh * V * 4.0,  # FP32 logits of every row are gathered to every rank
                      n_kernels=KERNELS["head"], category="head"))
        ops.append(Op("sample_accept", "head", -1, "vector", rows=rh, flops={"fp32": 8.0 * rh * V}, gather=rh * 64.0,
                      n_kernels=KERNELS["sample"], category="head"))
        return chain(ops)

    def _layer_ops(self, wl: Workload, l: int, rows: int, mailbox: bool = False) -> list[Op]:
        H, A, B = self.hidden, wl.act_bytes, wl.batch
        k_boundary = KERNELS["boundary"] + (0 if mailbox else KERNELS["all_reduce"])
        ap = self.attn_proj_params()
        full = self.is_full(l)
        ops: list[Op] = []

        # attention-residual mix + norm: FP32 vector work, fused into the preceding boundary kernel
        n_states = l // 6 + 2
        nq = 3 if (full and l > 0) else 1
        ops.append(Op("res_mix_attn", "verify", l, "vector", w_rep=nq * H * 4.0, rows=rows,
                      flops_rep={"fp32": 4.0 * rows * n_states * H * nq}, n_kernels=0, category="res_mix"))
        proj_out = 2 * self.q_width + 2 * self.kv_width + self.n_heads * self.d_rel + self.n_heads
        proj_bytes = (ap["q"] + ap["gate"] + ap["k"] + ap["v"] + ap["r"] + ap["sink"]) * 2.0
        ops.append(Op("qkvr_proj", "verify", l, "gemv", w_tp=proj_bytes, rows=rows,
                      flops={"bf16": 2.0 * rows * H * proj_out}, n_kernels=KERNELS["qkvr_proj"], category="attn_proj"))
        ctx = wl.context
        n_att = ctx if full else min(self.window, ctx)
        kv = B * n_att * 2 * self.kv_width * KV_BYTES[wl.kv_format]
        rel = 2.0 * rows * self.n_heads * self.d_rel * min(n_att, self.rel_extent)
        ops.append(Op("attn", "verify", l, "attn", kv_tp=kv, rows=rows,
                      flops={"bf16": 4.0 * rows * self.n_heads * self.head_dim * n_att + rel,
                             "fp32": 6.0 * rows * self.n_heads * n_att},  # softmax, bias add, sink
                      n_kernels=KERNELS["attn"], category="attn_full" if full else "attn_sliding"))
        ops.append(Op("o_proj", "verify", l, "gemv", w_tp=ap["o"] * 2.0 + self.layer_small_params(l) * 2.0, rows=rows,
                      flops={"bf16": 2.0 * rows * self.q_width * H}, n_kernels=KERNELS["o_proj"], category="attn_proj"))
        ops.append(Op("ar_attn", "verify", l, "collective", coll=rows * H * A, rows=rows,
                      n_kernels=k_boundary, category="collective"))

        ops.append(Op("res_mix_mlp", "verify", l, "vector", w_rep=H * 4.0, rows=rows,
                      flops_rep={"fp32": 4.0 * rows * (n_states + (1 if l % 6 == 0 else 0)) * H}, n_kernels=0, category="res_mix"))
        rw, rb = self.router_params()
        ops.append(Op("router", "verify", l, "router", w_rep=rw * 2.0 + rb * 4.0, rows=rows,
                      flops_rep={"fp32": 2.0 * rows * H * self.n_experts + 8.0 * rows * self.n_experts},
                      n_kernels=KERNELS["router"], category="router", out_w=self.n_experts * 2))
        d = distinct_experts_skewed(rows, self.top_k, self.n_experts, wl.locality, wl.skew)
        prec = self.expert_precision[l]
        ops.append(Op("routed_experts", "verify", l, "moe", w_dyn=d * self.expert_bytes(l), rows=rows,
                      rows_eff=rows * self.top_k / d, flops={prec: 2.0 * rows * self.top_k * self.expert_params()},
                      n_kernels=KERNELS["experts_" + prec], category="moe_nvfp4" if prec == "fp4" else "moe_mxfp8"))
        ops.append(Op("ar_mlp", "verify", l, "collective", coll=rows * H * A, rows=rows,
                      n_kernels=k_boundary, category="collective"))
        return ops

    def _draft_ops(self, wl: Workload) -> list[Op]:
        d, B, V, A = self.draft, wl.batch, self.vocab, wl.act_bytes
        rd = wl.draft_rows * B
        h, q, kvw = d.hidden, d.n_heads * d.head_dim, d.n_kv_heads * d.head_dim
        ops: list[Op] = [Op("draft_prep", "draft", -1, "vector", rows=rd, n_kernels=KERNELS["draft_prep"], category="draft")]
        for i in range(d.layers):
            lay = self.n_layers + i
            ops.append(Op("draft_proj", "draft", lay, "gemv", w_tp=(h * q + 4 * h * kvw) * 2.0, rows=rd,
                          flops={"bf16": 2.0 * rd * h * (q + 2 * kvw) + 2.0 * wl.tokens_per_step * B * h * 2 * kvw},
                          n_kernels=KERNELS["draft_proj"], category="draft"))
            n_att = wl.context if i in d.full_layer_ids else min(d.window, wl.context)
            ops.append(Op("draft_attn", "draft", lay, "attn", kv_tp=B * n_att * 2 * kvw * KV_BYTES["bf16"], rows=rd,
                          flops={"bf16": 4.0 * rd * d.n_heads * d.head_dim * (n_att + wl.draft_rows),
                                 "fp32": 6.0 * rd * d.n_heads * (n_att + wl.draft_rows)},
                          n_kernels=KERNELS["draft_attn"], category="draft_attn"))
            ops.append(Op("draft_o", "draft", lay, "gemv", w_tp=q * h * 2.0, rows=rd, flops={"bf16": 2.0 * rd * q * h},
                          n_kernels=KERNELS["draft_o"], category="draft"))
            ops.append(Op("draft_ar_attn", "draft", lay, "collective", coll=rd * h * A, rows=rd, n_kernels=0, category="collective"))
            ops.append(Op("draft_mlp", "draft", lay, "gemv", w_tp=3 * h * d.inter * 2.0, rows=rd,
                          flops={"bf16": 2.0 * rd * 3 * h * d.inter}, n_kernels=KERNELS["draft_mlp"], category="draft"))
            ops.append(Op("draft_ar_mlp", "draft", lay, "collective", coll=rd * h * A, rows=rd, n_kernels=0, category="collective"))
        # base logits of the proposals: a second read of the output head
        rp = max(1, wl.draft_rows - 1) * B
        ops.append(Op("draft_head", "draft", -1, "gemv", w_tp=V * self.hidden * 2.0, rows=rp,
                      flops={"bf16": 2.0 * rp * self.hidden * V}, n_kernels=KERNELS["draft_head"], category="head"))
        # low-rank recurrent head: one round per proposal, each needing the token chosen before it,
        # and each ending in a cross-rank argmax
        rounds = max(1, wl.draft_rows - 1)
        ops.append(Op("markov_head", "draft", -1, "gemv", w_tp=d.head_params(V) * 2.0, rows=B,
                      flops={"bf16": 2.0 * rd * d.markov_rank * V}, n_kernels=KERNELS["markov_round"] * rounds,
                      serial_rounds=rounds, category="draft"))
        for _ in range(rounds):
            ops.append(Op("draft_argmax", "draft", -1, "collective", coll=B * 16.0, rows=B, n_kernels=1, category="collective"))
        return ops

    # ------------------------------------------------------------------ summaries
    def step_bytes(self, wl: Workload, n_chips: int = 1) -> dict:
        """System-total bytes one step reads, by kind (no platform involved)."""
        ops = self.decode_step(wl)
        out: dict[str, float] = {}
        for o in ops:
            key = {"moe": "experts", "attn": "kv"}.get(o.kind, "dense")
            if o.phase == "draft" and o.kind != "attn":
                key = "draft"
            out[key] = out.get(key, 0.0) + o.w_tp + o.w_rep * n_chips + o.w_dyn + o.kv_tp + o.kv_rep * n_chips
        out["total"] = sum(out.values())
        return out


def load_manifest(path: pathlib.Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["elements"] = int(r["elements"])
            r["bytes"] = int(r["bytes"])
            rows.append(r)
    return rows


def apply_expert_parallel(ops: list[Op], shape: InklingShape, n_chips: int) -> list[Op]:
    """Scale routed-expert bytes for the busiest chip under expert parallelism (see module docstring).

    The engine simulates one chip and divides `w_dyn` by the group size; multiplying by the
    imbalance makes that chip the busiest one.
    """
    if n_chips <= 1:
        return ops
    for o in ops:
        if o.name == "routed_experts":
            d = o.w_dyn / shape.expert_bytes(o.layer)
            o.w_dyn *= ep_imbalance(d, shape.n_experts, n_chips)
    return ops


# The ladder: hypothetical rungs with Inkling 2 Small's ratios (hidden, layers).
LADDER = {
    "rung_100b": ("Inkling-2-shaped, ~100B (hypothetical)", 2304, 30),
    "rung_30b": ("Inkling-2-shaped, ~30B (hypothetical)", 1536, 24),
    "rung_8b": ("Inkling-2-shaped, ~8B (hypothetical)", 768, 24),
}


def ladder(base: InklingShape | None = None) -> dict[str, InklingShape]:
    base = base or InklingShape.from_checkpoint()
    out = {"inkling2_small": base}
    for key, (name, hidden, layers) in LADDER.items():
        out[key] = base.scaled(name, hidden, layers)
    return out
