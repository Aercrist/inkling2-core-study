"""Prefill (TTFT) model: chunked roofline over the Causal Encoder-Decoder prefill path.

With CED (tech report 2.2, 3.2.2) prefill runs the 20 encoder layers over every new token,
projects the decoder's global KV from the encoder output (layer-20 compressor), and replays
only the last `window_size` prompt tokens through the 20 decoder layers to rebuild decoder
SWA KV. Per op and chunk, time = max(FLOPs / (peak * mfu), bytes / (bw * eff)) plus kernel
overhead; collectives are costed per layer.
"""
from __future__ import annotations

from dataclasses import dataclass

from ...hardware import Hardware
from .modelspec import ModelSpec
from ...routing import distinct_experts


@dataclass
class PrefillResult:
    time: float
    flops: float
    hbm_bytes: float
    parts: dict


def _layer(s: ModelSpec, hw: Hardware, l: int, tokens: int, t_mid: float, mfu: float, decoder_replay: bool) -> tuple[float, float, float, dict]:
    n = hw.n_chips
    D = s.dim
    ab = s.attn_bytes()
    w = sum(ab.values()) / n + s.shared_expert_bytes / n
    w += 2 * s.tensors["layers.*.hc_attn_fn"].bytes_each + s.tensors["layers.*.ffn.gate.weight"].bytes_each
    dexp = distinct_experts(tokens, s.n_activated_experts, s.n_routed_experts)
    w_exp = dexp * s.expert_bytes / n
    attn_params = sum(s.params(f"layers.*.attn.{k}") for k in ("wq_a", "wkv", "wq_b", "wo_a", "wo_b"))
    f_fp8 = 2.0 * tokens * (attn_params + 3 * D * s.moe_inter_dim) / n
    f_fp4 = 2.0 * tokens * s.n_activated_experts * 3 * D * s.moe_inter_dim / n
    role = s.layer_role(l)
    ctx = int(t_mid)
    n_att = s.attended_entries(l, ctx)
    f_bf16 = 4.0 * tokens * s.n_heads * s.head_dim * n_att / n
    topk = 0.0
    kv = 0.0
    if role in ("full", "reindex"):
        scan = s.index_scan_entries(l, ctx)
        f_fp4 += 2.0 * tokens * s.index_n_heads * s.index_head_dim * scan / n
        topk = tokens * scan / n
        kv += scan * s.kv_entry_bytes["index"] / n
    if l in s.engram_layer_ids:
        w += s.wbytes("layers.*.engram.wkv") / n
        f_fp8 += 2.0 * tokens * (3 * s.engram_n_heads * s.engram_head_dim) * D * (s.hc_mult + 1) / n
    t_w = (w + w_exp + kv) / (hw.hbm_bw * hw.hbm_eff)
    t_c = f_fp8 / (hw.peak["fp8"] * mfu) + f_fp4 / (hw.peak["fp4"] * mfu) + f_bf16 / (hw.peak["bf16"] * mfu)
    t_k = topk / hw.topk_rate
    t_coll = 2 * hw.coll_time(tokens * D * 2)
    kernels = 13 + (6 if role in ("full", "reindex") else 0)
    ovh = kernels * hw.kernel_overhead if not hw.prefetch else 15 * hw.op_overhead
    t = max(t_w, t_c) + t_k + t_coll + ovh
    return t, (f_fp8 + f_fp4 + f_bf16) * n, (w + w_exp + kv) * n, {"weights": t_w, "compute": t_c, "topk": t_k, "coll": t_coll, "ovh": ovh}


def prefill_time(s: ModelSpec, hw: Hardware, new_tokens: int, prefix: int = 0, chunk: int = 8192, mfu: float = 0.5) -> PrefillResult:
    L2 = s.n_layers // 2
    tot = flops = byts = 0.0
    parts = {"weights": 0.0, "compute": 0.0, "topk": 0.0, "coll": 0.0, "ovh": 0.0}
    pos = prefix
    left = new_tokens
    while left > 0:
        c = min(chunk, left)
        t_mid = pos + c / 2
        for l in range(L2):
            t, f, b, p = _layer(s, hw, l, c, t_mid, mfu, False)
            tot += t; flops += f; byts += b
            for k in parts:
                parts[k] += p[k]
        # decoder global KV: layer-20 compressor over the chunk's encoder outputs
        f = 2.0 * c * s.dim * s.head_dim / hw.n_chips
        tot += f / (hw.peak["bf16"] * mfu)
        flops += f * hw.n_chips
        pos += c
        left -= c
    # decoder SWA bounded replay: last window_size tokens through the decoder layers
    rep = min(s.window_size, prefix + new_tokens)
    for l in range(L2, s.n_layers):
        t, f, b, p = _layer(s, hw, l, rep, prefix + new_tokens, mfu, True)
        tot += t; flops += f; byts += b
        for k in parts:
            parts[k] += p[k]
    return PrefillResult(tot, flops, byts, parts)
