"""Plain dense decoder-only transformers (Llama 3.1 8B) as an op graph.

Used for calibration: published per-user speeds for one dense model exist on four different
memory architectures, so the same engines can be checked against all four before they are
trusted on Inkling 2 Small. No experts, no draft: one row per request per step.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..ops import Op
from .base import Workload, chain

KERNELS = {"qkv_proj": 1, "attn": 1, "o_proj": 1, "mlp": 2, "boundary": 1, "embed": 1, "head": 1, "sample": 2}


@dataclass
class DenseShape:
    name: str
    hidden: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    inter: int
    vocab: int
    weight_bits: float = 16.0  # stored bits per parameter on the platform being modeled
    precision: str = "bf16"  # which compute precision the weights use: fp4 | fp8 | bf16
    kv_bytes_per_value: float = 2.0
    tied_embeddings: bool = False

    @property
    def q_width(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.n_kv_heads * self.head_dim

    def layer_params(self) -> int:
        h = self.hidden
        return 2 * h * self.q_width + 2 * h * self.kv_width + 3 * h * self.inter + 2 * h

    def total_params(self) -> int:
        emb = self.vocab * self.hidden * (1 if self.tied_embeddings else 2)
        return self.n_layers * self.layer_params() + emb + self.hidden

    def resident_bytes(self) -> dict:
        b = self.weight_bits / 8
        out = {"layers": self.n_layers * self.layer_params() * b, "embed": self.vocab * self.hidden * b,
               "unembed": 0.0 if self.tied_embeddings else self.vocab * self.hidden * b}
        out["total"] = sum(out.values())
        return out

    def kv_bytes_per_token(self, wl: Workload) -> dict:
        per_layer = 2 * self.kv_width * self.kv_bytes_per_value
        return {"target_full_per_token": self.n_layers * per_layer, "target_sliding_per_request": 0.0,
                "draft_stored_per_token": 0.0, "per_layer": per_layer, "draft_per_layer": 0.0}

    def decode_step(self, wl: Workload) -> list[Op]:
        rows, B, H, V, A = wl.batch, wl.batch, self.hidden, self.vocab, wl.act_bytes
        b, p = self.weight_bits / 8, self.precision
        ops = [Op("embed", "verify", -1, "lookup", w_tp=rows * H * b, rows=rows, n_kernels=KERNELS["embed"], category="misc")]
        for l in range(self.n_layers):
            ops.append(Op("qkv_proj", "verify", l, "gemv", w_tp=(H * self.q_width + 2 * H * self.kv_width) * b, rows=rows,
                          flops={p: 2.0 * rows * H * (self.q_width + 2 * self.kv_width)}, n_kernels=KERNELS["qkv_proj"], category="attn_proj"))
            ops.append(Op("attn", "verify", l, "attn", kv_tp=B * wl.context * 2 * self.kv_width * self.kv_bytes_per_value, rows=rows,
                          flops={"bf16": 4.0 * rows * self.n_heads * self.head_dim * wl.context},
                          n_kernels=KERNELS["attn"], category="attn_full"))
            ops.append(Op("o_proj", "verify", l, "gemv", w_tp=self.q_width * H * b, rows=rows,
                          flops={p: 2.0 * rows * self.q_width * H}, n_kernels=KERNELS["o_proj"], category="attn_proj"))
            ops.append(Op("ar_attn", "verify", l, "collective", coll=rows * H * A, rows=rows, n_kernels=KERNELS["boundary"], category="collective"))
            ops.append(Op("mlp_up", "verify", l, "gemv", w_tp=2 * H * self.inter * b, rows=rows,
                          flops={p: 2.0 * rows * 2 * H * self.inter}, n_kernels=KERNELS["mlp"] - 1, category="mlp"))
            ops.append(Op("mlp_down", "verify", l, "gemv", w_tp=H * self.inter * b, rows=rows,
                          flops={p: 2.0 * rows * H * self.inter}, n_kernels=1, category="mlp"))
            ops.append(Op("ar_mlp", "verify", l, "collective", coll=rows * H * A, rows=rows, n_kernels=KERNELS["boundary"], category="collective"))
        ops.append(Op("verify_head", "head", -1, "gemv", w_tp=V * H * b, rows=rows, flops={p: 2.0 * rows * H * V},
                      n_kernels=KERNELS["head"], category="head"))
        ops.append(Op("sample_accept", "head", -1, "vector", rows=rows, flops={"fp32": 8.0 * rows * V}, gather=rows * 64.0,
                      n_kernels=KERNELS["sample"], category="head"))
        return chain(ops)


def llama31_8b(weight_bits: float = 16.0, precision: str = "bf16") -> DenseShape:
    """Llama 3.1 8B: 32 layers, hidden 4096, 32 heads / 8 KV heads of 128, MLP 14336, vocab 128256."""
    return DenseShape("Llama 3.1 8B", hidden=4096, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128,
                      inter=14336, vocab=128256, weight_bits=weight_bits, precision=precision)
