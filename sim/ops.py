"""The op vocabulary every model builder emits and every engine consumes.

An `Op` carries the bytes it must read and the FLOPs it performs, split by how tensor
parallelism distributes them, so one op graph can be costed on GPUs, an HBM-based chip or a
weight-stationary chip. Lifted unchanged from the DeepSeek-V4.1 study's `sim/workload.py`,
where it was defined next to that model's op graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PRECISIONS = ("fp4", "fp8", "bf16", "fp32")


@dataclass
class Op:
    name: str
    phase: str  # draft | verify | head
    layer: int
    kind: str  # gemv | moe | attn | indexer | topk | vector | router | collective | lookup
    w_tp: float = 0.0  # weight bytes, system total, split across chips
    w_rep: float = 0.0  # weight bytes each chip reads in full
    w_dyn: float = 0.0  # data-dependent (routed expert) bytes, system total, split across chips
    kv_rep: float = 0.0  # cache bytes each chip reads in full
    kv_tp: float = 0.0  # cache bytes split across chips (indexer scan by position)
    host: float = 0.0  # bytes from the off-HBM tier (engram tables; HBM on the GPU baseline)
    flops: dict = field(default_factory=dict)  # precision -> system-total FLOPs (split across chips)
    flops_rep: dict = field(default_factory=dict)  # FLOPs every chip performs in full
    coll: float = 0.0  # all-reduce payload bytes per chip (0 = none)
    gather: float = 0.0  # all-gather payload bytes per chip
    rows: int = 1  # activation rows (GEMM M)
    n_kernels: int = 1  # GPU kernels this op maps to in deepseek-ultrafast (approximate)
    serial_rounds: int = 1  # inherently sequential rounds inside the op (Markov head)
    out_w: int = 0  # output width per row (for gathering a sharded replicated op)
    mb: int = 0  # micro-batch id
    deps: list = field(default_factory=list)  # op indices this op waits for (default: previous op)
    twin: int = -1  # op whose weight fetch this op reuses (second micro-batch)
    rows_eff: float = 0.0  # GEMM M actually seen by the matrix engine (routed: rows per expert)
    category: str = ""  # for breakdowns
    fanout: int = 1  # dies an op's payload is split across (weight-stationary engine only)

    def total_flops(self) -> float:
        return sum(self.flops.values()) + sum(self.flops_rep.values())
