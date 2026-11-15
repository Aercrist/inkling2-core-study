"""What every model builder provides.

A builder turns a model description (config + checkpoint manifest, or a scaled shape) into the
op graph of one decode step. The engines (`sim/engine.py` for HBM-based platforms,
`sim/stationary.py` for weight-stationary ones) consume that graph and nothing else, so every
platform is costed on the same workload.

The DeepSeek-V4.1 study had one hard-wired model (`sim/workload.py`). Here each model is a
builder: `sim/models/dsv41` (that model, unchanged), `sim/models/inkling2` (Inkling 2 Small and
scaled rungs of the same shape) and `sim/models/dense` (plain dense transformers such as
Llama 3.1 8B, used for calibration).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..ops import Op


@dataclass
class Workload:
    """One decode step's serving conditions. Fields a model does not use are ignored."""

    batch: int = 1  # concurrent requests stepping together
    context: int = 8192  # tokens of context per request
    speculation: bool = True  # run the model's draft and verify several rows per step
    verify_rows: int = 8  # rows verified per request per step when speculating
    draft_rows: int = 8  # positions the draft processes per request per step
    accept_len: float = 3.5  # tokens emitted per step when speculating
    locality: float = 0.0  # expert-routing locality across a step's rows, 0 = independent
    skew: float = 0.0  # Zipf exponent of expert popularity, 0 = uniform
    kv_format: str = "mxfp8"  # target model's KV cache: bf16 | mxfp8 | nvfp4
    act_bytes: int = 2  # activation bytes on the wire (BF16)
    expert_parallel: bool = True  # experts live whole on one chip (EP) instead of being sharded (TP)
    prefill_tokens: int = 0  # > 0: one prefill chunk of this many new tokens per request (no draft, no sampling)

    @property
    def rows(self) -> int:
        if self.prefill_tokens:
            return self.prefill_tokens * self.batch
        return (self.verify_rows if self.speculation else 1) * self.batch

    @property
    def tokens_per_step(self) -> float:
        """Tokens emitted per request per step (for a prefill chunk: tokens ingested)."""
        if self.prefill_tokens:
            return float(self.prefill_tokens)
        return self.accept_len if self.speculation else 1.0


class ModelBuilder(Protocol):
    name: str

    def decode_step(self, wl: Workload) -> list[Op]:
        """Ops of one decode step, in program order, with `deps` set."""

    def resident_bytes(self) -> dict:
        """Weight bytes that must be resident, by block and precision."""

    def kv_bytes_per_token(self, wl: Workload) -> dict:
        """KV-cache bytes per token of context, split into what grows with context and what does not."""


def chain(ops: list[Op]) -> list[Op]:
    """Make the ops a simple chain: each waits for the one before it."""
    for i, o in enumerate(ops):
        o.deps = [i - 1] if i else []
    return ops
