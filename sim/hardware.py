"""Hardware descriptions, loaded from configs/hw/*.yaml.

All rates are per chip. `n_chips` is the tensor-parallel width one sequence is spread over.
Every constant in the YAML carries a `# src:` comment pointing at docs/sources.md.
"""
from __future__ import annotations

import copy
import pathlib
from dataclasses import dataclass, field, fields

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass
class Hardware:
    name: str
    kind: str  # gpu | asic | fpga | sram
    n_chips: int = 4
    # memory
    hbm_bw: float = 8.0e12  # B/s peak
    hbm_eff: float = 0.9  # achieved fraction of peak on long streams
    hbm_cap: float = 288e9
    hbm_latency: float = 0.6e-6  # first-byte latency of a dependent (non-prefetched) fetch
    host_bw: float = 50e9  # off-HBM tier (LPDDR/host) bandwidth, B/s
    host_latency: float = 2e-6
    engram_in_hbm: bool = True
    # compute (dense FLOP/s)
    peak: dict = field(default_factory=lambda: {"fp4": 10e15, "fp8": 5e15, "bf16": 2.5e15, "fp32": 0.08e15})
    mm_eff: float = 0.7  # achievable matmul efficiency when compute-bound
    attn_eff: float = 0.0  # efficiency of attention kernels; 0 = same as mm_eff (the DeepSeek study's behaviour)
    array_rows: int = 0  # rows (GEMM M) the matrix engine processes per pass; 0 = ignore M padding
    topk_rate: float = 2e12  # elements/s through top-k selection
    # on-chip memory
    sram_bytes: float = 0.0
    sram_bw: float = 0.0
    stage_bytes: float = 0.0  # weight staging / prefetch window
    expert_cache_bytes: float = 0.0
    # interconnect
    link_bw: float = 900e9  # B/s per chip per direction
    coll_latency: float = 5e-6  # fixed latency of one small all-reduce across n_chips
    coll_latency_per_peer: float = 0.0  # extra latency per additional peer
    # execution model
    prefetch: bool = False  # cross-op weight prefetch (dataflow) vs per-kernel fetch (GPU)
    kernel_overhead: float = 0.0  # exposed s per kernel boundary (GPU)
    row_overhead: float = 0.0  # extra s per activation row in attention and MoE ops (fitted on batch data; 0 in the DeepSeek study)
    op_overhead: float = 0.0  # pipeline fill/drain per op (ASIC)
    router_latency: float = 0.0  # gate scores -> expert fetch issue
    serial_round_latency: float = 1e-6  # per sequential round (Markov head)
    step_overhead: float = 0.0  # per-step fixed cost (graph launch, host sync)
    # power / cost (used by energy.py / cost.py)
    tdp_w: float = 1000.0
    power: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    area: dict = field(default_factory=dict)
    notes: str = ""

    def coll_time(self, nbytes: float) -> float:
        n = self.n_chips
        if n <= 1 or nbytes <= 0:
            return 0.0
        lat = self.coll_latency + self.coll_latency_per_peer * (n - 2)
        # two-shot all-reduce: reduce-scatter + all-gather, each moves (n-1)/n of the payload
        return lat + 2.0 * nbytes * (n - 1) / n / self.link_bw

    def replace(self, **kw) -> "Hardware":
        h = copy.deepcopy(self)
        for k, v in kw.items():
            if not hasattr(h, k):
                raise KeyError(f"unknown hardware field {k}")
            setattr(h, k, v)
        return h


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _numeric(x):
    """PyYAML (YAML 1.1) reads `8.0e12` as a string; coerce such scalars to float."""
    if isinstance(x, dict):
        return {k: _numeric(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_numeric(v) for v in x]
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return x
    return x


def load_yaml(path: str | pathlib.Path) -> dict:
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = ROOT / p
    d = _numeric(yaml.safe_load(p.read_text()) or {})
    if "extends" in d:
        base = load_yaml(p.parent / d.pop("extends"))
        d = _deep_merge(base, d)
    return d


def load_hw(path: str | pathlib.Path, **overrides) -> Hardware:
    d = load_yaml(path)
    d = _deep_merge(d, overrides)
    known = {f.name for f in fields(Hardware)}
    unknown = set(d) - known
    if unknown:
        raise KeyError(f"{path}: unknown keys {sorted(unknown)}")
    return Hardware(**d)
