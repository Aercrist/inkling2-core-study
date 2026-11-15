"""Op-level, cycle-approximate scheduler for one decode step on one chip of a TP group.

All chips of the TP group run the same program on their shard, so one chip is simulated and
collectives are costed with `Hardware.coll_time`. Resources:

  * compute: issues ops in program order; an op occupies it for its compute time only, so a
    later independent op (other micro-batch) can run while an earlier op waits for data;
  * HBM: a bandwidth server that moves weight/KV bytes in chunks. Which chunk goes next is
    chosen work-conservingly by program order among the chunks that are *ready*;
  * links: collectives (latency + bytes / bandwidth), on the issuing chain's critical path.

An op starts when its dependencies have ended and the compute front has issued every earlier
op; it ends when its compute is done and all its bytes (and its twin's shared weights) have
arrived. The two execution policies differ only in when a chunk becomes ready:

  * `prefetch=False` (GPU kernels): an op's bytes are fetched by its own kernel, so they are
    ready when the op starts, and every op pays `kernel_overhead` per kernel boundary.
  * `prefetch=True` (statically scheduled dataflow): weights whose addresses are known ahead
    of time (everything except routed experts and top-k-selected KV) stream in as early as the
    staging buffer (`stage_bytes`) allows; routed-expert fetches are issued `router_latency`
    after their gate op ends; each op pays `op_overhead` of pipeline fill/drain.

For a single sequence (batch 1) the ops form a chain and this reduces to
end_i = max(end_{i-1} + overhead_i + t_comp_i, fetch_done_i) + t_coll_i. Times in seconds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .hardware import Hardware
from .ops import Op

CHUNK = 4 << 20  # HBM scheduling granularity (bytes)
LOOKAHEAD = 256  # chunks the prefetcher considers at once
INF = float("inf")


@dataclass
class OpTiming:
    op: Op
    start: float
    end: float
    fetch_done: float
    t_comp: float
    t_coll: float
    t_ovh: float
    hbm_bytes: float


@dataclass
class StepResult:
    step_time: float
    timings: list[OpTiming]
    hbm_busy: float
    hbm_bytes: float
    breakdown: dict = field(default_factory=dict)  # critical-path time by cause
    by_category: dict = field(default_factory=dict)  # critical-path time by op category
    segments: dict = field(default_factory=dict)  # op index -> critical-path time attributed to it


def _op_costs(op: Op, hw: Hardware, hit_rate: float):
    n = hw.n_chips
    static = op.w_tp / n + op.w_rep + op.kv_tp / n
    if hw.engram_in_hbm:
        static += op.host / n
    dyn = op.w_dyn / n * (1.0 - hit_rate) + op.kv_rep
    sram_t = (op.w_dyn / n * hit_rate) / hw.sram_bw if (hit_rate > 0 and hw.sram_bw > 0) else 0.0
    if op.kind == "topk":
        t_comp = op.flops.get("fp32", 0.0) / 4.0 / n / hw.topk_rate
    else:
        t_comp = 0.0
        for p in set(op.flops) | set(op.flops_rep):
            f = op.flops.get(p, 0.0) / n + op.flops_rep.get(p, 0.0)
            if f > 0:
                eff = 1.0 if op.kind in ("vector", "router") else hw.mm_eff
                if op.kind == "attn" and hw.attn_eff > 0:
                    eff = hw.attn_eff
                if hw.array_rows and op.kind in ("gemv", "moe", "attn", "indexer"):
                    m = op.rows_eff or op.rows  # routed experts: rows per expert
                    eff *= m / (math.ceil(m / hw.array_rows) * hw.array_rows) if m >= 1 else m / hw.array_rows
                t_comp += f / (hw.peak[p] * eff)
    t_coll = hw.coll_time(op.coll)
    if op.gather > 0 and n > 1:
        t_coll += hw.coll_latency + op.gather * (n - 1) / hw.link_bw
    if not hw.engram_in_hbm and op.host > 0 and not hw.prefetch:
        t_coll += hw.host_latency + op.host / n / hw.host_bw
    rounds = (op.serial_rounds - 1) * hw.serial_round_latency
    if hw.prefetch:
        ovh = (hw.op_overhead if op.kind != "collective" else 0.0) + rounds
    else:
        ovh = hw.kernel_overhead * op.n_kernels + rounds
    if hw.row_overhead and op.kind in ("moe", "attn"):
        ovh += hw.row_overhead * op.rows
    return static, dyn, sram_t, t_comp, t_coll, ovh


def simulate_step(ops: list[Op], hw: Hardware, hit_rate: float = 0.0, chunk: float | None = None) -> StepResult:
    n_ops = len(ops)
    CHUNK = chunk or globals()["CHUNK"]  # coarser scheduling granularity for very large steps
    costs = [_op_costs(o, hw, hit_rate) for o in ops]
    bw = hw.hbm_bw * hw.hbm_eff
    deps = [o.deps if (o.deps or i == 0) else [i - 1] for i, o in enumerate(ops)]

    # routed experts wait for the most recent gate of the same micro-batch
    src = [-1] * n_ops
    last_gate: dict[int, int] = {}
    for i, o in enumerate(ops):
        if o.kind == "router":
            last_gate[o.mb] = i
        if o.name == "routed_experts":
            src[i] = last_gate.get(o.mb, -1)

    chunks: list[tuple[int, bool, float, bool]] = []  # (op, dynamic, bytes, first of request)
    remaining = [0] * n_ops
    for i, (st, dy, *_rest) in enumerate(costs):
        for dyn, b in ((False, st), (True, dy)):
            first = True
            while b > 1e-9:
                c = min(b, CHUNK)
                chunks.append((i, dyn, c, first))
                remaining[i] += 1
                b -= c
                first = False

    starts: list[float | None] = [None] * n_ops
    ends: list[float | None] = [None] * n_ops
    fetch_done = [0.0] * n_ops
    hbm_bytes_op = [0.0] * n_ops
    compute_free = 0.0
    i_start = 0
    open_ops: list[int] = []  # started, not ended
    issued_static: dict[int, float] = {}  # op -> static bytes issued, not yet consumed

    def propagate() -> None:
        nonlocal i_start, compute_free
        changed = True
        while changed:
            changed = False
            # start ops in program order
            while i_start < n_ops:
                i = i_start
                dep_end = 0.0
                ok = True
                for d in deps[i]:
                    if ends[d] is None:
                        ok = False
                        break
                    dep_end = max(dep_end, ends[d])
                if not ok:
                    break
                _s, _d, sram_t, t_comp, _c, ovh = costs[i]
                starts[i] = max(compute_free, dep_end) + ovh
                if ops[i].kind != "collective":
                    compute_free = starts[i] + t_comp + sram_t
                open_ops.append(i)
                i_start += 1
                changed = True
            # end ops whose data has arrived
            for i in list(open_ops):
                tw = ops[i].twin
                if remaining[i] or (tw >= 0 and (remaining[tw] or ends[tw] is None and starts[tw] is None)):
                    continue
                static, dyn, sram_t, t_comp, t_coll, _ = costs[i]
                st = starts[i]
                fd = max(fetch_done[i], st) if (static + dyn) > 0 else st
                if tw >= 0:
                    fd = max(fd, fetch_done[tw])
                fetch_done[i] = fd
                ends[i] = max(st + t_comp + sram_t, fd + sram_t) + t_coll
                open_ops.remove(i)
                changed = True

    pending = list(range(len(chunks)))
    t_h = 0.0
    busy = 0.0
    propagate()
    while pending:
        best_pos, best_t = None, INF
        for pos in range(min(LOOKAHEAD, len(pending))):
            i, dyn, b, first = chunks[pending[pos]]
            if not hw.prefetch or (dyn and ops[i].name != "routed_experts"):
                rt = INF if starts[i] is None else starts[i]
            elif dyn:
                e = ends[src[i]] if src[i] >= 0 else 0.0
                rt = INF if e is None else e + hw.router_latency
            elif hw.stage_bytes <= 0:  # no staging SRAM: weights arrive with their op
                rt = INF if starts[i] is None else starts[i]
            elif pos == 0:
                rt = 0.0
            else:
                rt = _buffer_ready(issued_static, ends, hw.stage_bytes, b, t_h)
            t = max(t_h, rt)
            if t < best_t:
                best_pos, best_t = pos, t
                if t <= t_h:
                    break
        assert best_pos is not None and best_t < INF, "scheduler deadlock"
        i, dyn, b, first = chunks[pending.pop(best_pos)]
        lat = hw.hbm_latency if (dyn and first and hw.prefetch) else 0.0
        dur = b / bw
        t_h = best_t + lat + dur
        busy += dur
        if not dyn:
            issued_static[i] = issued_static.get(i, 0.0) + b
        hbm_bytes_op[i] += b
        fetch_done[i] = max(fetch_done[i], t_h)
        remaining[i] -= 1
        if remaining[i] == 0:
            propagate()
            for k in [k for k, _ in issued_static.items() if ends[k] is not None and ends[k] <= t_h]:
                del issued_static[k]
    propagate()
    assert all(e is not None for e in ends), "unfinished ops"

    step = max(ends) + hw.step_overhead
    timings = [OpTiming(o, starts[i], ends[i], fetch_done[i], costs[i][3], costs[i][4], costs[i][5], hbm_bytes_op[i])
               for i, o in enumerate(ops)]
    breakdown, by_cat, segs = _critical_path(timings, costs, deps, hw.step_overhead)
    return StepResult(step, timings, busy / step if step else 0.0, sum(hbm_bytes_op), breakdown, by_cat, segs)


def _buffer_ready(issued: dict[int, float], ends, cap: float, nbytes: float, t0: float) -> float:
    """Earliest t >= t0 at which `nbytes` more static bytes fit in the staging buffer."""
    held = 0.0
    releases = []
    for k, b in issued.items():
        e = ends[k]
        if e is None:
            held += b
        elif e > t0:
            held += b
            releases.append((e, b))
    if held + nbytes <= cap:
        return t0
    releases.sort()
    for e, b in releases:
        held -= b
        if held + nbytes <= cap:
            return e
    return INF


def _critical_path(timings: list[OpTiming], costs, deps, step_ovh: float):
    """Walk back from the last op along whatever bounded each op's start (a dependency's end,
    or the compute-issue front) and attribute every segment of the step to one cause."""
    breakdown = {"overhead": step_ovh, "collective": 0.0, "memory": 0.0, "compute": 0.0}
    by_cat: dict[str, float] = {}
    segs: dict[int, float] = {}
    i = max(range(len(timings)), key=lambda k: timings[k].end)
    T = timings[i].end
    while i is not None:
        tm = timings[i]
        _s, _d, sram_t, t_comp, t_coll, ovh = costs[i]
        prev, prev_T = None, 0.0
        for d in deps[i]:
            if timings[d].end >= prev_T:
                prev, prev_T = d, timings[d].end
        if i > 0 and tm.op.kind != "collective":
            j = i - 1
            while j >= 0 and timings[j].op.kind == "collective":
                j -= 1
            if j >= 0:
                issue = timings[j].start + costs[j][3] + costs[j][2]
                if issue > prev_T + 1e-15:
                    prev, prev_T = j, issue
        seg = max(0.0, T - prev_T)
        o = min(ovh, seg)
        c = min(t_coll, seg - o)
        rest = seg - o - c
        comp_end = tm.start + t_comp + sram_t
        mem = min(rest, max(0.0, min(T, tm.fetch_done) - comp_end)) if tm.fetch_done > comp_end else 0.0
        breakdown["overhead"] += o
        breakdown["collective"] += c
        breakdown["memory"] += mem
        breakdown["compute"] += rest - mem
        by_cat[tm.op.category] = by_cat.get(tm.op.category, 0.0) + seg
        segs[i] = segs.get(i, 0.0) + seg
        T = prev_T
        i = prev
    return breakdown, by_cat, segs
