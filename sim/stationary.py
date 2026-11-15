"""Engine for weight-stationary designs: every weight is resident on-die, tokens move.

It consumes the same op graph as `sim/engine.py`. The difference is the memory model. On an
HBM-based platform a step's bytes share one bandwidth server per chip. Here every weight
family sits in its own memory banks next to its own multiply-accumulate lanes, so reading an
op's weights does not compete with reading another op's.

Timing of one op (all assumptions are fields of `Stationary`, set in configs/hw/*.yaml):

  * A weight op (dense GEMV, expert GEMM, output head) scans its banks once per pass. A bank
    is `scan_rows` wordlines deep and reads one full wordline per clock, so a pass takes
    `scan_rows / f_clk` regardless of the tensor's size; a bigger tensor is more banks in
    parallel. Each bank column feeds `lane_rows` multipliers, so one pass serves that many
    activation rows; more rows need more passes.
  * Routed experts are two GEMMs in sequence (gate/up, then down), plus the hop that carries
    the quantized activations to the expert dies and the hop that brings the results back.
    A layer's experts and its dense weights share one multi-die tile, so these two hops use
    short die-to-die links and the payload is split across the tile's expert dies.
  * Attention reads the layer's KV cache from KV memory (on-die for the sliding window, HBM
    or LPDDR for the full-attention history) and runs on a dedicated attention engine; it
    takes the longer of the fetch and the arithmetic.
  * A tensor-parallel all-reduce does not exist here. The attention-to-MLP boundary stays on
    the layer's hub die; at the layer boundary the hidden state hops to the next layer's tile
    over a longer link.

Throughput. Different requests occupy different layers at the same time. A step visits the
ops in order, so up to `step_time / slowest_op` request groups circulate without waiting.
`serve()` picks the smallest group size that fits the requested batch, which is what keeps
per-user speed flat up to tens of concurrent requests.

Energy is bottom-up: bits scanned, FLOPs, link bits and KV bits times per-unit energies, plus
static power (storage leakage by bank state, logic, links, KV memory, host). Area is
bottom-up: storage by density, multiplier lanes by count, attention engines, link PHYs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

from .hardware import load_yaml
from .ops import Op

BITS_PER_WEIGHT = {"fp4": 4.5, "fp8": 8.25, "bf16": 16.0, "fp32": 32.0}
PREC_OF_CATEGORY = {"moe_nvfp4": "fp4", "moe_mxfp8": "fp8"}


@dataclass
class Stationary:
    name: str
    kind: str = "stationary"
    storage: str = "sram"  # sram (reloadable) | rom (fixed)
    # ---- bank geometry and clock
    f_clk: float = 2.0e9
    scan_rows: int = 512  # wordlines per bank = clocks per pass
    lane_rows_dense: int = 8  # activation rows served per pass by shared (dense) weights
    lane_rows_expert: int = 2  # rows served per pass by one expert
    t_pipe: float = 20.0e-9  # per GEMV: adder tree, accumulate, normalize, quantize
    router_latency: float = 0.1e-6
    vector_latency: float = 20.0e-9
    wake_latency: float = 0.0  # retention -> active before an expert can be scanned (managed banks)
    # ---- links
    d2d_latency: float = 5.0e-9  # one hop between dies of the same tile (short parallel link)
    d2d_bw: float = 800.0e9  # bytes/s leaving a layer's hub die for its expert dies, all links together
    hop_latency: float = 20.0e-9  # one hop between tiles (serial link)
    link_bw: float = 200.0e9  # bytes/s between two tiles, per direction
    act_bytes_dispatch: float = 0.5625  # bytes per value sent to an expert (NVFP4-quantized activations)
    # ---- attention and KV memory
    attn_peak: float = 2.0e14  # FLOP/s of one sliding-window attention engine (one per layer)
    attn_peak_full: float = 0.0  # FLOP/s of the pooled full-attention engine on the attention tile; 0 = attn_peak
    kv_bw: float = 1.0e12  # bytes/s of the pooled KV memory (set by serve(): stacks x kv_bw_per_stack)
    kv_bw_per_stack: float = 1.0e12
    kv_share: float = 1.0  # request groups using the pooled KV memory at the same moment (set by serve())
    tile_dies_max: int = 8  # dies one multi-die tile can hold
    tile_frac: float = 1.0  # share of layer boundaries that cross from one tile to another (set by serve())
    moe_cross_frac: float = 0.0  # share of a layer's expert dies outside its hub die's tile (set by serve())
    kv_eff: float = 0.84
    kv_latency: float = 0.5e-6
    kv_window_bw: float = 40.0e12  # on-die store for sliding windows
    kv_kind: str = "hbm"  # hbm | lpddr
    # ---- step-level
    step_overhead: float = 2.0e-6  # sampler hand-off and host sync
    # ---- area
    area: dict = field(default_factory=dict)
    # ---- energy
    power: dict = field(default_factory=dict)
    # ---- cost
    cost: dict = field(default_factory=dict)
    managed: bool = True  # idle banks held at retention voltage
    notes: str = ""

    def replace(self, **kw) -> "Stationary":
        import copy
        h = copy.deepcopy(self)
        for k, v in kw.items():
            if not hasattr(h, k):
                raise KeyError(f"unknown field {k}")
            setattr(h, k, v)
        return h

    t_scan_fixed: float = 0.0  # > 0: time of one pass given directly (capacity / bandwidth of a published machine)

    @property
    def t_scan(self) -> float:
        return self.t_scan_fixed or self.scan_rows / self.f_clk


def load_stationary(path: str, **overrides) -> Stationary:
    from .hardware import _deep_merge
    d = _deep_merge(load_yaml(path), overrides)
    known = {f.name for f in fields(Stationary)}
    unknown = set(d) - known
    if unknown:
        raise KeyError(f"{path}: unknown keys {sorted(unknown)}")
    return Stationary(**d)


@dataclass
class OpCost:
    op: Op
    time: float
    scan_bits: float = 0.0  # storage bits read (all passes)
    link_bits: float = 0.0  # between tiles
    d2d_bits: float = 0.0  # between dies of one tile
    kv_bits: float = 0.0
    kv_window_bits: float = 0.0
    cause: str = "scan"  # what set the time: scan | attn_fetch | attn_compute | hop | fixed


def op_cost(o: Op, hw: Stationary) -> OpCost:
    wbits = 8.0 * (o.w_tp + o.w_rep + o.w_dyn)
    if o.kind in ("gemv", "moe"):
        if o.kind == "moe":
            rows = max(1.0, o.rows_eff)
            passes = math.ceil(rows / hw.lane_rows_expert - 1e-9)
            disp_bits = 8.0 * o.out_w  # set by prepare(): bytes dispatched to the experts and returned
            t_scan = 2 * (passes * hw.t_scan + hw.t_pipe) + hw.wake_latency
            f = hw.moe_cross_frac  # share of a layer's expert dies on another tile than its hub die
            t_hop = 2 * ((1.0 - f) * hw.d2d_latency + f * hw.hop_latency) + disp_bits / 8.0 * ((1.0 - f) / hw.d2d_bw + f / hw.link_bw)
            return OpCost(o, t_scan + t_hop, scan_bits=wbits * passes, d2d_bits=disp_bits * (1.0 - f), link_bits=disp_bits * f,
                          cause="scan" if t_scan >= t_hop else "hop")
        passes = math.ceil(o.rows / hw.lane_rows_dense - 1e-9)
        t = passes * hw.t_scan + hw.t_pipe
        if o.serial_rounds > 1:  # low-rank recurrent head: each round rescans its (small) weights
            t = o.serial_rounds * (hw.t_scan + hw.t_pipe)
            return OpCost(o, t, scan_bits=wbits * o.serial_rounds, cause="scan")
        return OpCost(o, t, scan_bits=wbits * passes, cause="scan")
    if o.kind == "router":
        return OpCost(o, hw.t_scan + hw.t_pipe + hw.router_latency, scan_bits=wbits, cause="scan")
    if o.kind == "attn":
        flops = sum(o.flops.values()) + sum(o.flops_rep.values())
        t_comp = flops / hw.attn_peak
        kv_bytes = o.kv_tp + o.kv_rep
        if o.category in ("attn_sliding",) or (o.category == "draft_attn" and kv_bytes <= 0):
            t_fetch = kv_bytes / hw.kv_window_bw
            return OpCost(o, max(t_fetch, t_comp) + hw.vector_latency, kv_window_bits=8.0 * kv_bytes,
                          cause="attn_compute" if t_comp >= t_fetch else "attn_fetch")
        # Full attention runs on the attention tile: pooled KV memory and a pooled engine, shared by
        # the request groups that are in a full-attention layer at the same moment. With more than
        # one tile the hidden state hops there and back.
        share = max(1.0, hw.kv_share)
        t_comp = flops / ((hw.attn_peak_full or hw.attn_peak) / share)
        t_fetch = hw.kv_latency + kv_bytes / (hw.kv_bw * hw.kv_eff / share)
        t_hop = 2 * (hw.hop_latency + o.gather / hw.link_bw) if hw.tile_frac > 0 and hw.hop_latency > 0 else 0.0
        return OpCost(o, max(t_fetch, t_comp) + hw.vector_latency + t_hop, kv_bits=8.0 * kv_bytes, link_bits=16.0 * o.gather if t_hop else 0.0,
                      cause="attn_compute" if t_comp >= t_fetch else "attn_fetch")
    if o.kind == "collective":
        if o.name.endswith("ar_attn") or o.name.startswith(("draft_", "embed_")) or hw.hop_latency <= 0:
            # attention -> MLP stays on the layer's hub die; the draft and the embedding sit on one
            # tile, and there is no cross-rank argmax because nothing is sharded
            return OpCost(o, hw.vector_latency, cause="fixed")
        payload = o.coll  # layer boundary: to the next tile over a serial link, or to a neighbouring die of the same tile
        f = hw.tile_frac
        t = f * (hw.hop_latency + payload / hw.link_bw) + (1.0 - f) * (hw.d2d_latency + payload / hw.d2d_bw)
        return OpCost(o, t, link_bits=8.0 * payload * f, d2d_bits=8.0 * payload * (1.0 - f), cause="hop")
    if o.kind == "lookup":
        return OpCost(o, hw.vector_latency, scan_bits=wbits, cause="fixed")
    return OpCost(o, hw.vector_latency, scan_bits=wbits, cause="fixed")


def prepare(ops: list[Op], hw: Stationary, hidden: int, top_k: int, expert_dies: int = 1) -> list[Op]:
    """Annotate expert ops with the bytes dispatched to and returned from the expert dies.

    `out_w` is reused for that payload (it is unused by this engine otherwise): every row goes
    to top_k experts as NVFP4-quantized activations and comes back as BF16. `fanout` is the
    number of expert dies the payload is split across.
    """
    for o in ops:
        if o.kind == "moe":
            o.out_w = o.rows * top_k * hidden * (hw.act_bytes_dispatch + 2.0)
            touched = (o.w_dyn / max(1.0, o.w_dyn)) * o.rows * top_k / max(1.0, o.rows_eff)
            o.fanout = int(max(1, min(expert_dies, round(touched))))
        elif o.kind == "attn" and o.category == "attn_full":
            o.gather = o.rows * hidden * 2.0  # hidden state sent to the attention tile and back
    return ops


@dataclass
class StepCost:
    step_time: float
    costs: list[OpCost]
    slowest: float
    by_cause: dict
    by_category: dict
    flops: dict
    scan_bits: float
    link_bits: float
    d2d_bits: float
    kv_bits: float
    kv_window_bits: float


def simulate(ops: list[Op], hw: Stationary) -> StepCost:
    """One request group's step: ops in sequence, no contention between them."""
    costs = [op_cost(o, hw) for o in ops]
    by_cause: dict[str, float] = {}
    by_cat: dict[str, float] = {}
    flops = {"fp4": 0.0, "fp8": 0.0, "bf16": 0.0, "fp32": 0.0}
    for c in costs:
        by_cause[c.cause] = by_cause.get(c.cause, 0.0) + c.time
        by_cat[c.op.category] = by_cat.get(c.op.category, 0.0) + c.time
        for d in (c.op.flops, c.op.flops_rep):
            for k, v in d.items():
                flops[k] += v
    t = sum(c.time for c in costs) + hw.step_overhead
    by_cause["fixed"] = by_cause.get("fixed", 0.0) + hw.step_overhead
    return StepCost(t, costs, max(c.time for c in costs), by_cause, by_cat, flops,
                    sum(c.scan_bits for c in costs), sum(c.link_bits for c in costs), sum(c.d2d_bits for c in costs),
                    sum(c.kv_bits for c in costs), sum(c.kv_window_bits for c in costs))


# --------------------------------------------------------------------------------------------- area
def area(hw: Stationary, model) -> dict:
    """Silicon area by block (mm^2) and die count for a model builder with `resident_bytes()`."""
    a = hw.area
    rb = model.resident_bytes()
    resident = rb["total"] - (0.0 if a.get("vision_resident", True) else rb.get("vision_bf16", 0.0))
    mib = resident / 2**20
    storage = mib * (a["mm2_per_mib_rom"] if hw.storage == "rom" else a["mm2_per_mib_sram"])
    # multiplier lanes: one per weight column of every bank, times the rows served per pass
    lanes = _lanes(hw, model)
    mac = sum(lanes[p] * a["um2_per_mac"][p] for p in lanes) * 1e-6
    n_layers = getattr(model, "n_layers", 0)
    # Attention engines are sized for a rate (FLOP/s) at a reference clock; at the synthesized
    # clock the same rate needs proportionally more multipliers.
    clk = max(1.0, a.get("attn_ref_clk", hw.f_clk) / hw.f_clk)
    attn = (n_layers + (model.draft.layers if getattr(model, "draft", None) else 0)) * a["mm2_per_attn_engine"] * clk
    attn += a.get("mm2_attn_tile", 0.0) * clk
    # Sliding-window KV cache: reloadable SRAM on the hub dies, sized for a number of concurrent sessions.
    sessions = a.get("kv_window_sessions", 0)
    win_bytes = sessions * _window_bytes(model)
    kv_window = win_bytes / 2**20 * a["mm2_per_mib_sram"]
    n_stacks = kv_stacks(hw, model)
    kv_phy = n_stacks * a.get("mm2_per_kv_phy", 0.0)
    blocks = {"storage": storage, "mac_lanes": mac, "attention_engines": attn, "kv_window_sram": kv_window, "kv_phy": kv_phy}
    core = sum(blocks.values())
    usable = a["die_mm2"] * a["die_fill"] - a["mm2_d2d_phy_per_die"] - a["mm2_ctrl_per_die"]
    # Tiles hold at most `tile_dies_max` dies. A model that fits one tile needs no serial links.
    tiles = n_layers + 1
    for _ in range(4):
        phy = tiles * a["mm2_link_phy_per_tile"] if tiles > 1 else 0.0
        dies = math.ceil((core + phy) / usable)
        tiles = max(1, math.ceil(dies / hw.tile_dies_max))
    phy = tiles * a["mm2_link_phy_per_tile"] if tiles > 1 else 0.0
    dies = max(math.ceil((core + phy) / usable), 1)
    blocks["link_phy"] = phy + dies * a["mm2_d2d_phy_per_die"]
    blocks["control"] = dies * a["mm2_ctrl_per_die"]
    blocks["tiles"] = tiles
    blocks["dies_per_tile"] = dies / tiles
    blocks["dies_per_layer"] = dies / (n_layers + 1) if n_layers else float(dies)
    blocks["tile_frac"] = 0.0 if tiles <= 1 else min(1.0, tiles / (n_layers + 1))
    blocks["kv_stacks"] = n_stacks
    blocks["kv_window_sessions"] = sessions
    blocks["kv_window_gb"] = win_bytes / 1e9
    blocks["dies"] = dies
    blocks["die_mm2"] = a["die_mm2"]
    blocks["silicon_mm2"] = dies * a["die_mm2"]
    # share of the silicon that is repairable memory (SRAM with spare rows); mask-programmed storage is not
    sram = kv_window + (storage if hw.storage == "sram" else 0.0)
    blocks["repairable_share"] = sram / blocks["silicon_mm2"]
    blocks["lanes"] = lanes
    blocks["resident_gb"] = resident / 1e9
    blocks["gb_per_die"] = resident / 1e9 / dies
    return blocks


def _window_bytes(model) -> float:
    """Sliding-window KV bytes one session keeps on the hub dies (BF16)."""
    if not hasattr(model, "kv_bytes_per_token"):
        return 0.0
    from .models.base import Workload
    return float(model.kv_bytes_per_token(Workload()).get("target_sliding_per_request", 0.0))


def kv_stacks(hw: Stationary, model) -> int:
    """HBM stacks of the attention tile: a fixed count per full-history layer (target and draft)."""
    a = hw.area
    if "kv_stacks" in a:
        return int(a["kv_stacks"])
    full = len(getattr(model, "full_layer_ids", ())) or getattr(model, "n_layers", 0)
    draft = model.draft.layers if getattr(model, "draft", None) else 0
    return int(a.get("kv_stacks_per_layer", 0) * (full + draft))


def _lanes(hw: Stationary, model) -> dict:
    """Multiplier count by precision: weight columns x rows served per pass."""
    rb = model.resident_bytes()
    out = {"fp4": 0.0, "fp8": 0.0, "bf16": 0.0}
    def cols(nbytes, prec):
        return nbytes * 8.0 / BITS_PER_WEIGHT[prec] / hw.scan_rows
    out["fp4"] += cols(rb.get("experts_nvfp4", 0.0), "fp4") * hw.lane_rows_expert
    out["fp8"] += cols(rb.get("experts_mxfp8", 0.0), "fp8") * hw.lane_rows_expert
    dense = rb.get("dense_layers_bf16", 0.0) + rb.get("unembed_bf16", 0.0) + rb.get("draft_bf16", 0.0) + rb.get("layers", 0.0) + rb.get("unembed", 0.0)
    out["bf16"] += cols(dense, "bf16") * hw.lane_rows_dense
    return out


# --------------------------------------------------------------------------------------------- power
def static_power(hw: Stationary, model, ar: dict | None = None, busy_fraction: float = 0.0) -> dict:
    """Static power (W) by part. `busy_fraction` is the share of banks awake when banks are managed."""
    p = hw.power
    ar = ar or area(hw, model)
    gb = ar["resident_gb"]
    full = gb * p["leak_w_per_gb"] if hw.storage == "sram" else gb * p.get("leak_w_per_gb_rom", 0.0)
    if hw.managed and hw.storage == "sram":
        storage = full * (busy_fraction + (1.0 - busy_fraction) * p["retention_factor"])
    else:
        storage = full
    storage += ar.get("kv_window_gb", 0.0) * p["leak_w_per_gb"]  # the sliding-window cache is SRAM in both designs and stays awake
    dies = ar["dies"]
    parts = {
        "storage_leakage": storage,
        "logic_and_clock": dies * (p["logic_static_w_per_die"] if hw.managed else p["logic_static_w_per_die_unmanaged"]),
        "links_idle": (ar["tiles"] * p["link_idle_w_per_tile"] if ar["tiles"] > 1 else 0.0) + dies * p["d2d_idle_w_per_die"],
        "kv_memory": ar["kv_stacks"] * p["kv_static_w_per_stack"],
        "host": p["host_w"],
    }
    parts["total"] = sum(parts.values())
    return parts


def dynamic_energy(sc: StepCost, hw: Stationary) -> dict:
    """Joules for one request group's step, by part, before power-delivery loss."""
    p, pj = hw.power, 1e-12
    read = p["rom_read_pj_per_bit"] if hw.storage == "rom" else p["sram_read_pj_per_bit"]
    return {
        "storage_read": sc.scan_bits * read * pj,
        "compute": sum(sc.flops[k] * p["pj_per_flop"][k] for k in sc.flops) * pj,
        "links": sc.link_bits * p["link_pj_per_bit"] * pj + sc.d2d_bits * p["d2d_pj_per_bit"] * pj,
        "kv_memory": sc.kv_bits * p["kv_pj_per_bit"] * pj + sc.kv_window_bits * p["sram_read_pj_per_bit"] * pj,
    }


@dataclass
class Served:
    batch: int
    group: int  # requests stepping together through a layer
    groups: int  # request groups circulating
    step_time: float
    tok_s_user: float
    tok_s: float
    watts: float
    tokens_per_joule: float
    energy_parts: dict  # joules per token
    static: dict
    step: StepCost
    busy_fraction: float


def serve(model, wl, hw: Stationary, hidden: int | None = None, top_k: int | None = None) -> Served:
    """Serve `wl.batch` concurrent requests: smallest group size whose groups fit the pipeline."""
    from dataclasses import replace as dreplace
    hidden = hidden or model.hidden
    top_k = top_k if top_k is not None else getattr(model, "top_k", 0)
    B = wl.batch
    ar = area(hw, model)
    expert_dies = max(1, round(ar["dies_per_layer"]) - 1)
    hw = hw.replace(kv_bw=max(1, ar["kv_stacks"]) * hw.kv_bw_per_stack, tile_frac=ar["tile_frac"], kv_share=1.0,
                    moe_cross_frac=max(0.0, 1.0 - hw.tile_dies_max / max(1.0, ar["dies_per_layer"])))
    g = 1
    while True:
        sub = dreplace(wl, batch=g)
        ops = prepare(model.decode_step(sub), hw, hidden, top_k, expert_dies)
        groups = math.ceil(B / g)
        for _ in range(4):  # groups in a full-attention layer at the same moment share the attention tile
            sc = simulate(ops, hw)
            t_full = sum(c.time for c in sc.costs if c.op.kind == "attn" and c.kv_bits > 0)
            n_full = sum(1 for c in sc.costs if c.op.kind == "attn" and c.kv_bits > 0)
            share = min(max(1.0, 1.0 + (groups - 1) * t_full / sc.step_time), max(1.0, float(n_full)))
            if abs(share - hw.kv_share) < 0.02:
                break
            hw.kv_share = share
        depth = max(1, int(sc.step_time / sc.slowest))
        if groups <= depth or g >= B:
            break
        g = min(B, max(g + 1, math.ceil(B / depth)))
        hw.kv_share = 1.0
    groups = min(groups, max(1, depth)) if g >= B else groups
    tokens = wl.tokens_per_step
    tok_user = tokens / sc.step_time
    tok_s = tok_user * B
    # share of banks awake: time-average of bits being scanned over resident bits
    resident_bits = ar["resident_gb"] * 8e9
    busy = min(1.0, groups * (sc.scan_bits / max(1.0, resident_bits)) * (hw.t_scan / sc.step_time) * 4.0)
    st = static_power(hw, model, ar, busy)
    dyn = dynamic_energy(sc, hw)
    vr = hw.power.get("vr_efficiency", 1.0)
    per_step_tokens = tokens * g
    parts = {k: v / per_step_tokens for k, v in dyn.items()}
    for k, v in st.items():
        if k != "total":
            parts["static_" + k] = v / tok_s
    loss = sum(parts.values()) * (1.0 / vr - 1.0)
    parts["power_delivery_loss"] = loss
    j_per_token = sum(parts.values())
    return Served(B, g, groups, sc.step_time, tok_user, tok_s, j_per_token * tok_s, 1.0 / j_per_token, parts, st, sc, busy)


# --------------------------------------------------------------------------------------------- cost
def cost(hw: Stationary, model, ar: dict | None = None, volume_systems: float | None = None) -> dict:
    """Manufacturing cost of one system (USD), by part, plus amortized NRE.

    Dies are mostly memory, which can be repaired with spare rows and columns, so the Murphy
    yield model is applied to the non-repairable share of the die only. A fixed-weight design
    also pays for masks that are unique to one model version (`cost.unique_*`).
    """
    from .cost import dies_per_wafer, murphy_yield
    c = hw.cost
    ar = ar or area(hw, model)
    dies = ar["dies"]
    die_mm2 = ar["die_mm2"]
    dpw = dies_per_wafer(die_mm2)
    y = murphy_yield(die_mm2 * (1.0 - ar["repairable_share"]), c["d0_per_cm2"])
    die_usd = c["wafer_usd"] / (dpw * y)
    kv_gb = ar["kv_stacks"] * c["kv_gb_per_stack"]
    parts = {
        "dies": dies * die_usd,
        # tiles are multi-die advanced packages, costed per die at the rate of MDA-1's two-die package;
        # the attention tile carries the HBM stacks, eight to a package as on MDA-1
        "package_test": dies * c["package_test_usd_per_die"] + math.ceil(ar["kv_stacks"] / 8) * c.get("attn_package_usd", 0.0),
        "kv_memory": kv_gb * c["hbm_usd_per_gb"],
        "system": dies * c["system_usd_per_die"] + c["host_usd"],
    }
    manufacturing = sum(parts.values())
    systems = volume_systems if volume_systems is not None else c["volume_dies"] / dies
    nre = c["nre_usd"]
    unique = 0.0
    if hw.storage == "rom":  # every die holds different weights: masks unique to this model version
        unique = dies * c["unique_mask_layers"] * c["usd_per_unique_mask"]
    designs = int(c.get("die_designs", 1))
    nre += (designs - 1) * c.get("mask_set_usd", 0.0)
    parts["nre_share"] = (nre + unique) / systems
    parts["at_cost"] = manufacturing + parts["nre_share"]
    parts["total"] = parts["at_cost"] / (1.0 - c.get("gross_margin", 0.0))  # price to a buyer
    parts.update({"manufacturing": manufacturing, "die_usd": die_usd, "yield": y, "dies_per_wafer": dpw,
                  "systems": systems, "unique_mask_usd": unique, "nre_usd": nre,
                  "distinct_die_designs": (dies + designs - 1) if hw.storage == "rom" else designs})
    return parts


def usd_per_hour(hw: Stationary, system_usd: float, watts: float) -> float:
    c = hw.cost
    return system_usd / (c["lifetime_years"] * 8766.0) + watts / 1000.0 * c["pue"] * c["usd_per_kwh"]
