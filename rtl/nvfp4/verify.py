#!/usr/bin/env python3
"""Bit-exact verification of nvfp4_dot and nvfp4_dot_const against the NumPy reference.

    python rtl/nvfp4/verify.py            # everything (about 1.4 M dot products, a few minutes)
    python rtl/nvfp4/verify.py --quick    # a 20 k subset, seconds

Writes rtl/nvfp4/build/verify.json (assembled into results/silicon.json by run_all.py).
A "vector" is one dot product: one row of blocks streamed into the engine and one binary32 out.
The Verilator harness only drives pins and records outputs; every comparison is done here.
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import simlib as S  # noqa: E402

ref, P, ROOT = S.ref, S.P, S.ROOT
FIX = ROOT / "data" / "fixtures"
EXPERT = "data/fixtures/inkling2_small_L00_E000.npz"
ROWS_NPZ = "data/fixtures/inkling2_small_rows.npz"
CONST_ROWS = [0, 1, 577, 1154, 2305, 2882, 3459, 4607]      # single-row fixed-weight modules (w13)
WIDE_ROWS = list(range(16, 24))                             # eight rows sharing one activation stream
FINITE = np.array([b for b in range(256) if (b & 0x7F) != 0x7F], np.uint8)
TOKENSPEED = pathlib.Path("/home/user/scratch/ext/tokenspeed")


def blocks(packed: np.ndarray) -> np.ndarray:
    """[N, width/2] packed bytes -> [N, NB, 8]."""
    return np.ascontiguousarray(packed).reshape(packed.shape[0], -1, P["B"] // 2)


def codes_of(b: np.ndarray) -> np.ndarray:
    """[..., NB, 8] packed bytes -> [..., NB, 16] codes."""
    return ref.unpack_codes(b)


# ------------------------------------------------------------------------------- activations
def embedding_activations(n: int | None = None) -> np.ndarray:
    """Real-looking MoE inputs: real token-embedding rows, RMS-normalised and scaled by the real
    layer-0 `mlp_norm.weight`, stored as BF16. (The true input also carries the attention output,
    which cannot be computed without running the model.)"""
    aux = np.load(FIX / "inkling2_small_aux.npz")
    e = ref.bf16_to_f32(aux["embed_rows_bf16"]).astype(np.float64)
    g = ref.bf16_to_f32(aux["mlp_norm_weight_bf16"]).astype(np.float64)
    x = e / np.sqrt((e ** 2).mean(1, keepdims=True) + 1e-6) * g
    return ref.f32_to_bf16(x.astype(np.float32))[:n]


def synthetic_activations(rng, n: int, width: int, kind: str) -> np.ndarray:
    if kind == "gaussian":
        x = rng.normal(0, 1, (n, width))
    elif kind == "heavy":
        x = rng.standard_t(3, (n, width)) * 0.3
    elif kind == "sparse":
        x = rng.normal(0, 1, (n, width)) * (rng.random((n, width)) < 0.1)
    elif kind == "relu":
        x = np.maximum(rng.normal(-0.5, 1, (n, width)), 0) * rng.lognormal(0, 1, (n, 1))
    else:
        raise ValueError(kind)
    return ref.f32_to_bf16(x.astype(np.float32))


def swiglu_activations(x_tokens: np.ndarray, cfg: dict) -> np.ndarray:
    """Second-projection inputs produced by the REAL first projection of the fixture expert:
    quantise x, run the w13 GEMM with the reference, apply the outer scales and the capped SwiGLU
    exactly as TokenSpeed's reference kernel does (route weight 1), store BF16."""
    z = np.load(ROOT / EXPERT)
    xp, xs, amax = ref.quantize_activations(x_tokens)
    bits = ref.gemm_ref(codes_of(blocks(z["w13"])), z["w13_scale"], codes_of(blocks(xp)), xs)
    scale = (amax * z["w13_amax"]) * np.float32(1.0 / (np.float32(ref.ACT_DENOMINATOR) * z["w13_denominator"]))
    gu = ref.f32_to_bf16(bits.view(np.float32) * scale[:, None]).astype(np.float64)
    gate, up = gu[:, 0::2], gu[:, 1::2]
    sig = 1.0 / (1.0 + np.exp(-gate))
    gate = cfg["mlp_gate_softcap"] * np.tanh(gate / cfg["mlp_gate_softcap"])
    up = cfg["mlp_up_softcap"] * np.tanh(up / cfg["mlp_up_softcap"])
    return ref.f32_to_bf16((gate * sig * up).astype(np.float32))


def random_codes(rng, n: int, nb: int, scales: str = "finite", p_nan: float = 0.0):
    c = rng.integers(0, 256, (n, nb, 8), dtype=np.uint8)
    if scales == "finite":
        s = FINITE[rng.integers(0, len(FINITE), (n, nb))]
    elif scales == "real":           # the range real weights use: 48 .. 448
        s = rng.integers(100, 127, (n, nb)).astype(np.uint8)
    else:
        s = rng.integers(0, 256, (n, nb), dtype=np.uint8)
    if p_nan:
        hit = rng.random((n, nb)) < p_nan
        s = np.where(hit, np.where(rng.random((n, nb)) < 0.5, 0x7F, 0xFF), s).astype(np.uint8)
    return c, s


# ------------------------------------------------------------------------------- directed vectors
_PROD = {}
for _a, _qa in enumerate(ref.E2M1_Q[:8]):
    for _b, _qb in enumerate(ref.E2M1_Q[:8]):
        if _qa * _qb:
            _PROD.setdefault(int(_qa * _qb), (_a, _b))
_PRODS = sorted(_PROD, reverse=True)


def block_for(d: int):
    """Codes of one block whose exact sum of (2w)(2x) is d, or None if 16 lanes are not enough."""
    w, x, rem, neg = [], [], abs(d), 8 if d < 0 else 0
    while rem and len(w) < 16:
        q = next(v for v in _PRODS if v <= rem)
        w.append(_PROD[q][0] | neg); x.append(_PROD[q][1]); rem -= q
    if rem:
        return None
    w += [0] * (16 - len(w)); x += [0] * (16 - len(x))
    return np.array(w, np.uint8), np.array(x, np.uint8)


def e4m3_byte(m: int, e: int, sign: int = 0) -> int:
    """Significand m (8..15 normal, 0..7 with e = 1 subnormal) times 2^(e-10)."""
    return (sign << 7) | (((e if m >= 8 else 0) & 15) << 3) | (m & 7)


def directed_vectors(rng):
    """Named edge cases. Returns (list of (name, wc[NB,16], ws[NB], xc[NB,16], xs[NB]))."""
    out = []
    six, m6 = np.full(16, 7, np.uint8), np.full(16, 15, np.uint8)
    alt = np.where(np.arange(16) % 2, 15, 7).astype(np.uint8)
    rc = lambda: rng.integers(0, 16, 16).astype(np.uint8)  # noqa: E731

    def add(name, blks):
        out.append((name, np.array([b[0] for b in blks], np.uint8), np.array([b[1] for b in blks], np.uint8),
                    np.array([b[2] for b in blks], np.uint8), np.array([b[3] for b in blks], np.uint8)))

    for nb in (1, 2, 7, 288):
        for zs in (0x00, 0x80):
            add("zero scale", [(rc(), zs, rc(), 0x70) for _ in range(nb)])
            add("zero scale", [(rc(), 0x70, rc(), zs) for _ in range(nb)])
            add("zero scale", [(rc(), zs if b % 2 else 0x55, rc(), 0x40 if b % 3 else zs) for b in range(nb)])
        add("all-zero blocks", [(rng.choice([0, 8], 16).astype(np.uint8), 0x7E, rc(), 0x7E) for _ in range(nb)])
        add("all-zero blocks", [(rc(), 0x7E, rng.choice([0, 8], 16).astype(np.uint8), 0x7E) for _ in range(nb)])
        add("all-zero blocks", [((rc() if b == nb // 2 else np.zeros(16, np.uint8)), 0x60, rc(), 0x58) for b in range(nb)])
        for sgn_w, sgn_x in ((six, six), (m6, six), (six, m6), (m6, m6), (alt, six), (alt, alt)):
            add("saturation (all codes +-6, scales 448)", [(sgn_w, 0x7E, sgn_x, 0x7E) for _ in range(nb)])
            add("saturation, negative scales", [(sgn_w, 0xFE, sgn_x, 0x7E if nb % 2 else 0xFE) for _ in range(nb)])
        for sub in range(1, 8):
            add("subnormal scales", [(six, sub, six, sub) for _ in range(nb)])
            add("subnormal scales", [(rc(), sub, rc(), 0x80 | (8 - sub)) for _ in range(nb)])
            add("subnormal x maximum scale", [(rc(), sub, rc(), 0x7E) for _ in range(nb)])
        add("maximum scales", [(rc(), 0x7E, rc(), 0x7E) for _ in range(nb)])
        add("maximum scales", [(rc(), 0xFE, rc(), 0x7E) for _ in range(nb)])
        for nan in (0x7F, 0xFF):
            for pos in sorted({0, nb // 2, nb - 1}):
                add("NaN scale", [(rc(), nan if b == pos else 0x70, rc(), 0x68) for b in range(nb)])
                add("NaN scale", [(rc(), 0x70, rc(), nan if b == pos else 0x68) for b in range(nb)])
                add("NaN scale with zero codes", [(np.zeros(16, np.uint8), nan if b == pos else 0, rc(), 0x68) for b in range(nb)])
        add("NaN scale", [(rc(), 0x7F, rc(), 0xFF) for _ in range(nb)])
        # sign patterns and cancellation
        for pat in (0x0000, 0xFFFF, 0xAAAA, 0x5555, 0x00FF, 0x0F0F, 0x8001):
            sg = np.array([(pat >> j) & 1 for j in range(16)], np.uint8) << 3
            add("sign patterns", [((rc() & 7) | sg, 0x70, rc() & 7, 0x70) for _ in range(nb)])
            add("sign patterns", [(rc() & 7, 0x70 | (b & 1) << 7, (rc() & 7) | sg, 0x65) for b in range(nb)])
        w0, x0 = rc(), rc()
        add("cancellation across blocks", [(w0 ^ (8 if b % 2 else 0), 0x70, x0, 0x70) for b in range(nb + nb % 2)])
        add("cancellation inside a block", [(np.tile([7, 15], 8).astype(np.uint8), 0x7E, six, 0x7E) for _ in range(nb)])
    # binary32 rounding corners, built from exact block sums
    big = block_for(2048)
    for k in range(0, 12):
        small = block_for(1 << k)
        for sgn in (0, 0x80):
            for e_s in (1, 3, 6, 9, 12):
                add("rounding: large term then small terms",
                    [(big[0], e4m3_byte(8, 15), big[1], e4m3_byte(8, 15))] +
                    [(small[0], e4m3_byte(8, e_s, 0) | sgn, small[1], e4m3_byte(8, 1))] * 6)
                add("rounding: small terms then large term",
                    [(small[0], e4m3_byte(8, e_s) | sgn, small[1], e4m3_byte(8, 1))] * 5 +
                    [(big[0], e4m3_byte(8, 15), big[1], e4m3_byte(8, 15))])
    return out


def rounding_stress(rng, n: int, nb: int):
    """Random rows whose block terms are +-(1, 3, 2^k, 2^k +- 1, ...) * 2^e over all 29 term
    exponents, so that ties, long cancellations and sticky-only alignments are frequent."""
    wc = np.zeros((n, nb, 16), np.uint8); xc = np.zeros((n, nb, 16), np.uint8)
    ws = np.zeros((n, nb), np.uint8); xs = np.zeros((n, nb), np.uint8)
    cands = [d for d in ([1, 3, 5, 7, 9, 2303, 2304] + [1 << k for k in range(12)] + [(1 << k) + 1 for k in range(1, 12)]
                         + [(1 << k) - 1 for k in range(2, 12)] + [3 << k for k in range(10)]) if block_for(d) is not None]
    lib = [block_for(d) for d in cands]
    for i in range(n):
        mode = rng.integers(0, 3)
        e0 = rng.integers(1, 16)
        for b in range(nb):
            k = rng.integers(0, len(lib))
            w, x = lib[k]
            wc[i, b] = w ^ (8 if rng.random() < 0.5 else 0); xc[i, b] = x
            if mode == 0:      # full exponent range
                ew, ex = rng.integers(1, 16), rng.integers(1, 16)
            elif mode == 1:    # one exponent, power-of-two significands: exact ties and cancellations
                ew, ex = e0, e0
            else:              # slowly drifting exponent
                ew = int(np.clip(e0 + rng.integers(-1, 2), 1, 15)); ex = ew; e0 = ew
            mw = 8 if mode else rng.integers(8, 16)
            mx = 8 if mode else rng.integers(8, 16)
            ws[i, b] = e4m3_byte(int(mw), int(ew)); xs[i, b] = e4m3_byte(int(mx), int(ex), int(rng.random() < 0.3))
    return wc, ws, xc, xs


# ------------------------------------------------------------------------------- sections
class Tally:
    def __init__(self):
        self.sections, self.t0 = [], time.time()

    def add(self, dut: str, name: str, got: np.ndarray, exp: np.ndarray, cycles: int, note: str = ""):
        got, exp = np.asarray(got).reshape(-1), np.asarray(exp).reshape(-1)
        bad = int((got != exp).sum())
        self.sections.append({"dut": dut, "name": name, "vectors": int(got.size), "mismatches": bad,
                              "clock_cycles": int(cycles), "note": note})
        print(f"[{time.time() - self.t0:6.0f}s] {dut:22s} {name:58s} {got.size:9,d} vectors, {bad} mismatches", flush=True)
        if bad:
            i = int(np.nonzero(got != exp)[0][0])
            print(f"          first mismatch at {i}: rtl {int(got[i]):08x} ref {int(exp[i]):08x}")


def pairs_job(exe, nb, wc_b, ws, xc_b, xs, rng, gap_frac=0.1, tag="job", dut_rows=1, const=False):
    """All (weight row, token) pairs in random order; returns results reordered as [T, R]."""
    R = wc_b.shape[0] if not const else 1
    T = xc_b.shape[0]
    wi, xi = np.meshgrid(np.arange(R), np.arange(T), indexing="ij")
    order = rng.permutation(R * T)
    sched = np.stack([wi.ravel()[order], xi.ravel()[order],
                      np.where(rng.random(R * T) < gap_frac, rng.integers(1, 1 << 30, R * T), 0)], 1)
    res, cyc = S.run_job(exe, nb, None if const else wc_b, None if const else ws, xc_b, xs, sched, dut_rows, tag)
    out = np.zeros((R * T, dut_rows), np.uint32)
    out[order] = res
    return out.reshape(R, T, dut_rows), cyc


def verify(quick: bool = False) -> dict:
    rng = np.random.default_rng(20261115)
    t = Tally()
    cfg = json.loads((ROOT / "data" / "inkling2_small" / "config.json").read_text())["text_config"]
    z = np.load(ROOT / EXPERT)
    zr = np.load(ROOT / ROWS_NPZ)
    w13, w13s = blocks(z["w13"]), z["w13_scale"]
    w2, w2s = blocks(z["w2"]), z["w2_scale"]
    nb13, nb2 = P["nblk"]["w13"], P["nblk"]["w2"]
    sc = 16 if quick else 1  # scale-down factor for --quick

    # activations
    x_emb = embedding_activations(512 // sc)
    xe_p, xe_s, _ = ref.quantize_activations(x_emb)
    x_syn = np.concatenate([synthetic_activations(rng, 96 // sc + 1, 4608, k) for k in ("gaussian", "heavy", "sparse")])
    xs_p, xs_s, _ = ref.quantize_activations(x_syn)
    x_sw = swiglu_activations(x_emb[: 256 // sc], cfg)
    xw_p, xw_s, _ = ref.quantize_activations(x_sw)
    act_stats = {
        "embedding_tokens": int(x_emb.shape[0]),
        "embedding_blocks_scaled_to_4": float((np.abs(ref.e2m1_decode(codes_of(blocks(xe_p)))).max(-1) == 4).mean()),
        "swiglu_tokens": int(x_sw.shape[0]),
        "swiglu_zero_code_fraction": float((codes_of(blocks(xw_p)) & 7 == 0).mean()),
        "embedding_zero_code_fraction": float((codes_of(blocks(xe_p)) & 7 == 0).mean()),
    }

    # ---------------- reloadable, single row
    rel_dir = S.gen_rtl("nvfp4_dot")
    rel = S.build_verilator("nvfp4_dot", rel_dir, 1, False)
    rows = np.arange(0, 4608, 4608 // (1024 // sc))[: 1024 // sc] + (np.arange(1024 // sc) % 2)  # gate and up rows
    exp = ref.gemm_ref(codes_of(w13[rows]), w13s[rows], codes_of(blocks(xe_p)), xe_s).T
    got, cyc = pairs_job(rel, nb13, w13[rows], w13s[rows], blocks(xe_p), xe_s, rng, tag="r1")
    t.add("nvfp4_dot", "real w13 rows (L0 E0) x embedding-derived tokens", got[..., 0], exp, cyc,
          f"{len(rows)} rows x {x_emb.shape[0]} tokens, 288 blocks, 10% of vectors with idle gaps")
    rows2 = np.arange(0, 4608, 4608 // (512 // sc))[: 512 // sc]
    exp = ref.gemm_ref(codes_of(w2[rows2]), w2s[rows2], codes_of(blocks(xw_p)), xw_s).T
    got, cyc = pairs_job(rel, nb2, w2[rows2], w2s[rows2], blocks(xw_p), xw_s, rng, tag="r2")
    t.add("nvfp4_dot", "real w2 rows (L0 E0) x SwiGLU outputs of the real w13", got[..., 0], exp, cyc,
          f"{len(rows2)} rows x {x_sw.shape[0]} tokens, 144 blocks")
    for L, E in zr["samples"]:
        tag = f"L{L:02d}_E{E:03d}"
        wr, wrs = blocks(zr[f"{tag}_w13"]), zr[f"{tag}_w13_scale"]
        xa = np.concatenate([blocks(xs_p), blocks(xe_p)[: 64 // sc]])
        xas = np.concatenate([xs_s, xe_s[: 64 // sc]])
        exp = ref.gemm_ref(codes_of(wr), wrs, codes_of(xa), xas).T
        got, cyc = pairs_job(rel, nb13, wr, wrs, xa, xas, rng, tag="r3")
        t.add("nvfp4_dot", f"real w13 rows, layer {L} expert {E} x mixed tokens", got[..., 0], exp, cyc,
              f"{wr.shape[0]} rows x {xa.shape[0]} tokens (gaussian, heavy-tailed, sparse, embedding)")
        wr, wrs = blocks(zr[f"{tag}_w2"]), zr[f"{tag}_w2_scale"]
        xr = synthetic_activations(rng, 96 // sc + 1, 2304, "relu")
        xr_p, xr_s, _ = ref.quantize_activations(xr)
        xa, xas = np.concatenate([blocks(xr_p), blocks(xw_p)[: 64 // sc]]), np.concatenate([xr_s, xw_s[: 64 // sc]])
        exp = ref.gemm_ref(codes_of(wr), wrs, codes_of(xa), xas).T
        got, cyc = pairs_job(rel, nb2, wr, wrs, xa, xas, rng, tag="r3")
        t.add("nvfp4_dot", f"real w2 rows, layer {L} expert {E} x mixed tokens", got[..., 0], exp, cyc,
              f"{wr.shape[0]} rows x {xa.shape[0]} tokens")
    # random codes and scales, all row lengths
    for nb, n, kind, pn in ((1, 40000, "any", 0), (2, 30000, "any", 0), (3, 30000, "finite", 0.01), (5, 30000, "finite", 0.01),
                            (16, 30000, "finite", 0.002), (31, 20000, "finite", 0.001), (144, 12000, "finite", 2e-4),
                            (288, 24000, "finite", 1e-4), (288, 12000, "real", 0)):
        n //= sc
        wc, ws = random_codes(rng, n, nb, kind, pn)
        xc, xs = random_codes(rng, n, nb, kind, pn)
        sched = np.stack([np.arange(n), np.arange(n), np.where(rng.random(n) < 0.2, rng.integers(1, 1 << 30, n), 0)], 1)
        got, cyc = S.run_job(rel, nb, wc, ws, xc, xs, sched, tag="r4")
        t.add("nvfp4_dot", f"random codes, {kind} scale bytes, rows of {nb} block(s)", got[:, 0],
              ref.dot_ref(codes_of(wc), ws, codes_of(xc), xs), cyc,
              "uniform 4-bit codes (so +0 and -0 codes too); 'any' = all 256 scale bytes including NaN; "
              "'finite' = all 254 finite bytes (zeros, subnormals, negatives) with NaN injected at the stated rate")
    # directed edge cases, grouped by row length
    dv = directed_vectors(rng)
    cover: dict = {}
    by_name: dict = {}
    for nb in sorted({v[1].shape[0] for v in dv}):
        grp = [v for v in dv if v[1].shape[0] == nb]
        wc, ws = np.stack([v[1] for v in grp]), np.stack([v[2] for v in grp])
        xc, xs = np.stack([v[3] for v in grp]), np.stack([v[4] for v in grp])
        exp = ref.dot_ref(wc, ws, xc, xs)
        n = len(grp)
        for rep, gap in ((0, 0), (1, 1)):   # back to back, then with idle gaps
            sched = np.stack([np.arange(n), np.arange(n), np.full(n, gap * (rep + 7))], 1)
            got, cyc = S.run_job(rel, nb, ref.pack_codes(wc), ws, ref.pack_codes(xc), xs, sched, tag="r5")
            for i, v in enumerate(grp):
                s = by_name.setdefault(v[0], [0, 0])
                s[0] += 1; s[1] += int(got[i, 0] != exp[i])
        for i in range(n if nb < 100 else min(n, 40)):
            assert ref.dot_ref_int(wc[i], ws[i], xc[i], xs[i], stats=cover) == int(exp[i])
    for name, (n, bad) in sorted(by_name.items()):
        t.sections.append({"dut": "nvfp4_dot", "name": f"directed: {name}", "vectors": n, "mismatches": bad,
                           "clock_cycles": 0, "note": "each vector run back to back and again with idle gaps"})
        print(f"          nvfp4_dot directed: {name:52s} {n:6d} vectors, {bad} mismatches")
    for nb, n in ((4, 20000), (12, 20000), (40, 10000)):
        n //= sc
        wc, ws, xc, xs = rounding_stress(rng, n, nb)
        exp = ref.dot_ref(wc, ws, xc, xs)
        sched = np.stack([np.arange(n), np.arange(n), np.zeros(n)], 1)
        got, cyc = S.run_job(rel, nb, ref.pack_codes(wc), ws, ref.pack_codes(xc), xs, sched, tag="r6")
        t.add("nvfp4_dot", f"binary32 rounding stress, rows of {nb} blocks", got[:, 0], exp, cyc,
              "terms +-{1,3,2^k,2^k+-1,3*2^k,...} x 2^e: ties, long cancellations, sticky-only alignments")
        for i in range(min(n, 600)):
            assert ref.dot_ref_int(wc[i], ws[i], xc[i], xs[i], stats=cover) == int(exp[i])
    # coverage of the real-tensor adds, on a sample
    cover_real: dict = {}
    for i in range(40 // sc + 2):
        r_, x_ = int(rows[i * 7 % len(rows)]), i * 5 % x_emb.shape[0]
        ref.dot_ref_int(codes_of(w13[r_]), w13s[r_], codes_of(blocks(xe_p)[x_]), xe_s[x_], stats=cover_real)

    # ---------------- reloadable, eight rows sharing activations
    r8_dir = S.gen_rtl("nvfp4_dot_x8", rows=8)
    r8 = S.build_verilator("nvfp4_dot_x8", r8_dir, 8, False)
    n = 4096 // sc
    first = rng.integers(0, 4608 - 8, n)
    xi = rng.integers(0, xe_p.shape[0], n)
    sched = np.stack([first, xi, np.where(rng.random(n) < 0.1, 5, 0)], 1)
    got, cyc = S.run_job(r8, nb13, w13, w13s, blocks(xe_p), xe_s, sched, 8, tag="r8")
    exp = np.stack([ref.dot_ref(codes_of(w13[first + r]), w13s[first + r], codes_of(blocks(xe_p)[xi]), xe_s[xi]) for r in range(8)], 1)
    t.add("nvfp4_dot_x8", "8 consecutive real w13 rows x embedding-derived tokens", got, exp, cyc,
          "eight weight streams, one activation stream; each row result is one vector")

    # ---------------- fixed-weight modules
    def const_acts(nb, leaf):
        n_r = 8192 // sc
        rc_, rs_ = random_codes(rng, n_r, nb, "finite", 5e-4)
        edge_c, edge_s = random_codes(rng, 256 // sc + 8, nb, "any", 0)
        edge_s[:4] = 0; edge_s[4:8] = 0x7E; edge_c[8:12] = 0
        if leaf == "w13":
            real_c = np.concatenate([blocks(xe_p), blocks(xs_p)]); real_s = np.concatenate([xe_s, xs_s])
            more = np.concatenate([synthetic_activations(rng, 1536 // sc, 4608, k) for k in ("gaussian", "heavy", "sparse")])
        else:
            real_c, real_s = blocks(xw_p), xw_s
            more = synthetic_activations(rng, 2048 // sc, 2304, "relu")
        mp, ms, _ = ref.quantize_activations(more)
        return np.concatenate([real_c, blocks(mp), rc_, edge_c]), np.concatenate([real_s, ms, rs_, edge_s])

    xa13, xas13 = const_acts(nb13, "w13")
    xa2, xas2 = const_acts(nb2, "w2")
    const_designs = [(f"nvfp4_dot_const_r{r}", dict(fixture=EXPERT, leaf="w13", rows=[r]), w13[[r]], w13s[[r]]) for r in CONST_ROWS]
    const_designs.append(("nvfp4_dot_const_x8", dict(fixture=EXPERT, leaf="w13", rows=WIDE_ROWS), w13[WIDE_ROWS], w13s[WIDE_ROWS]))
    const_designs += [(f"nvfp4_dot_const_w2_r{r}", dict(fixture=EXPERT, leaf="w2", rows=[r]), w2[[r]], w2s[[r]]) for r in (0, 2303)]
    for L, E in zr["samples"][3:]:
        tag = f"L{L:02d}_E{E:03d}"
        const_designs.append((f"nvfp4_dot_const_{tag}", dict(fixture=ROWS_NPZ, leaf="w13", rows=[5], prefix=tag + "_"),
                              blocks(zr[f"{tag}_w13"])[[5]], zr[f"{tag}_w13_scale"][[5]]))
    for nblk in (1, 4, 16, 72):    # shorter hard-wired rows (for the storage-versus-datapath sweep)
        const_designs.append((f"nvfp4_dot_const_k{nblk}", dict(fixture=EXPERT, leaf="w13", rows=[0], blocks=(0, nblk)),
                              w13[[0], :nblk], w13s[[0], :nblk]))
    for tt, b0 in ((1, 0), (1, 100), (1, 200), (4, 0), (16, 0), (72, 0)):   # 288-block rows with a table of tt blocks
        name = f"nvfp4_dot_const_t{tt}" + (f"_b{b0}" if b0 else "")
        const_designs.append((name, dict(fixture=EXPERT, leaf="w13", rows=[0], blocks=(b0, b0 + tt), tile=True),
                              np.tile(w13[[0], b0:b0 + tt], (1, nb13 // tt, 1)), np.tile(w13s[[0], b0:b0 + tt], (1, nb13 // tt))))
    if quick:
        const_designs = [const_designs[0], const_designs[8], const_designs[-10], const_designs[-1]]
    for name, spec, wc_b, wsc in const_designs:
        d = S.gen_rtl(name, const=spec)
        exe = S.build_verilator(name, d, len(spec["rows"]), True)
        nb = wsc.shape[1]
        xa, xas = (xa13, xas13) if spec["leaf"] == "w13" else (xa2, xas2)
        xa, xas = xa[:, :nb], xas[:, :nb]
        n = xa.shape[0]
        order = rng.permutation(n)
        sched = np.stack([np.zeros(n), order, np.where(rng.random(n) < 0.1, rng.integers(1, 1 << 30, n), 0)], 1)
        got, cyc = S.run_job(exe, nb, None, None, xa, xas, sched, len(spec["rows"]), tag="c1")
        exp = ref.gemm_ref(codes_of(wc_b), wsc, codes_of(xa[order]), xas[order])
        t.add(name, f"{len(spec['rows'])} hard-wired row(s) x {nb} blocks: real, synthetic, random, edge tokens", got, exp, cyc,
              f"weights {spec['fixture'].split('/')[-1]} {spec.get('prefix', '')}{spec['leaf']} rows {spec['rows']}")

    total = sum(s["vectors"] for s in t.sections)
    bad = sum(s["mismatches"] for s in t.sections)
    res = {"pass": bad == 0, "vectors_total": total, "mismatches_total": bad,
           "vectors_reloadable": sum(s["vectors"] for s in t.sections if s["dut"].startswith("nvfp4_dot") and "const" not in s["dut"]),
           "vectors_fixed": sum(s["vectors"] for s in t.sections if "const" in s["dut"]),
           "clock_cycles_total": sum(s["clock_cycles"] for s in t.sections),
           "simulator": S.tool_version(["verilator", "--version"]), "quick": quick,
           "sections": t.sections, "activation_stats": act_stats,
           "adder_case_coverage_directed_and_stress": dict(sorted(cover.items())),
           "adder_case_coverage_real_tensor_sample": dict(sorted(cover_real.items())),
           "seconds": round(time.time() - t.t0, 1)}
    print(f"TOTAL {total:,} vectors, {bad} mismatches, {res['clock_cycles_total']:,} clock cycles, {res['seconds']} s")
    return res


# ------------------------------------------------------------------------------- other checks
def check_icarus() -> dict:
    """Cross-simulator check (Icarus) and the accumulator-overflow test that needs a test hook."""
    rng = np.random.default_rng(7)
    z = np.load(ROOT / EXPERT)
    w13, w13s = blocks(z["w13"]), z["w13_scale"]
    xe_p, xe_s, _ = ref.quantize_activations(embedding_activations(8))
    d = S.gen_rtl("nvfp4_dot")
    src = [d / "nvfp4_common.sv", d / "nvfp4_dot.sv", d / "tb_nvfp4_dot.sv"]
    vvp = S.build_icarus("nvfp4_dot", src)
    pairs = [(int(r), int(x)) for r in rng.integers(0, 4608, 24) for x in range(4)]
    got = S.run_icarus(vvp, S.stream(288, w13, w13s, blocks(xe_p), xe_s, pairs))
    exp = np.array([ref.dot_ref(codes_of(w13[r]), w13s[r], codes_of(blocks(xe_p)[x]), xe_s[x]) for r, x in pairs])
    out = {"simulator": S.tool_version(["iverilog", "-V"]),
           "reloadable_real_vectors": len(pairs), "reloadable_mismatches": int((got[:, 0] != exp).sum())}
    dc = S.gen_rtl("nvfp4_dot_const_r0", const=dict(fixture=EXPERT, leaf="w13", rows=[0]))
    vvpc = S.build_icarus("nvfp4_dot_const_r0", [dc / "nvfp4_common.sv", dc / "nvfp4_dot_const_r0.sv", dc / "tb_nvfp4_dot_const_r0.sv"])
    pc = [(0, x) for x in range(8)]
    gotc = S.run_icarus(vvpc, S.stream(288, None, None, blocks(xe_p), xe_s, pc), tag="icc")
    expc = ref.gemm_ref(codes_of(w13[[0]]), w13s[[0]], codes_of(blocks(xe_p)), xe_s)[:, 0]
    out.update(fixed_real_vectors=len(pc), fixed_mismatches=int((gotc[:, 0] != expc).sum()))
    # overflow: not reachable from the ports. Block 0 is a zero term; during the idle clocks after it
    # the testbench overwrites the accumulator register and places one large term in front of the adder
    vvpp = S.build_icarus("nvfp4_dot", src, defines=("NVFP4_POKE",), tag="_poke")
    six = np.full(16, 7, np.uint8)
    cases = []
    FMAX, HALF_ULP = 0x7F7FFFFF, 0x73000000          # largest finite binary32; 2^103, half its last place
    for what, init, term, sgn, nan_at in (
            ("max + max", FMAX, 0x7F7FFFE0, 0, None), ("-max - max", FMAX | 1 << 31, 0xFF7FFFE0, 8, None),
            ("max + half an ulp: the tie rounds up and carries into infinity", FMAX, HALF_ULP, 0, None),
            ("max + a quarter ulp: stays finite", FMAX, 0x72800000, 0, None),
            ("-max - half an ulp", FMAX | 1 << 31, HALF_ULP | 1 << 31, 0, None),
            ("max - max: exact cancellation, no overflow", FMAX, 0xFF7FFFE0, 0, None),
            ("2^127 + 2^127", 0x7F000000, 0x7F000000, 0, None), ("2^126 + 2^126: finite", 0x7E800000, 0x7E800000, 8, None),
            ("infinity, then a NaN scale", FMAX, 0x7F7FFFE0, 0, 5), ("no injected term: finite", FMAX, None, 8, None)):
        nb = 12
        wc = np.tile(six | sgn, (nb, 1)); xc = np.tile(six, (nb, 1))
        ws = np.full(nb, 0x7E, np.uint8); xs = np.full(nb, 0x7E, np.uint8)
        wc[0] = 0
        if nan_at is not None:
            xs[nan_at] = 0x7F
        words = S.stream(nb, ref.pack_codes(wc)[None], ws[None], ref.pack_codes(xc)[None], xs[None], [(0, 0)], gaps={0: 10})
        args = ["+poke_cycle=7", f"+poke_val={init:08x}"] + ([f"+poke_term={term:08x}"] if term is not None else [])
        got = S.run_icarus(vvpp, words, plusargs=args, tag="icp")
        exp = ref.dot_ref_int(wc, ws, xc, xs, acc_init_bits=init, inject_term_bits=term)
        with np.errstate(over="ignore", invalid="ignore"):
            f0 = np.array([init], np.uint32).view(np.float32)[0]
            if term is not None:
                f0 = np.float32(f0 + np.array([term], np.uint32).view(np.float32)[0])
        exp_np = int(ref.dot_ref(wc, ws, xc, xs, acc_init=f0)) if nan_at is None else int(ref.QNAN_BITS)
        cases.append({"case": what, "accumulator_preload": f"{init:08x}", "injected_term": None if term is None else f"{term:08x}",
                      "then": "11 terms of " + ("-" if sgn else "+") + "1.156e8" + (", NaN scale in block 5" if nan_at is not None else ""),
                      "rtl": f"{int(got[0, 0]):08x}", "integer_model": f"{exp:08x}", "numpy_float32": f"{exp_np:08x}",
                      "match": bool(int(got[0, 0]) == exp == exp_np)})
    out["overflow_cases"] = cases
    out["pass"] = out["reloadable_mismatches"] == 0 and out["fixed_mismatches"] == 0 and all(c["match"] for c in cases)
    return out


def check_formats() -> dict:
    """Byte layout and scale formula, from the checkpoint bytes, torch and TokenSpeed's reference."""
    import torch
    z = np.load(ROOT / EXPERT)
    aux = np.load(FIX / "inkling2_small_aux.npz")
    out: dict = {}
    b = np.arange(256, dtype=np.uint8)
    tv = torch.from_numpy(b.copy()).view(torch.float8_e4m3fn).float().numpy()
    mv = ref.e4m3_decode(b)
    out["e4m3_decode_all_256_bytes_equal_torch"] = bool(
        np.array_equal(np.isnan(tv), np.isnan(mv)) and np.array_equal(tv[~np.isnan(tv)].view(np.uint32), mv[~np.isnan(mv)].view(np.uint32)))
    out["e4m3_nan_bytes"] = [f"0x{int(v):02x}" for v in b[np.isnan(tv)]]
    out["e4m3_max_finite"] = float(np.nanmax(tv))
    out["e4m3_min_subnormal"] = float(tv[1])
    rng = np.random.default_rng(1)
    x = np.clip(np.concatenate([rng.uniform(-448, 448, 1_000_000), np.exp(rng.uniform(np.log(1e-4), np.log(448), 1_000_000)),
                                ((mv[:126].astype(np.float64) + mv[1:127]) / 2)]).astype(np.float32), -448, 448)
    out["e4m3_encode_equal_torch"] = {"values": int(x.size), "equal": bool(np.array_equal(
        torch.from_numpy(x).to(torch.float8_e4m3fn).view(torch.uint8).numpy(), ref.e4m3_encode(x)))}
    # checkpoint statistics
    lay = {}
    for leaf in ("w13", "w2"):
        w, s = z[leaf], z[leaf + "_scale"]
        c = ref.unpack_codes(w)
        mag = np.abs(ref.e2m1_decode(c)).reshape(w.shape[0], -1, 16).max(-1)
        sv = ref.e4m3_decode(s)
        hist = np.bincount(s.ravel(), minlength=256)
        lay[leaf] = {
            "code_fraction_plus_zero": float((c == 0).mean()), "code_fraction_minus_zero": float((c == 8).mean()),
            "blocks": int(mag.size), "blocks_whose_largest_code_is_6": int((mag == 6).sum()),
            "scale_byte_min": int(np.nonzero(hist)[0].min()), "scale_byte_max": int(np.nonzero(hist)[0].max()),
            "scale_value_min": float(sv.min()), "scale_value_max": float(sv.max()),
            "scale_zero_bytes": int(hist[0] + hist[128]), "scale_subnormal_bytes": int(hist[1:8].sum() + hist[129:136].sum()),
            "scale_negative_bytes": int(hist[128:].sum()), "scale_nan_bytes": int(hist[127] + hist[255]),
            "amax": float(z[leaf + "_amax"]), "outer_scale_denominator": float(z[leaf + "_denominator"]),
            "max_abs_code_times_block_scale": float((mag * sv).max()),
            "max_abs_weight_by_amax_over_denominator": float(np.float32((mag * sv).max()) * ref.weight_outer_scale(z[leaf + "_amax"], z[leaf + "_denominator"])),
        }
        lay[leaf]["formula_reproduces_amax"] = bool(abs(lay[leaf]["max_abs_weight_by_amax_over_denominator"] / lay[leaf]["amax"] - 1) < 1e-6)
    out["checkpoint_layer0_expert0"] = lay
    # nibble order from the data: per-input-column RMS of w13 against the real mlp_norm.weight
    def deq(w, s, lo_first):
        lo, hi = w & 15, w >> 4
        c = np.stack((lo, hi) if lo_first else (hi, lo), -1).reshape(w.shape[0], -1)
        return (ref.e2m1_decode(c) * np.repeat(ref.e4m3_decode(s), 16, 1)).astype(np.float64)
    g = np.abs(ref.bf16_to_f32(aux["mlp_norm_weight_bf16"]).astype(np.float64))
    nib = {}
    w13d = {o: deq(z["w13"], z["w13_scale"], o) for o in (True, False)}
    for o in (True, False):
        nib[f"w13_column_rms_vs_mlp_norm_weight_corr_{'low' if o else 'high'}_nibble_first"] = float(
            np.corrcoef(np.sqrt((w13d[o] ** 2).mean(0)), g)[0, 1])
    u = w13d[True] / np.sqrt((w13d[True] ** 2).sum(1))[:, None]
    for o in (True, False):
        w2d = deq(z["w2"], z["w2_scale"], o)
        v = w2d / np.sqrt((w2d ** 2).sum(0))[None]
        cgate = np.einsum("ki,ik->i", v, u[0::2])
        nib[f"w2_column_vs_w13_gate_row_fraction_abs_cos_over_3sigma_{'low' if o else 'high'}_nibble_first"] = float(
            (np.abs(cgate) > 3 / np.sqrt(4608)).mean())
    nib["null_expectation_correlation_sigma"] = float(1 / np.sqrt(4608))
    nib["null_expectation_fraction_over_3sigma"] = 0.0027
    nib["low_nibble_first_supported"] = bool(
        nib["w13_column_rms_vs_mlp_norm_weight_corr_low_nibble_first"] > 0.3
        and abs(nib["w13_column_rms_vs_mlp_norm_weight_corr_high_nibble_first"]) < 0.06
        and nib["w2_column_vs_w13_gate_row_fraction_abs_cos_over_3sigma_low_nibble_first"] > 0.02
        and nib["w2_column_vs_w13_gate_row_fraction_abs_cos_over_3sigma_high_nibble_first"] < 0.01)
    out["nibble_order_from_data"] = nib
    # TokenSpeed's reference quantizer and decoder, executed from its own source
    ts = {"available": False}
    try:
        src = subprocess.run(["git", "-C", str(TOKENSPEED), "show",
                              "HEAD:tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/block_scaled.py"],
                             capture_output=True, text=True, check=True).stdout
        ns = {"torch": torch}
        for node in ast.parse(src).body:
            if isinstance(node, ast.FunctionDef) and node.name in ("_fp4_round", "reference_block_quantize", "_decode_values"):
                exec(compile(ast.Module([node], []), "tokenspeed", "exec"), ns)  # noqa: S102
        commit = subprocess.run(["git", "-C", str(TOKENSPEED), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        ts = {"available": True, "commit": commit, "file": "tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/block_scaled.py", "sets": {}}
        sets = {"embedding-derived": embedding_activations(), "gaussian": synthetic_activations(rng, 256, 4608, "gaussian"),
                "heavy-tailed": synthetic_activations(rng, 256, 4608, "heavy"), "sparse": synthetic_activations(rng, 256, 4608, "sparse"),
                "zero and extreme rows": ref.f32_to_bf16(np.concatenate([np.zeros((2, 4608)), rng.normal(0, 1e-30, (4, 4608)),
                                                                         rng.normal(0, 1e30, (4, 4608))]).astype(np.float32))}
        ok = True
        for name, xx in sets.items():
            pk, sc, am = ns["reference_block_quantize"](torch.from_numpy(xx.copy()).to(torch.bfloat16), "nvfp4")
            mp, ms, ma = ref.quantize_activations(xx)
            dec = ns["_decode_values"](pk, sc, "nvfp4").numpy()
            same = bool(np.array_equal(pk.numpy(), mp) and np.array_equal(sc.view(torch.uint8).numpy(), ms)
                        and np.array_equal(am.numpy(), ma) and np.array_equal(dec, ref.dequantize(mp, ms)))
            ts["sets"][name] = {"tokens": int(xx.shape[0]), "codes_scales_amax_and_decode_identical": same}
            ok &= same
        ts["pass"] = ok
    except Exception as e:  # noqa: BLE001
        ts["error"] = str(e)
    out["tokenspeed_reference"] = ts
    out["pass"] = bool(out["e4m3_decode_all_256_bytes_equal_torch"] and out["e4m3_encode_equal_torch"]["equal"]
                       and all(v["formula_reproduces_amax"] for v in lay.values()) and nib["low_nibble_first_supported"]
                       and ts.get("pass", True))
    return out


def check_torch() -> dict:
    """The reference against torch.nn.functional.linear on a whole real expert (both projections)."""
    import torch
    torch.set_num_threads(4)
    rng = np.random.default_rng(5)
    cfg = json.loads((ROOT / "data" / "inkling2_small" / "config.json").read_text())["text_config"]
    z = np.load(ROOT / EXPERT)
    x_emb = embedding_activations()
    sets = [("w13 (4608 x 4608) x 512 embedding-derived tokens", "w13", x_emb),
            ("w13 (4608 x 4608) x 128 heavy-tailed synthetic tokens", "w13", synthetic_activations(rng, 128, 4608, "heavy")),
            ("w2 (4608 x 2304) x 256 SwiGLU outputs of the real w13", "w2", swiglu_activations(x_emb[:256], cfg))]
    tol = 1e-6
    out = {"expert": "layer 0 expert 0", "torch": torch.__version__, "threads": 4,
           "tolerance_error_over_sum_of_abs_terms": tol, "sets": {}}

    def stats(a, b, l1):
        d = np.abs(a - b)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(b != 0, d / np.abs(b), 0.0)
        return {"worst_relative_error": float(rel.max()), "median_relative_error": float(np.median(rel)),
                "p999_relative_error": float(np.quantile(rel, 0.999)),
                "worst_error_over_sum_of_abs_terms": float((d / np.maximum(l1, 1e-300)).max()),
                "bit_identical_fraction": float((a.astype(np.float32).view(np.uint32) == b.astype(np.float32).view(np.uint32)).mean())}

    ok = True
    for name, leaf, x in sets:
        xp, xs, amax = ref.quantize_activations(x)
        bits = ref.gemm_ref(codes_of(blocks(z[leaf])), z[leaf + "_scale"], codes_of(blocks(xp)), xs)   # [T, R]
        eng = bits.view(np.float32).astype(np.float64)
        wd, xd = ref.dequantize(z[leaf], z[leaf + "_scale"]), ref.dequantize(xp, xs)   # code * block scale: exact in float32
        wt, xt = torch.from_numpy(wd), torch.from_numpy(xd)
        tl = torch.nn.functional.linear(xt, wt).numpy()
        exact = xd.astype(np.float64) @ wd.astype(np.float64).T        # exact: multiples of 2^-20 far below 2^53
        l1 = np.abs(xd).astype(np.float64) @ np.abs(wd).astype(np.float64).T
        r = {"outputs": int(eng.size),
             "engine_vs_torch_linear": stats(eng, tl.astype(np.float64), l1),
             "engine_vs_exact_value": stats(eng, exact, l1),
             "torch_linear_vs_exact_value": stats(tl.astype(np.float64), exact, l1),
             "outputs_exactly_representable_in_binary32": float((exact.astype(np.float32).astype(np.float64) == exact).mean()),
             "activation_block_scales_below_8_fraction": float((ref.e4m3_decode(xs) < 8).mean())}
        # with the outer scales, as the engines use the result
        outer = (amax * z[leaf + "_amax"]) * np.float32(1.0 / (np.float32(ref.ACT_DENOMINATOR) * z[leaf + "_denominator"]))
        y_eng = (bits.view(np.float32) * outer[:, None]).astype(np.float64)
        wfull = ref.dequantize(z[leaf], z[leaf + "_scale"], ref.weight_outer_scale(z[leaf + "_amax"], z[leaf + "_denominator"]))
        xfull = (xd * (amax * np.float32(1.0 / ref.ACT_DENOMINATOR))[:, None]).astype(np.float32)
        y_t = torch.nn.functional.linear(torch.from_numpy(xfull), torch.from_numpy(wfull)).numpy().astype(np.float64)
        l1f = np.abs(xfull).astype(np.float64) @ np.abs(wfull).astype(np.float64).T
        r["with_outer_scales_vs_torch_linear_on_dequantized_floats"] = stats(y_eng, y_t, l1f)
        # torch on CPU is not guaranteed batch invariant: the same tokens alone and inside the batch
        nb1 = min(16, x.shape[0])
        one = np.concatenate([torch.nn.functional.linear(xt[i:i + 1], wt).numpy() for i in range(nb1)])
        vec = np.stack([torch.mv(wt, xt[i]).numpy() for i in range(nb1)])
        r["torch_batch_of_1_vs_full_batch_bit_identical_fraction"] = float((one.view(np.uint32) == tl[:nb1].view(np.uint32)).mean())
        r["torch_mv_vs_full_batch_bit_identical_fraction"] = float((vec.view(np.uint32) == tl[:nb1].view(np.uint32)).mean())
        out["sets"][name] = r
        ok &= r["engine_vs_torch_linear"]["worst_error_over_sum_of_abs_terms"] < tol
        print(f"   {name}: engine==torch {r['engine_vs_torch_linear']['bit_identical_fraction']:.4f}, "
              f"worst rel {r['engine_vs_torch_linear']['worst_relative_error']:.3g}, "
              f"worst err/L1 {r['engine_vs_torch_linear']['worst_error_over_sum_of_abs_terms']:.3g}", flush=True)
    # a case where rounding does occur: random codes and random finite block scales
    wc, ws = random_codes(rng, 512, 288, "finite")
    xc, xs = random_codes(rng, 64, 288, "finite")
    bits = ref.gemm_ref(codes_of(wc), ws, codes_of(xc), xs)
    wd = ref.dequantize(wc.reshape(512, -1), ws); xd = ref.dequantize(xc.reshape(64, -1), xs)
    wt, xt = torch.from_numpy(wd), torch.from_numpy(xd)
    tl = torch.nn.functional.linear(xt, wt).numpy()
    exact = xd.astype(np.float64) @ wd.astype(np.float64).T
    l1 = np.abs(xd).astype(np.float64) @ np.abs(wd).astype(np.float64).T
    one = np.concatenate([torch.nn.functional.linear(xt[i:i + 1], wt).numpy() for i in range(64)])
    r = {"outputs": int(bits.size), "engine_vs_torch_linear": stats(bits.view(np.float32).astype(np.float64), tl.astype(np.float64), l1),
         "engine_vs_exact_value": stats(bits.view(np.float32).astype(np.float64), exact, l1),
         "torch_linear_vs_exact_value": stats(tl.astype(np.float64), exact, l1),
         "outputs_exactly_representable_in_binary32": float((exact.astype(np.float32).astype(np.float64) == exact).mean()),
         "torch_batch_of_1_vs_full_batch_bit_identical_fraction": float((one.view(np.uint32) == tl.view(np.uint32)).mean())}
    out["sets"]["random codes and random finite scales, 512 rows x 64 tokens (rounding occurs)"] = r
    ok &= r["engine_vs_torch_linear"]["worst_error_over_sum_of_abs_terms"] < tol
    out["pass"] = bool(ok)
    return out


def check_batch_invariance() -> dict:
    """The same token gives the same bits whatever the batch size and its position in the batch."""
    rng = np.random.default_rng(99)
    z = np.load(ROOT / EXPERT)
    w13, w13s = blocks(z["w13"]), z["w13_scale"]
    xe_p, xe_s, _ = ref.quantize_activations(embedding_activations())
    xe = blocks(xe_p)
    probes = [(int(r), int(x)) for r, x in zip(rng.integers(0, 4608, 48), rng.integers(0, 512, 48))]
    exp = np.array([ref.dot_ref(codes_of(w13[r]), w13s[r], codes_of(xe[x]), xe_s[x]) for r, x in probes], np.uint32)
    rel = S.build_verilator("nvfp4_dot", S.gen_rtl("nvfp4_dot"), 1, False)
    wide_spec = dict(fixture=EXPERT, leaf="w13", rows=WIDE_ROWS)
    wide = S.build_verilator("nvfp4_dot_const_x8", S.gen_rtl("nvfp4_dot_const_x8", const=wide_spec), 8, True)
    exp_w = ref.gemm_ref(codes_of(w13[WIDE_ROWS]), w13s[WIDE_ROWS], codes_of(xe), xe_s)   # [512, 8]
    out = {"probes": len(probes), "placements": [], "distinct_results_per_probe_max": 0}
    seen = [set() for _ in probes]
    seen_w = [set() for _ in range(48)]
    n_place = 0
    for bsz in (1, 2, 3, 8, 32, 128):
        for pos in sorted({0, bsz // 2, bsz - 1}):
            for gaps in (0, 1):
                # reloadable: a batch = bsz tokens against one weight row, back to back
                sched, where = [], []
                for (r, x) in probes:
                    others = rng.integers(0, 512, bsz)
                    for k in range(bsz):
                        if k == pos:
                            where.append(len(sched))
                        sched.append((r, x if k == pos else int(others[k]), (rng.integers(1, 1 << 30) if gaps else 0)))
                got, _ = S.run_job(rel, 288, w13, w13s, xe, xe_s, np.array(sched), tag="bi")
                pr = got[where, 0]
                for i, v in enumerate(pr):
                    seen[i].add(int(v))
                # fixed, 8 rows: a batch = bsz tokens streamed through the same hard-wired rows
                sched, where = [], []
                for i in range(48):
                    others = rng.integers(0, 512, bsz)
                    for k in range(bsz):
                        if k == pos:
                            where.append(len(sched))
                        sched.append((0, i if k == pos else int(others[k]), (rng.integers(1, 1 << 30) if gaps else 0)))
                gw, _ = S.run_job(wide, 288, None, None, xe, xe_s, np.array(sched), 8, tag="biw")
                for i in range(48):
                    seen_w[i].add(tuple(int(v) for v in gw[where[i]]))
                n_place += 1
                out["placements"].append({"batch_size": bsz, "position": pos, "idle_gaps": bool(gaps),
                                          "reloadable_probe_mismatches_vs_reference": int((pr != exp).sum()),
                                          "fixed_x8_probe_mismatches_vs_reference": int(sum(
                                              tuple(int(v) for v in exp_w[i]) != tuple(int(v) for v in gw[where[i]]) for i in range(48)))})
    out["placements_per_probe"] = n_place
    out["distinct_results_per_probe_max"] = max(max(len(s) for s in seen), max(len(s) for s in seen_w))
    out["reloadable_probe_results"] = len(probes) * n_place
    out["fixed_x8_probe_results"] = 48 * 8 * n_place
    out["pass"] = bool(out["distinct_results_per_probe_max"] == 1 and all(
        p["reloadable_probe_mismatches_vs_reference"] == 0 and p["fixed_x8_probe_mismatches_vs_reference"] == 0 for p in out["placements"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["vectors", "icarus", "formats", "torch", "batch"])
    a = ap.parse_args()
    S.BUILD.mkdir(parents=True, exist_ok=True)
    path = S.BUILD / ("verify_quick.json" if a.quick else "verify.json")
    res = json.loads(path.read_text()) if path.exists() and a.only else {}
    steps = {"formats": check_formats, "vectors": lambda: verify(a.quick), "icarus": check_icarus,
             "batch": check_batch_invariance, "torch": check_torch}
    for name, fn in steps.items():
        if a.only in (None, name):
            t0 = time.time()
            res[name] = fn()
            print(f"== {name}: {'PASS' if res[name].get('pass') else 'FAIL'} ({time.time() - t0:.0f} s)", flush=True)
            path.write_text(json.dumps(res, indent=1) + "\n")
    print("written", path)


if __name__ == "__main__":
    main()
