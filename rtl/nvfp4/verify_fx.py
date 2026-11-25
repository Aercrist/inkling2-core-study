#!/usr/bin/env python3
"""Verification of the exact-sum variant nvfp4_dot_fx (SPEC.md section 5).

    python rtl/nvfp4/verify_fx.py            # everything (about 1.2 M dot products, a few minutes)
    python rtl/nvfp4/verify_fx.py --quick    # a small subset, seconds
    python rtl/nvfp4/verify_fx.py --only vectors|icarus|definitions|mutation

Writes rtl/nvfp4/build/verify_fx.json (merged into results/silicon.json by run_all.py --exact-sum).

* vectors      RTL (Verilator) against `ref.dot_fx_ref` on the vector groups verify.py uses for the
               reloadable block, plus directed cases and a stress group for the final rounding.
* icarus       the same RTL in a second simulator on a subset.
* definitions  the two definitions compared in the references, no RTL: one rounding per block
               (`ref.gemm_ref`) against one rounding per row (`ref.gemm_fx_ref`).
* mutation     two deliberate faults written into a copy of the generated RTL in the build
               directory; the checker must report mismatches, and none on the unmodified RTL.
"""
from __future__ import annotations

import argparse
import bisect
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import simlib as S   # noqa: E402
import verify as V   # noqa: E402

ref, P, ROOT = S.ref, S.P, S.ROOT
TOP = "nvfp4_dot_fx"
FXW = S.gen.derive_fx(P)
SAT_TERM = 2304 * 14 * 14 << 28          # all codes +-6, both scales 448: the largest finite term, 2^46.8 units


# ------------------------------------------------------------------------------- rows with a chosen exact sum
_REP = {d: V.block_for(d) for d in range(1, 2305)}
_REP = {d: b for d, b in _REP.items() if b is not None}       # block sums 16 lanes can produce (greedy)
_REP_SORTED = sorted(_REP)


def pow2_scale(a: int) -> int:
    """E4M3 byte whose value is 2^(a-9), i.e. M * 2^(E-1) = 2^a: subnormal for a <= 2, M = 8 above."""
    assert 0 <= a <= 17
    return (1 << a) if a <= 2 else ((a - 2) << 3)


def term_block(d: int, k: int, rng):
    """One block whose term is exactly d * 2^k units of 2^-20 (|d| a producible block sum, k in 0..34)."""
    w, x = V.block_for(abs(d))
    aw = int(rng.integers(max(0, k - 17), min(17, k) + 1))
    ws, xs = pow2_scale(aw), pow2_scale(k - aw)
    if d < 0:                                                 # sign in the weight codes, or in one of the scales
        how = int(rng.integers(0, 3))
        if how == 0:
            w = np.where(w & 7, w | 8, w).astype(np.uint8)
        elif how == 1:
            ws |= 0x80
        else:
            xs |= 0x80
    return w, ws, x, xs


def blocks_for_sum(s: int, rng) -> list:
    """Blocks whose terms add up to exactly s units of 2^-20 (|s| < 2^56), in random order."""
    sg, m, out = (-1 if s < 0 else 1), abs(s), []
    assert m < ref.FX_SUM_BOUND
    if m >= 1 << 46:                                          # bulk: saturated blocks with scales 448 x 448
        n = m // SAT_TERM
        m -= n * SAT_TERM
        six = np.full(16, 7, np.uint8)
        out += [(six | (8 if sg < 0 else 0), 0x7E, six, 0x7E)] * n
    hi = m >> 34
    while hi:                                                 # multiples of 2^34
        d = _REP_SORTED[bisect.bisect_right(_REP_SORTED, min(hi, 2304)) - 1]
        out.append(term_block(sg * d, 34, rng))
        hi -= d
    lo = m & ((1 << 34) - 1)
    for k in (0, 8, 16, 24, 32):                              # 8-bit digits of the low 34 bits
        d = (lo >> k) & 0xFF
        if d:
            out.append(term_block(sg * d, k, rng))
    assert len(out) <= ref.FX_MAX_BLOCKS, (s, len(out))
    return [out[i] for i in rng.permutation(len(out))]


def pad_rows(rows: list, nb: int, rng):
    """Rows (lists of blocks) -> arrays [n, nb, 16], [n, nb], ...; padded with zero-term blocks at random places."""
    n = len(rows)
    wc = np.zeros((n, nb, 16), np.uint8); xc = np.zeros((n, nb, 16), np.uint8)
    ws = np.zeros((n, nb), np.uint8); xs = np.zeros((n, nb), np.uint8)
    for i, blks in enumerate(rows):
        assert len(blks) <= nb
        pos = np.sort(rng.permutation(nb)[: len(blks)])
        fill = np.ones(nb, bool); fill[pos] = False
        for j, b in zip(pos, blks):
            wc[i, j], ws[i, j], xc[i, j], xs[i, j] = b
        k = int(fill.sum())
        if k:                                                 # a zero term: zero codes, or a zero scale under any codes
            kind = rng.integers(0, 3, k)
            wc[i, fill] = np.where((kind == 0)[:, None], 0, rng.integers(0, 16, (k, 16))).astype(np.uint8)
            xc[i, fill] = rng.integers(0, 16, (k, 16)).astype(np.uint8)
            ws[i, fill] = np.where(kind == 1, rng.choice([0x00, 0x80], k), rng.integers(0, 127, k)).astype(np.uint8)
            xs[i, fill] = np.where(kind == 2, rng.choice([0x00, 0x80], k), rng.integers(0, 127, k)).astype(np.uint8)
    return wc, ws, xc, xs


def directed_fx(rng) -> list:
    """Named cases for the final rounding and the ends of the accumulator: [(name, [blocks])]."""
    out = []
    even, odd, ones = 0xA5A5A4, 0xA5A5A5, 0xFFFFFF
    for sh in (1, 2, 7, 13, 19, 25, 31):                      # sums of 25 to 55 bits
        half = 1 << (sh - 1)
        for sg in (1, -1):
            for name, s in (
                    ("tie, rounded down to even", (even << sh) | half),
                    ("tie, rounded up to even", (odd << sh) | half),
                    ("tie at an all-ones significand, carries into the exponent", (ones << sh) | half),
                    ("tie at significand 2^24 - 2, rounded down, no carry", ((ones - 1) << sh) | half),
                    ("one unit below a tie, rounded down", (odd << sh) | (half - 1) if sh > 1 else None),
                    ("one unit above a tie on an even significand, rounded up (sticky far below)", (even << sh) | half | 1 if sh > 1 else None),
                    ("only the lowest unit set below the significand, rounded down (sticky far below)", (odd << sh) | 1 if sh > 1 else None),
                    ("all ones (2^k - 1), rounded up into the next power of two", (1 << (24 + sh)) - 1),
                    ("exact: low bits zero", odd << sh)):
                if s is not None:
                    out.append((f"final rounding: {name}", blocks_for_sum(sg * s, rng)))
    six, m6 = np.full(16, 7, np.uint8), np.full(16, 15, np.uint8)
    sat_p, sat_n = (six, 0x7E, six, 0x7E), (m6, 0x7E, six, 0x7E)
    for nb in (1, 2, 288, 512):
        out.append((f"maximum magnitude row ({nb} blocks, all codes +-6, scales 448)", [sat_p] * nb))
        out.append((f"maximum magnitude row ({nb} blocks, all codes +-6, scales 448)", [sat_n] * nb))
    for nb in (2, 16, 512):                                   # largest terms, sum exactly zero
        out.append(("exact cancellation to +0", [sat_p, sat_n] * (nb // 2)))
        out.append(("exact cancellation to +0", [sat_n] * (nb // 2) + [sat_p] * (nb // 2)))
        out.append(("exact cancellation to +0", [sat_p] * (nb // 2) + [sat_n] * (nb // 2)))
    for s in (1, 3, (1 << 24) + 1, (1 << 40) - 1):            # a = b + c + ... with opposite signs
        a = blocks_for_sum(s, rng)
        out.append(("exact cancellation to +0", a + blocks_for_sum(-s, rng)))
        out.append(("exact cancellation to +0", blocks_for_sum(-s, rng) + a))
    for big in (SAT_TERM, 9 << 42, 200 * SAT_TERM):           # large terms cancel and leave a few units
        for res in (1, -1, 3, (1 << 24) + 1, -((1 << 25) + 1)):
            out.append(("cancellation of large terms down to a small sum", blocks_for_sum(big, rng) + blocks_for_sum(res - big, rng)))
    # rows of one block: every alignment shift with the largest product, either sign
    for es in range(2, 31):
        ew = min(15, es - 1); ex = es - ew
        for sgn in (0, 8):
            mk = lambda e: (e << 3) | (7 if e < 15 else 6)    # noqa: E731  M = 15, or 14 at the top exponent (0x7F is NaN)
            out.append(("row of 1 block, largest product at each of the 29 alignment shifts", [(six | sgn, mk(ew), six, mk(ex))]))
    out.append(("row of 1 block, smallest non-zero term (2^-20)", [term_block(1, 0, rng)]))
    out.append(("row of 1 block, smallest non-zero term (2^-20)", [term_block(-1, 0, rng)]))
    return out


def final_rounding_stress(rng, n: int, long_rows: bool) -> list:
    """Random rows whose exact sum has a chosen leading bit and a chosen pattern below the 24-bit
    significand: ties, one unit either side of a tie, all ones, sticky only far below, exact."""
    rows = []
    for _ in range(n):
        h = int(rng.integers(46, 56)) if long_rows else int(rng.integers(24, 46))      # leading bit of |sum|
        sh = h - 23
        mant = (1 << 23) | int(rng.integers(0, 1 << 23))
        mode = int(rng.integers(0, 4))
        if mode == 0:
            mant = (1 << 24) - 1 - int(rng.integers(0, 2))    # all ones, or all ones but the last bit
        elif mode == 1:
            mant = (1 << 23) + int(rng.integers(0, 2))
        if h == 55:
            mant &= 0xBFFFFF                                  # 512 saturated blocks reach 2^55.8: keep the sum below 1.5 * 2^55
        half = 1 << (sh - 1)
        kind = int(rng.integers(0, 8))
        low = (half, half, half - 1, half + 1, 1, (1 << sh) - 1, 0, int(rng.integers(0, 1 << sh)))[kind]
        s = (mant << sh) | (low & ((1 << sh) - 1))
        assert s <= 496 * SAT_TERM                            # so that the row needs at most 512 blocks
        blks = blocks_for_sum(-s if rng.random() < 0.5 else s, rng)
        if not long_rows and rng.random() < 0.5:              # a pair of large terms that cancels
            c = int(rng.integers(1, 2305)); c = _REP_SORTED[bisect.bisect_right(_REP_SORTED, c) - 1]
            k = int(rng.integers(20, 35))
            blks += [term_block(c, k, rng), term_block(-c, k, rng)]
            blks = [blks[i] for i in rng.permutation(len(blks))]
        rows.append(blks)
    return rows


# ------------------------------------------------------------------------------- the checker
def run_groups(exe, quick: bool, seed: int = 20261125, real_scale: int | None = None, cover: dict | None = None) -> V.Tally:
    """Drive the Verilator model `exe` of nvfp4_dot_fx with every vector group and compare with the
    reference. `real_scale` shrinks only the real-row and random groups (the mutation check)."""
    rng = np.random.default_rng(seed)
    t = V.Tally()
    cfg = json.loads((ROOT / "data" / "inkling2_small" / "config.json").read_text())["text_config"]
    z = np.load(ROOT / V.EXPERT)
    zr = np.load(ROOT / V.ROWS_NPZ)
    w13, w13s = V.blocks(z["w13"]), z["w13_scale"]
    w2, w2s = V.blocks(z["w2"]), z["w2_scale"]
    nb13, nb2 = P["nblk"]["w13"], P["nblk"]["w2"]
    sc = real_scale or (16 if quick else 1)
    sd = 16 if quick else 1                                   # the directed and stress groups shrink only with --quick

    def scalar_check(wc, ws, xc, xs, exp, count):
        for i in range(min(len(exp), count)):
            assert ref.dot_fx_ref_int(wc[i], ws[i], xc[i], xs[i]) == int(exp[i])

    def classify(wc, ws, xc, xs):
        if cover is None:
            return
        terms, nan = ref.fx_terms(wc, ws, xc, xs)
        for s_, n_, live in zip(terms.sum(-1).tolist(), nan.tolist(), (terms != 0).any(-1).tolist()):
            for key in (["NaN scale in the row"] if n_ else ["exact cancellation to +0"] if s_ == 0 and live else ref.fx_round_case(s_)):
                cover[key] = cover.get(key, 0) + 1

    # ---------------- real rows (the groups of verify.py for the reloadable block)
    x_emb = V.embedding_activations(512 // sc)
    xe_p, xe_s, _ = ref.quantize_activations(x_emb)
    x_syn = np.concatenate([V.synthetic_activations(rng, 96 // sc + 1, 4608, k) for k in ("gaussian", "heavy", "sparse")])
    xs_p, xs_s, _ = ref.quantize_activations(x_syn)
    x_sw = V.swiglu_activations(x_emb[: 256 // sc], cfg)
    xw_p, xw_s, _ = ref.quantize_activations(x_sw)
    rows = np.arange(0, 4608, 4608 // (1024 // sc))[: 1024 // sc] + (np.arange(1024 // sc) % 2)
    exp = ref.gemm_fx_ref(V.codes_of(w13[rows]), w13s[rows], V.codes_of(V.blocks(xe_p)), xe_s).T
    got, cyc = V.pairs_job(exe, nb13, w13[rows], w13s[rows], V.blocks(xe_p), xe_s, rng, tag="f1")
    t.add(TOP, "real w13 rows (L0 E0) x embedding-derived tokens", got[..., 0], exp, cyc,
          f"{len(rows)} rows x {x_emb.shape[0]} tokens, 288 blocks, 10% of vectors with idle gaps")
    rows2 = np.arange(0, 4608, 4608 // (512 // sc))[: 512 // sc]
    exp = ref.gemm_fx_ref(V.codes_of(w2[rows2]), w2s[rows2], V.codes_of(V.blocks(xw_p)), xw_s).T
    got, cyc = V.pairs_job(exe, nb2, w2[rows2], w2s[rows2], V.blocks(xw_p), xw_s, rng, tag="f2")
    t.add(TOP, "real w2 rows (L0 E0) x SwiGLU outputs of the real w13", got[..., 0], exp, cyc,
          f"{len(rows2)} rows x {x_sw.shape[0]} tokens, 144 blocks")
    for L, E in zr["samples"]:
        tag = f"L{L:02d}_E{E:03d}"
        wr, wrs = V.blocks(zr[f"{tag}_w13"]), zr[f"{tag}_w13_scale"]
        if real_scale:
            wr, wrs = wr[:8], wrs[:8]
        xa = np.concatenate([V.blocks(xs_p), V.blocks(xe_p)[: 64 // sc]])
        xas = np.concatenate([xs_s, xe_s[: 64 // sc]])
        exp = ref.gemm_fx_ref(V.codes_of(wr), wrs, V.codes_of(xa), xas).T
        got, cyc = V.pairs_job(exe, nb13, wr, wrs, xa, xas, rng, tag="f3")
        t.add(TOP, f"real w13 rows, layer {L} expert {E} x mixed tokens", got[..., 0], exp, cyc,
              f"{wr.shape[0]} rows x {xa.shape[0]} tokens (gaussian, heavy-tailed, sparse, embedding)")
        wr, wrs = V.blocks(zr[f"{tag}_w2"]), zr[f"{tag}_w2_scale"]
        if real_scale:
            wr, wrs = wr[:8], wrs[:8]
        xr = V.synthetic_activations(rng, 96 // sc + 1, 2304, "relu")
        xr_p, xr_s, _ = ref.quantize_activations(xr)
        xa, xas = np.concatenate([V.blocks(xr_p), V.blocks(xw_p)[: 64 // sc]]), np.concatenate([xr_s, xw_s[: 64 // sc]])
        exp = ref.gemm_fx_ref(V.codes_of(wr), wrs, V.codes_of(xa), xas).T
        got, cyc = V.pairs_job(exe, nb2, wr, wrs, xa, xas, rng, tag="f3")
        t.add(TOP, f"real w2 rows, layer {L} expert {E} x mixed tokens", got[..., 0], exp, cyc,
              f"{wr.shape[0]} rows x {xa.shape[0]} tokens")
    # ---------------- random codes and scale bytes, all row lengths (verify.py's list, plus rows of 512 blocks)
    for nb, n, kind, pn in ((1, 40000, "any", 0), (2, 30000, "any", 0), (3, 30000, "finite", 0.01), (5, 30000, "finite", 0.01),
                            (16, 30000, "finite", 0.002), (31, 20000, "finite", 0.001), (144, 12000, "finite", 2e-4),
                            (288, 24000, "finite", 1e-4), (288, 12000, "real", 0), (512, 4000, "finite", 1e-4)):
        n //= sc
        wc, ws = V.random_codes(rng, n, nb, kind, pn)
        xc, xs = V.random_codes(rng, n, nb, kind, pn)
        sched = np.stack([np.arange(n), np.arange(n), np.where(rng.random(n) < 0.2, rng.integers(1, 1 << 30, n), 0)], 1)
        got, cyc = S.run_job(exe, nb, wc, ws, xc, xs, sched, tag="f4")
        exp = ref.dot_fx_ref(V.codes_of(wc), ws, V.codes_of(xc), xs)
        scalar_check(V.codes_of(wc), ws, V.codes_of(xc), xs, exp, 200 if nb <= 31 else 20)
        t.add(TOP, f"random codes, {kind} scale bytes, rows of {nb} block(s)", got[:, 0], exp, cyc,
              "uniform 4-bit codes (so +0 and -0 codes too); 'any' = all 256 scale bytes including NaN; "
              "'finite' = all 254 finite bytes (zeros, subnormals, negatives) with NaN injected at the stated rate; "
              "20% of vectors with idle gaps")
    # ---------------- directed: verify.py's edge cases, then the cases for the final rounding
    by_name: dict = {}

    def directed(cases, prefix, tag):
        for nb in sorted({len(c[1]) if isinstance(c[1], list) else c[1].shape[0] for c in cases}):
            grp = [c for c in cases if (len(c[1]) if isinstance(c[1], list) else c[1].shape[0]) == nb]
            if isinstance(grp[0][1], list):
                wc, ws, xc, xs = (np.stack([np.stack([b[k] for b in c[1]]) for c in grp]).astype(np.uint8) for k in range(4))
            else:
                wc, ws = np.stack([v[1] for v in grp]), np.stack([v[2] for v in grp])
                xc, xs = np.stack([v[3] for v in grp]), np.stack([v[4] for v in grp])
            exp = ref.dot_fx_ref(wc, ws, xc, xs)
            scalar_check(wc, ws, xc, xs, exp, len(grp) if nb < 100 else 12)
            n = len(grp)
            for rep, gap in ((0, 0), (1, 1)):   # back to back, then with idle gaps
                sched = np.stack([np.arange(n), np.arange(n), np.full(n, gap * (rep + 7))], 1)
                got, _ = S.run_job(exe, nb, ref.pack_codes(wc), ws, ref.pack_codes(xc), xs, sched, tag=tag)
                classify(wc, ws, xc, xs)
                for i, v in enumerate(grp):
                    s = by_name.setdefault(prefix + v[0], [0, 0])
                    s[0] += 1; s[1] += int(got[i, 0] != exp[i])

    directed(V.directed_vectors(rng), "directed: ", "f5")
    directed(directed_fx(rng), "directed, exact sum: ", "f6")
    for name, (n, bad) in by_name.items():
        t.sections.append({"dut": TOP, "name": name, "vectors": n, "mismatches": bad,
                           "clock_cycles": 0, "note": "each vector run back to back and again with idle gaps"})
        print(f"          {name:100s} {n:6d} vectors, {bad} mismatches")
    # ---------------- stress: verify.py's per-term patterns, then sums built to hit the final rounding
    for nb, n in ((4, 20000), (12, 20000), (40, 10000)):
        n //= sd
        wc, ws, xc, xs = V.rounding_stress(rng, n, nb)
        exp = ref.dot_fx_ref(wc, ws, xc, xs)
        scalar_check(wc, ws, xc, xs, exp, 300)
        classify(wc, ws, xc, xs)
        sched = np.stack([np.arange(n), np.arange(n), np.zeros(n)], 1)
        got, cyc = S.run_job(exe, nb, ref.pack_codes(wc), ws, ref.pack_codes(xc), xs, sched, tag="f7")
        t.add(TOP, f"rounding stress (the per-term patterns of verify.py), rows of {nb} blocks", got[:, 0], exp, cyc,
              "terms +-{1,3,2^k,2^k+-1,3*2^k,...} x 2^e")
    for nb, n, long_rows in ((12, 40000, False), (512, 4000, True)):
        n //= sd
        wc, ws, xc, xs = pad_rows(final_rounding_stress(rng, n, long_rows), nb, rng)
        exp = ref.dot_fx_ref(wc, ws, xc, xs)
        scalar_check(wc, ws, xc, xs, exp, 300 if nb < 100 else 12)
        classify(wc, ws, xc, xs)
        sched = np.stack([np.arange(n), np.arange(n), np.where(rng.random(n) < 0.2, rng.integers(1, 1 << 30, n), 0)], 1)
        got, cyc = S.run_job(exe, nb, ref.pack_codes(wc), ws, ref.pack_codes(xc), xs, sched, tag="f8")
        t.add(TOP, f"final-rounding stress, rows of {nb} blocks, leading bit of the sum {'46..55' if long_rows else '24..45'}", got[:, 0], exp, cyc,
              "sums built bit by bit: ties, one unit either side of a tie, all ones, sticky only in the lowest unit, exact; "
              "half of the short rows also carry a pair of large terms that cancels; 20% of vectors with idle gaps")
    return t


CATEGORIES = [
    ("real expert rows x real-looking tokens", lambda x: x["name"].startswith("real")),
    ("random codes and scale bytes, rows of 1 to 512 blocks", lambda x: x["name"].startswith("random")),
    ("directed edge cases of verify.py", lambda x: x["name"].startswith("directed: ")),
    ("directed cases for the exact sum and the final rounding", lambda x: x["name"].startswith("directed, exact sum: ")),
    ("rounding stress (per-term patterns of verify.py)", lambda x: x["name"].startswith("rounding stress")),
    ("final-rounding stress", lambda x: x["name"].startswith("final-rounding stress")),
]


def summarize(t: V.Tally) -> dict:
    return {name: {"vectors": sum(x["vectors"] for x in t.sections if pred(x)),
                   "mismatches": sum(x["mismatches"] for x in t.sections if pred(x))} for name, pred in CATEGORIES}


def verify(quick: bool = False) -> dict:
    exe = S.build_verilator(TOP, S.gen_rtl(TOP, fx=True), 1, False)
    cover: dict = {}
    t = run_groups(exe, quick, cover=cover)
    total = sum(s["vectors"] for s in t.sections)
    bad = sum(s["mismatches"] for s in t.sections)
    res = {"pass": bad == 0, "dut": TOP, "vectors_total": total, "mismatches_total": bad,
           "clock_cycles_total": sum(s["clock_cycles"] for s in t.sections),
           "simulator": S.tool_version(["verilator", "--version"]), "quick": quick,
           "groups": summarize(t), "sections": t.sections,
           "final_rounding_coverage_directed_and_stress": dict(sorted(cover.items())),
           "seconds": round(time.time() - t.t0, 1)}
    print(f"TOTAL {total:,} vectors, {bad} mismatches, {res['clock_cycles_total']:,} clock cycles, {res['seconds']} s")
    return res


# ------------------------------------------------------------------------------- second simulator
def check_icarus() -> dict:
    """Icarus Verilog on a subset: real rows, random rows, the short directed cases for the final rounding."""
    rng = np.random.default_rng(7)
    z = np.load(ROOT / V.EXPERT)
    w13, w13s = V.blocks(z["w13"]), z["w13_scale"]
    xe_p, xe_s, _ = ref.quantize_activations(V.embedding_activations(8))
    d = S.gen_rtl(TOP, fx=True)
    vvp = S.build_icarus(TOP, [d / "nvfp4_common.sv", d / f"{TOP}.sv", d / f"tb_{TOP}.sv"])
    pairs = [(int(r), int(x)) for r in rng.integers(0, 4608, 24) for x in range(4)]
    got = S.run_icarus(vvp, S.stream(288, w13, w13s, V.blocks(xe_p), xe_s, pairs), tag="icf")
    exp = np.array([ref.dot_fx_ref(V.codes_of(w13[r]), w13s[r], V.codes_of(V.blocks(xe_p)[x]), xe_s[x]) for r, x in pairs])
    out = {"simulator": S.tool_version(["iverilog", "-V"]),
           "real_vectors": len(pairs), "real_mismatches": int((got[:, 0] != exp).sum())}
    n_r = n_d = bad_r = bad_d = 0
    for nb in (1, 3, 16):
        n = 64
        wc, ws = V.random_codes(rng, n, nb, "finite", 0.02)
        xc, xs = V.random_codes(rng, n, nb, "finite", 0.02)
        got = S.run_icarus(vvp, S.stream(nb, wc, ws, xc, xs, [(i, i) for i in range(n)], gaps={5: 3, 9: 1}), tag="icf")
        n_r += n; bad_r += int((got[:, 0] != ref.dot_fx_ref(V.codes_of(wc), ws, V.codes_of(xc), xs)).sum())
    short = [c for c in directed_fx(rng) if len(c[1]) <= 16]
    wc, ws, xc, xs = pad_rows([c[1] for c in short], 16, rng)
    got = S.run_icarus(vvp, S.stream(16, ref.pack_codes(wc), ws, ref.pack_codes(xc), xs, [(i, i) for i in range(len(short))],
                                     gaps={2: 4}), tag="icf")
    n_d, bad_d = len(short), int((got[:, 0] != ref.dot_fx_ref(wc, ws, xc, xs)).sum())
    out.update(random_vectors=n_r, random_mismatches=bad_r, directed_final_rounding_vectors=n_d, directed_final_rounding_mismatches=bad_d)
    out["pass"] = out["real_mismatches"] == 0 and bad_r == 0 and bad_d == 0
    return out


# ------------------------------------------------------------------------------- the two definitions
def compare_definitions() -> dict:
    """One binary32 rounding per block (nvfp4_dot) against one per row (nvfp4_dot_fx), in the references."""
    rng = np.random.default_rng(5)
    cfg = json.loads((ROOT / "data" / "inkling2_small" / "config.json").read_text())["text_config"]
    z = np.load(ROOT / V.EXPERT)
    x_emb = V.embedding_activations()
    xe_p, xe_s, _ = ref.quantize_activations(x_emb)
    xw_p, xw_s, _ = ref.quantize_activations(V.swiglu_activations(x_emb[:256], cfg))
    rw, rws = V.random_codes(rng, 512, 288, "finite")
    rx, rxs = V.random_codes(rng, 64, 288, "finite")
    qw, qws = V.random_codes(rng, 512, 288, "real")
    qx, qxs = V.random_codes(rng, 64, 288, "real")
    sets = [("real", "w13 of layer 0 expert 0 (4608 rows x 288 blocks) x 512 embedding-derived tokens",
             V.blocks(z["w13"]), z["w13_scale"], V.blocks(xe_p), xe_s),
            ("real", "w2 of layer 0 expert 0 (4608 rows x 144 blocks) x 256 SwiGLU outputs of the real w13",
             V.blocks(z["w2"]), z["w2_scale"], V.blocks(xw_p), xw_s),
            ("random", "random codes and random finite scale bytes, 512 rows x 64 tokens, 288 blocks", rw, rws, rx, rxs),
            ("random", "random codes, scale bytes in the range real weights use (100..126), 512 rows x 64 tokens, 288 blocks",
             qw, qws, qx, qxs)]
    out = {"what": ("per-block-rounded reference (ref.gemm_ref: one binary32 round-to-nearest-even addition per block) against the "
                    "exact-sum reference (ref.gemm_fx_ref: exact integer sum, one rounding per row); no RTL involved; relative "
                    "difference = |per-block - exact-sum| / |exact-sum| over outputs whose exact-sum result is not zero"),
           "sets": {}}
    for kind, name, wc, ws, xc, xs in sets:
        t0 = time.time()
        a = ref.gemm_ref(V.codes_of(wc), ws, V.codes_of(xc), xs)
        b, sums = ref.gemm_fx_ref(V.codes_of(wc), ws, V.codes_of(xc), xs, return_sums=True)
        fa, fb = a.view(np.float32).astype(np.float64), b.view(np.float32).astype(np.float64)
        nz = fb != 0
        rel = np.abs(fa - fb)[nz] / np.abs(fb[nz])
        exact = sums.astype(np.float64) * 2.0 ** -20            # for the two counts below only; rounds above 2^53 units
        with np.errstate(over="ignore"):
            representable = (np.abs(sums) < (1 << 53)) & (exact.astype(np.float32).astype(np.float64) == exact)
        out["sets"][name] = {
            "kind": kind, "outputs": int(a.size), "bit_identical": int((a == b).sum()),
            "bit_identical_fraction": float((a == b).mean()),
            "worst_relative_difference": float(rel.max()) if rel.size else 0.0,
            "median_relative_difference_where_different": float(np.median(rel[rel > 0])) if (rel > 0).any() else 0.0,
            "exact_sum_zero_but_per_block_not": int(((fb == 0) & (fa != 0)).sum()),
            "outputs_whose_exact_sum_is_a_binary32_number": int(representable.sum()),
            "largest_abs_exact_sum_log2_units": float(np.log2(float(np.abs(sums).max()))) if np.abs(sums).max() else None}
        r = out["sets"][name]
        print(f"   {name}: {r['bit_identical_fraction'] * 100:.4f}% bit-identical of {r['outputs']:,}, worst relative "
              f"difference {r['worst_relative_difference']:.3g} ({time.time() - t0:.0f} s)", flush=True)
    for kind in ("real", "random"):
        ks = [v for v in out["sets"].values() if v["kind"] == kind]
        out[f"{kind}_rows"] = {"outputs": sum(v["outputs"] for v in ks), "bit_identical": sum(v["bit_identical"] for v in ks),
                               "bit_identical_fraction": sum(v["bit_identical"] for v in ks) / sum(v["outputs"] for v in ks),
                               "worst_relative_difference": max(v["worst_relative_difference"] for v in ks)}
    # "the same bits whenever no intermediate addition of the existing block rounds", on rows where that can be counted
    n_clean = n_same = n_rows = 0
    for nb in (2, 3, 5, 16, 40):
        wc, ws = V.random_codes(rng, 300, nb, "finite")
        xc, xs = V.random_codes(rng, 300, nb, "finite")
        fxb = ref.dot_fx_ref(V.codes_of(wc), ws, V.codes_of(xc), xs)
        for i in range(300):
            st: dict = {}
            old = ref.dot_ref_int(V.codes_of(wc[i]), ws[i], V.codes_of(xc[i]), xs[i], stats=st)
            n_rows += 1
            if "inexact result" not in st:
                n_clean += 1
                n_same += int(old == int(fxb[i]))
    out["rows_where_no_per_block_addition_rounds"] = {
        "rows_examined": n_rows, "rows_without_an_inexact_addition": n_clean, "of_which_bit_identical": n_same,
        "note": "random codes and finite scale bytes, rows of 2, 3, 5, 16 and 40 blocks; the integer model of the "
                "binary32 adder reports whether any addition of a row was inexact"}
    out["pass"] = bool(n_clean == n_same)
    return out


# ------------------------------------------------------------------------------- mutation check
MUTANTS = {
    "final rounding: round half up instead of ties to even": (
        "wire        inc_c = n_c[{g}] & (n_c[{l}] | (|n_c[{s}:0]));", "wire        inc_c = n_c[{g}];"),
    "S3: alignment shift off by one (Ew + Ex - 1)": (
        "wire [4:0]  sh_c  = es2_q - 5'd2;", "wire [4:0]  sh_c  = es2_q - 5'd1;"),
}


def mutation_check() -> dict:
    """Write each fault into a copy of the generated RTL (build directory only), run the checker,
    then run it on the unmodified RTL and confirm that the generator's output is the committed file."""
    mw = FXW["MW"]
    fmt = dict(g=mw - 25, l=mw - 24, s=mw - 26)
    clean_text = S.gen.fx_sv(P, TOP)
    out = {"vector_set": "the --quick directed and stress groups in full, plus 1/64 of the real-row and random groups",
           "mutants": {}}
    for name, (old, new) in MUTANTS.items():
        old, new = old.format(**fmt), new.format(**fmt)
        assert clean_text.count(old) == 1, f"mutation site not found: {old}"
        d = S.BUILD / "rtl" / ("nvfp4_dot_fx_mutant_" + str(len(out["mutants"])))
        S.gen_rtl(TOP, fx=True, out=d)
        (d / f"{TOP}.sv").write_text(clean_text.replace(old, new))
        t = run_groups(S.build_verilator(TOP, d, 1, False), quick=True, real_scale=64)
        out["mutants"][name] = {"replaced": old, "by": new, "vectors": sum(s["vectors"] for s in t.sections),
                                "mismatches": sum(s["mismatches"] for s in t.sections), "by_group": summarize(t)}
        (d / f"{TOP}.sv").unlink()                    # the faulty file does not outlive the check
    d = S.gen_rtl(TOP, fx=True)
    t = run_groups(S.build_verilator(TOP, d, 1, False), quick=True, real_scale=64)
    committed = HERE / "generated" / f"{TOP}.sv"
    out["unmodified"] = {"vectors": sum(s["vectors"] for s in t.sections), "mismatches": sum(s["mismatches"] for s in t.sections),
                         "generator_output_equals_committed_rtl": bool(committed.exists() and committed.read_text() == clean_text
                                                                       and (d / f"{TOP}.sv").read_text() == clean_text)}
    out["pass"] = bool(all(m["mismatches"] > 0 for m in out["mutants"].values()) and out["unmodified"]["mismatches"] == 0
                       and out["unmodified"]["generator_output_equals_committed_rtl"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["vectors", "icarus", "definitions", "mutation"])
    a = ap.parse_args()
    S.BUILD.mkdir(parents=True, exist_ok=True)
    path = S.BUILD / ("verify_fx_quick.json" if a.quick else "verify_fx.json")
    res = json.loads(path.read_text()) if path.exists() and a.only else {}
    steps = {"vectors": lambda: verify(a.quick), "icarus": check_icarus, "definitions": compare_definitions,
             "mutation": mutation_check}
    for name, fn in steps.items():
        if a.only in (None, name):
            t0 = time.time()
            res[name] = fn()
            print(f"== {name}: {'PASS' if res[name].get('pass') else 'FAIL'} ({time.time() - t0:.0f} s)", flush=True)
            path.write_text(json.dumps(res, indent=1) + "\n")
    print("written", path)


if __name__ == "__main__":
    main()
