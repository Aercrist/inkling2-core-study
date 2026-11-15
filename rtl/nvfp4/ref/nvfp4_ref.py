#!/usr/bin/env python3
"""Bit-exact NumPy reference for the NVFP4 block-scaled dot-product engine (rtl/nvfp4/SPEC.md).

Formats (verified against the checkpoint and against TokenSpeed's reference kernel, see REPORT.md):

* E2M1 code, 4 bits: bit 3 sign, bits 2:0 index into (0, 0.5, 1, 1.5, 2, 3, 4, 6). Codes 0 and 8
  are +0 and -0. Two codes per byte, the EARLIER element in the LOW nibble.
* E4M3 block scale, 8 bits (torch.float8_e4m3fn): sign, 4-bit exponent e, 3-bit mantissa m.
  e = 0: m * 2^-9 (zero and subnormals); e >= 1: (8 + m) * 2^(e - 10); 0x7F and 0xFF are NaN;
  there is no infinity; the largest finite value is 0x7E = 448.
* Weights: w = code * block_scale * (amax / outer_scale_denominator), amax per expert.
* Activations: x = code * block_scale * (amax_token / 1536), block scale chosen by four-over-six.

The engine itself works on codes and block scales only:

    D_b   = sum_j (2*w_bj) * (2*x_bj)                 exact integer, |D_b| <= 2304
    T_b   = D_b/4 * scale_w_b * scale_x_b             exact, representable in binary32
    A_0   = +0;  A_(b+1) = RNE_binary32(A_b + T_b)    block order b = 0, 1, 2, ...
    out   = A_NB, or 0x7FC00000 if any block scale of the row is NaN

Two independent implementations are given: `dot_ref` (NumPy float32, which is IEEE-754 binary32
with round-to-nearest-even on every platform NumPy supports) and `dot_ref_int` (Python integers,
mirroring the RTL adder bit by bit). The tests check them against each other.
"""
from __future__ import annotations

import numpy as np

E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
E2M1_Q = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.int32)  # 2*value, by 4-bit code
QNAN_BITS = np.uint32(0x7FC00000)
ACT_DENOMINATOR = 1536.0  # per-token outer scale of NVFP4 activations (TokenSpeed block_scaled.py)
F32_MAX = np.finfo(np.float32).max


# ----------------------------------------------------------------------------- E2M1
def unpack_codes(packed: np.ndarray) -> np.ndarray:
    """[..., n] bytes -> [..., 2n] 4-bit codes; element 2k is the low nibble of byte k."""
    packed = np.asarray(packed, np.uint8)
    return np.stack((packed & 15, packed >> 4), -1).reshape(*packed.shape[:-1], -1)


def pack_codes(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, np.uint8)
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)


def e2m1_decode(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes)
    v = E2M1_VALUES[codes & 7]
    return np.where(codes & 8, -v, v).astype(np.float32)


def e2m1_encode(x: np.ndarray) -> np.ndarray:
    """Round to the nearest E2M1 value, ties to the even code, saturating at 6. -0 keeps its sign."""
    x = np.asarray(x, np.float32)
    m = np.abs(x)
    code = ((m > 0.25).astype(np.uint8) + (m >= 0.75) + (m > 1.25) + (m >= 1.75)
            + (m > 2.5) + (m >= 3.5) + (m > 5.0)).astype(np.uint8)
    return (code | (np.signbit(x).astype(np.uint8) << 3)).astype(np.uint8)


# ----------------------------------------------------------------------------- E4M3
def e4m3_fields(b: np.ndarray):
    """byte -> (sign, M, E, nan) with value = (-1)^sign * M * 2^(E - 10), M in 0..15, E in 1..15."""
    b = np.asarray(b, np.uint8)
    e = (b >> 3) & 15
    m = b & 7
    nz = e != 0
    return (b >> 7).astype(np.int32), np.where(nz, 8 + m, m).astype(np.int32), \
        np.where(nz, e, 1).astype(np.int32), (b & 0x7F) == 0x7F


def e4m3_decode(b: np.ndarray) -> np.ndarray:
    s, m, e, nan = e4m3_fields(b)
    v = np.ldexp(m.astype(np.float32), e - 10).astype(np.float32)
    v = np.where(s == 1, -v, v)
    return np.where(nan, np.float32(np.nan), v).astype(np.float32)


_E4M3_POS = None


def e4m3_encode(x: np.ndarray) -> np.ndarray:
    """float32 -> E4M3 byte: nearest, ties to even, saturating at 448 (0x7E); NaN -> 0x7F.
    Matches torch's cast to float8_e4m3fn for |x| <= 448 (callers clamp, as TokenSpeed does)."""
    global _E4M3_POS
    if _E4M3_POS is None:
        _E4M3_POS = e4m3_decode(np.arange(127, dtype=np.uint8)).astype(np.float64)  # 0 .. 448, increasing
    x = np.asarray(x, np.float32)
    a = np.abs(x).astype(np.float64)
    hi = np.clip(np.searchsorted(_E4M3_POS, a, side="left"), 0, 126)
    lo = np.clip(hi - 1, 0, 126)
    dlo, dhi = a - _E4M3_POS[lo], _E4M3_POS[hi] - a
    pick_hi = (dhi < dlo) | ((dhi == dlo) & (hi % 2 == 0))
    code = np.where(a >= 448.0, 126, np.where(pick_hi, hi, lo)).astype(np.uint8)
    code = np.where(np.isnan(x), 0x7F, code | (np.signbit(x).astype(np.uint8) << 7)).astype(np.uint8)
    return code


# ----------------------------------------------------------------------------- activations
def quantize_activations(x: np.ndarray, block: int = 16):
    """NVFP4 activation quantizer: per-token amax, per-block E4M3 scale, four-over-six by MAE.

    x: float32 [tokens, width] (the engines quantize BF16 rows; pass BF16-representable values).
    Returns (packed codes uint8 [tokens, width/2], block scales uint8 [tokens, width/block],
    amax float32 [tokens]). Dequantized value = code * scale * amax / 1536.
    A float32 transliteration of `reference_block_quantize(x, "nvfp4")` in TokenSpeed's
    tokenspeed_kernel/ops/moe/block_scaled.py; tests check it bit for bit against that function.
    """
    x = np.ascontiguousarray(x, np.float32)
    rows, width = x.shape
    g = x.reshape(rows, width // block, block)
    maxima = np.abs(g).max(-1)
    amax = np.abs(x).max(1)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        encoder = np.minimum(np.float32(ACT_DENOMINATOR) / amax, F32_MAX).astype(np.float32)
        decoder = (np.float32(1.0) / encoder).astype(np.float32)
        outer = (amax * np.float32(1.0 / ACT_DENOMINATOR)).astype(np.float32)
        cand = []
        for target in (6.0, 4.0):
            sc = np.minimum(maxima * (encoder[:, None] * np.float32(1.0 / target)), np.float32(448.0))
            sbyte = e4m3_encode(sc.astype(np.float32))
            sc = e4m3_decode(sbyte)
            inv = np.minimum(np.float32(1.0) / (sc * decoder[:, None]), F32_MAX).astype(np.float32)
            code = e2m1_encode(g * inv[..., None])
            recon = (e2m1_decode(code) * sc[..., None]) * outer[:, None, None]
            err = np.abs(g - recon).astype(np.float32)
            while err.shape[-1] > 1:  # pairwise tree, as the kernel does
                err = (err[..., 0::2] + err[..., 1::2]).astype(np.float32)
            cand.append((code, sbyte, err[..., 0]))
    choose4 = cand[1][2] < cand[0][2]
    code = np.where(choose4[..., None], cand[1][0], cand[0][0]).reshape(rows, width)
    scale = np.where(choose4, cand[1][1], cand[0][1]).astype(np.uint8)
    return pack_codes(code), scale, amax


def dequantize(packed: np.ndarray, scales: np.ndarray, outer: float | np.ndarray = 1.0, block: int = 16):
    """code * block scale * outer, float32. outer = amax/denominator (weights) or amax/1536 (tokens)."""
    v = e2m1_decode(unpack_codes(packed)) * np.repeat(e4m3_decode(scales), block, axis=-1)
    return (v * np.asarray(outer, np.float32)).astype(np.float32)


def weight_outer_scale(amax, denominator) -> np.float32:
    return np.float32(np.float32(amax) / np.float32(denominator))


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 (nearest even), returned as float32."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16).astype(np.uint32)
    return r.view(np.float32)


# ----------------------------------------------------------------------------- the engine
def block_terms(wc, ws, xc, xs):
    """Codes [..., NB, B] (4-bit), scales [..., NB] (bytes) -> (terms float32 [..., NB], nan [...])."""
    d = (E2M1_Q[np.asarray(wc)] * E2M1_Q[np.asarray(xc)]).sum(-1)  # exact integer, 4x the real sum
    sw, mw, ew, nw = e4m3_fields(ws)
    sx, mx, ex, nx = e4m3_fields(xs)
    p = d * mw * mx * np.where(sw ^ sx, -1, 1)                      # |p| < 2^19
    t = np.ldexp(p.astype(np.float32), ew + ex - 22).astype(np.float32)   # exact
    nan = (nw | nx)
    return np.where(nan, np.float32(0), t), nan.any(-1)


def accumulate(terms: np.ndarray, nan: np.ndarray, acc_init=None) -> np.ndarray:
    """Sequential binary32 accumulation over the last axis; returns uint32 bit patterns."""
    acc = np.zeros(terms.shape[:-1], np.float32) if acc_init is None else \
        np.broadcast_to(np.asarray(acc_init, np.float32), terms.shape[:-1]).copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for b in range(terms.shape[-1]):
            acc = (acc + terms[..., b]).astype(np.float32)
    return np.where(nan, QNAN_BITS, np.asarray(acc, np.float32).view(np.uint32)).astype(np.uint32)


def dot_ref(wc, ws, xc, xs, acc_init=None) -> np.ndarray:
    """The engine: codes [..., NB, B], block scales [..., NB] -> binary32 bit patterns [...]."""
    t, nan = block_terms(wc, ws, xc, xs)
    return accumulate(t, nan, acc_init)


def gemm_ref(wc, ws, xc, xs) -> np.ndarray:
    """All dot products of R weight rows with T tokens: wc [R, NB, B], ws [R, NB], xc [T, NB, B],
    xs [T, NB] -> uint32 [T, R]. Same arithmetic and block order as `dot_ref`."""
    qw = E2M1_Q[np.asarray(wc)].astype(np.float32)   # products and block sums are small integers:
    qx = E2M1_Q[np.asarray(xc)].astype(np.float32)   # exact in float32 whatever BLAS does
    sw, mw, ew, nw = e4m3_fields(ws)
    sx, mx, ex, nx = e4m3_fields(xs)
    fw = np.where(nw, 0, np.ldexp((mw * np.where(sw, -1, 1)).astype(np.float32), ew - 11)).astype(np.float32)
    fx = np.where(nx, 0, np.ldexp((mx * np.where(sx, -1, 1)).astype(np.float32), ex - 11)).astype(np.float32)
    T, R, NB = qx.shape[0], qw.shape[0], qw.shape[1]
    acc = np.zeros((T, R), np.float32)
    for b in range(NB):
        d = qx[:, b, :] @ qw[:, b, :].T                              # [T, R] exact integers
        acc = (acc + (d * fx[:, b, None]) * fw[None, :, b]).astype(np.float32)
    bits = acc.view(np.uint32).copy()
    bits[nx.any(1)[:, None] | nw.any(1)[None, :]] = QNAN_BITS
    return bits


def exact_dot_f64(wc, ws, xc, xs) -> np.ndarray:
    """The mathematically exact value of the same dot product (float64 holds it exactly:
    every term is a multiple of 2^-20 and the sum of 2^16 blocks stays below 2^44)."""
    d = (E2M1_Q[np.asarray(wc)].astype(np.int64) * E2M1_Q[np.asarray(xc)]).sum(-1)
    sw, mw, ew, _ = e4m3_fields(ws)
    sx, mx, ex, _ = e4m3_fields(xs)
    p = d * mw * mx * np.where(sw ^ sx, -1, 1)
    return np.ldexp(p.astype(np.float64), ew + ex - 22).sum(-1)


# ----------------------------------------------------------------------------- integer model of the RTL
def _fp32_add_int(a, t, stats: dict | None = None):
    """One accumulator step on (sign, exp, 24-bit mantissa) triples, mirroring the RTL adder.
    Zero is (0, 0, 0). Returns (result triple, overflow). `stats` counts which adder cases occur."""
    (sa, ea, ma), (st, et, mt) = a, t

    def hit(k):
        if stats is not None:
            stats[k] = stats.get(k, 0) + 1
    if (ea, ma) >= (et, mt):
        xs_, xe, xm, ys, ye, ym = sa, ea, ma, st, et, mt
    else:
        xs_, xe, xm, ys, ye, ym = st, et, mt, sa, ea, ma
    d = xe - ye
    yext = ym << 3
    if d >= 32:
        yal, sticky = 0, int(ym != 0)
    else:
        yal, sticky = yext >> d, int(yext & ((1 << d) - 1) != 0)
    yal |= sticky
    sub = xs_ != ys and ym != 0
    if ym != 0 and xm != 0:
        hit("effective subtraction" if sub else "effective addition")
        if d >= 27:
            hit("smaller operand entirely below the sticky bit (shift >= 27)")
        if d == 0:
            hit("equal exponents")
    s = (xm << 3) - yal if xs_ != ys else (xm << 3) + yal
    if s == 0:
        if xm != 0:
            hit("exact cancellation to +0")
        return (0, 0, 0), False
    if s >> 27:
        n, e = (s >> 1) | (s & 1), xe + 1
        hit("carry out, right shift by 1")
    else:
        lz = 27 - s.bit_length()
        n, e = s << lz, xe - lz
        if lz == 1:
            hit("left shift by 1")
        elif lz >= 2:
            hit("cancellation, left shift by 2 or more")
            if lz >= 12:
                hit("cancellation, left shift by 12 or more")
    g, rs, lsb = (n >> 2) & 1, n & 3, (n >> 3) & 1
    if n & 7:
        hit("inexact result")
    if g and not rs:
        hit("tie, rounded up to even" if lsb else "tie, rounded down to even")
    inc = g and (rs != 0 or lsb)
    if inc:
        hit("rounded up")
    r = (n >> 3) + (1 if inc else 0)
    if r >> 24:
        r, e = r >> 1, e + 1
        hit("rounding carried into the exponent")
    if e >= 255:
        hit("overflow to infinity")
    return (xs_, e & 0xFF, r), e >= 255


def dot_ref_int(wc, ws, xc, xs, acc_init_bits: int | None = None, stats: dict | None = None,
                inject_term_bits: int | None = None) -> int:
    """One dot product in pure integer arithmetic. wc, xc: [NB, B] codes; ws, xs: [NB] bytes.
    acc_init_bits / inject_term_bits mirror the testbench hook that overwrites the accumulator
    register and places one extra binary32 term in front of the adder (overflow tests)."""
    wc, xc = np.asarray(wc), np.asarray(xc)
    acc, nan, inf, inf_s = (0, 0, 0), False, False, 0

    def triple(bits):
        e = (bits >> 23) & 0xFF
        return (bits >> 31, e, ((bits & 0x7FFFFF) | (1 << 23)) if e else 0)

    if acc_init_bits is not None:
        acc = triple(acc_init_bits)
    if inject_term_bits is not None:
        acc, ovf = _fp32_add_int(acc, triple(inject_term_bits), stats)
        if ovf:
            inf, inf_s = True, acc[0]
    for b in range(wc.shape[0]):
        d = int((E2M1_Q[wc[b]] * E2M1_Q[xc[b]]).sum())
        sw, mw, ew, nw = (int(v) for v in e4m3_fields(ws[b]))
        sx, mx, ex, nx = (int(v) for v in e4m3_fields(xs[b]))
        nan |= bool(nw or nx)
        p = abs(d) * mw * mx
        if p == 0:
            t = (0, 0, 0)
        else:
            h = p.bit_length() - 1
            t = (int(d < 0) ^ sw ^ sx, 105 + h + ew + ex, (p << (18 - h)) << 5)
        acc, ovf = _fp32_add_int(acc, t, stats)
        if ovf and not inf:
            inf, inf_s = True, acc[0]
    if nan:
        return int(QNAN_BITS)
    if inf:
        return (inf_s << 31) | 0x7F800000
    return (acc[0] << 31) | (acc[1] << 23) | (acc[2] & 0x7FFFFF)
