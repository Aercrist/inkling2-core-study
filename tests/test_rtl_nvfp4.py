"""NVFP4 dot-product block: a fast bit-exact check, and the shape of results/silicon.json.

The simulation tests run when Icarus Verilog is installed (a few seconds); the others always run.
The million-vector run, synthesis and place-and-route live in rtl/nvfp4/run_all.py.
"""
import json
import pathlib
import shutil
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NV = ROOT / "rtl" / "nvfp4"
sys.path.insert(0, str(NV))
sys.path.insert(0, str(NV / "ref"))

import gen_nvfp4_dot as gen  # noqa: E402
import nvfp4_ref as ref      # noqa: E402

FIXTURE = ROOT / "data" / "fixtures" / "inkling2_small_L00_E000.npz"
needs_fixture = pytest.mark.skipif(not FIXTURE.exists(), reason="run tools/fetch_expert_fixture.py")
needs_iverilog = pytest.mark.skipif(shutil.which("iverilog") is None, reason="Icarus Verilog not installed")


def _random_rows(rng, n, nb):
    finite = np.array([b for b in range(256) if (b & 0x7F) != 0x7F], np.uint8)
    wc, xc = rng.integers(0, 16, (n, nb, 16)).astype(np.uint8), rng.integers(0, 16, (n, nb, 16)).astype(np.uint8)
    ws, xs = finite[rng.integers(0, 254, (n, nb))], finite[rng.integers(0, 254, (n, nb))]
    ws[0, 0], xs[1, nb - 1] = 0x7F, 0xFF          # NaN scales
    ws[2], xs[3] = 0, 0x80                         # zero scales
    ws[4], xs[4] = 0x01, 0x87                      # subnormal scales
    wc[5], xc[5], ws[5], xs[5] = 7, 15, 0x7E, 0x7E  # saturation: -6 * 6 everywhere, scales 448
    return wc, ws, xc, xs


def test_formats():
    assert list(ref.e2m1_decode(np.arange(16))) == [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6]
    assert list(ref.unpack_codes(np.array([0x21, 0xF8], np.uint8))) == [1, 2, 8, 15]   # low nibble first
    v = ref.e4m3_decode(np.arange(256, dtype=np.uint8))
    assert v[0x7E] == 448 and v[0x01] == 2.0 ** -9 and v[0x38] == 1.0 and v[0x08] == 2.0 ** -6
    assert np.isnan(v[0x7F]) and np.isnan(v[0xFF]) and np.isnan(v).sum() == 2
    assert np.array_equal(ref.e4m3_encode(v[:127]), np.arange(127))                    # round trip
    x = np.array([[6.0, -3.0, 0.26, 0.0] * 4, [1.0] * 16], np.float32)
    packed, scale, amax = ref.quantize_activations(x)
    assert list(amax) == [6.0, 1.0] and list(ref.e4m3_decode(scale)[:, 0]) == [256.0, 256.0]
    assert np.allclose(ref.dequantize(packed, scale, (amax / 1536)[:, None]), [[6, -3, 0.5, 0] * 4, [1] * 16])


def test_reference_models_agree():
    """NumPy float32 accumulation equals the integer model of the RTL adder, bit for bit."""
    rng = np.random.default_rng(3)
    for nb in (1, 3, 40):
        wc, ws, xc, xs = _random_rows(rng, 64, nb)
        bits = ref.dot_ref(wc, ws, xc, xs)
        assert all(ref.dot_ref_int(wc[i], ws[i], xc[i], xs[i]) == int(bits[i]) for i in range(64))
        assert bits[0] == bits[1] == 0x7FC00000 and bits[2] == bits[3] == 0
    assert np.array_equal(ref.gemm_ref(wc, ws, xc[:8], xs[:8]).diagonal(), bits[:8])


def test_exact_sum_reference():
    """nvfp4_dot_fx: the Python-integer model, the NumPy model and the definition of
    round-to-nearest-even on rationals agree; the per-block-rounded result is the same whenever
    none of its additions is inexact."""
    from fractions import Fraction
    rng = np.random.default_rng(5)
    f32 = lambda bits: np.array([bits], np.uint32).view(np.float32)[0]   # noqa: E731
    assert [ref.rn_even_binary32(n) for n in (0, 1 << 20, -(1 << 20), 1)] == [0, 0x3F800000, 0xBF800000, 0x35800000]
    assert f32(ref.rn_even_binary32((1 << 24) + 1)) == 16.0                       # tie, down to the even significand
    assert f32(ref.rn_even_binary32((1 << 24) + 3)) == (2 ** 24 + 4) / 2 ** 20    # tie, up to the even significand
    assert f32(ref.rn_even_binary32((1 << 25) - 1)) == 32.0                       # carries into the exponent
    assert f32(ref.rn_even_binary32(-((1 << 56) - 1))) == -(2.0 ** 36)            # the top of the accumulator
    cases = [0, 1, -1, 3, (1 << 24) - 1, (1 << 24) + 1, (1 << 24) + 3, (1 << 25) - 1, -((1 << 25) + 2), (1 << 56) - 1,
             (0xFFFFFF << 31) | (1 << 30), (0xFFFFFE << 31) | (1 << 30), (0xA5A5A4 << 19) | (1 << 18) | 1, (0xA5A5A5 << 19) | 1]
    for _ in range(400):
        n = int(rng.integers(1, 1 << 62)) >> int(rng.integers(6, 62))
        sh = max(n.bit_length() - 24, 0)
        if sh and rng.random() < 0.5:                                             # an exact tie
            n = (n >> sh << sh) | (1 << (sh - 1))
        cases.append(-n if rng.random() < 0.5 else n)
    for n in cases:                                   # nearest neighbour, and the even one of two nearest
        bits = ref.rn_even_binary32(n)
        x, c = Fraction(n, 1 << 20), Fraction(float(f32(bits)))
        lo, hi = (Fraction(float(np.nextafter(f32(bits), np.float32(s * np.inf)))) for s in (-1, 1))
        assert abs(x - c) <= min(abs(x - lo), abs(x - hi)), n
        assert abs(x - c) < min(abs(x - lo), abs(x - hi)) or bits & 1 == 0, n
    assert list(ref.fx_round(np.array(cases, np.int64))) == [ref.rn_even_binary32(n) for n in cases]
    for nb in (1, 3, 40, 512):
        k = 64 if nb < 100 else 6
        wc, ws, xc, xs = _random_rows(rng, k, nb)
        bits = ref.dot_fx_ref(wc, ws, xc, xs)
        assert all(ref.dot_fx_ref_int(wc[i], ws[i], xc[i], xs[i]) == int(bits[i]) for i in range(k))
        assert bits[0] == bits[1] == 0x7FC00000 and bits[2] == bits[3] == 0
        assert np.array_equal(ref.gemm_fx_ref(wc, ws, xc[:6], xs[:6]).diagonal(), bits[:6])
        for i in range(k if nb < 100 else 0):
            st = {}
            old = ref.dot_ref_int(wc[i], ws[i], xc[i], xs[i], stats=st)
            assert "inexact result" in st or old == int(bits[i])
    six = np.full((512, 16), 7, np.uint8)             # largest row: 512 blocks of 16 x (6 * 6), scales 448 * 448
    big = ref.dot_fx_ref_int(six, np.full(512, 0x7E, np.uint8), six, np.full(512, 0x7E, np.uint8))
    assert f32(big) == np.float32(512 * 16 * 36 * 448.0 * 448.0) and 512 * (2304 * 14 * 14 << 28) < ref.FX_SUM_BOUND == 1 << 56
    assert ref.dot_fx_ref_int(np.concatenate([six[:4], six[:4] | 8]), np.full(8, 0x7E, np.uint8), six[:8], np.full(8, 0x7E, np.uint8)) == 0
    with pytest.raises(AssertionError):               # a row of 513 blocks is outside the specification
        ref.dot_fx_ref_int(np.zeros((513, 16), np.uint8), np.zeros(513, np.uint8), np.zeros((513, 16), np.uint8), np.zeros(513, np.uint8))


def test_generated_rtl_is_in_sync(tmp_path):
    """rtl/nvfp4/generated/ is what the generator writes from the config and the fixture."""
    p = gen.model_params()
    assert (p["B"], p["DW"], p["PW"], p["nblk"]) == (16, 13, 19, {"w13": 288, "w2": 144})
    want = {"nvfp4_common.sv": gen.common_sv(p), "nvfp4_dot.sv": gen.reloadable_sv(p),
            "nvfp4_dot_fx.sv": gen.fx_sv(p), "tb_nvfp4_dot_fx.sv": gen.tb_sv(p, "nvfp4_dot_fx", 1, False, poke=False)}
    q = gen.derive_fx(p)
    assert (q["TW"], q["MW"], q["ACCW"], q["MAX_BLOCKS"], q["LATENCY"]) == (47, 56, 57, 512, 7)
    assert gen.front_sv(p) in want["nvfp4_common.sv"] and gen.front_sv(p) in want["nvfp4_dot_fx.sv"]   # S1 and S2 are shared text
    if FIXTURE.exists():
        codes, scales, origin = gen.fixture_rows(gen.PRIMARY["fixture"], gen.PRIMARY["leaf"], [gen.PRIMARY["row"]], None, p)
        want["nvfp4_dot_const.sv"] = gen.const_sv(p, "nvfp4_dot_const", codes, scales, origin)
    for name, text in want.items():
        assert (NV / "generated" / name).read_text() == text, f"{name}: run gen_nvfp4_dot.py committed"
    with pytest.raises(ValueError):
        gen.derive(1024, p["nblk"])      # a block this large would no longer be exact in binary32
    assert gen.derive(32, p["nblk"])["PW"] == 20


@needs_iverilog
@needs_fixture
def test_rtl_bit_exact_small(tmp_path, monkeypatch):
    """Both modules against the reference in Icarus: random, edge-case and real-tensor rows."""
    import simlib as S
    import verify as V
    monkeypatch.setattr(S, "BUILD", tmp_path)
    rng = np.random.default_rng(11)
    gdir = NV / "generated"
    z = np.load(FIXTURE)
    w13, w13s = V.blocks(z["w13"]), z["w13_scale"]
    xp, xs, _ = ref.quantize_activations(V.embedding_activations(4))
    xb = V.blocks(xp)
    # reloadable: short random rows (with NaN, zero, subnormal, saturated blocks), then two real rows
    vvp = S.build_icarus("nvfp4_dot", [gdir / "nvfp4_common.sv", gdir / "nvfp4_dot.sv", gdir / "tb_nvfp4_dot.sv"])
    for nb in (1, 5):
        wc, ws, xc, xs_ = _random_rows(rng, 48, nb)
        got = S.run_icarus(vvp, S.stream(nb, ref.pack_codes(wc), ws, ref.pack_codes(xc), xs_, [(i, i) for i in range(48)],
                                         gaps={3: 2, 7: 5}))
        assert np.array_equal(got[:, 0], ref.dot_ref(wc, ws, xc, xs_))
    pairs = [(0, 0), (4607, 1), (0, 0)]
    got = S.run_icarus(vvp, S.stream(288, w13, w13s, xb, xs, pairs))
    exp = [int(ref.dot_ref(V.codes_of(w13[r]), w13s[r], V.codes_of(xb[x]), xs[x])) for r, x in pairs]
    assert list(got[:, 0]) == exp and exp[0] == exp[2]
    # fixed weights: row 0 of the fixture expert, the same tokens
    vvp = S.build_icarus("nvfp4_dot_const", [gdir / "nvfp4_common.sv", gdir / "nvfp4_dot_const.sv", gdir / "tb_nvfp4_dot_const.sv"])
    got = S.run_icarus(vvp, S.stream(288, None, None, xb, xs, [(0, i) for i in range(4)]), tag="c")
    assert np.array_equal(got[:, 0], ref.gemm_ref(V.codes_of(w13[[0]]), w13s[[0]], V.codes_of(xb), xs)[:, 0])
    assert got[0, 0] == exp[0]            # the two variants agree on the same row and token


@needs_iverilog
@needs_fixture
def test_exact_sum_rtl_bit_exact_small(tmp_path, monkeypatch):
    """nvfp4_dot_fx against its reference in Icarus: random, edge-case, final-rounding and real rows."""
    import simlib as S
    import verify as V
    import verify_fx as F
    monkeypatch.setattr(S, "BUILD", tmp_path)
    rng = np.random.default_rng(12)
    gdir = NV / "generated"
    vvp = S.build_icarus("nvfp4_dot_fx", [gdir / "nvfp4_common.sv", gdir / "nvfp4_dot_fx.sv", gdir / "tb_nvfp4_dot_fx.sv"])
    for nb in (1, 5):
        wc, ws, xc, xs_ = _random_rows(rng, 48, nb)
        got = S.run_icarus(vvp, S.stream(nb, ref.pack_codes(wc), ws, ref.pack_codes(xc), xs_, [(i, i) for i in range(48)],
                                         gaps={3: 2, 7: 5}))
        assert np.array_equal(got[:, 0], ref.dot_fx_ref(wc, ws, xc, xs_))
    # the final rounding: ties both ways, carry into the exponent, sticky far below, cancellation to +0
    cases = [c for c in F.directed_fx(rng) if len(c[1]) <= 16 and not c[0].startswith("row of 1 block, largest")]
    assert {"final rounding: tie, rounded down to even", "final rounding: tie, rounded up to even",
            "exact cancellation to +0"} <= {c[0] for c in cases}
    wc, ws, xc, xs_ = F.pad_rows([c[1] for c in cases], 16, rng)
    exp = ref.dot_fx_ref(wc, ws, xc, xs_)
    got = S.run_icarus(vvp, S.stream(16, ref.pack_codes(wc), ws, ref.pack_codes(xc), xs_, [(i, i) for i in range(len(cases))]))
    assert np.array_equal(got[:, 0], exp)
    assert all(e == 0 for c, e in zip(cases, exp) if c[0] == "exact cancellation to +0")
    # two real rows: the reference, and the same bits as the per-block-rounded block (no addition rounds there)
    z = np.load(FIXTURE)
    w13, w13s = V.blocks(z["w13"]), z["w13_scale"]
    xp, xs, _ = ref.quantize_activations(V.embedding_activations(2))
    xb = V.blocks(xp)
    pairs = [(0, 0), (4607, 1)]
    got = S.run_icarus(vvp, S.stream(288, w13, w13s, xb, xs, pairs))
    exp = [int(ref.dot_fx_ref(V.codes_of(w13[r]), w13s[r], V.codes_of(xb[x]), xs[x])) for r, x in pairs]
    assert list(got[:, 0]) == exp == [int(ref.dot_ref(V.codes_of(w13[r]), w13s[r], V.codes_of(xb[x]), xs[x])) for r, x in pairs]


def test_silicon_json_is_well_formed():
    path = ROOT / "results" / "silicon.json"
    if not path.exists():
        pytest.skip("results/silicon.json not generated yet (rtl/nvfp4/run_all.py)")
    d = json.loads(path.read_text())
    v = d["verification"]
    assert v["vectors"]["pass"] and v["vectors"]["mismatches_total"] == 0
    assert v["vectors"]["vectors_total"] >= 1_000_000
    assert v["vectors"]["vectors_reloadable"] > 0 and v["vectors"]["vectors_fixed"] > 0
    for k in ("formats", "icarus", "batch", "torch"):
        assert v[k]["pass"], k
    assert d["layout"]["nibble_order"] == "low nibble first"
    s = d["silicon"]
    assert s["process"] == "ASAP7" and isinstance(s["place_and_route_ran"], bool)
    for key in ("reloadable", "fixed"):
        m = s["headline"][key]
        assert m["cell_area_um2"] > 0 and m["cells"] > 0 and m["clock_period_ps"] > 0
        assert m["power_w"]["dynamic"] > 0 and m["power_w"]["leakage"] > 0
        assert m["post_route_netlist_bit_exact"] is True
        assert abs(m["energy_per_block_pj"] - m["power_w"]["total"] * m["clock_period_ps"]) < 1e-6 * m["energy_per_block_pj"] + 1e-9
    r = s["ratios_reloadable_over_fixed"]
    for key in ("cell_area", "energy_per_block"):
        assert r[key]["min"] <= r[key]["median"] <= r[key]["max"]
    assert set(d["tools"]) >= {"verilator", "iverilog", "yosys", "openroad", "opensta"}
