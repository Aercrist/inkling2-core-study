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


def test_generated_rtl_is_in_sync(tmp_path):
    """rtl/nvfp4/generated/ is what the generator writes from the config and the fixture."""
    p = gen.model_params()
    assert (p["B"], p["DW"], p["PW"], p["nblk"]) == (16, 13, 19, {"w13": 288, "w2": 144})
    want = {"nvfp4_common.sv": gen.common_sv(p), "nvfp4_dot.sv": gen.reloadable_sv(p)}
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
