"""Measure placed-and-routed designs: area, timing, a post-route bit-exact check and
activity-based power under real-tensor workloads. One JSON per design, cached in the ORFS work
directory; run_all.py assembles them into results/silicon.json.

    python rtl/nvfp4/flow/silicon.py rel:1800 const_r0:1800 ...
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import orfs       # noqa: E402
import pnr_jobs   # noqa: E402
import power      # noqa: E402
import simlib as S  # noqa: E402
import verify as V  # noqa: E402

ref = S.ref
POWER_ROWS = [0, 1, 577, 1154, 2305]     # rows compared as reloadable vs fixed, same tokens
TOKENS = list(range(0, 512, 64))         # 8 embedding-derived tokens, one per fetched id run
K1_BLOCKS = [0, 100, 200]
T_DESIGNS = {"t1_b0": (1, 0), "t1_b100": (1, 100), "t1_b200": (1, 200), "t4": (4, 0), "t16": (16, 0), "t72": (72, 0)}
REL_T = ["t1_b0", "t1_b100"]             # repeated-table workloads the reloadable module is also given
_cache: dict = {}


def tensors():
    if not _cache:
        z = np.load(S.ROOT / V.EXPERT)
        cfg = json.loads((S.ROOT / "data" / "inkling2_small" / "config.json").read_text())["text_config"]
        x = V.embedding_activations()
        xp, xs, _ = ref.quantize_activations(x)
        sw = V.swiglu_activations(x[TOKENS], cfg)
        sp, ss, _ = ref.quantize_activations(sw)
        _cache.update(w13=V.blocks(z["w13"]), w13s=z["w13_scale"], w2=V.blocks(z["w2"]), w2s=z["w2_scale"],
                      x=V.blocks(xp), xs=xs, sx=V.blocks(sp), sxs=ss)
    return _cache


def _segment(wc, ws, xc, xs, k: int):
    """Rows of k blocks: weights wc [k, 8] (or None) against every k-block window of xc [n, 8]."""
    n = (xc.shape[0] // k) * k
    xcs, xss = xc[:n].reshape(-1, k, 8), xs[:n].reshape(-1, k)
    if wc is None:
        return S.stream(k, None, None, xcs, xss, [(0, i) for i in range(xcs.shape[0])]), xcs, xss
    return S.stream(k, wc[None], ws[None], xcs, xss, [(0, i) for i in range(xcs.shape[0])]), xcs, xss


def workloads(name: str) -> list:
    """[(workload name, stimulus words, expected [n, rows] uint32, description)] for one design."""
    t = tensors()
    top, kw, rows, const = pnr_jobs.design(name)
    out = []
    xt, xst = t["x"][TOKENS], t["xs"][TOKENS]
    flat_x, flat_xs = xt.reshape(-1, 8), xst.reshape(-1)          # 2304 real activation blocks

    def row_wl(leaf, r, label):
        w, ws_, xa, xas = (t["w13"], t["w13s"], xt, xst) if leaf == "w13" else (t["w2"], t["w2s"], t["sx"], t["sxs"])
        nb = ws_.shape[1]
        pairs = [(r, i) for i in range(xa.shape[0])]
        words = S.stream(nb, None if const else w, None if const else ws_, xa, xas, pairs)
        exp = ref.gemm_ref(V.codes_of(w[[r]]), ws_[[r]], V.codes_of(xa), xas)
        out.append((label, words, exp, f"{leaf} row {r} of layer 0 expert 0 x {xa.shape[0]} tokens, {nb} blocks each, no idle clocks"))

    def k_wl(k, b0, label):
        wc, ws_ = t["w13"][0, b0:b0 + k], t["w13s"][0, b0:b0 + k]
        nx = min(512, 2304 // k)
        xc = t["x"][:nx, b0:b0 + k].reshape(-1, 8) if k > 1 else flat_x
        xs_ = t["xs"][:nx, b0:b0 + k].reshape(-1) if k > 1 else flat_xs
        words, xcs, xss = _segment(None if const else wc, None if const else ws_, xc, xs_, k)
        exp = ref.dot_ref(V.codes_of(np.broadcast_to(wc, xcs.shape)), np.broadcast_to(ws_, xss.shape), V.codes_of(xcs), xss)[:, None]
        out.append((label, words, exp, f"w13 row 0 blocks {b0}..{b0 + k - 1} hard-wired or held on the weight pins; "
                                       f"{xcs.shape[0]} rows of {k} real activation block(s)"))

    def t_wl(label):
        tt, b0 = T_DESIGNS[label]
        wc = np.tile(t["w13"][0, b0:b0 + tt], (288 // tt, 1))[None]
        ws_ = np.tile(t["w13s"][0, b0:b0 + tt], 288 // tt)[None]
        words = S.stream(288, None if const else wc, None if const else ws_, xt, xst, [(0, i) for i in range(xt.shape[0])])
        exp = ref.gemm_ref(V.codes_of(wc), ws_, V.codes_of(xt), xst)
        out.append((label, words, exp, f"288-block rows whose weights are block(s) {b0}..{b0 + tt - 1} of w13 row 0 repeated "
                                       f"{288 // tt} times (hard-wired, or driven on the weight pins) x {xt.shape[0]} tokens"))

    if name == "rel":
        for r in POWER_ROWS:
            row_wl("w13", r, f"w13_row{r}")
        row_wl("w2", 0, "w2_row0")
        for label in REL_T:
            t_wl(label)
        for b in K1_BLOCKS:
            k_wl(1, b, f"k1_b{b}")
    elif name in ("rel_x8", "const_x8"):
        w, ws_ = t["w13"][pnr_jobs.WIDE_ROWS], t["w13s"][pnr_jobs.WIDE_ROWS]
        if const:
            words = S.stream(288, None, None, xt, xst, [(0, i) for i in range(len(TOKENS))])
        else:   # eight weight streams side by side
            v, last, xcs, xss, wcs, wss = [], [], [], [], [], []
            for i in range(len(TOKENS)):
                for b in range(288):
                    v.append(1); last.append(int(b == 287)); xcs.append(xt[i, b]); xss.append(xst[i, b])
                    wcs.append(w[:, b]); wss.append(ws_[:, b])
            words = S.stim_words(v, last, np.array(xcs), np.array(xss), np.array(wcs), np.array(wss))
        out.append(("w13_rows16_23", words, ref.gemm_ref(V.codes_of(w), ws_, V.codes_of(xt), xst),
                    "w13 rows 16..23 of layer 0 expert 0 x 8 tokens"))
    elif name.startswith("const_r"):
        row_wl("w13", int(name[7:]), f"w13_row{int(name[7:])}")
    elif name.startswith("const_w2_r"):
        row_wl("w2", int(name[10:]), f"w2_row{int(name[10:])}")
    elif name.startswith("const_t"):
        t_wl(name[6:] if "_b" in name else (name[6:] + "_b0" if name == "const_t1" else name[6:]))
    elif name.startswith("const_k"):
        parts = name[7:].split("_b")
        k, b0 = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        k_wl(k, b0, f"k1_b{b0}" if k == 1 else f"k{k}")
    else:
        raise KeyError(name)
    nidle = 256
    idle = S.stim_words([0] * nidle, [0] * nidle, np.zeros((nidle, 8), np.uint8), np.zeros(nidle, np.uint8),
                        None if const else np.zeros((nidle, rows, 8), np.uint8), None if const else np.zeros((nidle, rows), np.uint8))
    out.append(("idle", idle, np.zeros((0, rows), np.uint32), "in_valid low, data pins held at zero"))
    return out


def measure(name: str, period: float, force: bool = False) -> dict:
    tag = pnr_jobs.tag_of(name, period)
    work = orfs.WORK / tag
    cache = work / "power" / "measure.json"
    top, kw, rows, const = pnr_jobs.design(name)
    if cache.exists() and not force:
        old = json.loads(cache.read_text())
        if set(old["workloads"]) == {w[0] for w in workloads(name)}:
            return old
    res = power.collect(tag)
    res.update(design=name, top=top, rows=rows, fixed_weights=const)
    codes = kw.get("const")
    res["hardwired_weights"] = 0 if not const else rows * 16 * (
        (codes["blocks"][1] - codes["blocks"][0]) if codes.get("blocks") else S.P["nblk"][codes["leaf"]])
    res["distinct_blocks_in_table"] = None if not const else res["hardwired_weights"] // (16 * rows)
    wls = workloads(name)
    checks, vcds = {}, []
    for wl, words, exp, desc in wls:
        got, vcd = power.gate_sim(tag, top, rows, const, words, period, wl)
        ok = got.shape == exp.shape and bool((got == exp).all())
        checks[wl] = {"description": desc, "clocks": len(words), "outputs": int(exp.size),
                      "post_route_netlist_bit_exact": ok}
        vcds.append(wl)
    st = power.sta(tag, vcds)
    for wl in vcds:
        (work / "power" / f"{wl}.vcd").unlink(missing_ok=True)
        (work / "power" / f"{wl}.hex").unlink(missing_ok=True)
    res["sta"] = {k: st[k] for k in ("setup_wns_ps", "hold_wns_ps", "min_period_ps", "fmax_mhz", "critical_path") if k in st}
    res["vectorless_power_w"] = st["vectorless"]["total"]["total_w"]
    res["workloads"] = {}
    for wl in vcds:
        p = st[wl]["power"]
        tot = p["total"]
        dyn = tot["internal_w"] + tot["switching_w"]
        e_block = tot["total_w"] * period * 1e-12            # one block per clock
        res["workloads"][wl] = {
            **checks[wl],
            "power_w": {"internal": tot["internal_w"], "switching": tot["switching_w"], "dynamic": dyn,
                        "leakage": tot["leakage_w"], "total": tot["total_w"]},
            "power_by_group_w": {g: p[g]["total_w"] for g in ("sequential", "combinational", "clock")},
            "energy_per_block_pj": e_block * 1e12, "energy_per_mac_pj": e_block * 1e12 / (16 * rows),
            "dynamic_energy_per_block_pj": dyn * period, "leakage_energy_per_block_pj": tot["leakage_w"] * period,
            "vcd_pins_annotated": st[wl]["activity_annotation"].get("vcd"),
            "pins_unannotated": st[wl]["activity_annotation"].get("unannotated"),
        }
    res["post_route_netlist_bit_exact"] = all(c["post_route_netlist_bit_exact"] for c in checks.values())
    cache.write_text(json.dumps(res, indent=1) + "\n")
    return res


if __name__ == "__main__":
    for job in sys.argv[1:]:
        name, period = job.split(":")
        r = measure(name, float(period), force="--force" in sys.argv)
        main_wl = next(iter(r["workloads"]))
        w = r["workloads"][main_wl]
        print(f"{job:22s} area {r['cell_area_um2']:8.1f} um2, {r['cells']:6d} cells, WNS {r['setup_wns_ps']:7.1f} ps, "
              f"min period {r['min_period_ps']:7.1f} ps, {main_wl}: {w['power_w']['total'] * 1e3:.4f} mW, "
              f"{w['energy_per_block_pj']:.3f} pJ/block, netlist bit-exact {r['post_route_netlist_bit_exact']}", flush=True)

