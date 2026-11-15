"""Build and run the simulations of rtl/nvfp4 (Verilator for volume, Icarus for file-driven runs)."""
from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = pathlib.Path(os.environ.get("NVFP4_BUILD", HERE / "build"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "ref"))

import gen_nvfp4_dot as gen  # noqa: E402
import nvfp4_ref as ref      # noqa: E402

P = gen.model_params()


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))}\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}")
    return r.stdout


def tool_version(cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        return "not installed"


# ------------------------------------------------------------------------------- RTL generation
def gen_rtl(name: str, *, const: dict | None = None, rows: int = 1, out: pathlib.Path | None = None):
    """Write common + top (+ file-driven testbench) for one design into a directory; returns it.
    const = dict(fixture=, leaf=, rows=[...], blocks=(b0, b1) | None, prefix="", tile=False)."""
    d = out or BUILD / "rtl" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "nvfp4_common.sv").write_text(gen.common_sv(P))
    if const is None:
        (d / f"{name}.sv").write_text(gen.reloadable_sv(P, rows, name))
        nrows = rows
    else:
        codes, scales, origin = gen.fixture_rows(const["fixture"], const["leaf"], const["rows"],
                                                 const.get("blocks"), P, const.get("prefix", ""))
        if const.get("tile"):   # a full-length row whose weight table has only these blocks, repeated
            reps = P["nblk"][const["leaf"]] // scales.shape[1]
            codes, scales = np.tile(codes, (1, reps, 1)), np.tile(scales, (1, reps))
            origin += f", repeated {reps} times"
        (d / f"{name}.sv").write_text(gen.const_sv(P, name, codes, scales, origin))
        nrows = len(const["rows"])
    (d / f"tb_{name}.sv").write_text(gen.tb_sv(P, name, nrows, const is not None))
    return d


# ------------------------------------------------------------------------------- Verilator
def build_verilator(name: str, rtl_dir: pathlib.Path, rows: int, const: bool) -> pathlib.Path:
    srcs = [rtl_dir / "nvfp4_common.sv", rtl_dir / f"{name}.sv", HERE / "tb" / "sim_main.cpp"]
    key = hashlib.sha256(b"".join(s.read_bytes() for s in srcs)).hexdigest()[:16]
    obj = BUILD / "verilator" / f"{name}_{key}"
    exe = obj / f"V{name}"
    if exe.exists():
        return exe
    obj.mkdir(parents=True, exist_ok=True)
    cflags = f'-O2 -DVTOP=V{name} -DVTOP_H=\\"V{name}.h\\" -DDUT_ROWS={rows}' + (" -DCONST_VARIANT" if const else "")
    run(["verilator", "--cc", "--exe", "--build", "-j", "4", "-O3", "--x-assign", "fast", "--x-initial", "fast",
         "-Wno-fatal", "--top-module", name, "--Mdir", str(obj), "-CFLAGS", cflags,
         str(srcs[0]), str(srcs[1]), str(srcs[2])])
    return exe


def run_job(exe: pathlib.Path, nb: int, wc, ws, xc, xs, sched, dut_rows: int = 1, tag: str = "job"):
    """wc [RW, NB, 8] packed bytes (or None), ws [RW, NB], xc [A, NB, 8], xs [A, NB],
    sched [NVEC, 3] (first weight row, activation row, gap seed) -> uint32 [NVEC, dut_rows]."""
    jd = BUILD / "jobs"
    jd.mkdir(parents=True, exist_ok=True)
    job, out = jd / f"{tag}.bin", jd / f"{tag}.out"
    wc = np.zeros((0, nb, 8), np.uint8) if wc is None else np.ascontiguousarray(wc, np.uint8)
    ws = np.zeros((0, nb), np.uint8) if ws is None else np.ascontiguousarray(ws, np.uint8)
    xc, xs = np.ascontiguousarray(xc, np.uint8), np.ascontiguousarray(xs, np.uint8)
    sched = np.ascontiguousarray(sched, np.uint32)
    assert wc.shape[1:] == (nb, 8) and xc.shape[1:] == (nb, 8) and ws.shape[1:] == (nb,) and xs.shape[1:] == (nb,)
    with open(job, "wb") as f:
        np.array([0x3144564E, nb, wc.shape[0], xc.shape[0], sched.shape[0], dut_rows, 0, 0], np.uint32).tofile(f)
        for a in (wc, ws, xc, xs, sched):
            a.tofile(f)
    try:
        log = run([str(exe), str(job), str(out)])
        res = np.fromfile(out, np.uint32).reshape(sched.shape[0], dut_rows)
    finally:
        for p in (job, out):
            if p.exists():
                p.unlink()
    return res, int(log.split("cycles=")[1].split()[0])


# ------------------------------------------------------------------------------- Icarus (file-driven)
def stim_words(valid, last, xc, xs, wc=None, ws=None) -> list[int]:
    """Per-clock stimulus words for the generated testbench. xc [N, 8] bytes, xs [N];
    wc [N, R, 8], ws [N, R] for the reloadable top."""
    n = len(valid)
    words = []
    xcw = np.ascontiguousarray(xc, np.uint8).view("<u8").reshape(n)
    for i in range(n):
        w = int(valid[i]) | (int(last[i]) << 1) | (int(xs[i]) << 2) | (int(xcw[i]) << 10)
        if wc is not None:
            for r in range(wc.shape[1]):
                base = 74 + r * 72
                w |= int(ws[i, r]) << base
                w |= int(np.ascontiguousarray(wc[i, r]).view("<u8")[0]) << (base + 8)
        words.append(w)
    return words


def stream(nb: int, wc, ws, xc, xs, pairs, gaps=None):
    """Lay vectors out clock by clock. wc [RW, NB, 8] or None (const), pairs = [(w_row, x_row)].
    gaps: optional {vector index: idle clocks inserted after its first block}."""
    v, l, xcs, xss, wcs, wss = [], [], [], [], [], []
    for k, (wr, xr) in enumerate(pairs):
        for b in range(nb):
            v.append(1); l.append(int(b == nb - 1)); xcs.append(xc[xr, b]); xss.append(xs[xr, b])
            if wc is not None:
                wcs.append(wc[wr, b][None]); wss.append(ws[wr, b][None])
            if gaps and b == 0 and gaps.get(k):
                for _ in range(gaps[k]):
                    v.append(0); l.append(0); xcs.append(np.zeros(8, np.uint8)); xss.append(0)
                    if wc is not None:
                        wcs.append(np.zeros((1, 8), np.uint8)); wss.append(np.zeros(1, np.uint8))
    return stim_words(v, l, np.array(xcs), np.array(xss),
                      np.array(wcs) if wc is not None else None, np.array(wss) if wc is not None else None)


def build_icarus(name: str, sources: list[pathlib.Path], defines=(), tag: str = "") -> pathlib.Path:
    out = BUILD / "icarus" / f"{name}{tag}.vvp"
    out.parent.mkdir(parents=True, exist_ok=True)
    run(["iverilog", "-g2012", "-s", "tb", "-o", str(out), *[f"-D{d}" for d in defines], *map(str, sources)])
    return out


def run_icarus(vvp: pathlib.Path, words: list[int], rows: int = 1, vcd: pathlib.Path | None = None,
               plusargs=(), tag: str = "ic") -> np.ndarray:
    d = BUILD / "icarus"
    stim, out = d / f"{tag}.hex", d / f"{tag}.out"
    stim.write_text("\n".join(f"{w:x}" for w in words) + "\n")
    cmd = ["vvp", "-n", str(vvp), f"+stim={stim}", f"+out={out}", f"+n={len(words)}", *plusargs]
    if vcd is not None:
        cmd.append(f"+vcd={vcd}")
    run(cmd)
    res = [int(x, 16) for x in out.read_text().split()]
    return np.array([[(r >> (32 * k)) & 0xFFFFFFFF for k in range(rows)] for r in res], np.uint32).reshape(-1, rows)
