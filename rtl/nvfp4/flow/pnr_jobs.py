"""Run a list of place-and-route jobs, two at a time.

    python rtl/nvfp4/flow/pnr_jobs.py rel:1800 const_r0:1800 const_k1:1800 ...

A job is <design>:<clock period in ps>; designs are named in DESIGNS below. Results stay in the
ORFS work directory (scratch); flow/power.py and run_all.py read them.
"""
from __future__ import annotations

import concurrent.futures as cf
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import orfs    # noqa: E402
import simlib  # noqa: E402

EXPERT = "data/fixtures/inkling2_small_L00_E000.npz"
ROWS_NPZ = "data/fixtures/inkling2_small_rows.npz"
WIDE_ROWS = list(range(16, 24))


def design(name: str):
    """name -> (top module, generator arguments, DUT rows, is_const)."""
    if name == "rel":
        return "nvfp4_dot", dict(), 1, False
    if name == "rel_x8":
        return "nvfp4_dot_x8", dict(rows=8), 8, False
    if name == "const_x8":
        return "nvfp4_dot_const_x8", dict(const=dict(fixture=EXPERT, leaf="w13", rows=WIDE_ROWS)), 8, True
    if name.startswith("const_r"):                       # one full w13 row of layer 0 expert 0
        r = int(name[7:])
        return f"nvfp4_dot_const_r{r}", dict(const=dict(fixture=EXPERT, leaf="w13", rows=[r])), 1, True
    if name.startswith("const_w2_r"):
        r = int(name[10:])
        return f"nvfp4_dot_const_w2_r{r}", dict(const=dict(fixture=EXPERT, leaf="w2", rows=[r])), 1, True
    if name.startswith("const_L"):                       # const_L13_E005: row 5 of that sample
        tag = name[6:]
        return f"nvfp4_dot_const_{tag}", dict(const=dict(fixture=ROWS_NPZ, leaf="w13", rows=[5], prefix=tag + "_")), 1, True
    if name.startswith("const_t"):                       # const_t16 or const_t1_b100: a 288-block row whose table
        parts = name[7:].split("_b")                     # holds t distinct blocks of row 0 (from block b), repeated
        t, b0 = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        return f"nvfp4_dot_{name}", dict(const=dict(fixture=EXPERT, leaf="w13", rows=[0], blocks=(b0, b0 + t), tile=True)), 1, True
    if name.startswith("const_k"):                       # const_k16 or const_k1_b37: first k blocks (or block b) of row 0
        parts = name[7:].split("_b")
        k, b0 = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        return f"nvfp4_dot_{name}", dict(const=dict(fixture=EXPERT, leaf="w13", rows=[0], blocks=(b0, b0 + k))), 1, True
    raise KeyError(name)


def tag_of(name: str, period: float) -> str:
    return f"{name}_p{int(period)}"


def run_one(job: str, cores: int = 2):
    name, period = job.split(":")
    top, kw, _, _ = design(name)
    d = simlib.gen_rtl(top, **kw)
    t0 = time.time()
    try:
        orfs.run_flow(tag_of(name, float(period)), top, d, float(period), cores)
        return job, "ok", time.time() - t0
    except Exception as e:  # noqa: BLE001
        return job, f"FAILED: {str(e)[:300]}", time.time() - t0


if __name__ == "__main__":
    jobs = sys.argv[1:]
    with cf.ThreadPoolExecutor(2) as ex:
        for job, status, dt in ex.map(run_one, jobs):
            print(f"{job:24s} {status} ({dt:.0f} s)", flush=True)
