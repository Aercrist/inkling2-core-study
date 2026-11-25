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
    if name == "fx":                                     # exact-sum variant (SPEC.md section 5)
        return "nvfp4_dot_fx", dict(fx=True), 1, False
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


def final_timing(name: str, period: float):
    """(setup slack, hold slack, cell area) of a finished run from its final ORFS report, or None."""
    import json
    tag = tag_of(name, period)
    base = orfs.WORK / tag / "out"
    rep = base / "logs" / "asap7" / tag / "base" / "6_report.json"
    if not (base / "results" / "asap7" / tag / "base" / "6_final.v").exists() or not rep.exists():
        return None
    r = json.loads(rep.read_text())
    area = r["finish__design__instance__area__stdcell"] - (r.get("finish__design__instance__area__class:tap_cell") or 0)
    return r["finish__timing__setup__ws"], r["finish__timing__hold__ws"], round(area, 3)


def flow_ended_without_result(name: str, period: float) -> bool:
    """True if the flow of this run ran to its end (flow.log has its exit line) and left no routed design."""
    log = orfs.WORK / tag_of(name, period) / "flow.log"
    return (final_timing(name, period) is None and log.exists()
            and log.read_text(errors="replace").rstrip().rsplit("\n", 1)[-1].startswith("exit="))


def search_points(lo: int, hi: int, step: int = 10) -> list[int]:
    """The two constraints that cut the bracket (lo fails, hi closes) in three, on the step grid."""
    n = (hi - lo) // step
    return sorted({lo + step * int(n * i / 3 + 0.5) for i in (1, 2)} - {lo, hi})


def clock_search(name: str, lo: int, hi: int, step: int = 10, start: bool = True) -> dict:
    """Smallest clock constraint that ends with non-negative setup and hold slack after routing,
    to `step` ps. `lo` is a constraint known to fail (or 0), `hi` one known to close. Each round
    runs the two constraints of `search_points` (two flows at a time, two cores each); the new
    bracket is the smallest constraint that closed and the largest that failed below it. Finished
    runs in the work directory are reused, so a second call starts nothing. start = False never
    starts a flow and returns an incomplete search if a run is missing."""
    rounds, points, complete = [], {}, True
    while hi - lo > step:
        pts = search_points(lo, hi, step)
        status = {q: "flow ended without a routed design (see flow.log in the work directory)"
                  for q in pts if flow_ended_without_result(name, q)}
        missing = [q for q in pts if final_timing(name, q) is None and q not in status]
        if missing and start:
            with cf.ThreadPoolExecutor(2) as ex:
                for job, st, _ in ex.map(run_one, [f"{name}:{q}" for q in missing]):
                    status[int(job.split(":")[1])] = st
                    print(f"{job:24s} {st}", flush=True)
        res = {}
        for q in pts:
            ft = final_timing(name, q)
            if ft is None and q not in status:
                complete = False                           # not allowed to start it
                continue
            res[q] = ({"constraint_ps": q, "setup_wns_ps": ft[0], "hold_wns_ps": ft[1], "cell_area_um2": ft[2],
                       "closed": bool(ft[0] >= 0 and ft[1] >= 0)} if ft else
                      {"constraint_ps": q, "setup_wns_ps": None, "hold_wns_ps": None, "cell_area_um2": None, "closed": False,
                       "flow_failed": status[q]})
        if not complete:
            break
        points.update(res)
        closed = [q for q in pts if res[q]["closed"]]
        new_hi = min(closed) if closed else hi
        failed = [q for q in pts if not res[q]["closed"] and q < new_hi]
        new_lo = max(failed) if failed else lo
        rounds.append({"bracket_ps": [lo, hi], "tried_ps": pts, "closed_ps": closed, "new_bracket_ps": [new_lo, new_hi]})
        lo, hi = new_lo, new_hi
    ordered = [points[q] for q in sorted(points)]
    return {"step_ps": step, "rounds": rounds, "points": ordered, "complete": complete,
            "largest_failing_constraint_ps": lo, "smallest_closing_constraint_ps": hi if complete else None,
            "monotonic": all(not a["closed"] or b["closed"] for a, b in zip(ordered, ordered[1:]))}


if __name__ == "__main__":
    jobs = sys.argv[1:]
    with cf.ThreadPoolExecutor(2) as ex:
        for job, status, dt in ex.map(run_one, jobs):
            print(f"{job:24s} {status} ({dt:.0f} s)", flush=True)
