#!/usr/bin/env python3
"""Re-run the DeepSeek-V4.1 study from a clean clone and record whether it reproduces itself.

This study reuses that repo's simulator and render pipeline, so the first step (Phase 0) is to
show that the starting point regenerates its own committed results. The script runs
`make reproduce` and `make test` in the clone, then compares the SHA-256 of every committed
results/*.json and generated doc before and after.

    git clone --depth 1 https://github.com/Aercrist/dsv41-accel-study ../dsv41-accel-study
    python tools/check_dsv41_reproduction.py --repo ../dsv41-accel-study

Not part of `make reproduce` (needs the other repo); results/phase0_dsv41_reproduction.json is committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(cmd: list[str], cwd: pathlib.Path, env: dict) -> tuple[int, str, float]:
    t = time.time()
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr, time.time() - t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to a clean clone of Aercrist/dsv41-accel-study")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    repo = pathlib.Path(a.repo).resolve()
    env = dict(os.environ, PATH=f"{pathlib.Path(a.python).parent}{os.pathsep}{os.environ['PATH']}")

    git = lambda *args: subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).stdout.strip()
    commit = git("rev-parse", "HEAD")
    dirty_before = git("status", "--porcelain")
    tracked = [repo / p for p in git("ls-files").splitlines()
               if p.startswith("results/") and p.endswith(".json") or p.endswith(".md")]
    before = {str(p.relative_to(repo)): sha(p) for p in tracked}

    rc1, out1, t1 = run(["make", "reproduce", f"PY={a.python}"], repo, env)
    rc2, out2, t2 = run(["make", "test", f"PY={a.python}"], repo, env)
    after = {k: sha(repo / k) for k in before}
    changed = sorted(k for k in before if before[k] != after[k])

    m = re.search(r"(\d+) passed", out2)
    rtl = re.search(r"router RTL vs golden: (\d+)/(\d+) tokens match", out1)
    rec = {
        "repo": "https://github.com/Aercrist/dsv41-accel-study",
        "commit": commit,
        "clean_before": dirty_before == "",
        "make_reproduce_exit": rc1,
        "make_reproduce_seconds": round(t1, 1),
        "make_test_exit": rc2,
        "tests_passed": int(m.group(1)) if m else None,
        "rtl_ran": shutil.which("iverilog") is not None and rtl is not None,
        "rtl_match": f"{rtl.group(1)}/{rtl.group(2)}" if rtl else None,
        "files_compared": len(before),
        "json_results_compared": sum(1 for k in before if k.endswith(".json")),
        "files_changed": changed,
        "byte_identical": not changed,
        "clean_after": git("status", "--porcelain") == "",
        "figures_note": "PNG figures are regenerated but not hashed: their bytes depend on the matplotlib build.",
        "python": subprocess.run([a.python, "--version"], capture_output=True, text=True).stdout.strip(),
    }
    out = ROOT / "results" / "phase0_dsv41_reproduction.json"
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    sys.exit(0 if (rc1 == 0 and rc2 == 0 and not changed) else 1)


if __name__ == "__main__":
    main()
