"""Place-and-route one generated design with OpenROAD-flow-scripts (ASAP7) inside Docker.

    python rtl/nvfp4/flow/orfs.py <tag> <top> <rtl_dir> <clock_ps> [cores]

Needs a running Docker daemon and the image `openroad/orfs` (free, Docker Hub). Work files go
to $NVFP4_ORFS (default /home/user/scratch/rtl/orfs_work/<tag>), never into the repo.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

WORK = pathlib.Path(os.environ.get("NVFP4_ORFS", "/home/user/scratch/rtl/orfs_work"))
IMAGE = os.environ.get("NVFP4_ORFS_IMAGE", "openroad/orfs:latest")
CORNER = "TC"  # ASAP7 RVT, TT libraries, 0.70 V

CONFIG = """export PLATFORM = asap7
export DESIGN_NAME = {top}
export DESIGN_NICKNAME = {tag}
export VERILOG_FILES = /work/rtl/nvfp4_common.sv /work/rtl/{top}.sv
export SDC_FILE = /work/constraint.sdc
export CORNER = {corner}
export CORE_UTILIZATION = {util}
export CORE_ASPECT_RATIO = 1
export CORE_MARGIN = 1
export PLACE_DENSITY = {density}
"""

SDC = """current_design {top}
set clk_period {period}
create_clock -name clk -period $clk_period [get_ports clk]
set_input_delay  [expr {{0.2 * $clk_period}}] -clock clk [all_inputs -no_clocks]
set_output_delay [expr {{0.2 * $clk_period}}] -clock clk [all_outputs]
"""


DOCKER_ROOT = pathlib.Path(os.environ.get("NVFP4_DOCKER_ROOT", "/home/user/scratch/rtl/docker"))


def docker_ok(start: bool = True) -> bool:
    """True if the daemon answers and the ORFS image is present. Starts `dockerd` (data root in
    scratch) when it is not running; pulls nothing."""
    def image() -> bool:
        try:
            return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0
        except FileNotFoundError:
            return False
    if image():
        return True
    if not start or shutil.which("dockerd") is None:
        return False
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        DOCKER_ROOT.mkdir(parents=True, exist_ok=True)
        with open(DOCKER_ROOT / "dockerd.log", "ab") as log:
            subprocess.Popen(["dockerd", "--data-root", str(DOCKER_ROOT / "data")], stdout=log, stderr=log,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        for _ in range(30):
            time.sleep(1)
            if subprocess.run(["docker", "info"], capture_output=True).returncode == 0:
                break
    return image()


def in_container(work: pathlib.Path, script: str, timeout: int = 7200) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "run", "--rm", "-v", f"{work}:/work", IMAGE, "bash", "-lc",
         "cd /OpenROAD-flow-scripts && source ./env.sh >/dev/null 2>&1; " + script],
        capture_output=True, text=True, timeout=timeout)


def run_flow(tag: str, top: str, rtl_dir: pathlib.Path, period_ps: float, cores: int = 2,
             util: int = 50, density: float = 0.60, force: bool = False, synth_only: bool = False) -> pathlib.Path:
    """Synthesis, floorplan, placement, CTS, global and detailed routing, final report.
    synth_only stops after Yosys/ABC (reports/.../synth_stat.txt)."""
    if synth_only:
        work = WORK / tag
        stat = work / "out" / "reports" / "asap7" / tag / "base" / "synth_stat.txt"
        if stat.exists() and not force:
            return work
        if not docker_ok():
            raise RuntimeError(f"Docker daemon or image {IMAGE} not available")
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
        (work / "rtl").mkdir(parents=True)
        for f in ("nvfp4_common.sv", f"{top}.sv"):
            shutil.copy(pathlib.Path(rtl_dir) / f, work / "rtl" / f)
        (work / "config.mk").write_text(CONFIG.format(top=top, tag=tag, corner=CORNER, util=util, density=density))
        (work / "constraint.sdc").write_text(SDC.format(top=top, period=period_ps))
        in_container(work, f"cd flow && make DESIGN_CONFIG=/work/config.mk WORK_HOME=/work/out NUM_CORES={cores} synth "
                           f"> /work/flow.log 2>&1; echo exit=$? >> /work/flow.log; chmod -R a+rwX /work")
        if not stat.exists():
            raise RuntimeError(f"ORFS synthesis failed for {tag}: see {work}/flow.log")
        return work
    if not docker_ok():
        raise RuntimeError(f"Docker daemon or image {IMAGE} not available (docker pull {IMAGE})")
    work = WORK / tag
    done = work / "out" / "results" / "asap7" / tag / "base" / "6_final.v"
    if done.exists() and not force:
        return work
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    (work / "rtl").mkdir(parents=True)
    for f in ("nvfp4_common.sv", f"{top}.sv"):
        shutil.copy(pathlib.Path(rtl_dir) / f, work / "rtl" / f)
    (work / "config.mk").write_text(CONFIG.format(top=top, tag=tag, corner=CORNER, util=util, density=density))
    (work / "constraint.sdc").write_text(SDC.format(top=top, period=period_ps))
    t0 = time.time()
    r = in_container(work, f"cd flow && make DESIGN_CONFIG=/work/config.mk WORK_HOME=/work/out NUM_CORES={cores} "
                           f"> /work/flow.log 2>&1; echo exit=$? >> /work/flow.log; chmod -R a+rwX /work")
    (work / "flow_seconds.txt").write_text(f"{time.time() - t0:.0f}\n")
    if not done.exists():
        raise RuntimeError(f"ORFS flow failed for {tag}: see {work}/flow.log\n{r.stderr[-2000:]}")
    return work


if __name__ == "__main__":
    tag, top, rtl_dir, period = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]), float(sys.argv[4])
    cores = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    w = run_flow(tag, top, rtl_dir, period, cores)
    rep = json.loads((w / "out" / "logs" / "asap7" / tag / "base" / "6_report.json").read_text())
    for k in sorted(rep):
        if any(s in k for s in ("design__instance__area", "instance__count", "timing__setup__ws", "power__total", "clock__period")):
            print(k, rep[k])


def synth_stat(tag: str) -> dict:
    """Cell area and count after Yosys/ABC, from ORFS's synth_stat.txt."""
    import re
    txt = (WORK / tag / "out" / "reports" / "asap7" / tag / "base" / "synth_stat.txt").read_text()
    m = re.search(r"Chip area for module.*?:\s*([\d.]+)", txt)
    mc = re.search(r"^\s*(\d+)\s+\S+\s+\d+\s+\S+\s+cells\s*$", txt, re.M)
    seq = sum(int(x) for x in re.findall(r"^\s*(\d+)\s+\S+\s+\d+\s+\S+\s+DFF\w+\s*$", txt, re.M))
    return {"synth_cell_area_um2": float(m.group(1)), "synth_cells": int(mc.group(1)), "synth_sequential_cells": seq}
