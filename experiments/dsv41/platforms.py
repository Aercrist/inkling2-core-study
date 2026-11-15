"""The platforms compared in every experiment, built once with the GPU calibration applied.

calibration mode:
  full  both fitted GPU parameters (HBM efficiency, per-stage overhead): the as-built engine
  eff   only the fitted HBM efficiency (same HBM, same physics): projections and MDA-1
  none  the YAML's own values (FPGA: different memory technology)
layout:
  engine  stream deepseek-ultrafast's padded in-memory expert layout (as built)
  disk    stream on-disk bytes (any new software or silicon can pick an unpadded layout)
"""
from __future__ import annotations

from functools import lru_cache

from sim.models.dsv41.calibrate import Calibration, calibrate
from sim.hardware import Hardware, load_hw
from sim.models.dsv41.modelspec import ModelSpec

PLATFORMS = {
    "gpu_tp4": ("configs/hw/dsv41/gb300_tp4.yaml", "4x GB300, deepseek-ultrafast (calibrated)", "full", "engine"),
    "gpu_tp8": ("configs/hw/dsv41/b300_tp8.yaml", "8x B300, deepseek-ultrafast (calibrated)", "full", "engine"),
    "gpu_mk_tp4": ("configs/hw/dsv41/gpu_megakernel_tp4.yaml", "4x GB300, barrier-synced persistent kernel (projected)", "eff", "engine"),
    "gpu_df_tp4": ("configs/hw/dsv41/gpu_dataflow_tp4.yaml", "4x GB300, dataflow-style software (projected)", "eff", "disk"),
    "asic": ("configs/hw/dsv41/asic_mda1.yaml", "MDA-1 x4 (proposed)", "eff", "disk"),
    "fpga": ("configs/hw/dsv41/fpga_v80.yaml", "16x Alveo V80 FPGA", "none", "disk"),
    "rubin_tp4": ("configs/hw/dsv41/rubin_tp4.yaml", "4x Rubin, today's software (projected)", "full", "engine"),
    "rubin_df_tp4": ("configs/hw/dsv41/rubin_dataflow_tp4.yaml", "4x Rubin, dataflow-style software (projected)", "eff", "disk"),
    "asic_hbm4": ("configs/hw/dsv41/asic_mda1_hbm4.yaml", "MDA-1 HBM4 x4 (projected)", "eff", "disk"),
}


@lru_cache(maxsize=1)
def spec() -> ModelSpec:
    return ModelSpec.load()


@lru_cache(maxsize=1)
def cal() -> Calibration:
    return calibrate(spec())


def platform(key: str, **overrides) -> Hardware:
    path, lab, mode, _layout = PLATFORMS[key]
    c = cal()
    over: dict = {}
    if mode == "full":
        over = c.as_overrides()
    elif mode == "eff":
        over = {"hbm_eff": c.hbm_eff}
    over.update(overrides)
    hw = load_hw(path, **over)
    hw.name = lab if "name" not in overrides else overrides["name"]
    hw.notes = key
    return hw


def layout_tp(hw: Hardware) -> int:
    """Expert layout to stream on this platform (0 = on-disk bytes)."""
    key = hw.notes
    if key in PLATFORMS and PLATFORMS[key][3] == "engine":
        return hw.n_chips
    return 0


def label(key: str) -> str:
    return PLATFORMS[key][1]
