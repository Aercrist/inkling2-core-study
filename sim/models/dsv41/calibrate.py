"""Calibrate the GPU execution model against deepseek-ultrafast's reported decode step times.

Data: two measured step times (4 x GB300 TP4: 3.30 ms; 8 x B300 TP8: 2.71 ms). Peak HBM
bandwidth, link bandwidth and the collective latency floor are fixed from published specs
and microbenchmarks (configs/hw/*.yaml). Two parameters are free:

  * hbm_eff          achieved fraction of peak HBM bandwidth on weight streams
  * kernel_overhead  exposed time per kernel boundary (launch gap, ramp, tail, grid sync)

With two parameters and two measurements, the joint fit is exact by construction, so it is
a calibration and not a validation. We validate the model's structure in two ways:

  1. leave-one-out: fix hbm_eff at its published microbenchmark value, fit kernel_overhead
     on one configuration and predict the other (reported as the calibration error);
  2. plausibility: the jointly fitted parameters must land inside independently published
     ranges (hbm_eff vs. measured stream bandwidth; kernel_overhead vs. launch latency).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, least_squares

from ...engine import simulate_step
from ...hardware import Hardware, load_hw, load_yaml
from .modelspec import ModelSpec
from .workload import DecodeWorkload, decode_step

BASELINE = "data/dsv41/reported_baseline.yaml"
CONFIGS = {"gb300_tp4": "configs/hw/dsv41/gb300_tp4.yaml", "b300_tp8": "configs/hw/dsv41/b300_tp8.yaml"}


@dataclass
class Calibration:
    hbm_eff: float
    kernel_overhead: float
    predicted_ms: dict
    measured_ms: dict
    loo: dict  # leave-one-out: config -> (predicted, measured, error %)
    eff_prior: float

    def as_overrides(self) -> dict:
        return {"hbm_eff": self.hbm_eff, "kernel_overhead": self.kernel_overhead}


def workload_for(cfg_name: str, rep: dict, **kw) -> DecodeWorkload:
    """The as-built engine: its verify-head precision and its padded expert layout."""
    c = rep["configs"][cfg_name]
    tp = 4 if cfg_name == "gb300_tp4" else 8
    return DecodeWorkload(verify_head=c["verify_head"], accept_len=c["accept_len"], layout_tp=tp, **kw)


def step_ms(spec: ModelSpec, hw: Hardware, wl: DecodeWorkload) -> float:
    return simulate_step(decode_step(spec, wl), hw).step_time * 1e3


def calibrate(spec: ModelSpec | None = None, context: int = 2600, locality: float = 0.0,
              measured: dict | None = None, hw_overrides: dict | None = None) -> Calibration:
    spec = spec or ModelSpec.load()
    rep = load_yaml(BASELINE)
    hws = {k: load_hw(v, **(hw_overrides or {})) for k, v in CONFIGS.items()}
    wls = {k: workload_for(k, rep, context=context, locality=locality) for k in CONFIGS}
    meas = measured or {k: rep["configs"][k]["step_ms"] for k in CONFIGS}
    ops = {k: decode_step(spec, wls[k]) for k in CONFIGS}

    def model(k, eff, ko):
        return simulate_step(ops[k], hws[k].replace(hbm_eff=eff, kernel_overhead=ko)).step_time * 1e3

    def resid(x):
        return [(model(k, x[0], x[1] * 1e-6) - meas[k]) / meas[k] for k in CONFIGS]

    fit = least_squares(resid, x0=[0.85, 2.0], bounds=([0.3, 0.0], [1.0, 40.0]))
    eff, ko = fit.x[0], fit.x[1] * 1e-6
    pred = {k: model(k, eff, ko) for k in CONFIGS}

    # leave-one-out with the published bandwidth efficiency fixed
    eff0 = hws["gb300_tp4"].hbm_eff
    loo = {}
    names = list(CONFIGS)
    for fit_on, held in ((names[0], names[1]), (names[1], names[0])):
        ko1 = brentq(lambda x: model(fit_on, eff0, x * 1e-6) - meas[fit_on], 0.0, 80.0) * 1e-6
        p = model(held, eff0, ko1)
        loo[held] = {"fit_on": fit_on, "kernel_overhead_us": ko1 * 1e6, "predicted_ms": p,
                     "measured_ms": meas[held], "error_pct": 100.0 * (p - meas[held]) / meas[held]}
    return Calibration(eff, ko, pred, meas, loo, eff0)


def calibrated_hw(name: str, cal: Calibration | None = None, **kw) -> Hardware:
    cal = cal or calibrate()
    return load_hw(CONFIGS[name], **cal.as_overrides(), **kw)


if __name__ == "__main__":
    c = calibrate()
    print(f"joint fit: hbm_eff={c.hbm_eff:.3f}  kernel_overhead={c.kernel_overhead * 1e6:.2f} us")
    for k in c.measured_ms:
        print(f"  {k}: model {c.predicted_ms[k]:.3f} ms vs measured {c.measured_ms[k]:.3f} ms")
    for k, v in c.loo.items():
        print(f"  LOO predict {k} from {v['fit_on']}: {v['predicted_ms']:.3f} vs {v['measured_ms']:.3f} ms "
              f"({v['error_pct']:+.1f}%), kernel_overhead={v['kernel_overhead_us']:.2f} us")
    print(np.round([c.hbm_eff], 3))
