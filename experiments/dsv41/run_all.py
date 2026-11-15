"""Re-run the DeepSeek-V4.1 study's experiments on the shared engine and regenerate results/dsv41/.

The outputs must stay byte-identical to that study's committed results (tests/test_dsv41_unchanged.py)."""
from __future__ import annotations

import time


def main() -> None:
    t0 = time.time()
    from experiments.dsv41 import decode_experiments, prefill_tradestudy, scalesim_crosscheck, workload_analysis
    steps = [("workload analysis (Phase 1)", workload_analysis.main),
             ("SCALE-Sim cross-check", scalesim_crosscheck.run),
             ("decode experiments (calibration, comparison, sweeps, tornado, ablation)", decode_experiments.main),
             ("prefill / TTFT / trade study", prefill_tradestudy.main)]
    for name, fn in steps:
        t = time.time()
        fn()
        print(f"[{time.time() - t:6.1f}s] {name}")
    print(f"done in {time.time() - t0:.1f}s; see results/dsv41/")


if __name__ == "__main__":
    main()
