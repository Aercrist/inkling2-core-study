"""Run every experiment and regenerate results/ and the generated docs. `make reproduce`."""
from __future__ import annotations

import time


def main() -> None:
    t0 = time.time()
    from experiments import calibration_gpu, calibration_llama, inkling, napkin, phase0_workload, predictions, render_docs, reviews, silicon_scaling, uncertainty
    from experiments.dsv41 import run_all as dsv41
    steps = [("workload from the checkpoint headers", phase0_workload.main),
             ("DeepSeek-V4.1 experiments on the shared engine (must stay byte-identical)", dsv41.main),
             ("GPU calibration on published Inkling 2 Small figures", calibration_gpu.main),
             ("Llama 3.1 8B on four memory architectures", calibration_llama.main),
             ("constants from the synthesized block, scaled to N4P-class", silicon_scaling.main),
             ("Inkling 2 Small experiments", inkling.main),
             ("uncertainty: ablation, tornado, Monte Carlo", uncertainty.main),
             ("review log and how far the headline moved", reviews.main),
             ("score the pre-registered predictions", predictions.main),
             ("napkin arithmetic beside the simulator", napkin.main),
             ("render docs from results", render_docs.main)]
    for name, fn in steps:
        t = time.time()
        fn()
        print(f"[{time.time() - t:6.1f}s] {name}")
    print(f"done in {time.time() - t0:.1f}s; see results/ and docs/")


if __name__ == "__main__":
    main()
