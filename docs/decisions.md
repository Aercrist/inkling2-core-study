<!-- generated from docs/templates/decisions.md.j2 by experiments/render_docs.py; edit the template -->
# Decisions log

These are the judgment calls, in the order that we made them. Each entry gives the decision, the reason, and
the condition that reverses it.

## Phase 0 · 2026-11-14

**D1 · Clean room.** The inputs are public sources and our own DeepSeek-V4.1 study
(`https://github.com/Aercrist/dsv41-accel-study` at `04617dc1`), nothing else. We cloned that study read-only and ran it again
before we reused anything. `make reproduce` and `make test` leave all 14 tracked
result and doc files byte-identical (`results/phase0_dsv41_reproduction.json`).

**D2 · Read at source.** We downloaded the model card and the announcement and read them in full as
text, not through a summarizer. We read the serving material in the same way. A first automated summary
of the announcement missed its one line on draft and verify. The full text has that line, and it links to the
engine write-ups that publish speeds.

**D3 · How to count parameters.** We report the convention of the Hub (594,050,672,720, which the
headers reproduce exactly) because that is the published 594B. For arithmetic we use weights only
(594,050,641,920). Block scales and second-level scales are format overhead. We count them in bytes but
not as parameters.

**D4 · What "17B active" means.** It means the attention and router weights of every layer plus the routed
experts: 17.03B. It is the only definition that gives the published figure.
Every token also uses the output projection, which is outside this definition. The simulator counts it.

**D5 · Which GPU baseline.** The model card names 2 × B300 as a memory minimum. Every published speed
is on four or eight GPUs. The rule is to give the GPU its best published serving setup. Thus we report G0
at 4 × B300 (the comparison gives it first) and at 2 × B300 (the smallest recommended
deployment). Reverse this decision if a source publishes a 2-GPU measurement.

**D6 · Published engine speeds are calibration points.** They name the hardware, the layout and the
workload. Benchmark-site figures from providers that do not disclose hardware stay a sanity band.

**D7 · Speculation is in every design.** The checkpoint ships a draft model and every published speed
uses it. Acceptance comes from the published 3.3-3.5.
Where an engine does not publish it (TokenSpeed), 3.5 is an assumption, and
we label it as an assumption.

**D8 · Phase 0 arithmetic is not the simulator.** `experiments/phase0_workload.py` does header
arithmetic and first-order ceilings. This let us write the predictions before any simulation.
Its napkin numbers use one borrowed constant, the HBM efficiency calibrated in the DeepSeek study
(83.8%). Phase 2 refits it.

**D9 · KV-cache formats.** MXFP8 is the quantized cache that the public engines expose, thus it is the
default. We keep an NVFP4-style cache as a sensitivity until we find a public implementation.

**D10 · We did not change predictions to fit the hypotheses.** Two predictions depart from the hypotheses
as first written (energy at low batch in H2, convergence in H5). We recorded them as written in
[00](00_hypotheses.md) and will score them the same as the others.

**D12 · Tools.** Yosys, Verilator and Icarus Verilog came from system packages. We did not buy or rent
anything, and we did not call a paid API.

## Phases 1 to 5 · 2026-11-15

**D13 · Lead baseline: the best B300 layout for each metric.** The predictions named 2x B300 as G0.
At first, our lead baseline was 4x B300. The second review showed that an HGX node has eight GPUs and that the
fit itself uses 8-GPU figures. The GPUs now get 2x, 4x or 8x B300, whichever is best for each metric:
8x for one user, 4x at the interactive point. Against 2x B300 alone the ratios are larger
(6.4x on cost).

**D14 · The speed of S comes from a published machine, not from our own constant.** The first pass used an
assumed 0.3 µs per operation and put S near 8,000 tok/s per user, above our prior. The Llama 3.1 8B
check showed that a shipped wafer-scale SRAM machine runs at 54%
of its memory bound. S now takes the whole time per weight operation of that machine (3.91 µs).
This decision reverses if a reloadable weight-stationary design publishes a measured figure.

**D15 · A pays the same latency as S.** Beyond its own memory time, A pays what S pays beyond its bank
pass. The physics is the same, thus the constant is the same.

**D16 · We report the cross-architecture check as failed.** 0 of 4 have an
error of 30% or less. We did not tune the simulator for each machine to make the check pass.

**D17 · One pooled attention tile.** The first S design gave each full-attention layer its own two HBM
stacks and was slower than the GPUs at 1M context. A design that pools the KV memory and the attention engine is
the better design, and a builder will use it. The rate of the pooled engine is an assumption. We set it to
the rate that four A packages supply, and it is in the sweep.

**D18 · Tiles by capacity.** The model does not charge a small model for one tile per layer.

**D19 · Measured GPU power, not a flat share of TDP.** The measured power is 43% of TDP
at one request and increases to 86% at 128. The measurement is from H100, and we carried it
to B300 (gpu-power-batch). Compared with the 60% that we first assumed, this decreases GPU power at batch 1
and increases it at high batch.

**D20 · The attention-kernel timings stay out of the fit.** With them in the fit, the HBM efficiency stayed
at 1.0 and every throughput point became worse. They are a held-out check that the model fails. We report
this in [06](06_simulator.md), and the correction is in the sweep.

**D21 · We cut the hybrid design.** The plan allowed this. The routing sweep bounds its possible gain.

**D22 · The dense 8B rung uses published speeds, not the simulator.** The simulator missed all four
published Llama 3.1 8B figures. Thus the ladder shows the published ratios for that row and says so.

**D23 · We checked a cost figure above the prior before we wrote anything.** With the lane
constants assumed before synthesis, the central cost-per-token gain against the GPUs was
2.2x on the model as it was then. That is above the 2-10x prior. We checked it by hand, added its dominant assumption
to the sweep, and drafted no post. The synthesized lane brought it inside the prior. Then the
first adversarial review brought it below the prior.

**D25 · The synthesized block replaces assumptions; we did not re-tune the design to it.** Bank depth and
lanes per expert stayed as we chose them before synthesis. We built only the NVFP4 lane. The MXFP8 and BF16
lanes take the same area correction and keep their assumed energy.

**D26 · We accepted the first adversarial review almost in full.** There were twelve findings. We changed the
model for eleven and listed the twelfth as a limitation ([05](05_limitations.md)). It was our task to
catch the two largest findings. The fit did not include the prefill load of the GPU, and the cost headline was decode
only. The decode cost gain went from 9.5x to 1.8x.

**D27 · Packaging from the sourced parts of the HBM chip, not its per-die rate.** The reviewer proposed
to charge S the $1,250 a die of the HBM chip. That rate is mostly HBM attach and the yield loss on HBM, which
a weight tile does not have. We rebuilt the cost from the same three sourced parts:

- substrate by area
- yield loss as a share of content
- test per die

We put the rate of the reviewer at the top of the sweep.

**D28 · One token demand for every chip.** We spread design cost over the systems that a stated demand needs
(5 trillion tokens a day), not over a number of dies. The demand is an assumption and has the
largest effect on cost of any one input. We report the break-even.

**D29 · We did not re-balance the design after review.** The review showed that the attention tile of S is much too
small for its weights. A larger tile restores much of the cost gain ([04](04_results.md),
section 19). We report the design as specified and the sweep beside it. Thus a redesign that increases
the result does not follow a review that decreased it.

**D30 · The write-up gives medians first.** The central case is beside them.
