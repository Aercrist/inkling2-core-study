<!-- generated from docs/templates/decisions.md.j2 by experiments/render_docs.py; edit the template -->
# Decisions log

Judgment calls, in the order they were made. Each entry says what was decided, why, and what would
reverse it.

## Phase 0 · 2026-11-14

**D1 · Clean room.** Inputs are public sources and our own DeepSeek-V4.1 study
(`https://github.com/Aercrist/dsv41-accel-study` at `bc1e959a`), nothing else. That study was cloned read-only and re-run
before anything was reused: `make reproduce` and `make test` leave all 14 tracked
result and doc files byte-identical (`results/phase0_dsv41_reproduction.json`).

**D2 · Read at source.** The model card and the announcement were downloaded and read in full as
text, not through a summarizer. The serving material was read the same way. A first automated summary
of the announcement missed its one line on draft and verify; the full text has it, and links to the
engine write-ups that publish speeds.

**D3 · How to count parameters.** We report the Hub's convention (594,050,672,720, which the
headers reproduce exactly) because that is the published 594B. For arithmetic we use weights only
(594,050,641,920): block scales and second-level scales are format overhead, counted in bytes but
not as parameters.

**D4 · What "17B active" means.** The attention and router weights of every layer plus the routed
experts: 17.03B. It is the only definition that lands on the published figure.
The output projection, which every token also uses, is outside it; the simulator counts it anyway.

**D5 · Which GPU baseline.** The model card names 2 × B300 as a memory minimum. Every published speed
is on four or eight GPUs. The rule is to give the GPU its best published serving setup, so G0 is
reported at 4 × B300 (the comparison leads with it) and at 2 × B300 (the smallest recommended
deployment). Reverse if a 2-GPU measurement is published.

**D6 · Published engine speeds are calibration points.** They name the hardware, the layout and the
workload. Benchmark-site figures from providers that do not disclose hardware remain a sanity band.

**D7 · Speculation is in every design.** The checkpoint ships a draft model and every published speed
uses it. Acceptance comes from the published 3.3-3.5;
where an engine does not publish it (TokenSpeed), 3.5 is an assumption and
is labelled as one.

**D8 · Phase 0 arithmetic is not the simulator.** `experiments/phase0_workload.py` does header
arithmetic and first-order ceilings so that the predictions could be written before any simulation.
Its napkin numbers carry one borrowed constant, the HBM efficiency calibrated in the DeepSeek study
(83.8%); Phase 2 refits it.

**D9 · KV-cache formats.** MXFP8 is the quantized cache the public engines expose, so it is the
default. An NVFP4-style cache is carried as a sensitivity until we find a public implementation.

**D10 · Predictions were not bent to fit the hypotheses.** Two predictions depart from the hypotheses
as first written (energy at low batch in H2, convergence in H5). They are recorded as written in
[00](00_hypotheses.md) and will be scored like the rest.

**D12 · Tools.** Yosys, Verilator and Icarus Verilog came from system packages. Nothing was bought,
rented or called through a paid API.

## Phases 1 to 5 · 2026-11-15

**D13 · Lead baseline: the best B300 layout for each metric.** The predictions named 2x B300 as G0.
We first led with 4x B300; the second review pointed out that an HGX node has eight GPUs and that the
fit itself uses 8-GPU figures. The GPUs now get 2x, 4x or 8x B300, whichever is best for each metric:
8x for one user, 4x at the interactive point. Against 2x B300 alone the ratios are larger
(6.4x on cost).

**D14 · S's speed comes from a published machine, not from our own constant.** The first pass used an
assumed 0.3 µs per operation and put S near 8,000 tok/s per user, above our prior. The Llama 3.1 8B
check showed that a shipping wafer-scale SRAM machine runs at 54%
of its memory bound. S now takes that machine's whole time per weight operation (3.91 µs).
Reverses if a reloadable weight-stationary design publishes a measured figure.

**D15 · A pays the same latency as S.** Beyond its own memory time, A pays what S pays beyond its bank
pass. Same physics, same constant.

**D16 · The cross-architecture check is reported as failed.** 0 of 4 within
30%. We did not tune the simulator per machine to make it pass.

**D17 · One pooled attention tile.** The first S design gave each full-attention layer its own two HBM
stacks and was slower than the GPUs at 1M context. Pooling the KV memory and the attention engine is
the better design and is what a builder would do. The pooled engine's rate is an assumption, set to
what four A packages provide, and is in the sweep.

**D18 · Tiles by capacity.** A small model is not charged for one tile per layer.

**D19 · Measured GPU power, not a flat share of TDP.** 43% of TDP
at one request rising to 86% at 128, measured on H100 and carried
to B300 (gpu-power-batch). This lowers GPU power at batch 1 and raises it at high batch against the
60% first assumed.

**D20 · The attention-kernel timings stay out of the fit.** Including them pinned the HBM efficiency
at 1.0 and worsened every throughput point. They are a held-out check that the model fails, reported
in [06](06_simulator.md), with the correction in the sweep.

**D21 · The hybrid design was cut.** The plan allowed it. The routing sweep bounds what it could gain.

**D22 · The dense 8B rung uses published speeds, not the simulator.** The simulator missed all four
published Llama 3.1 8B figures, so the ladder shows the published ratios for that row and says so.

**D23 · A cost figure above the prior was checked before anything was written.** With the lane
constants assumed before synthesis, the central cost-per-token gain against the GPUs was
2.2x on the model as it then stood, above the 2-10x prior. It was checked by hand, its dominant assumption
was added to the sweep, and no post was drafted. The synthesized lane brought it inside the prior; the
first adversarial review then took it below.

**D25 · The synthesized block replaces assumptions; the design was not re-tuned to it.** Bank depth and
lanes per expert stayed as chosen before synthesis. Only the NVFP4 lane was built: the MXFP8 and BF16
lanes take the same area correction and keep their assumed energy.

**D26 · The first adversarial review was accepted almost in full.** Twelve findings; we changed the
model for eleven and listed the twelfth as a limitation ([05](05_limitations.md)). The largest were
ours to have caught: the GPU's prefill load was left out of the fit, and the cost headline was decode
only. The decode cost gain went from 9.5x to 1.8x.

**D27 · Packaging from the HBM chip's sourced parts, not its per-die rate.** The reviewer proposed
charging S the HBM chip's $1,250 a die. That rate is mostly HBM attach and the yield loss on HBM, which
a weight tile does not have. We rebuilt the cost from the same three sourced parts (substrate by area,
yield loss as a share of content, test per die) and put the reviewer's rate at the top of the sweep.

**D28 · One token demand for every chip.** Design cost is spread over the systems a stated demand needs
(5 trillion tokens a day), not over a number of dies. The demand is an assumption and the
largest single lever on cost; the break-even is reported.

**D29 · The design was not re-balanced after review.** The review showed S's attention tile is far too
small for its weights. A larger tile would restore much of the cost gain ([04](04_results.md),
section 19). We report the design as specified and the sweep beside it, so that a review that lowered
the result is not followed by a redesign that raises it.

**D30 · The write-up leads with medians.** The central case is shown beside them.
