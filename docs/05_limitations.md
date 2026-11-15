<!-- generated from docs/templates/05_limitations.md.j2 by experiments/render_docs.py; edit the template -->
# 05 · Limitations and review log

## What this study is

A model of five platforms serving one model, calibrated where public measurements exist. One small
circuit block was synthesized. No chip was built, no GPU was run, no token was generated. The GPU rows
reproduce other people's published measurements; everything else is a design on paper.

## Limits, in order of how much they could move the headline

1. **The time per weight operation of S is borrowed.** It is what a published wafer-scale SRAM machine
   takes on a dense 8B model, applied to a different design running a 594B mixture-of-experts model.
   Across the plausible range (half again S's own bank pass, to a many-chip SRAM system's time) the
   batch-1 speed gain runs from 1.4x to 10.4x.
2. **Cost depends on a token demand nobody knows.** Design cost is spread over the systems a demand of
   5 trillion tokens a day needs. At 0.5 trillion the cost gain is 0.42x; at 20 trillion,
   2.47x. The 5th to 95th percentile of the decode cost gain is 0.31x to 2.63x: the study does
   not show a cost advantage at 100 tok/s per user.
3. **The chips are costed without a seller's margin; GPUs are at street price.** At a 50% gross margin
   S's decode cost gain falls to 0.91x.
   Spares, tile assembly yield, software and bring-up are not costed either.
4. **The shared attention tile is sized by assumption and caps S's throughput.** A tile eight times
   larger gives 3.4x in cost instead of 1.8x. The tile runs close to full with no
   queueing modeled.
5. **The GPU model is fitted to 9 published figures from two engines.** It reproduces them to
   8.0% RMS. Held out one at a time, the worst is 64% (the only figure without the
   draft model); the rest are within 24%. How much prefill sat on the GPUs during the published
   throughput runs decides the GPU's attention efficiency, and with it the high-batch comparison; our
   reading now favours the GPU against the engine's own kernel timings ([06](06_simulator.md)).
6. **The GPU model is least constrained at batches of 2 to 16**, which is where the GPUs sit when every
   user must get 150 or 200 tok/s. The large cost gains at those thresholds ([04](04_results.md),
   section 17) carry that weakness.
7. **The simulator did not predict any of four published Llama 3.1 8B speeds within 30%** from public
   specifications. We used the misses as measurements ([06](06_simulator.md)). A reader may fairly
   weigh that as evidence the simulator's absolute numbers for unbuilt machines carry wide error bars.
   The Monte Carlo ranges are our statement of those error bars.
8. **GPU software is still improving.** The engines published their Inkling 2 Small numbers within a
   week of the weights. G1 is our estimate of where software can go (1.7x at batch 1).
9. **No routing traces.** Expert popularity and locality are swept across values measured on other
   models, without refitting the GPU model, which favours the GPU. The most skewed case lowers the
   speed gain to 3.3x.
10. **Power management is assumed to work.** Without it S's batch-1 tokens per joule fall from
    2.7x to 1.2x of the GPU's. SRAM leakage at operating temperature has no
    public figure at this node.
11. **Full utilization on every platform, and the same lifetime in the central case.** A machine of
    521 dies needs hundreds of concurrent sessions to exist. A chip laid out for one
    architecture may be retired sooner than a GPU; lifetime is in the sweep (2 to 5 years).
12. **The design was not re-tuned.** Under the latency rule used here (S's whole time per operation is a
    published machine's), deeper banks would cost no speed and would cut the die count, and so would
    sharing lanes between experts or a larger attention tile. The design is reported as specified.
13. **Small simplifications in the chip engine**: the router takes one pass whatever the row count,
    and expert passes use the average rows per expert, not the busiest. A reviewer's patch for both
    moved results by under 2%.

## Not modeled

- **Vision input.** The vision encoder is resident in the area count; no image workload was run.
- **Reasoning effort and output length.** One acceptance length (3.5 tokens per step, swept 3 to 4).
- **Reliability.** A 521-die system will have failed dies, links and banks. Repair,
  sparing and availability are not modeled.
- **Time.** A chip takes years. The model will have successors before silicon exists, which is the
  argument for reloadable weights and also a risk to any design sized to one model.
- **The fixed-weight design beyond one die.** F carries one shipped 8B chip's density and speed to
  93 dies. Nothing public confirms that.
- **Sampling, tokenization, networking, the host.** A fixed 2 µs per
  step and 500 W.
- **Quality.** All platforms run the same checkpoint at the same precision; none changes the model's
  outputs. The circuit block is checked bit-for-bit against a reference ([07](07_silicon.md)), but we
  did not run the model end to end.
- **Layout.** No floorplan, no timing closure, no signal or power integrity.

## What would change the conclusions

- A measured time per weight operation for a reloadable weight-stationary design. At 13.1 µs the
  speed gain is 1.4x.
- Real token demand for one model. Below about 1.6 trillion tokens a day S costs more than GPUs.
- A product requirement for fast tokens. At 200 tok/s per user S is 4x cheaper; at 100 it is level.
- A GPU engine that removes dependent launches. That takes 1.7x off the batch-1 speed gain,
  leaving 3.2x.
- A larger attention tile, or lanes shared between experts. Either moves cost in S's favour
  ([04](04_results.md), section 19).
- Routing traces showing strong skew. Then a hybrid (hot experts on-chip, cold in HBM) deserves the
  study we cut.

## Review log

Adversarial reviews were run by fresh agents that saw only a clean copy of the repository. Their
findings and what we did with each:

| # | review | severity | finding | what we did | effect |
|---|---|---|---|---|---|
| 1 | 1 | blocker | The GPU's high-batch decode was calibrated as if prefill took none of the GPUs' time, although the published throughput workload prefills about 3K new tokens per turn. Everything slow was absorbed into a low attention efficiency. | Refitted with each session's prefill counted on the same GPUs. The GPU's attention efficiency rose about fourfold; the first reading is kept as an alternative in docs/06. | tokens per joule and cost at the best point fell; speed unchanged |
| 2 | 1 | blocker | The cost headline was decode-only. On the published agentic workload prefill is a large share of the cost per output token and was never added. | Added the end-to-end cost and energy per output token with prefill, for each platform and for S paired with GPUs or HBM chips for prefill (docs/04). The decode-only figure is labeled as such everywhere. | end to end, S alone is worse than GPUs on cost; S for decode with HBM chips for prefill is ahead |
| 3 | 1 | major | S's packaging was an unsourced $300 a die while the HBM chip paid a sourced $2,500 for a two-die package; the attention tile's package was free; the sweep never reached the HBM chip's rate. | Packaging rebuilt from the same sourced parts as the HBM chip's package ($700 a die), the attention tile's packages added, board and cooling cost raised, and the sweep widened to the HBM chip's per-die rate. | cost gain down |
| 4 | 1 | major | Design cost was spread over 2 million dies, which implied a fleet serving 35 trillion tokens a day of one model, and a different demand than the HBM chip's. | Every chip's fleet is now sized for the same token demand (5 trillion a day, swept 0.5 to 20), with the break-even demand reported. | cost gain down; cost is now most sensitive to demand |
| 5 | 1 | major | S's per-operation latency subtracted a memory pass computed from the published machine's marketing bandwidth, then added S's own shorter pass. | S now takes the published machine's whole time per weight operation, with its own bank pass inside it. The best published single-user GPU figure (630 tok/s) is shown beside the modeled baseline. | speed gain down |
| 6 | 1 | major | Lifetime and utilization were equal by fiat and not swept; the chips were at cost while GPUs are at street price. | S's lifetime (2 to 5 years) and a seller's gross margin (0 to 50%) are in the sweep, and the margin's effect is reported on its own. Utilization stays equal and is listed as a limitation. | cost range wider and lower |
| 7 | 1 | major | The 100 tok/s per user threshold never binds S and always binds the GPU, and the batch grid skipped the GPU's true best batch. | The best batch is now found exactly by bisection for every platform, and the gains are reported at five thresholds. | tokens per joule and cost at the best point down slightly; the dependence on the threshold is shown |
| 8 | 1 | major | The central values sat high in the study's own Monte Carlo. | The write-up leads with the median and the 5th to 95th percentile; the latency range is centred on the published machine. | presentation |
| 9 | 1 | major | The synthesized clock was applied to the multiplier lanes but the attention engines kept their area at 1.5 GHz. | Attention engines' area now grows in proportion at the synthesized clock. | a few more dies |
| 10 | 1 | minor | A shallow merge of overrides made six rows of the tornado silently revert S to its pre-synthesis lanes. | Overrides are deep-merged; the tornado was regenerated. | tornado figure corrected; Monte Carlo was not affected |
| 11 | 1 | minor | Yield and the text assumed 80% of the die was repairable memory; after synthesis it is about half. | The repairable share is computed from the area model; the text is rendered from it. | die cost up |
| 12 | 1 | minor | No area or leakage for the sliding-window KV cache, KV PHY area outside the die count, KV capacity not enforced, tiles above their die limit, no queueing at the shared attention tile, router and busiest-expert passes simplified. | Window-cache SRAM, KV PHY area, the tile limit and KV capacity checks added. Queueing at the attention tile and the two pass simplifications (under 2% by the reviewer's own patch) are listed as limitations. | more dies; small |
| 13 | 2 | blocker | S's prefill ran one prompt through its 66-tile pipeline, so all but one tile sat idle. With prompts pipelined its prefill is about seven times cheaper, and the conclusions that S should not do its own prefill did not survive. | Every platform's prefill is now costed at the number of prompts in flight that makes it cheapest. The prefill, end-to-end, README, claims and post text were rewritten. | end to end, S alone moved from 2.7x more expensive than GPUs to cheaper; decode headline unchanged |
| 14 | 2 | major | The GPU baseline compared only 2x and 4x B300, although an HGX node has eight GPUs and the fit itself uses 8-GPU figures. 8x B300 is faster for one user. | The GPUs now get 2x, 4x or 8x B300, whichever is best for each metric, in the headline, the thresholds, the ladder and the uncertainty analysis. | speed gain down by about a quarter; gains at high per-user speed down |
| 15 | 2 | major | Attention operations paid the borrowed per-weight-operation time on top of their own, although that time is a published machine's time per token divided by its weight operations and so already contains its attention. The router, a weight operation, did not pay it. | Attention no longer pays it; the router does. | speed gain up |
| 16 | 2 | major | The claim that reloadable weights cost about 4x in speed compares two vendors' published latencies; the synthesized fixed-weight datapath closes at the same clock. The claim that the price is in memory and not in arithmetic ignored that multiplier lanes are a large share of S's silicon. | The speed claim was withdrawn everywhere and replaced with a statement that the two speeds are not separable; the memory-versus-arithmetic sentence was rewritten. | presentation |
| 17 | 2 | major | The text claimed two adversarial reviews when only one had been run, and scored the prediction on their effect. | That prediction stays pending until both reviews are logged; this log now has both. | presentation |
| 18 | 2 | major | Prediction scoring: P25 was scored although no rung fits one package; P28 landed inside only with an LPDDR pool chosen after the prediction; P12 is inside only against 2x B300. | P25 is scored untestable; the notes on P28 and P12 say so. | 21 of 34 scorable predictions inside, from 21 of 35 |
| 19 | 2 | major | Public wording: claims stronger than the results in places (a range rounded up, a cost figure at the least certain operating point stated without its range, 'built' for a block that was simulated and placed and routed, published figures miscounted), and the independence statement missing from some public text. | All corrected in the public text, which is kept outside this repository. | presentation |
| 20 | 2 | minor | The headline table showed batch and per-user speed from the cost-optimal point beside tokens per joule from another batch; the best-batch search tried only three candidates; the attention-tile table labeled the batch as sessions. | The table gives each metric with its own batch; the search tries seven candidates below the limit; the column was fixed. | small |
| 21 | 2 | minor | The tornado held the latency beyond the bank pass fixed while the Monte Carlo held the whole time per operation fixed, so bank depth and clock moved speed in one and not the other. | Both now hold the whole time per operation fixed. That deeper banks would then be cheaper at no cost in speed is listed as a design not re-tuned. | tornado ranking changed; Monte Carlo immaterial |
| 22 | 2 | minor | 'Every die the same design' was untrue (weight, hub and attention-tile dies); the diagram said seven expert dies when a layer averages more than ten and spans tiles, while dispatch was charged on the die-to-die link; wake latency did not follow the clock; MXFP8 and BF16 lanes took the synthesized area penalty but not its energy ratio. | Three die designs, each further one costing a mask set; experts on a second tile are reached over the serial link; wake latency is 15 cycles at the synthesized clock; the MXFP8 and BF16 lanes take both ratios. | small |
| 23 | 2 | minor | The packaging source key used by the first review's fix did not resolve in the sources page; a table of corrections quoted post-review numbers as if they were the pre-review state. | Source added; the correction row now gives die counts only. | none |

How far the headline (S against the GPUs, central case) moved, review by review:

| | before the reviews | after | factor |
|---|---|---|---|
| speed for one user | 7.4x | 4.1x | 1.81x |
| tokens per joule, one user | 4.4x | 2.7x | 1.64x |
| tokens per joule, best point | 6.7x | 4.4x | 1.52x |
| decode cost per token, best point | 9.5x | 1.8x | 5.27x |

The largest change was 5.3x. We had predicted 1.5x (range
1.2x to 3x): the study was less robust to review than we expected, and almost all of
the movement was in cost.

### Corrections made before review

These are changes to our own first results, each made because a check failed.

| what we first had | what was wrong | what it became |
|---|---|---|
| S at about 8,000 tok/s per user, 15x the GPUs | per-operation latency was an unsourced 0.3 µs | latency taken from a published SRAM machine: 2,960 tok/s, 4.1x |
| GPU power at a flat 60% of TDP | measured data shows 43% at one request rising to 86% at 128 (gpu-power-batch) | power follows the measured curve |
| B300 at $45,000 | reseller quotes are $56,000 to $76,000 with networking (b300-price) | $60,000, swept |
| S slower than the GPUs at 1M context | each full-attention layer had its own two HBM stacks | one pooled attention tile |
| small models charged for one tile per layer | a model that fits one package needs no board links | tiles sized by die count |
| A with 0.3 µs per operation while S paid a published machine's latency | same physics, different constants | A pays the same latency as S |
| multiplier lane assumed at 12 µm² and 1.5 GHz | the block was placed and routed ([07](07_silicon.md)) | 34 µm² and 0.62 GHz; in today's model that is 384 against 521 dies |
| best batch searched on a coarse grid | it penalized the GPU, whose best batch fell between grid points | one fine grid for every platform |
