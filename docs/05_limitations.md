<!-- generated from docs/templates/05_limitations.md.j2 by experiments/render_docs.py; edit the template -->
# 05 · Limitations and review log

## What this study is

This study models five platforms that serve one model. We calibrated the model where public
measurements exist. We synthesized one small circuit block. We built no chip, we ran no GPU, and we
generated no token. The GPU rows reproduce published measurements from other people. All other
results are a design on paper.

## Limits, in order of how much they can move the headline

1. **We borrowed the time per weight operation of S.** A published wafer-scale SRAM machine takes
   this time on a dense 8B model. We apply it to a different design that runs a 594B
   mixture-of-experts model. The plausible range goes from 1.5 times the bank pass of S to the time
   of a many-chip SRAM system. Across this range, the batch-1 speed gain goes from
   1.4x to 10.4x.
2. **Cost depends on a token demand that nobody knows.** We spread the design cost over the systems
   that a demand of 5 trillion tokens a day needs.
   At 0.5 trillion, the cost gain is 0.42x. At 20 trillion, it is
   2.47x. The 5th to 95th percentile of the decode cost gain is 0.31x to 2.63x.
   The study does not show a cost advantage at 100 tok/s per user.
3. **The chip cost has no seller's margin, but the GPU cost is the street price.** At a 50% gross
   margin, the decode cost gain of S decreases to 0.91x.
   The chip cost also does not include spares, tile assembly yield, software and bring-up.
4. **An assumption sets the size of the shared attention tile, and the tile limits the throughput
   of S.** A tile that is eight times larger gives 3.4x in cost instead of 1.8x.
   The tile operates near its full capacity, and we did not model queueing.
5. **We fitted the GPU model to 9 published figures from two engines.** The model reproduces them to
   8.0% RMS. With one figure held out at a time, the worst error is 64% (the only
   figure without the draft model). The other errors are 24% or less. The quantity of prefill
   on the GPUs during the published throughput runs sets the attention efficiency of the GPU, and
   thus the high-batch comparison. Compared with the engine's own kernel timings, our
   reading now favours the GPU ([06](06_simulator.md)).
6. **The GPU model has the fewest constraints at batches of 2 to 16.** The GPUs operate at these
   batches when each user must get 150 or 200 tok/s. The large cost gains at those thresholds
   ([04](04_results.md), section 17) have that weakness.
7. **From public specifications, the simulator predicted none of four published Llama 3.1 8B
   speeds with an error of 30% or less.** We used the misses as measurements ([06](06_simulator.md)). A reader
   can fairly use that as evidence that the simulator's absolute numbers for unbuilt machines have
   wide error bars. The Monte Carlo ranges are our statement of those error bars.
8. **GPU software continues to improve.** The engines published their Inkling 2 Small numbers less
   than a week after the weights. G1 is our estimate of the level that software can get to
   (1.7x at batch 1).
9. **There are no routing traces.** We sweep expert popularity and locality across values measured
   on other models. We do not refit the GPU model for this sweep, which favours the GPU. The most
   skewed case decreases the speed gain to 3.3x.
10. **We assume that power management works.** Without it, the batch-1 tokens per joule of S
    decrease from 2.7x to 1.2x of the GPU's. There is no public figure for SRAM leakage at
    operating temperature at this node.
11. **Each platform has full utilization, and each has the same lifetime in the central case.** A
    machine of 521 dies has full utilization only if hundreds of concurrent
    sessions exist. It is possible that an operator retires a chip laid out for one architecture
    sooner than a GPU. Lifetime is in the sweep (2 to 5 years).
12. **We did not tune the design again.** The latency rule used here gives S the whole time per
    operation of a published machine. With this rule, deeper banks cost no speed and decrease the
    die count. Lanes shared between experts, or a larger attention tile, have the same effect. We
    report the design as specified.
13. **The chip engine has small simplifications.** The router takes one pass for any row count.
    Expert passes use the average rows per expert, not the rows of the busiest expert. A reviewer's
    patch for the two simplifications moved results by less than 2%.

## Not modeled

- **Vision input.** The area count includes the vision encoder. We ran no image workload.
- **Reasoning effort and output length.** There is one acceptance length (3.5 tokens per step, swept
  3 to 4).
- **Reliability.** Some dies, links and banks will fail in a 521-die system. We did not
  model repair, sparing and availability.
- **Time.** A chip takes years. The model will have successors before silicon exists. That is the
  argument for reloadable weights. It is also a risk to any design sized to one model.
- **The fixed-weight design beyond one die.** F carries the density and speed of one shipped 8B chip
  to 93 dies. No public data confirms that.
- **Sampling, tokenization, networking, the host.** These get a fixed 2 µs per
  step and 500 W.
- **Quality.** All platforms run the same checkpoint at the same precision. No platform changes the
  outputs of the model. We checked the circuit block bit-for-bit against a reference
  ([07](07_silicon.md)), but we did not run the model end to end.
- **Layout.** There is no floorplan, no timing closure, no signal or power integrity.

## What can change the conclusions

- A measured time per weight operation for a reloadable weight-stationary design. At 13.1 µs, the
  speed gain is 1.4x.
- Real token demand for one model. Below about 1.6 trillion tokens a day, S costs more than GPUs.
- A product requirement for fast tokens. At 200 tok/s per user, S is 4x cheaper. At 100 tok/s
  per user, the cost is equal.
- A GPU engine that removes dependent launches. That removes 1.7x from the batch-1 speed gain
  and leaves 3.2x.
- A larger attention tile, or lanes shared between experts. Each one moves cost in favour of S
  ([04](04_results.md), section 19).
- Routing traces that show strong skew. Then it is necessary to do the study that we cut: a hybrid
  design with hot experts on-chip and cold experts in HBM.

## Review log

Fresh agents did the adversarial reviews. Each agent saw only a clean copy of the repository. The
table shows their findings and our action for each finding.

| # | review | severity | finding | what we did | effect |
|---|---|---|---|---|---|
| 1 | 1 | blocker | The calibration of the GPU's high-batch decode assumed that prefill took none of the GPUs' time. But the published throughput workload prefills about 3K new tokens per turn. A low attention efficiency absorbed everything slow. | We refitted with each session's prefill counted on the same GPUs. The GPU's attention efficiency increased about fourfold. docs/06 keeps the first reading as an alternative. | tokens per joule and cost at the best point decreased; speed did not change |
| 2 | 1 | blocker | The cost headline was decode-only. On the published agentic workload, prefill is a large share of the cost per output token. We never added it. | We added the end-to-end cost and energy per output token with prefill (docs/04). We give them for each platform, and for S paired with GPUs or HBM chips for prefill. All decode-only figures now have a decode-only label. | end to end, S alone is worse than GPUs on cost; S for decode with HBM chips for prefill is better than GPUs |
| 3 | 1 | major | S's packaging was $300 a die without a source, but the HBM chip paid a sourced $2,500 for a two-die package. The attention tile's package was free. The sweep never reached the HBM chip's rate. | We rebuilt the packaging cost from the same sourced parts as the HBM chip's package ($700 a die). We added the attention tile's packages and raised the board and cooling cost. We widened the sweep to the HBM chip's per-die rate. | cost gain decreased |
| 4 | 1 | major | We spread the design cost over 2 million dies. That implied a fleet that serves 35 trillion tokens a day of one model, and a demand different from the HBM chip's. | We now size every chip's fleet for the same token demand (5 trillion a day, swept 0.5 to 20). We report the break-even demand. | cost gain decreased; cost is now most sensitive to demand |
| 5 | 1 | major | S's per-operation latency subtracted a memory pass computed from the published machine's marketing bandwidth. It then added S's own shorter pass. | S now takes the published machine's whole time per weight operation, with its own bank pass inside it. We show the best published single-user GPU figure (630 tok/s) beside the modeled baseline. | speed gain decreased |
| 6 | 1 | major | We set lifetime and utilization equal by decision and did not sweep them. The chips were at cost, but GPUs are at street price. | S's lifetime (2 to 5 years) and a seller's gross margin (0 to 50%) are in the sweep. We report the margin's effect separately. Utilization stays equal, and we list it as a limitation. | cost range wider and lower |
| 7 | 1 | major | The threshold of 100 tok/s per user never binds S and always binds the GPU. The batch grid skipped the GPU's true best batch. | Bisection now finds the exact best batch for every platform. We report the gains at five thresholds. | tokens per joule and cost at the best point decreased slightly; the results show the dependence on the threshold |
| 8 | 1 | major | The central values were high in the range of the study's own Monte Carlo. | The write-up starts with the median and the 5th to 95th percentile. The centre of the latency range is the published machine. | presentation |
| 9 | 1 | major | We applied the synthesized clock to the multiplier lanes, but the attention engines kept their area at 1.5 GHz. | The area of the attention engines now grows in proportion at the synthesized clock. | a few more dies |
| 10 | 1 | minor | A shallow merge of overrides made six rows of the tornado revert S to its pre-synthesis lanes without a warning. | We now deep-merge overrides. We regenerated the tornado. | tornado figure corrected; no effect on the Monte Carlo |
| 11 | 1 | minor | Yield and the text assumed that 80% of the die was repairable memory. After synthesis, it is about half. | We compute the repairable share from the area model. We render the text from it. | die cost increased |
| 12 | 1 | minor | The sliding-window KV cache had no area or leakage. KV PHY area was outside the die count. Nothing enforced KV capacity. Tiles were above their die limit. The shared attention tile had no queueing. The router and busiest-expert passes were simplified. | We added window-cache SRAM, KV PHY area, the tile limit and KV capacity checks. We list queueing at the attention tile and the two pass simplifications as limitations. By the reviewer's own patch, the two simplifications change results by less than 2%. | more dies; small |
| 13 | 2 | blocker | S's prefill ran one prompt through its 66-tile pipeline, so all tiles but one were idle. With pipelined prompts, its prefill is about seven times cheaper. The conclusions that S must not do its own prefill did not hold. | We now calculate every platform's prefill cost at the number of prompts in flight that makes it cheapest. We rewrote the prefill, end-to-end, README, claims and post text. | end to end, S alone moved from 2.7x more expensive than GPUs to cheaper; decode headline did not change |
| 14 | 2 | major | The GPU baseline compared only 2x and 4x B300. But an HGX node has eight GPUs, and the fit itself uses 8-GPU figures. 8x B300 is faster for one user. | The GPUs now get 2x, 4x or 8x B300, whichever is best for each metric. This applies in the headline, the thresholds, the ladder and the uncertainty analysis. | speed gain decreased by about a quarter; gains at high per-user speed decreased |
| 15 | 2 | major | Attention operations paid the borrowed per-weight-operation time in addition to their own time. But that time is a published machine's time per token divided by its weight operations. Thus it already contains that machine's attention. The router is a weight operation, and it did not pay that time. | Attention no longer pays it. The router pays it. | speed gain increased |
| 16 | 2 | major | The claim that reloadable weights cost about 4x in speed compares two vendors' published latencies. The synthesized fixed-weight datapath closes at the same clock. The claim that the cost is in memory and not in arithmetic ignored that multiplier lanes are a large share of S's silicon. | We withdrew the speed claim everywhere. A statement that the two speeds are not separable replaces it. We rewrote the memory-versus-arithmetic sentence. | presentation |
| 17 | 2 | major | The text claimed two adversarial reviews when only one was complete. It scored the prediction on their effect. | That prediction stays pending until the log has both reviews. This log now has both. | presentation |
| 18 | 2 | major | Prediction scoring: P25 had a score although no rung fits one package. P28 was inside only with an LPDDR pool chosen after the prediction. P12 is inside only against 2x B300. | P25 now has the score untestable. The notes on P28 and P12 state these conditions. | 21 of 34 scorable predictions inside, from 21 of 35 |
| 19 | 2 | major | Public wording: some claims were stronger than the results. A range was rounded up. A cost figure at the least certain operating point had no range. The text said 'built' for a block that we simulated, placed and routed. The count of published figures was incorrect. Some public text did not have the independence statement. | We corrected all of these in the public text. The public text is outside this repository. | presentation |
| 20 | 2 | minor | The headline table showed batch and per-user speed from the cost-optimal point beside tokens per joule from another batch. The best-batch search tried only three candidates. The attention-tile table labeled the batch as sessions. | The table gives each metric with its own batch. The search tries seven candidates below the limit. We corrected the column. | small |
| 21 | 2 | minor | The tornado held the latency beyond the bank pass fixed. The Monte Carlo held the whole time per operation fixed. Thus bank depth and clock moved speed in one and not in the other. | Both now hold the whole time per operation fixed. Deeper banks are then cheaper at no cost in speed. We list that as a design that we did not tune again. | tornado ranking changed; no material effect on the Monte Carlo |
| 22 | 2 | minor | 'Every die the same design' was untrue (weight, hub and attention-tile dies). The diagram said seven expert dies, but a layer averages more than ten and spans tiles. The model charged dispatch on the die-to-die link. Wake latency did not follow the clock. MXFP8 and BF16 lanes took the synthesized area penalty but not its energy ratio. | There are three die designs, and each design after the first costs a mask set. The serial link reaches experts on a second tile. Wake latency is 15 cycles at the synthesized clock. The MXFP8 and BF16 lanes take both ratios. | small |
| 23 | 2 | minor | The packaging source key used by the first review's fix did not resolve in the sources page. A table of corrections quoted post-review numbers as the pre-review state. | We added the source. The correction row now gives die counts only. | none |

The next table shows how far the reviews moved the headline (S against the GPUs, central case).

| | before the reviews | after | factor |
|---|---|---|---|
| speed for one user | 7.4x | 4.1x | 1.81x |
| tokens per joule, one user | 4.4x | 2.7x | 1.64x |
| tokens per joule, best point | 6.7x | 4.4x | 1.52x |
| decode cost per token, best point | 9.5x | 1.8x | 5.27x |

The largest change was 5.3x. We predicted 1.5x (range
1.2x to 3x). The study was less robust to review than we expected. Almost all of
the movement was in cost.

### Corrections made before review

These are changes to our own first results. We made each change because a check failed.

| our first result | the error | the corrected result |
|---|---|---|
| S at about 8,000 tok/s per user, 15x the GPUs | per-operation latency was 0.3 µs without a source | latency from a published SRAM machine: 2,960 tok/s, 4.1x |
| GPU power at a flat 60% of TDP | measured data shows 43% at one request and 86% at 128 (gpu-power-batch) | power follows the measured curve |
| B300 at $45,000 | reseller prices are $56,000 to $76,000 with networking (b300-price) | $60,000, swept |
| S slower than the GPUs at 1M context | each full-attention layer had its own two HBM stacks | one pooled attention tile |
| small models paid for one tile per layer | a model that fits one package needs no board links | tiles sized by die count |
| A with 0.3 µs per operation while S paid a published machine's latency | same physics, different constants | A pays the same latency as S |
| multiplier lane assumed at 12 µm² and 1.5 GHz | we placed and routed the block ([07](07_silicon.md)) | 34 µm² and 0.62 GHz; in today's model that is 384 against 521 dies |
| best batch found on a coarse grid | the grid gave a penalty to the GPU, because the best GPU batch fell between grid points | one fine grid for every platform |
