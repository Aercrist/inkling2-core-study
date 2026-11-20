<!-- generated from docs/templates/00_hypotheses.md.j2 by experiments/render_docs.py; edit the template -->
# 00 · Hypotheses and pre-registered predictions

**Frozen on 2026-11-14, before any simulation of Inkling 2 Small.** We wrote the predictions below
from the checkpoint headers, the published serving figures and first-order arithmetic
([01 · Workload](01_workload.md)). They are in [`data/preregistration.yaml`](../data/preregistration.yaml).
We did not edit them after the commit that added them (`make test` checks a hash).
`experiments/predictions.py` fills the "observed" column from `results/predictions.json`.

## Scorecard

**21 of 35 scored predictions are inside their range**; 1 are pending.

| group | inside / scored |
|---|---|
| baseline | 5 / 5 |
| H1 | 1 / 3 |
| H2 | 7 / 8 |
| H3 | 3 / 5 |
| H4 | 0 / 6 |
| H5 | 2 / 4 |
| limits | 2 / 2 |
| method | 1 / 3 |

| hypothesis | verdict | in one line |
|---|---|---|
| H1 · same memory, better wiring | **held** | a chip on HBM has 0.85x the speed of good software on the same GPUs at batch 1; it is better on power and price, not speed |
| H2 · moving the weights | **held for speed and energy, not for cost** | 4.1x speed and 4.4x tokens per joule against the best layout of the GPUs; decode cost is about equal at 100 tok/s per user (1.8x central, 1.00x median) |
| H3 · the price of reloadable weights | **held in silicon; not in a single lane** | 5.6x the dies; but a fixed weight in a multiplier saves only 1.13x in area (P15): the cost is in the memory |
| H4 · the ladder | **failed** for Inkling-shaped rungs | the gain decreases from 4.1x to 2.8x as the model becomes smaller; we predicted an increase to 40x |
| H5 · long context | **held in part, as predicted** | the lead of S over an HBM chip decreases from 3.8x to 2.9x at 1M tokens |

The predictions named 2x B300 as G0 (the smallest deployment). The study gives first the best B300 layout for each metric, which is
the better layout of the GPUs. Thus several observed values show both figures.

### Why the misses missed

We scored several predictions against results that two adversarial reviews changed later
([05](05_limitations.md)). The verdicts below are against the final results.

- **P04** A / G1 per-user speed at equal HBM bandwidth, batch 1: predicted 1 (0.9 to 1.15), observed 0.848. The model gives A the latency per operation of a published SRAM machine, and we did not expect that.
- **P05** A / G1 tokens per joule, batch 1: predicted 1.7 (1.2 to 2.5), observed 0.952. The cause is the same as for P04: the batch-1 energy advantage of A over good GPU software disappears.
- **P07** S: number of reticle-class dies holding every weight in SRAM, N4P-class, checkpoint precision: predicted 250 (160 to 330), observed 521. The prediction counted storage only. The multiplier lanes, at their synthesized size, and the window cache of the sessions add the remainder.
- **P15** Synthesis: reloadable / fixed-weight cell area of the dot-product datapath (logic only, same function): predicted 1.8 (1.3 to 3), observed 1.13. A fixed weight in a lane folds the multipliers away. But the adder tree and the 32-bit accumulator stay, and they are most of the block.
- **P16** Synthesis: reloadable / fixed-weight dynamic energy per dot product: predicted 2 (1.3 to 3.5), observed 1.12. The cause is the same as for P15: the accumulator dominates and does not become smaller.
- **P20** Ladder, S / best GPU per-user speed at batch 1, at 594B: predicted 12 (5 to 24), observed 4.11. The fastest GPU layout for one user is 8x B300, not the 2x that the prediction assumed. Also, the two reviews decreased the speed of S.
- **P21** Ladder, S / best GPU per-user speed at batch 1, hypothetical ~100B rung: predicted 15 (7 to 30), observed 4. We reasoned from dense models. A sparse model with a draft reads little per token at small sizes. Thus the speed of the GPU increases faster than the speed of the chip.
- **P22** Ladder, S / best GPU per-user speed at batch 1, hypothetical ~30B rung: predicted 25 (10 to 50), observed 3.29. The cause is the same as for P21.
- **P23** Ladder, S / best GPU per-user speed at batch 1, hypothetical ~8B rung: predicted 40 (15 to 80), observed 2.77. The cause is the same as for P21.
- **P24** Ladder: largest rung at which S fits on 8 dies or fewer: predicted 16 (8 to 30), observed 0. The window cache for 256 sessions alone needs more than eight dies. No rung in the scan fits.
- **P26** S / A per-user speed at 8K context, batch 1: predicted 8 (4 to 15), observed 3.77. S takes the full time per operation of a published machine, which is slower than our assumption.
- **P29** G0 per-user speed at 1M context / at 8K context: predicted 0.3 (0.15 to 0.5), observed 0.781. When the fit counts prefill, the GPUs attend about four times faster than our first inference. Thus long context costs them less.
- **P32** Llama 3.1 8B on four architectures: how many of the four published speeds the simulator reproduces within 30% on the first pass: predicted 2 (1 to 3), observed 0. Public specifications do not predict these machines. Refer to docs/06 for what we did about that.
- **P34** Largest change in any headline ratio caused by the two adversarial reviews: predicted 1.5 (1.2 to 3), observed 5.27. The first review alone moved the cost figure by more than the full predicted range.

## The designs

| name | what it is |
|---|---|
| **G0** | GPUs that run the best serving engine published today, calibrated to published speeds |
| **G1** | the same GPUs with dataflow-style software: one persistent program, no per-kernel waits |
| **A** | a decode chip that keeps the weights in HBM, as MDA-1 did in our DeepSeek-V4.1 study |
| **S** | a reloadable weight-stationary chip: every weight in on-die SRAM, loaded at boot or after a retrain |
| **F** | a fixed-weight chip: weights in the masks. A comparison point for the cost of reloadability |

## Operating points

| name | meaning |
|---|---|
| `batch1` | one request, 8K context, text only, DSpark on (8 verified rows per step), NVFP4 checkpoint |
| `best` | each platform at its own best batch subject to at least 100 tok/s per user, 32K context |
| `long` | one request at 1,048,576 tokens of context |
| `g0` | 2x B300, the smallest deployment the model card gives, running the best published engine (calibrated) |
| `g0x4` | 4x B300 at TP4 + EP4, the layout the engines publish numbers for |

## What the predictions rest on

Five facts from the workload page are the most important. We render all of them from
`results/phase0_workload.json`.

1. One token reads 19.3 GB of weights at batch 1 (derived). Half is 16-bit attention
   weight, not 4-bit expert weight.
2. One speculative step reads about 73 GB and gives about
   3.5 tokens (napkin). Speculation saves latency, not memory traffic.
3. Published engines get 476 to
   630 tok/s per user on four GPUs (measured, source). That is
   39% to 49%
   of the bandwidth ceiling (napkin).
4. To hold every weight in SRAM, a chip needs 160 to 310
   reticle-class dies at N4P-class density (napkin). One token uses
   4.9% of those bytes.
5. At full context a step reads 39 GB
   of KV cache with an MXFP8 target cache (derived).

## H1 · Same memory, better wiring

*At equal HBM bandwidth, a decode chip matches good GPU software on per-user speed. It wins only on
energy and cost.*

We expect H1 to hold, as it did for DeepSeek-V4.1.

| id | prediction | predicted (range) | why | observed | verdict |
|---|---|---|---|---|---|
| P04 | A / G1 per-user speed at equal HBM bandwidth, batch 1 | **1x** (0.9x to 1.15x) | Same memory, same physics, same execution model. The DeepSeek study found 0.92x-1.19x. | **0.848x** (four packages against four GPUs) | below |
| P05 | A / G1 tokens per joule, batch 1 | **1.7x** (1.2x to 2.5x) | No general-purpose compute to keep powered. The DeepSeek study found 1.8x (1.2x-3.0x across GPU power). | **0.952x** (four packages against four GPUs) | below |
| P06 | G1 / A cost per million tokens (above 1 means A is cheaper), batch 1 | **1.4x** (0.9x to 2x) | The DeepSeek study found 1.5x (0.9x-2.1x across GPU price). | **1.55x** (four packages against four GPUs) | inside |


## H2 · Moving the weights

*Keeping the weights on the chip multiplies per-user speed. At low batch it loses on energy, unless
idle weight storage is power-managed.*

We expect the first sentence to hold. We expect the second sentence to be **too pessimistic as worded**. Refer to
"Where we expect to differ" below.

| id | prediction | predicted (range) | why | observed | verdict |
|---|---|---|---|---|---|
| P07 | S: number of reticle-class dies holding every weight in SRAM, N4P-class, checkpoint precision [dies] | **250** (160 to 330) | 393 GB of weights at 0.25-0.5 mm2 per MiB of SRAM, about 600 mm2 of SRAM per die. | **521** | above |
| P08 | S per-user decode speed at batch 1 [tok/s per user] | **4,500** (2,000 to 9,000) | About 12 microseconds per layer once weights stop moving, which existing weight-stationary systems roughly achieve on dense models; 48 layers plus the draft; 3.5 tokens accepted per step. | **2,963** | inside |
| P09 | S / G0 per-user speed at batch 1 | **12x** (5x to 24x) | P08 over P01. | **7.84x** (against the best B300 layout: 4.1x) | inside |
| P10 | S / G0 tokens per joule at batch 1, idle weight banks power-managed | **6x** (2x to 15x) | S is about 12x faster, and a system of about 250 dies with links, clocks, KV memory and a host draws about twice what two GPUs draw at 60% of TDP. | **2.68x** (against the best B300 layout: 2.7x) | inside |
| P11 | S / G0 tokens per joule at batch 1, no power management (every bank at full voltage, clocks running) | **3.5x** (1x to 8x) | Physical leakage for 3.1 trillion bitcells is about a kilowatt (our estimate, not yet sourced), not the hundreds of kilowatts that product-level figures imply. We therefore expect S not to lose to a batch-1 GPU even unmanaged. This departs from the wording of H2 and from the DeepSeek study's bound, and is the prediction most likely to be wrong. | **1.23x** (against the best B300 layout: 1.2x) | inside |
| P12 | S at batch 1, unmanaged, against G0 at its own best batch: tokens per joule | **1x** (0.3x to 2x) | A GPU serving many users is several times more efficient than the same GPU serving one. | **0.445x** (against the best B300 layout, the study's lead baseline: 0.13x, below the range) | inside |
| P13 | S / G0 tokens per joule, each at its best operating point | **6x** (3x to 15x) | Memory read energy dominates S; streaming every expert through HBM dominates the GPU. | **14.7x** (against the best B300 layout: 4.4x) | inside |
| P14 | G0 / S cost per million tokens, each at its best operating point, NRE included | **3x** (1x to 8x) | About 250 large dies cost about as much as three or four GPUs; NRE is spread over few systems. | **6.4x** (against the best B300 layout: 1.8x) | inside |


## H3 · The price of reloadable weights

*Fixed weights beat reloadable ones on datapath area and energy by a bounded factor. The gap is
smaller at system level.*

We expect H3 to hold. We expect a smaller datapath factor than 8-bit intuition suggests, because a 4-bit
floating-point multiply is already tiny.

| id | prediction | predicted (range) | why | observed | verdict |
|---|---|---|---|---|---|
| P15 | Synthesis: reloadable / fixed-weight cell area of the dot-product datapath (logic only, same function) | **1.8x** (1.3x to 3x) | An E2M1 by E2M1 product is already a few dozen gates, so folding constants saves less than it would for 8-bit weights. Most of the remaining logic is the adder tree and the FP32 accumulation. | **1.13x** (ASAP7, placed and routed; a lane whose weight never changes. With a row's weights time-shared through one lane the fixed variant is larger) | below |
| P16 | Synthesis: reloadable / fixed-weight dynamic energy per dot product | **2x** (1.3x to 3.5x) | Folded constants remove toggling on the weight inputs and prune zero terms. | **1.12x** (same designs, gate-level activity on real tensors) | below |
| P17 | System: S die count / F die count | **4x** (2.5x to 6x) | Six transistors per stored bit against about one for a mask-programmed bit, less shared overhead. | **5.6x** | inside |
| P18 | System: F / S tokens per joule at the best operating point | **1.8x** (1.2x to 3x) | The gap shrinks at system level: attention, KV memory, links and the host are the same on both. | **1.29x** | inside |
| P19 | F: distinct die designs needed for one model version [designs] | **60** (40 to 100) | Every die holds different weights, so the count of designs equals the count of dies. | **93** | inside |


## H4 · The ladder

*The gain grows as models shrink. It jumps where the whole model fits on a handful of dies.*

The smaller rungs are hypothetical models with the ratios of Inkling 2 Small. We expect the per-user speed
gain to increase smoothly. We expect the jump to show in energy and cost, not in speed.

| id | prediction | predicted (range) | why | observed | verdict |
|---|---|---|---|---|---|
| P20 | Ladder, S / best GPU per-user speed at batch 1, at 594B | **12x** (5x to 24x) | P09. | **4.11x** (best GPU layout at batch 1: 8x B300) | below |
| P21 | Ladder, S / best GPU per-user speed at batch 1, hypothetical ~100B rung | **15x** (7x to 30x) | Fewer dies per layer shorten the hops; the GPU stays bound by per-kernel latency. | **4x** | below |
| P22 | Ladder, S / best GPU per-user speed at batch 1, hypothetical ~30B rung | **25x** (10x to 50x) | As above. | **3.29x** | below |
| P23 | Ladder, S / best GPU per-user speed at batch 1, hypothetical ~8B rung | **40x** (15x to 80x) | Speeds reported for Llama 3.1 8B, which we have yet to verify at source, span about 8x (wafer-scale SRAM) to about 74x (fixed-weight) over a single GPU. | **2.77x** | below |
| P24 | Ladder: largest rung at which S fits on 8 dies or fewer [billion parameters] | **16** (8 to 30) | Die count scales with weight bytes: about 250 dies at 594B. | **0** (Inkling-shaped rungs only) | below |
| P25 | Ladder: step in S / GPU tokens per joule (best operating point) between the last multi-board rung and the first rung that fits one package | **1.8x** (1.2x to 3x) | Board-level links, their idle power and most of the host disappear. | pending | untestable |


## H5 · Long context

*Near 1M tokens, KV traffic in the global-attention layers dominates, and the architectures converge.*

We expect H5 to hold **only in part**. With speculation on, an HBM-based design reads about
73 GB of weights per step. KV traffic does not match that until about
2.0M tokens, which is beyond the
1M limit. Without speculation and with a BF16 cache, the crossover is at
about 590,000 tokens. Thus KV traffic dominates
the weight-stationary design at long context. The designs converge, but not to parity.

| id | prediction | predicted (range) | why | observed | verdict |
|---|---|---|---|---|---|
| P26 | S / A per-user speed at 8K context, batch 1 | **8x** (4x to 15x) | P08 against a chip that still streams 73 GB of weights per step. | **3.77x** | below |
| P27 | S / A per-user speed at 1M context, batch 1, both with the same HBM bandwidth for KV | **2.3x** (1.3x to 4x) | At 1M tokens a step reads about 39 GB of KV. That is all of S's memory traffic but only about a third of A's, because A still streams its weights. The designs converge by about 3x, not to parity. | **2.88x** (S's attention tile has 26 HBM stacks; A's four packages have their own) | inside |
| P28 | Context above which S with LPDDR-only KV is slower per user than A [tokens of context] | **65,000** (30,000 to 150,000) | LPDDR at about 0.5 TB/s against HBM at about 13 TB/s effective. | **146,449** (with an LPDDR pool of 1.09 TB/s, chosen after the prediction, which assumed about 0.5 TB/s) | inside |
| P29 | G0 per-user speed at 1M context / at 8K context | **0.3x** (0.15x to 0.5x) | Published throughput already falls 2.4x from 32K to 256K sessions. | **0.781x** (on 4x B300: 0.83) | above |


## Baselines

| id | prediction | predicted (range) | why | observed | verdict |
|---|---|---|---|---|---|
| P01 | G0 (2x B300), per-user decode speed at batch 1 [tok/s per user] | **380** (300 to 470) | Published 4-GPU steps are about half weight streaming. Two GPUs have half the bandwidth, so the streaming half doubles and the fixed half does not. | **378** | inside |
| P02 | G1 / G0 per-user speed at batch 1 on 2x B300 (same GPUs, dataflow-style software) | **1.45x** (1.15x to 1.8x) | G1 reaches about 90% of the bandwidth ceiling; G0 on two GPUs is already more than half streaming. | **1.4x** | inside |
| P03 | G1 / G0 per-user speed at batch 1 on 4x B300 | **2x** (1.5x to 2.6x) | On four GPUs the published engines run at 39-49% of the bandwidth ceiling. | **1.68x** | inside |
| P35 | G0 (2x B300) tokens per joule at batch 1, GPU power at 60% of TDP [tokens per joule] | **0.29** (0.2 to 0.5) | P01 over two GPUs at 60% of an 1,100 W TDP. GPU power is an assumption and will be swept from 40% to 100%. | **0.262** (GPU power from the measured batch curve, not a flat 60%) | inside |
| P36 | G0 (2x B300) cost per million tokens at batch 1, owned, fully utilized [USD per million tokens] | **2** (1.4 to 3) | P01 with the DeepSeek study's cost assumptions: $45K per GPU all-in over 4 years, plus energy. | **2.63** | inside |


## Where we expect that purpose-built silicon is not better

| id | prediction | predicted (range) | why | observed | verdict |
|---|---|---|---|---|---|
| P30 | Prefill: S / G0 prefill tokens per second per dollar of hardware | **0.6x** (0.2x to 1.5x) | Prefill is compute-bound and GPUs use their tensor cores well there. We expect to keep prefill on GPUs. | **0.912x** | inside |
| P31 | S / G0 tokens per joule at batch 256 and above, 32K context | **4x** (2x to 10x) | At high batch a GPU shares each weight read across many tokens. | **2.41x** (batch 256; GPU at the better of its two layouts) | inside |


## The method itself

| id | prediction | predicted (range) | why | observed | verdict |
|---|---|---|---|---|---|
| P32 | Llama 3.1 8B on four architectures: how many of the four published speeds the simulator reproduces within 30% on the first pass [of 4] | **2** (1 to 3) | The single-GPU figure depends on software overheads we have not calibrated for that engine. | **0** | below |
| P33 | GPU model, leave-one-out on the published Inkling 2 Small speeds: median absolute error [percent] | **20** (10 to 35) | Twelve points from two engines, with workloads only partly specified. | **12.9** (largest 64%) | inside |
| P34 | Largest change in any headline ratio caused by the two adversarial reviews | **1.5x** (1.2x to 3x) | The DeepSeek study's reviews removed the proposed chip's speed lead entirely. | **5.27x** (after both reviews) | above |


## Where we expect to differ from the hypotheses as first written

- **H2, energy at low batch.** Our DeepSeek-V4.1 study found that an SRAM-only system was 12-13x worse on energy
  than GPUs at batch 1. That bound used the wall power of current SRAM-based products and scaled it
  by capacity. A count of bitcells gives about a kilowatt of leakage for this model.

  Thus we predict that S does not lose to a batch-1 GPU, even without power management (P11). We
  predict that S loses only against a GPU that runs at its own best batch (P12). If the simulator, with
  sourced leakage and link power, shows a different result, the DeepSeek bound was right and this prediction
  was wrong.
- **H5, convergence.** Refer to the text above: partial.
- **Above the prior.** Every central value above 10x is per-user speed at batch 1, where a GPU is
  furthest from its own best operating point. These values are S at full size (P09, central 12x)
  and the smaller rungs of the ladder (P21 to P23). The central energy and cost
  predictions for the full-size model are inside the prior, but their ranges go up to
  15x. We will check any result above 10x for the full-size model two times before we
  write it in a report.

## How this page is scored

For each prediction, the page gives the observed value and tells if it was inside the range. If it was not,
the reason is above. We report a prediction that misses as a miss. The ranges are wide on purpose. If every
central value is correct in a study, its ranges do no work.
