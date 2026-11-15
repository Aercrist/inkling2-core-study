<!-- generated from docs/templates/02_design_rationale.md.j2 by experiments/render_docs.py; edit the template -->
# 02 · Design rationale

What the workload asks of hardware, and the five platforms the study compares. Numbers from the
checkpoint are **derived** ([01](01_workload.md)); design constants are in `configs/hw/*.yaml` with a
source key or the word *assumption*; results are **modeled** ([04](04_results.md)).

## What the workload asks for

1. **Almost every byte is a weight, and almost none of them are used.** The checkpoint is
   393 GB. One token touches 19.3 GB of it:
   8 of 384 experts in each of 48 layers, plus attention.
2. **A GPU has to move those bytes every step.** The weights sit in HBM and the arithmetic sits in the
   GPU, so each token's weights cross the memory interface. With the draft model on, a step verifies
   8 rows and reads about 73 GB. At 4x B300 that is a floor of several
   milliseconds per step before any arithmetic.
3. **The arithmetic itself is small and narrow.** 71% of the
   weight bytes are NVFP4: a 4-bit code times a 4-bit code, scaled per block of 16. On the GPU,
   arithmetic is 0.9% of a batch-1 step.
4. **A single request is launch-bound.** The serving engine issues about 940 dependent kernel launches
   per speculative step (counted in its source; tokenspeed-src), and the fit puts each at
   2.5 µs.
5. **Long context is a separate problem.** Eight of the 48 layers attend over the full history. At the
   1,048,576-token limit a step reads 22 GB
   of KV cache, wherever the weights live.

So there are three costs to attack: launches (software), weight movement (memory architecture) and
KV movement (memory bandwidth). They call for different remedies, and the design space below takes
them one at a time.

## The platforms

| | what it is | weights live in | what it removes | status |
|---|---|---|---|---|
| **G0** | B300 GPUs running today's best published engines | HBM | nothing: the baseline | calibrated to published speeds |
| **G1** | the same GPUs with dataflow-style software: one compiled schedule, no dependent launches | HBM | launch waits | projected |
| **A** | a decode chip on HBM (MDA-1 from our DeepSeek-V4.1 study, re-targeted) | HBM | launch waits, GPU power and price | proposed |
| **S** | reloadable weights on the chip (MDS-1) | on-die SRAM, next to the multipliers | weight movement | proposed |
| **F** | fixed weights (MDF-1) | mask-programmed storage | weight movement and reloadability | comparison point only |

**G0 gets its best published setup.** DSpark speculation on, MXFP8 KV cache, TP4 with expert
parallelism, the faster of two B300 layouts as the lead baseline. Its four free constants are fitted
to nine published figures ([06](06_simulator.md)).

**G1 is the software counterfactual.** Anything software can do on installed GPUs should not be
credited to silicon. G1 keeps the GPUs' HBM efficiency and attention kernels and removes dependent
launches. It is worth 1.7x at batch 1.

**A tests whether a better datapath on the same memory is enough.** For speed it is not: at batch 1 it
is 0.85x of G1, because both wait on HBM. This was hypothesis H1 and it held. A's value
is power and price: with prefill counted it is the cheapest whole system in the study
(2.2x lower cost than GPUs).

**S is the design the study is about.** It is the speed design. Every weight sits in SRAM beside the arithmetic that uses it.
A layer's weights are spread over a tile of dies; a token's hidden state moves from tile to tile, and
the weights never move. Weights are loaded at boot, so the same hardware serves a new model version,
a fine-tune or another model of the family. It is the right comparison for a customer who retrains.

**F is the bound.** Fix the weights in the masks and storage gets several times denser and the
datapath simpler. It shows how much of S's cost is the price of reloadability. It is not a proposal:
a model with fixed weights cannot follow a retrain, and this model's own card describes several
checkpoints already.

## Same physics, same constants

Where two platforms use the same thing, they get the same number.

| constant | value | used by |
|---|---|---|
| HBM efficiency (share of peak bandwidth delivered) | 0.802 (fitted on the GPUs) | G0, G1, A, and the KV memory of S and F |
| HBM3E stack rate | 1.0 TB/s | A, S, F |
| time per weight operation of a dataflow chip | 3.91 µs (a published SRAM wafer machine's) for S; A pays the part beyond S's bank pass on top of its own memory time | A, S |
| process | N4P-class, the node B300 is built on | A, S, F |
| facility overhead (PUE) | 1.25 | all |
| electricity | $0.0862/kWh | all |
| host server | 500 W, counted once per system | all |
| lifetime, utilization | 4 years, fully used | all |
| draft model, acceptance length, routing | identical operation graph | all |

N3-class is reported as a sensitivity only ([04](04_results.md), section 2).

## Why these memory choices

| memory | bandwidth to the arithmetic | density | reloadable | role here |
|---|---|---|---|---|
| HBM | 7.7 TB/s per B300 (vendor figure) | highest | yes | weights on G0, G1, A; KV cache everywhere |
| on-die SRAM | every bank read in parallel, in 0.83 µs per pass | 2.0 MiB/mm² including overheads | yes | weights on S |
| mask-programmed storage | as SRAM | 7.9 MiB/mm² including arithmetic | no | weights on F |
| LPDDR | 1.09 TB/s for a 16-package pool | high, cheap | yes | KV option; loses above about 146,000 tokens |

The answer to "which memory architecture" is a split: **weights in on-die SRAM, KV cache in HBM.** The
weights are large, mostly idle and read in fixed patterns, so they can be spread across many dies and
held at retention voltage. The KV cache is written every token and read in bulk by eight layers, so it
wants one pooled, high-bandwidth memory beside one shared attention engine.

A hybrid (hot experts in SRAM, cold in HBM) was considered and cut for time. The routing sweep
([04](04_results.md), section 10) bounds what it could gain: with the most skewed routing measured on
open models, the top 20% of experts take 80% of the picks.

## What decides the result

In order of effect on the headline ([04](04_results.md), section 12):

1. The time per weight operation of S. Taken from a published machine, not assumed. It decides speed.
2. Token demand. It decides how many systems share the design cost, and so the cost per token.
3. How fast "interactive" has to be. S is level with GPUs on cost at 100 tok/s per user and far ahead at 200.
4. The size of the shared attention tile. It caps S's throughput and is cheap to enlarge.
5. Power management of idle weight banks. It decides the batch-1 energy result.
