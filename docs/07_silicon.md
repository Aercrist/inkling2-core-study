<!-- generated from docs/templates/07_silicon.md.j2 by experiments/render_docs.py; edit the template -->
# 07 · One block in silicon: the NVFP4 dot product

The system model needs three numbers for a multiplier lane: its area, its energy and its clock. For
the first six phases, these numbers were assumptions. This page replaces them for the NVFP4 lane,
which serves 71% of the weight bytes.

We built one block. It is the inner loop of an expert's matrix product in layers 0 to 39. In these
layers, the weights and the activations are NVFP4 (4-bit codes with a shared scale per 16 values).

We generated the block as SystemVerilog from the model's config. We checked it bit for bit against
a reference on real expert weights. We placed and routed it on an open 7 nm predictive kit (ASAP7).
The full engineering report, with every command, is [`rtl/nvfp4/REPORT.md`](../rtl/nvfp4/REPORT.md).
Every number is in [`results/silicon.json`](../results/silicon.json).

**This is one block, not a chip.** It has no memory, no clock distribution beyond the block, and no
floorplan.

## Two variants

| | reloadable (`nvfp4_dot`) | fixed (`nvfp4_dot_const`) |
|---|---|---|
| weights | arrive on ports from a memory, 16 at a time | constants taken from a real expert row |
| used for | S | F, as the comparison |

## Verification

| check | result |
|---|---|
| vectors compared against the bit-exact reference | 1,589,011 (1,165,348 reloadable, 423,663 fixed), **0 mismatches** |
| what the vectors are | real rows of a real expert (layer 0, expert 0, and row samples from other layers) against activations derived from the model's own embeddings, plus directed edge cases: zeros, signed zero, largest and smallest scales, cancellation, rounding ties, overflow |
| simulators | Verilator 5.020 2024-01-01 rev (Debian 5.020-1); second simulator: Icarus Verilog version 12.0 (stable) () |
| reference against PyTorch on a whole real expert | pass (torch 2.15.0+cpu) |
| stored format | checked against the checkpoint bytes and against a serving engine's own quantizer run from source: pass |
| batch invariance | 48 probe requests, each placed in 30 different batch sizes, positions and idle patterns: 1 distinct result per probe (pass) |
| placed-and-routed netlist against the reference | reloadable: bit-exact; fixed: bit-exact |

Batch invariance is important for this model's customers. The output of a request must not depend
on the other requests in the batch. The design of this block gives batch invariance: each row has
its own accumulator and a fixed order of additions. The test confirms it.

## Placed and routed (measured in an open flow)

Library: asap7sc7p5t RVT standard cells, NLDM, TT corner (0.70 V, 25 C); ORFS CORNER=TC. The two variants use a 1,800 ps clock
(556 MHz) and do one 16-value block per clock. Power comes from gate-level activity on real
tensors.

| | reloadable | fixed, one whole row time-shared through the lanes | reloadable / fixed |
|---|---|---|---|
| cell area | 511.1 µm² | 882.8 µm² | 0.58 |
| cells (flip-flops) | 4,548 (273) | 8,234 (279) | |
| setup / hold slack | 23.3 / 38.1 ps | 21.8 / 35.7 ps | |
| dynamic power | 1.092 mW | 1.831 mW | 0.60 |
| leakage | 0.41 µW | 0.72 µW | |
| energy per 16-value block | 1.97 pJ | 3.30 pJ | 0.60 |
| energy per multiply-accumulate | 0.123 pJ | 0.206 pJ | |
| area per multiplier lane | 31.9 µm² | 55.2 µm² | |
| weights held in the module | none (they come from memory) | 4,608 hard-wired, 0.19 µm² each with the datapath | |

The best clock period that the flow reached was 1,617 ps. In the two variants, the
critical path is the single-cycle 32-bit floating-point accumulate.

## The gain from fixed weights depends on their use

We predicted that a fixed-weight datapath is 1.8x smaller and uses
2x less energy. The synthesis shows that the result depends on the use of the weights.

| how the fixed weights reach the lanes | reloadable / fixed, area | reloadable / fixed, energy |
|---|---|---|
| **a lane always sees the same weight** (the same 16 weights in every block of the row, so the constants fold into the multipliers) | **1.13** (1.12 to 1.13, n=2) | **1.12** (1.11 to 1.13) |
| **a whole row of 4,608 weights time-shared through 16 lanes** (the weights become a table in logic) | 0.58 | 0.60 |

- When every lane has its own fixed weight, the block is 1.13x smaller and
  uses 1.12x less energy. The multipliers fold away. The adder tree
  and the 32-bit accumulator stay, and they are most of the block. This is the comparison that the
  predictions meant. Area was below the predicted range, and energy was below it.

  A first version of this test used rows one block long. Synthesis then also removed the
  accumulator and showed a larger decrease. That result stays in the JSON with a warning, and we do
  not use it.
- When the fixed weights are time-shared, the fixed module is *larger* than the reloadable module.
  The fixed module contains its weights, and the reloadable module does not. As storage, the fixed
  module is dense: 0.19 µm² per hard-wired weight in standard cells, with the datapath.
  For scale, the one shipped fixed-weight chip holds 8B weights on 815 mm². That is about
  0.10 µm² per weight on an older process with custom cells (taalas-hc1).

Thus a fixed weight in a lane saves little in that lane. The reloadable design is still larger, for
two reasons:

- An SRAM bit is larger than a mask-programmed bit.
- S has a multiplier lane on every bank, which is 32% of its silicon. The density figure of the
  fixed design comes from one shipped chip with its arithmetic folded in.

Public data does not show how much of the difference comes from reloadability and how much comes
from the design of that chip.

![synthesized block](../results/fig_silicon.png)

We measured the middle bar (one weight per lane) on its own one-block workload. On that same
workload, the reloadable block uses 1.83 pJ. That value gives the
1.12 in the table.

## From the open kit to an N4P-class process

ASAP7 is a predictive kit. Its numbers are reliable relative to one another, but not as predictions
of a foundry process. Three factors convert them, and all three are in the uncertainty sweep.

| | on ASAP7 | factor | N4P-class value used | assumed before synthesis | new / old |
|---|---|---|---|---|---|
| area per NVFP4 lane | 31.9 µm² of cells | × 0.69 (cell pitches; asap7-scaling), ÷ 0.65 placement utilization (assumption) | 33.9 µm² | 12 µm² | 2.8x |
| energy per NVFP4 operation | 0.061 pJ | × 0.82 (N5 to N4P; n4p) | 0.050 pJ | 0.15 pJ | 0.34x |
| clock | 556 MHz, timing closed | × 1.11 (N5 to N4P; n4p) | 617 MHz | 1,500 MHz | 0.41x |

We apply only the published N5-to-N4P step to speed and energy. Apart from that step, we treat
ASAP7 as N5-class, which is conservative. We built only the NVFP4 lane. The MXFP8 and BF16 lanes
take the same area correction and keep their assumed energy.

## What it changed

| S against the GPUs | with the assumed lane | with the synthesized lane |
|---|---|---|
| time for one pass over a weight bank | 0.34 µs | 0.83 µs |
| multiplier lanes | 47,500 mm² | 134,300 mm² |
| dies | 384 | 521 |
| system cost | $1,141,000 | $1,382,000 |
| speed for one user | 4.1x | 4.1x |
| tokens per joule, one user | 3.8x | 3.0x |
| tokens per joule, best point | 3.7x | 4.4x |
| cost per token, best point | 2.2x | 1.8x |

The lane is larger and has a slower clock than we assumed, and it uses less energy. The machine
grew by 137 dies, and its cost gain decreased from 2.2x to 1.8x. Speed changed little. S takes
its time per weight operation whole from a published machine, and its own bank pass is inside that
figure in the two cases. Every result in [04](04_results.md) uses the synthesized values.

We did not tune the design again after the synthesis. A lane of this size is an argument for deeper
weight banks or for lanes shared between experts. One request touches about 60 of
384 experts per layer per step, and the other lanes are idle. Each change
decreases the die count and is the clear next step. These results include neither change.

## For context

A published NVFP4 multiply-accumulate block at 28 nm reports 21.5 pJ
per 16-value block at 524 MHz in 0.024 mm²
(nvfp4-mac-28nm). Our block on ASAP7 uses 1.97 pJ and 0.00051 mm² of
cells. The node, library, design and method are different. The comparison shows only that the two
blocks are in the range that process scaling can explain.

## Not done

- We built no memory. The weights of the reloadable lane come from an SRAM. The SRAM cost comes from
  published macros, not from synthesis.
- One block and one precision. We did not build the MXFP8 lane, the BF16 attention datapath, the
  router's top-8 select or the softmax here. The top-8 select comes from the previous study.
- No pipeline in the accumulator. A carry-save accumulate clocks faster. It must stay bit-exact with
  the arithmetic of the serving engines, and we did not try that.
- One corner (typical, 0.70 V, 25 °C). No signoff, no variation, no test logic.
- ASAP7 only. We ran nothing on a foundry kit.
- Power comes from gate-level activity without delays. Thus the power does not include glitches,
  and absolute power is optimistic. The sweep covers the arithmetic energy from 0.2x to 2x.
- The activations are realistic but do not come from a forward pass. They are real embedding rows
  through the real layer-0 norm, and the outputs of the real first matrix. We fetched one expert
  whole.
- One placement seed per design. Timing closes at 1,800 ps and fails at 1,600 ps.

## Reproduce

If Icarus Verilog is installed, `make test` runs a short version of the verification. For the full
run, use `python rtl/nvfp4/run_all.py`. The full run has the vectors, the second simulator, the
PyTorch check, and place and route in the Docker image of the OpenROAD flow. See section 11 of the
[report](../rtl/nvfp4/REPORT.md).
