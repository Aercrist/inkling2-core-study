<!-- generated from docs/templates/README.md.j2 by experiments/render_docs.py; edit the template -->
# Inkling 2 Small on purpose-built silicon

This is a hardware-architecture study by Aercrist. The study is about a chip built around
[Inkling 2 Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) (594B parameters, 17B
active per token). It asks four questions:

- What does this chip deliver against the best published GPU deployment?
- Which memory architecture gets that result?
- What is the cost to keep the weights reloadable?
- How does the answer change for smaller models?

This study is the sequel to our [DeepSeek-V4.1 study](https://github.com/Aercrist/dsv41-accel-study)
and follows its rules:

- We render every number from `results/*.json`.
- Every constant has a source or a sweep.
- We wrote the predictions before we simulated anything.

This study is independent. It is not affiliated with or endorsed by Thinking Machines Lab.

**We fabricated nothing, and we measured nothing on hardware.** The GPU figures reproduce the published
measurements of other people. The chips are designs on paper. We synthesized one circuit block.

## Findings

The table compares a chip that keeps every weight in reloadable on-die memory with B300 GPUs. The
GPUs run the best engines published today, in the best layout for each metric (2x, 4x or 8x B300).
The median and the range are from 400 joint draws of
30 uncertain inputs.

| | median | 5th to 95th percentile | central case | confidence |
|---|---|---|---|---|
| **speed for one user** | **3.4x** | 1.3x to 8.2x | 4.1x (3,000 against 720 tok/s) | medium: the GPU side is calibrated; the chip side takes its time per operation from a published machine |
| **tokens per joule**, decode, at each platform's best interactive point | **4.0x** | 2.0x to 8.2x | 4.4x | medium |
| **cost per token**, decode, at the same point | **1.00x** | 0.31x to 2.63x | 1.8x | low: rests on token demand, lifetime and margin |
| **cost of reloadable weights** against fixed weights in the masks | | | 5.6x the silicon; speed not separable: the two designs' speeds come from two different published machines | medium on silicon: the fixed-weight side scales one shipped 8B chip |

1. **On-chip weights give speed and energy gains, not cheap tokens.** One user gets about
   3,000 tokens a second instead of 720. A busy system delivers 4.4x the
   tokens per joule. At 100 tok/s per user, the decode cost of the chip is less than 2x above or below the GPU decode cost (median 1.00x). At
   200 tok/s per user and above, the GPUs hold only a few users each, and the chip is
   4x cheaper. Our GPU model is least certain at these batch sizes.
2. **The memory architecture makes the difference, and software comes first.** Better software on the same
   GPUs gives 1.7x for a single user. A purpose-built chip that keeps its weights in HBM
   adds no speed to that gain (0.85x). But it is the cheapest whole system. Its cost is 2.2x lower
   than the cost of GPUs when we count prefill.
3. **The chip can do its own prefill if it pipelines the prompts.** On the published agentic workload, prefill is
   34% of the GPU cost per output token. With 48 prompts in flight, the on-chip design
   prefills at 0.94x the cost per token of the GPUs. End to end, it is 1.4x lower in cost and 3.3x
   better in tokens per joule than GPUs. With HBM chips for prefill, the gains are 1.9x and 3.0x.
4. **Reloadable weights cost 5.6x the silicon.** In return, there are three die designs instead of
   95. A new model version needs a load of weights, not $56M of new masks. The
   synthesized block shows that a fixed weight in a multiplier saves only 1.13x in the area of that lane.

   The silicon
   difference is SRAM against mask-programmed storage, plus the multiplier lanes of S (32% of its silicon).
   The density of the fixed design comes from one shipped chip that has its arithmetic folded in. The higher speed of the fixed design in our tables is not separable. The speeds of the two designs come from two different published machines, not from a property of reloadability.
5. **The chip needs scale.** A system has 521 dies and costs about $1.4M with its share of design cost.
   It breaks even with GPUs at about 1.6 trillion tokens a day of this one model.
6. **The gain becomes smaller down the ladder for models with the shape of this one.** We predicted the opposite.
   At about 8B parameters the speed gain is 2.8x. For dense models, the published machines show
   that the gain becomes larger.

Our prior was a gain of about 2-10x. The speed gain and the energy gain are in that range. The cost gain is below it.
Two adversarial reviews changed these numbers by a large amount ([05](docs/05_limitations.md)). Before
the reviews, the cost figure was 9.5x.

## Key results

| platform | one user: tok/s | tokens/J | $/M tokens | best interactive point: batch | tokens/J | $/M tokens |
|---|---|---|---|---|---|---|
| G0 · 4x B300, today's engines (calibrated) | 550 | 0.23 | $3.58 | 101 | 2.42 | $0.200 |
| G0 · 2x B300, today's engines (calibrated) | 380 | 0.26 | $2.63 | 14 | 0.72 | $0.712 |
| G1 · 4x B300, dataflow-style software (projected) | 930 | 0.39 | $2.13 | 153 | 3.58 | $0.132 |
| A · 4x MDA-1, HBM decode chip (proposed) | 790 | 0.37 | $1.37 | 128 | 5.27 | $0.085 |
| S · MDS-1, reloadable weights on-die (proposed) | 2,960 | 0.70 | $3.74 | 96 | 10.64 | $0.111 |
| F · MDF-1, fixed weights (comparison point) | 12,320 | 4.36 | $0.55 | 256 | 13.70 | $0.069 |

All values are modeled. We calibrated the two G0 rows to published speeds. Costs are for decode only.

The one-user columns use 8K context with the draft model on. The best interactive point is the best
batch of each platform at 32K context with at least
100 tok/s per user. The process is N4P-class, which is the node of B300. An N3-class process
changes the gain by 1.28x in tokens per joule and
1.00x in speed.

![frontier](results/fig_frontier.png)

**Purpose-built silicon is not better in these cases:**

- decode cost at ordinary interactive speed (about equal)
- prefill (0.91x the GPUs per dollar)
- demand below about 1.6 trillion tokens a day
- small sparse models
- the longest contexts on cheap KV memory
- a single request (0.29x of the tokens per joule of a busy GPU cluster)

## Method

1. **Workload from the checkpoint.** We compute parameter counts, precisions and bytes per token
   from the safetensors headers. The results reproduce the published 594.05B exactly
   ([01](docs/01_workload.md)).
2. **A GPU baseline calibrated to public data.** We fitted four constants to 9 published
   figures from two serving engines. The RMS error is 8.0%. The leave-one-out error has a median of
   12.9% and a worst value of 64% ([06](docs/06_simulator.md)). The GPU gets its best
   published setup, and we count its prefill load. We also model a software-only counterfactual on the same GPUs.
3. **A check on four real machines.** The machines are Llama 3.1 8B on a GPU, two SRAM machines and a fixed-weight
   chip. With public specifications as input, 0 of 4 predictions have an error of 30% or less. We report that as a
   failure. We use the published speeds, not our own assumption, to set the time per operation of the chip.
4. **Five platforms on one simulator.** The constants are the same where the physics is the same
   ([02](docs/02_design_rationale.md), [03](docs/03_architecture.md)).
5. **One block in silicon.** We wrote the NVFP4 block-scaled dot product in SystemVerilog. We checked it bit for bit
   against a reference on real expert weights. We synthesized it on an open 7 nm kit
   ([07](docs/07_silicon.md)).
6. **Predictions first.** We froze 36 numeric predictions before simulation.
   21 of 35 scored predictions are inside their range, and we list the misses
   ([00](docs/00_hypotheses.md)).
7. **Uncertainty.** 30 inputs have ranges that we state. We moved them one at a time and all
   together (400 draws). A napkin calculation is beside each headline ([napkin](docs/napkin.md)).
8. **Adversarial review.** Two fresh agents attacked a clean copy of the repo. [05](docs/05_limitations.md) gives their findings, our
   changes and the size of the headline change.

## Read next

| page | what it is |
|---|---|
| [docs/04_results.md](docs/04_results.md) | every result, and the cases where silicon is not better |
| [docs/00_hypotheses.md](docs/00_hypotheses.md) | the predictions and how they scored |
| [docs/01_workload.md](docs/01_workload.md) | the model from its checkpoint, and what the serving engine's source adds |
| [docs/02_design_rationale.md](docs/02_design_rationale.md) | the five platforms and why |
| [docs/03_architecture.md](docs/03_architecture.md) | the reloadable design and the fixed-weight comparison: area, power, cost |
| [docs/05_limitations.md](docs/05_limitations.md) | limits, what can change the conclusions, review log |
| [docs/06_simulator.md](docs/06_simulator.md) | the simulator, the calibration, the check it failed |
| [docs/07_silicon.md](docs/07_silicon.md) | the synthesized block |
| [docs/napkin.md](docs/napkin.md) | hand arithmetic for each headline |
| [docs/decisions.md](docs/decisions.md) · [docs/sources.md](docs/sources.md) | judgment calls; every source by key |
| [CLAIMS.md](CLAIMS.md) | each public claim mapped to the result behind it |

## Reproduce

```bash
pip install -r requirements.txt
```

```bash
make reproduce
```

```bash
make test
```

`make reproduce` regenerates every number, figure and document from the committed inputs and does not need
a network. The checkpoint manifest (headers only, 1,854 tensors) is in the repo.
`tools/fetch_checkpoint_manifest.py` rebuilds it from the Hub. The simulation and
synthesis of the circuit block need Icarus Verilog, Verilator, Yosys and, for place and route, the OpenROAD flow. Refer to
[07](docs/07_silicon.md). The results of these tools are in the repo, thus `make reproduce` does not need the tools.

Edit `docs/templates/`, not the generated files.

## Layout

```
data/         model config, checkpoint manifest, published figures, frozen predictions, test fixtures
sim/          operation-level simulator: model builders, the HBM engine, the weight-stationary engine
configs/hw/   one file per platform; every constant has a source key or the word "assumption"
experiments/  calibration, experiments, uncertainty, predictions, napkin, doc rendering
rtl/          the NVFP4 dot-product block and the router block from the previous study
results/      generated JSON and figures
docs/         generated pages; templates/ holds what to edit
tests/        invariants
```

## Limitations

This is a model, not a measurement. The speed of the chip depends on a time per operation. We inferred that time from
the machine of another company, which runs a different model. The cost of the chip depends on an assumed token demand, an
assumed attention engine and street prices for GPUs. The cost does not include a seller's margin.

We did no
layout. We did not model the reliability of a 521-die system. GPU software will continue to
improve. [05](docs/05_limitations.md) has the full list and the things that can change the conclusions.

## Acknowledgments

This study uses the public work of Thinking Machines Lab (Inkling 2 Small, its model card and announcement).
It also uses the public work of the vLLM, SGLang and TokenSpeed projects. Their published recipes, measurements and source make
a calibrated GPU baseline possible. The synthesis uses the ASAP7 predictive kit, Yosys and OpenROAD.
The errors are ours. We welcome corrections.

Ben Blaine, Aercrist.
