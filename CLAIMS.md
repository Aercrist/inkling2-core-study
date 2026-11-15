<!-- generated from docs/templates/CLAIMS.md.j2 by experiments/render_docs.py; edit the template -->
# Claims map

Every claim this study makes in public, the number behind it, where that number is computed, and how
far to trust it. A sentence about the results that has no row here is not supported by the study.

Labels: **measured** (by the named publisher; we ran nothing), **derived** (arithmetic on the
checkpoint), **modeled** (our simulator), **assumption**.

## The model and today's serving

| id | claim | value | label | where | confidence |
|---|---|---|---|---|---|
| C1 | Inkling 2 Small's parameter count is reproduced from the checkpoint headers | 594,050,672,720 | derived | `results/phase0_workload.json` → `params.hub_total`; [01](docs/01_workload.md) | high: exact match to the Hub |
| C2 | One token reads about 19 GB of a 393 GB checkpoint | 19.3 GB of 393.5 GB | derived | `phase0_workload.json` → `touched_per_token.total`, `bytes.total_with_draft` | high |
| C3 | Published single-user speed on 4x B300 with the draft model | 476 tok/s | measured (tokenspeed-blog) | `data/public_measurements.yaml` | high: quoted |
| C4 | Published single-user speed on 4x GB200 | about 630 tok/s | measured (vllm-blog) | `data/public_measurements.yaml` | high: quoted |
| C5 | Our GPU model reproduces 9 published figures | 8.0% RMS error; leave-one-out median 12.9%, worst 64% | modeled against measured | `results/calibration_gpu.json`; [06](docs/06_simulator.md) | medium: four constants, nine points, two engines |
| C6 | On the GPU a single request is bound by kernel launches and weight streaming, not arithmetic | arithmetic is 0.9% of the step | modeled; agrees with the engine's own statement (tokenspeed-blog) | `results/inkling.json` → `headline.rows.g0_x4.time_by_cause_batch1_ms` | medium |

## The headline comparisons (S, reloadable weights on the chip, against the best B300 layout for each metric)

Lead with the median and the range. The central case is one setting of the inputs.

| id | claim | median | range (5th to 95th percentile) | central case | label | where | confidence |
|---|---|---|---|---|---|---|---|
| C7 | Single-user speed | 3.4x | 1.3x to 8.2x | 4.1x: 3,000 against 720 tok/s; 4.7x against the best published 630 tok/s | modeled | `inkling.json` → `headline.ratios["s/g0_x4"].speed_batch1`; `uncertainty.json` → `monte_carlo.speed_batch1` | medium |
| C8 | Tokens per joule, decode, at each platform's best interactive point | 4.0x | 2.0x to 8.2x | 4.4x | modeled | `…ratios["s/g0_x4"].tokens_per_joule_best` | medium |
| C9 | Decode cost per token at 100 tok/s per user is about level with GPUs | 1.00x | 0.31x to 2.63x | 1.8x | modeled | `…ratios["s/g0_x4"].cost_best` | **low**: rests on token demand, lifetime, margin. Never publish as a cost advantage without the range |
| C10 | Tokens per joule for a single request | 1.7x | 0.40x to 5.9x | 2.7x with power management, 1.2x without | modeled | `inkling.json` → `power_management` | medium-low |
| C29 | As the per-user speed required rises, S's decode cost advantage grows | | | 4.3x at 200 tok/s per user, GPUs on 8x B300 | modeled; the GPU model is least constrained at small batches | `inkling.json` → `thresholds` | low-medium |
| C30 | End to end with prefill, S for decode plus HBM chips for prefill | | | 1.9x lower cost, 3.0x tokens per joule | modeled | `inkling.json` → `end_to_end.rows.s_decode_a_prefill` | low-medium |
| C31 | With prompts pipelined, S doing its own prefill is cheaper end to end than GPUs | | | 1.38x cost, 3.28x tokens per joule | modeled | `end_to_end.rows.s` | low-medium |
| C32 | The HBM chip is the cheapest whole system | | | 2.2x lower cost end to end | modeled | `end_to_end.rows.a_x4` | medium-low |
| C33 | S breaks even with GPUs on decode cost at about 1.6 trillion tokens a day | | | 1.6 trillion | modeled; demand is an assumption | `inkling.json` → `demand.breakeven_tokens_per_day` | low |

## Where the gain comes from

| id | claim | value | label | where | confidence |
|---|---|---|---|---|---|
| C11 | Better software on the same GPUs is worth about 1.7x for one user | 1.7x | modeled (projection) | `…ratios["g1_x4/g0_x4"].speed_batch1` | medium |
| C12 | A chip that keeps weights in HBM is no faster than that software for one user | 0.85x | modeled | `…ratios["a_x4/g1_x4"].speed_batch1` | medium |
| C13 | That chip's gain is in energy and cost | 2.2x tokens per joule, 2.4x decode cost | modeled | `…ratios["a_x4/g0_x4"]` | medium |
| C14 | Moving weights onto the chip is the large step | 3.8x over the HBM chip | modeled | `uncertainty.json` → `ablation.steps[3]` | medium |
| C15 | The process node is not the lever | N3-class changes speed by 1.00x, tokens per joule by 1.28x | modeled; scaling factors are assumptions | `uncertainty.json` → `ablation.process_node` | low-medium |

## Reloadable against fixed weights

| id | claim | value | label | where | confidence |
|---|---|---|---|---|---|
| C16 | Reloadable weights cost about 6x the silicon | 521 dies against 93 | modeled; the fixed side scales one shipped 8B chip (taalas-hc1) | `inkling.json` → `designs.s_over_f.dies` | medium |
| C17 | The fixed design's higher speed is not a cost of reloadability | the two designs take their time per operation from two different published machines; the synthesized fixed datapath closes at the same clock | modeled and synthesized | docs/04 section 7; `results/silicon.json` → `fmax` | medium |
| C18 | but little in tokens per joule when busy | 1.29x | modeled | `headline.rows.*.best_energy` | medium |
| C35 | Fixing a multiplier's weight saves little in that lane | 1.13x in area, 1.12x in energy | measured by us in an open flow (ASAP7) | `results/silicon.json` → `hardwired_row_length_sweep` | medium: one kit, one corner |
| C19 | A fixed-weight design needs new masks for each model version | 95 die designs, $56M | modeled; mask price is an assumption | `designs.f.cost` | low-medium |
| C20 | The NVFP4 dot-product block was designed, checked bit for bit, and placed and routed on an open 7 nm kit (not fabricated) | 1,589,011 vectors, 0 mismatches | measured by us in simulation and an open flow | `results/silicon.json` → `verification`; [07](docs/07_silicon.md) | high for the block; it is one block, not a chip |

## The ladder, context and limits

| id | claim | value | label | where | confidence |
|---|---|---|---|---|---|
| C21 | For models shaped like this one the gain narrows as the model shrinks | 4.1x at 594B to 2.8x at about 8B | modeled; the smaller rungs are hypothetical | `inkling.json` → `ladder` | medium-low |
| C22 | We predicted the opposite | predicted 40x at about 8B | our own pre-registration | `data/preregistration.yaml` P23 | high |
| C23 | For a dense 8B model published machines show a wide gap | 8.4x (SRAM wafer) and 72x (fixed-weight die) over one H200 | measured by others; conditions differ | `inkling.json` → `dense_8b`; `data/published_llama31_8b.yaml` | medium: vendor and relayed figures |
| C24 | The lead over an HBM chip shrinks with context | 3.8x at 8K to 2.9x at 1M tokens | modeled | `inkling.json` → `context.s_over_a` | medium |
| C25 | Prefill is where S's gain is smallest | 0.94x the GPUs' prefill cost per token, at its best concurrency | modeled | `inkling.json` → `prefill.s_over_g0_cost` | medium-low |
| C26 | The machine is large | 521 dies of 800 mm², about $1.4M per system with its share of design cost | modeled | `designs.s` | medium on area, low on cost |
| C27 | 21 of 35 predictions landed inside their range | 21 / 35 | our own scoring | `results/predictions.json` | high |
| C28 | The simulator did not reproduce four published Llama 3.1 8B speeds within 30% | 0 of 4 | modeled against measured | `results/calibration_llama.json` | high |
| C34 | Adversarial review cut the headline | cost gain from 9.5x to 1.8x; speed from 7.4x to 4.1x | our own log | `results/reviews.json`; [05](docs/05_limitations.md) | high |

## Things we do not claim

- That any of this was built, taped out or measured by us, apart from the simulation and open-flow
  place and route of one block.
- That a chip would reach these figures. They are a model with stated ranges.
- Anything about Thinking Machines' plans, hardware or opinions. This study is independent and is
  not affiliated with or endorsed by Thinking Machines Lab.
- A gain of "orders of magnitude". For a model this size the supported gain is a single-digit multiple
  in speed and energy, and no reliable gain in cost at ordinary interactive speed.
- That the chip is cheaper per token. At 100 tok/s per user it is about level; its case is speed,
  energy, and cost only where every user must be served fast.
- That fixed-weight silicon is a product we propose. Aercrist's chips load the customer's weights.
- That reloadable weights cost speed. The speed difference between S and F comes from two vendors' machines.
