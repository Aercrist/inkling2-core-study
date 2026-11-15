<!-- generated from docs/templates/napkin.md.j2 by experiments/render_docs.py; edit the template -->
# Napkin checks

One page of hand arithmetic per headline, from a few inputs, set beside the simulator. If a napkin
and the simulator disagree by more than about 20%, one of them is wrong or the reason is stated.
The arithmetic is done by `experiments/napkin.py` so that no figure here is typed by hand.

| headline | napkin | simulator | gap |
|---|---|---|---|
| GPUs at batch 1 | 660 tok/s | 550 tok/s | +20% |
| S at batch 1 | 3,170 tok/s | 2,960 tok/s | +7% |
| speed, S over GPUs | 4.8x | 5.4x | -11% |
| tokens/J at the best point, S over GPUs | 4.2x | 4.4x | -4% |
| cost per token at the best point, GPUs over S | 1.8x | 1.8x | +2% |
| dies, reloadable over fixed | 5.7x | 5.6x | +2% |

## 1 · GPUs, one request: 4x B300 with the draft model on

1. A speculative step verifies 8 rows and reads 73.2 GB of weights (checkpoint arithmetic, docs/01).
2. Four B300 deliver 4 x 7.7 TB/s x 0.80 (fitted) = 24.7 TB/s, so the read takes 2.96 ms.
3. The step issues 937 dependent kernel launches at 2.50 us (fitted): 2.35 ms.
4. Step = 5.31 ms; 3.5 tokens accepted per step gives 660 tok/s per user.

**Napkin 660 tok/s per user; simulator 550 tok/s per user (+20%).**
Published: 476 tok/s on 4x B300 at 94K context (tokenspeed-blog); about 630 on 4x GB200 (vllm-blog).
Why they differ: The napkin leaves out collectives between GPUs and the uneven split of experts across the four GPUs (0.31 ms and about 0.63 ms in the simulator).

## 2 · Reloadable weights on the chip (S), one request

1. About 5 weight operations in series per layer (query/key/value projection, output projection, router, expert up, expert down): 5 x 48 layers + 5 x 5 draft layers + 4 for embedding and heads = 269.
2. Each takes 3.91 us: the whole time per weight operation that a published wafer-scale SRAM machine shows on Llama 3.1 8B (docs/06). S's own bank pass, 512 cycles at 617 MHz = 0.83 us, is inside that figure.
3. Weight operations: 269 x 3.91 us = 1.053 ms.
4. Attention: 53 attention operations pay a hand-off each (0.005 ms), their arithmetic takes 0.029 ms and the KV fetch 0.016 ms.
5. Step = 1.103 ms; 3.5 tokens per step gives 3,173 tok/s per user.

**Napkin 3,173 tok/s per user; simulator 2,963 tok/s per user (+7%).**

## 3 · Tokens per joule at the best interactive point (32K context, at least 100 tok/s per user)

1. GPUs: at batch 101 every step touches every expert, so it streams the whole 393 GB: 15.9 ms before attention and launches. The simulator's step is 34.9 ms, 10,130 tok/s in total.
2. GPU power: 4 x 1100 W x 0.84 (measured share of TDP at this batch, carried from H100) + 500 W host = 4,192 W. 10,130 / 4,192 = 2.42 tokens/J.
3. S: throughput is capped by the shared attention tile. A token needs 17.0 GFLOP of full attention at 32K, so 2 PFLOP/s gives 117,479 tok/s; it reads 0.20 GB of KV, so 26 HBM stacks at 0.80 give 103,628 tok/s. The lower: 103,628 tok/s.
4. S energy per token: reading weights 36.8 mJ (20.9 GB at 0.22 pJ/bit), weight arithmetic 9.4 mJ, attention and KV 9.3 mJ; with power-delivery loss 63.0 mJ.
5. S power: 3,576 W static + 103,628 tok/s x 63.0 mJ = 10,108 W. 10.3 tokens/J.
6. Ratio: 4.2x.

**Napkin 4.24x; simulator 4.4x (-4%).**

Why they differ: The napkin takes the attention tile's limit as the throughput and ignores groups waiting for the tile; the simulator also lets one bank pass serve several rows of a group.

## 4 · Cost per million tokens at the best interactive point

1. GPUs: 4 x $60,000 = $240,000 over 4 years = $6.84/h, plus 4.2 kW x 1.25 x $0.0862/kWh = $7.30/h. At 10,130 tok/s that is $0.200 per million tokens.
2. S: 521 dies, $869,173 to build plus $512,923 of design cost ($300M over the 624 systems that 5 trillion tokens a day need) = $1,382,100; $39.42/h, plus 10.1 kW of power = $40.51/h. At 103,628 tok/s (napkin 3) that is $0.1086 per million tokens.
3. Ratio: 1.8x.

**Napkin 1.84x; simulator 1.8x (+2%).**

Why they differ: Same throughput cap as napkin 3: the napkin lets the attention tile run at its limit; the simulator reaches 100,982 tok/s.

## 5 · What reloadability costs in silicon

1. The checkpoint is 393.5 GB = 375,252 MiB.
2. Both designs also keep 256 sessions' sliding-window KV cache in SRAM: 10,560 mm2.
3. S: weights in SRAM at 0.5 mm2/MiB = 187,626 mm2, plus 134,274 mm2 of multiplier lanes and the window cache. A die offers 650 mm2 after pads, links and control: 511 dies.
4. F: mask-programmed storage with its arithmetic at 0.127 mm2/MiB (from one shipped chip, scaled) = 47,657 mm2: 90 dies.
5. Ratio: 5.7x.

**Napkin 5.71x; simulator 5.6x (+2%).**

Why they differ: The simulator adds the attention engines, the HBM PHYs and the links between tiles to both designs.

## What the napkins show

- The GPU's batch-1 step is two terms of about equal size: streaming weights and waiting on launches.
  Neither is arithmetic.
- S's batch-1 step is the number of operations in series times a time per operation taken from a
  published machine. S's own bank pass is 21% of that time.
- At the best interactive point the GPUs stream the whole checkpoint every step and S does not. S's
  throughput is set by its attention tile, and its cost by how many systems share the design cost.
- The die ratio is the ratio of two storage densities, plus the multiplier lanes and the window cache both designs carry.
