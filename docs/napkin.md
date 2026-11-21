<!-- generated from docs/templates/napkin.md.j2 by experiments/render_docs.py; edit the template -->
# Napkin checks

Each headline has one page of hand arithmetic from a few inputs, set beside the simulator. If a napkin
and the simulator disagree by more than about 20%, one of them is wrong or we give the reason.
`experiments/napkin.py` does the arithmetic, so that no person types a figure here by hand.

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
Why they differ: The napkin does not include collectives between GPUs and the uneven split of experts across the four GPUs. These are 0.31 ms and about 0.63 ms in the simulator.

## 2 · Reloadable weights on the chip (S), one request

1. Each layer has about 5 weight operations in series (query/key/value projection, output projection, router, expert up, expert down). The total is 5 x 48 layers + 5 x 5 draft layers + 4 for embedding and heads = 269.
2. Each operation takes 3.91 us. That is the whole time per weight operation that a published wafer-scale SRAM machine shows on Llama 3.1 8B (docs/06). The bank pass of S, 512 cycles at 617 MHz = 0.83 us, is inside that figure.
3. Weight operations: 269 x 3.91 us = 1.053 ms.
4. Attention: 53 attention operations each pay a hand-off (0.005 ms). Their arithmetic takes 0.029 ms and the KV fetch takes 0.016 ms.
5. Step = 1.103 ms; 3.5 tokens per step gives 3,173 tok/s per user.

**Napkin 3,173 tok/s per user; simulator 2,963 tok/s per user (+7%).**

## 3 · Tokens per joule at the best interactive point (32K context, at least 100 tok/s per user)

1. GPUs: at batch 101, every step touches every expert. Thus the step streams the whole 393 GB, which takes 15.9 ms before attention and launches. The step of the simulator is 34.9 ms, which gives 10,130 tok/s in total.
2. GPU power: 4 x 1100 W x 0.84 (measured share of TDP at this batch, carried from H100) + 500 W host = 4,192 W. 10,130 / 4,192 = 2.42 tokens/J.
3. S: the shared attention tile limits throughput. A token needs 17.0 GFLOP of full attention at 32K, so 2 PFLOP/s gives 117,479 tok/s. A token reads 0.20 GB of KV, so 26 HBM stacks at 0.80 give 103,628 tok/s. The lower value is 103,628 tok/s.
4. S energy per token: 36.8 mJ to read weights (20.9 GB at 0.22 pJ/bit), 9.4 mJ for weight arithmetic, 9.3 mJ for attention and KV. With power-delivery loss, the total is 63.0 mJ.
5. S power: 3,576 W static + 103,628 tok/s x 63.0 mJ = 10,108 W. 10.3 tokens/J.
6. Ratio: 4.2x.

**Napkin 4.24x; simulator 4.4x (-4%).**

Why they differ: The napkin uses the limit of the attention tile as the throughput and ignores groups that wait for the tile. The simulator also lets one bank pass serve several rows of a group.

## 4 · Cost per million tokens at the best interactive point

1. GPUs: 4 x $60,000 = $240,000 over 4 years = $6.84/h, plus 4.2 kW x 1.25 x $0.0862/kWh = $7.30/h. At 10,130 tok/s that is $0.200 per million tokens.
2. S: 521 dies. The build cost is $869,173. The share of design cost is $512,923 ($300M over the 624 systems that 5 trillion tokens a day need). The total is $1,382,100. That is $39.42/h, plus 10.1 kW of power = $40.51/h. At 103,628 tok/s (napkin 3) that is $0.1086 per million tokens.
3. Ratio: 1.8x.

**Napkin 1.84x; simulator 1.8x (+2%).**

Why they differ: The throughput cap is the same as in napkin 3. The napkin lets the attention tile run at its limit. The simulator gets 100,982 tok/s.

## 5 · What reloadability costs in silicon

1. The checkpoint is 393.5 GB = 375,252 MiB.
2. The two designs also keep the sliding-window KV cache of 256 sessions in SRAM: 10,560 mm2.
3. S: weights in SRAM at 0.5 mm2/MiB = 187,626 mm2, plus 134,274 mm2 of multiplier lanes and the window cache. A die gives 650 mm2 after pads, links and control: 511 dies.
4. F: mask-programmed storage with its arithmetic at 0.127 mm2/MiB (from one shipped chip, scaled) = 47,657 mm2: 90 dies.
5. Ratio: 5.7x.

**Napkin 5.71x; simulator 5.6x (+2%).**

Why they differ: The simulator adds the attention engines, the HBM PHYs and the links between tiles to both designs.

## What the napkins show

- The batch-1 step of the GPU is two terms of about equal size: weight streaming and waits on launches.
  The two terms are not arithmetic.
- The batch-1 step of S is the number of operations in series times a time per operation. That time is from a
  published machine. The bank pass of S is 21% of that time.
- At the best interactive point, the GPUs stream the whole checkpoint every step and S does not. The
  attention tile of S sets its throughput. The number of systems that share the design cost sets its cost.
- The die ratio is the ratio of two storage densities, plus the multiplier lanes and the window cache that the two designs have.
