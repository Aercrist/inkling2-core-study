<!-- generated from docs/templates/03_architecture.md.j2 by experiments/render_docs.py; edit the template -->
# 03 · Architecture: MDS-1 (reloadable) and MDF-1 (fixed)

Two designs on paper. Neither has been laid out; one block has been synthesized ([07](07_silicon.md)).
Every figure here is **modeled** from the constants in `configs/hw/s_mds1.yaml` and `f_mdf1.yaml`,
each of which cites a source or is marked *assumption*. MDF-1 is a comparison point, not a proposal.

## MDS-1 in one picture

```
            host: sampler, tokenizer, weight loader
                         |
   +---------------------+----------------------------------------------+
   |                                                                    |
   v                                                                    |
 tile 0        layer 0                   layer 47                       |
 draft,        (about 11 dies, over  ... (MXFP8 layers                 |
 embedding, -> two tiles)               need more)                     |
 head          [hub die: attention  -> ... [hub die]  -> output head ->-+
                (window 1024),               |
                router, norms]               |            attention tile
                 |   |   |   |               |          +------------------------+
                 v   v   v   v               +--------->| pooled full-attention  |
               expert dies:                             | engine                 |
               384 experts, each                        | 26 HBM stacks: KV cache  |
               bank beside its                          | of 8 full layers and   |
               own multipliers                          | 5 draft layers         |
                                                        +------------------------+
```

- **Layers over tiles.** 66 tiles of at most 8 dies. A layer takes about 10.6 dies on
  average: one hub die for attention, routing and normalization, the rest holding that layer's 384
  experts, so a layer spans more than one tile and the experts on the second tile are reached over the
  serial link (modeled). 521 dies in all, of 3 designs (weight die, hub die, attention-tile die),
  each further design costing another mask set.
- **Weights never move.** Each weight tensor is stored in SRAM banks 512 rows deep with a
  multiplier lane on the bank's output. A matrix-vector product is one pass over the bank:
  512 cycles at 0.6 GHz, 0.83 µs, whatever the
  matrix size.
- **Tokens move.** A layer's hub die sends a row to the 8 experts the router picked, already quantized
  to NVFP4 (4.5 bits per value), over die-to-die links inside the tile. The hidden state then hops to
  the next tile over a serial link.
- **One attention tile.** The 40 sliding-window layers attend inside their own hub die (window 1,024,
  kept in on-die SRAM). The 8 full-attention layers and the draft model's 5 layers send their queries
  to a shared attention tile that holds the whole KV cache in HBM.
- **Reloadable.** Weights are written at boot. At 10 GB/s from storage (assumption) a full load takes
  about 39 seconds.

## Area

| block | mm² | share |
|---|---|---|
| weight storage (SRAM, 393 GB) | 187,600 | 53% |
| multiplier lanes (from the synthesized block) | 134,300 | 38% |
| sliding-window KV cache for 256 sessions (SRAM, 22 GB) | 10,600 | 3.0% |
| attention engines (per layer, plus the pooled engine) and HBM PHYs | 2,380 | 0.7% |
| link PHYs (between tiles and between dies) | 8,540 | 2.4% |
| control, clocking, power management, test | 10,420 | 2.9% |
| **used** | **353,800** | |
| **silicon: 521 dies of 800 mm²** | **416,800** | |

That is 5.9 wafers' worth of area per system, and
0.76 GB of weights per die. Memory is 48% of the silicon and the
multiplier lanes most of the rest. SRAM is costed at 0.5 mm² per
MiB, twice the published macro density at this class of node, to cover error correction, routing and
the power grid (sources: sram-density, sram-3nm-macro, vcache).

We predicted 250 dies (160 to 330); the design
came out at 521. The prediction left out the multiplier lanes, and the synthesized lane is 2.8x the area we had assumed (384 dies before synthesis).

## Timing of one step at batch 1

| | time | share |
|---|---|---|
| passes over weight banks, with the per-operation latency | 1.096 ms | 93% |
| full attention: arithmetic | 0.026 ms | 2% |
| full attention: KV fetch | 0.025 ms | 2% |
| hops between tiles | 0.016 ms | 1% |
| step boundary, sampler | 0.019 ms | 2% |
| **step** | **1.181 ms** | 2,960 tok/s per user |

A weight operation takes 3.91 µs in all. That is not a design target: it is the whole
time per weight operation that a published wafer-scale SRAM machine shows on Llama 3.1 8B
([06](06_simulator.md)). S's own bank pass (0.83 µs) is inside it. A design that needed only
half again its own pass would be
2.5x
faster than the figure reported.

Because every layer has its own tile, requests are pipelined: while one request is in layer 20,
another can be in layer 3. That is why throughput scales with batch until the shared attention tile
saturates ([04](04_results.md), section 4).

## Power

| state | weight banks | logic and clock | links, idle | KV memory | host | total |
|---|---|---|---|---|---|---|
| idle, managed (banks in retention, lanes clock-gated) | 71 W | 2,080 W | 840 W | 78 W | 500 W | **3,580 W** |
| idle, unmanaged (full voltage, clocks running) | 274 W | 6,250 W | 840 W | 78 W | 500 W | **7,950 W** |

| serving | system power |
|---|---|
| one request, 8K context | 4,230 W |
| 256 requests, 32K context | 9,100 W |

- **SRAM leakage is small once managed.** Measured leakage of a 3 nm macro at 25 °C is 27.5 pW per bit
  (sram-3nm-macro): 87 W for this model. We multiply by 3 for
  operating temperature (assumption; no hot figure is public) and hold idle banks at a retention
  voltage that cuts it to 20% (published range 5% to 31%;
  sram-retention).
- **Idle logic is the larger term**: 4.0 W per die managed,
  12.0 W unmanaged (assumptions, swept 1 to 15 W).
- **Power density is low.** Fully busy, the system draws about 17 W per die,
  2.2 W/cm². A B300 is 1,100 W in one package. MDS-1
  can be air-cooled; its difficulty is the number of dies, not heat.

## Packaging, power delivery, cooling

- **Tiles.** About 8 large dies on one fan-out or interposer tile
  (8 is the assumed limit; a layer spans more than one tile). Die-to-die links inside a
  tile are short-reach, 0.25 pJ/bit (d2d-energy). Packaging and test are costed at
  $700 per die, built from the same sourced parts as the HBM chip's
  package (cowos-cost); the attention tile's HBM packages are costed as the HBM chip's.
- **Between tiles.** 72 lanes of 112G serial links per neighbour, 900 GB/s,
  250 ns per hop including error correction (ucie-targets, with margin;
  swept 0.1 to 2 µs).
- **System.** 66 tiles plus the attention tile: several chassis, not one board.
  Boards, power delivery, cooling and chassis are costed at $500 per
  die (assumption).
- **Power delivery** at 88% efficiency (assumption, as MDA-1).

## Cost

| per system | MDS-1 | MDF-1 |
|---|---|---|
| dies | $204,900 (521 at $393, yield 67%) | $46,600 (93 at $501, yield 52%) |
| package and test | $374,700 | $75,100 |
| KV memory (HBM) | $14,000 | $14,000 |
| boards, power, cooling, host | $275,500 | $61,500 |
| **manufacturing** | **$869,000** | **$197,000** |
| share of design cost (NRE) at 620 / 580 systems | $512,900 | $647,800 |
| **total** | **$1,382,000** | **$845,000** |

For scale: four B300 with networking are about $240,000
(reseller range $224,000 to $304,000; b300-price).

Memory is repairable with spare rows, so only the 52% of an MDS-1 die
that is logic must be defect-free. A fixed-weight die's storage has no spares.

Vendor margin, spares, the yield of assembling a tile, and software and bring-up are not in these
figures. GPUs are at street price, which includes all of them.

Design cost is spread over the systems a token demand needs (NRE of $300M; assumption, swept $30M to $436M).
The central demand is 5 trillion tokens a day for this one model, the same for every chip:

| tokens a day | MDS-1 systems | per system | of which NRE | MDF-1 systems | per system | of which NRE and unique masks |
|---|---|---|---|---|---|---|
| 0.5 trillion | 62 | $6,000,000 | $5,130,000 | 58 | $6,680,000 | $6,480,000 |
| 1.0 trillion | 125 | $3,430,000 | $2,560,000 | 116 | $3,440,000 | $3,240,000 |
| 2.0 trillion | 250 | $2,150,000 | $1,280,000 | 232 | $1,820,000 | $1,620,000 |
| 5.0 trillion | 624 | $1,380,000 | $510,000 | 580 | $850,000 | $650,000 |
| 10.0 trillion | 1,248 | $1,130,000 | $260,000 | 1,160 | $520,000 | $320,000 |
| 20.0 trillion | 2,496 | $1,000,000 | $130,000 | 2,321 | $360,000 | $160,000 |

Design cost alone is $0.041 per million tokens at the central demand, whatever the
design. MDS-1 breaks even with the GPUs' decode cost at about 1.6 trillion tokens a day
([04](04_results.md), section 18). This is a chip for a model served at very large scale.

## MDF-1: what changes when the weights are fixed

| | MDS-1 | MDF-1 |
|---|---|---|
| weight storage | SRAM, 0.5 mm²/MiB, plus multiplier lanes | mask-programmed, 0.127 mm²/MiB including arithmetic |
| dies | 521 | 93 |
| distinct die designs | 3 | 95 |
| tiles | 66 | 12 |
| time per weight operation | 3.91 µs (a published SRAM wafer machine's) | 0.55 µs (a shipped fixed-weight die's) |
| idle weight storage power | 71 W | 15 W |
| new model version | load weights | 95 dies × 2 mask layers: $55.8M and a fab cycle |

MDF-1's density comes from one shipped chip: 8B parameters at 3 and 6 bits on an 815 mm² N6 die
(taalas-hc1), scaled by 1.5 to N4P-class (assumption). Its latency is what that chip's published
speed implies. Both are single-die results at 8B parameters carried to 93 dies; nothing
public confirms they carry. The vendor's own figure for a model of this class is labeled a simulation
by the vendor.

## What has not been validated

- No floorplan, no place-and-route of a die, no timing closure of a 512-row bank at
  0.6 GHz.
- The pooled attention engine (2 PFLOP/s) is sized by assumption, and it caps the system's throughput. Queueing in front of it is not modeled.
- A 521-die system has 66 tiles and their links to bring up, test and keep
  running. Availability is not modeled.
- Wake from retention in 10 ns across a whole bank.
- The weight loader, error correction and repair flow.
