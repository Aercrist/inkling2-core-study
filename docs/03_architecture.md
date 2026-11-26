<!-- generated from docs/templates/03_architecture.md.j2 by experiments/render_docs.py; edit the template -->
# 03 · Architecture: MDS-1 (reloadable) and MDF-1 (fixed)

Two designs exist on paper. We did not lay out either design. We synthesized one block
([07](07_silicon.md)). Each figure here is **modeled** from the constants in `configs/hw/s_mds1.yaml`
and `f_mdf1.yaml`. Each constant cites a source or has the mark *assumption*. MDF-1 is a comparison
point, not a proposal.

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

- **Layers over tiles.** 66 tiles of a maximum of 8 dies. A layer uses about 10.6 dies on
  average: one hub die for attention, routing and normalization, and other dies for its 384
  experts. Thus a layer spans more than one tile. The hub die gets access to the experts on the
  second tile through the serial link (modeled). There are 521 dies in all, of 3 designs (weight die, hub die, attention-tile die),
  and each added design costs one more mask set.
- **Weights never move.** Each weight tensor is in SRAM banks 512 rows deep, with a
  multiplier lane on the output of the bank. A matrix-vector product is one pass over the bank:
  512 cycles at 0.6 GHz, 0.83 µs, for any
  matrix size.
- **Tokens move.** The hub die of a layer sends a row to the 8 experts that the router picked. The
  row is already quantized to NVFP4 (4.5 bits per value). It goes through die-to-die links in the
  tile. Then the hidden state goes to the next tile through a serial link.
- **One attention tile.** The 40 sliding-window layers attend in their own hub die (window 1,024,
  kept in on-die SRAM). The 8 full-attention layers and the 5 layers of the draft model send their
  queries to a shared attention tile. This tile holds the whole KV cache in HBM.
- **Reloadable.** The weight loader writes the weights at boot. At 10 GB/s from storage (assumption),
  a full load takes about 39 seconds.

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

That is the area of 5.9 wafers per system, and
0.76 GB of weights per die. Memory is 48% of the silicon, and the
multiplier lanes are most of the remainder. We set the area of SRAM at 0.5 mm² per
MiB. This is two times the published macro density at this class of node. The margin is for error
correction, routing and the power grid (sources: sram-density, sram-3nm-macro, vcache).

We predicted 250 dies (160 to 330). The design
has 521 dies. The prediction did not include the multiplier lanes. Also, the synthesized lane is 2.8x the area that we assumed (384 dies before synthesis).

## Timing of one step at batch 1

| | time | share |
|---|---|---|
| passes over weight banks, with the per-operation latency | 1.096 ms | 93% |
| full attention: arithmetic | 0.026 ms | 2% |
| full attention: KV fetch | 0.025 ms | 2% |
| hops between tiles | 0.016 ms | 1% |
| step boundary, sampler | 0.019 ms | 2% |
| **step** | **1.181 ms** | 2,960 tok/s per user |

A weight operation takes 3.91 µs in all. That is not a design target. It is the whole
time per weight operation that a published wafer-scale SRAM machine shows on Llama 3.1 8B
([06](06_simulator.md)). This time includes the bank pass of S (0.83 µs). A design that needs only
one and a half times its own pass is
2.5x
faster than the reported figure.

Each layer has its own tile, and thus the design pipelines requests. While one request is in layer 20,
a different request can be in layer 3. For this reason, throughput scales with batch until the shared
attention tile saturates ([04](04_results.md), section 4).

## Power

| state | weight banks | logic and clock | links, idle | KV memory | host | total |
|---|---|---|---|---|---|---|
| idle, managed (banks in retention, lanes clock-gated) | 71 W | 2,080 W | 840 W | 78 W | 500 W | **3,580 W** |
| idle, unmanaged (full voltage, clocks on) | 274 W | 6,250 W | 840 W | 78 W | 500 W | **7,950 W** |

| serving | system power |
|---|---|
| one request, 8K context | 4,230 W |
| 256 requests, 32K context | 9,100 W |

- **Managed SRAM leakage is small.** The measured leakage of a 3 nm macro at 25 °C is 27.5 pW per bit
  (sram-3nm-macro). This is 87 W for this model. We multiply by 3 for
  operating temperature (assumption; no hot figure is public). We hold idle banks at a retention
  voltage that decreases the leakage to 20% (published range 5% to 31%;
  sram-retention).
- **Idle logic is the larger term**: 4.0 W per die managed,
  12.0 W unmanaged (assumptions, swept 1 to 15 W).
- **Power density is low.** When it is fully busy, the system uses about 17 W per die,
  2.2 W/cm². A B300 is 1,100 W in one package. Air
  cooling is possible for MDS-1. Its difficulty is the number of dies, not heat.

## Packaging, power delivery, cooling

- **Tiles.** About 8 large dies on one fan-out or interposer tile
  (8 is the assumed limit; a layer spans more than one tile). Die-to-die links in a
  tile are short-reach, 0.25 pJ/bit (d2d-energy). We set the cost of packaging and test at
  $700 per die. We built this cost from the same sourced parts as the
  package of the HBM chip (cowos-cost). The HBM packages of the attention tile have the same cost as
  those of the HBM chip.
- **Between tiles.** 72 lanes of 112G serial links per neighbour, 900 GB/s,
  250 ns per hop with error correction (ucie-targets, with margin;
  swept 0.1 to 2 µs).
- **System.** 66 tiles and the attention tile: several chassis, not one board.
  We set the cost of boards, power delivery, cooling and chassis at $500 per
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

For comparison, four B300 with networking are about $240,000
(reseller range $224,000 to $304,000; b300-price).

Memory is repairable with spare rows. Thus only the 52% of an MDS-1 die
that is logic must be defect-free. The storage of a fixed-weight die has no spares.

These figures do not include vendor margin, spares, the yield of tile assembly, or software and
bring-up. GPUs are at street price, which includes all of them.

We spread the design cost over the systems that a token demand needs (NRE of $300M; assumption, swept $30M to $436M).
The central demand is 5 trillion tokens a day for this one model, the same for each chip:

| tokens a day | MDS-1 systems | per system | of which NRE | MDF-1 systems | per system | of which NRE and unique masks |
|---|---|---|---|---|---|---|
| 0.5 trillion | 62 | $6,000,000 | $5,130,000 | 58 | $6,680,000 | $6,480,000 |
| 1.0 trillion | 125 | $3,430,000 | $2,560,000 | 116 | $3,440,000 | $3,240,000 |
| 2.0 trillion | 250 | $2,150,000 | $1,280,000 | 232 | $1,820,000 | $1,620,000 |
| 5.0 trillion | 624 | $1,380,000 | $510,000 | 580 | $850,000 | $650,000 |
| 10.0 trillion | 1,248 | $1,130,000 | $260,000 | 1,160 | $520,000 | $320,000 |
| 20.0 trillion | 2,496 | $1,000,000 | $130,000 | 2,321 | $360,000 | $160,000 |

Design cost alone is $0.041 per million tokens at the central demand, for any
design. The cost of MDS-1 becomes equal to the decode cost of the GPUs at about 1.6 trillion tokens a day
([04](04_results.md), section 18). This is a chip for a model served at extremely large scale.

## MDF-1: what changes when the weights are fixed

| | MDS-1 | MDF-1 |
|---|---|---|
| weight storage | SRAM, 0.5 mm²/MiB, plus multiplier lanes | mask-programmed, 0.127 mm²/MiB with arithmetic |
| dies | 521 | 93 |
| distinct die designs | 3 | 95 |
| tiles | 66 | 12 |
| time per weight operation | 3.91 µs (from a published SRAM wafer machine) | 0.55 µs (from a shipped fixed-weight die) |
| idle weight storage power | 71 W | 15 W |
| new model version | load weights | 95 dies × 2 mask layers: $55.8M and a fab cycle |

The density of MDF-1 comes from one shipped chip. That chip has 8B parameters at 3 and 6 bits on an
815 mm² N6 die (taalas-hc1). We scaled its density by 1.5 to N4P-class (assumption). Its latency is
the value that the published speed of that chip implies. The two values are single-die results at 8B
parameters that we extend to 93 dies, and no public source confirms that they extend. The vendor labels its own
figure for a model of this class a simulation.

## What is not validated

- No floorplan, no place-and-route of a die, no timing closure of a 512-row bank at
  0.6 GHz.
- The size of the pooled attention engine (2 PFLOP/s) is an assumption, and this engine limits the throughput of the system. We did not model the queue in front of it.
- A 521-die system has 66 tiles and their links to bring up, test and keep
  in operation. We did not model availability.
- Wake from retention in 10 ns across a whole bank.
- The weight loader, error correction and repair flow.
