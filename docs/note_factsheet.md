# Fact sheet of the technical note on the NVFP4 block

One row per number in the note, generated from the files of `Aercrist/inkling2-core-study` at `a425f8b`.
Evidence classes: laid out (report of the place-and-route tools on the routed design), simulated (RTL or
gate-level simulation, including power from simulated activity), counted (arithmetic on the headers of the
published checkpoint), spec (the definition of the block), arithmetic (computed from other rows, formula in
the row), assumed, cited, modelled (output of the study's system simulator, shown only to say what was
withdrawn).

| key | value | what it is | condition | evidence | source |
|---|---|---|---|---|---|
| volt | 0.70 V | supply voltage of the library corner | typical corner | laid out | results/silicon.json silicon.library |
| temp | 25 °C | temperature of the library corner | typical corner | laid out | results/silicon.json silicon.library |
| ioPct | 20 % | input and output delay as a share of the clock period | SDC of every design | laid out | results/silicon.json silicon.constraints.sdc |
| clockMargin | none | clock uncertainty or derating in the constraint | the SDC sets the clock and the port delays only | laid out | results/silicon.json silicon.constraints.sdc |
| utilTarget | 50 % | target core utilization | floorplan | laid out | results/silicon.json silicon.constraints.core_utilization_percent |
| clockPs | 1,800 ps | clock period the block closes timing at | ASAP7 RVT, typical corner, 0.70 V, 25 C, 1,800 ps clock, after routing | laid out | results/silicon.json silicon.runs['rel:1800'].clock_period_constraint_ps |
| clockMhz | 556 MHz | the same as a frequency | 1e6 / period in ps | arithmetic | silicon.headline.reloadable.clock_mhz |
| setupSlack | 23.3 ps | worst setup slack | ASAP7 RVT, typical corner, 0.70 V, 25 C, 1,800 ps clock, after routing | laid out | results/silicon.json silicon.runs['rel:1800'].setup_wns_ps |
| holdSlack | 38.1 ps | worst hold slack | ASAP7 RVT, typical corner, 0.70 V, 25 C, 1,800 ps clock, after routing | laid out | results/silicon.json silicon.runs['rel:1800'].hold_wns_ps |
| failPs | 1,600 ps | tighter clock that was tried | same flow and constraints | laid out | silicon.fmax.rel.points[0].constraint_ps |
| failSlack | 16.6 ps | setup violation at that clock | after routing | laid out | silicon.fmax.rel.points[0].setup_wns_ps |
| failArea | 546.8 µm² | cell area of that run | after routing | laid out | silicon.fmax.rel.points[0].cell_area_um2 |
| failAreaExtra | 36 µm² | cell area the tools added at the tighter clock | failArea - cellArea | arithmetic | same |
| cellArea | 511.1 µm² | standard-cell area without fill and tap cells | ASAP7 RVT, typical corner, 0.70 V, 25 C, 1,800 ps clock, after routing | laid out | results/silicon.json silicon.runs['rel:1800'].cell_area_um2 |
| cells | 4,548 cells | standard cells without fill and tap cells | ASAP7 RVT, typical corner, 0.70 V, 25 C, 1,800 ps clock, after routing | laid out | results/silicon.json silicon.runs['rel:1800'].cells |
| flops | 273 flip-flops | sequential cells | ASAP7 RVT, typical corner, 0.70 V, 25 C, 1,800 ps clock, after routing | laid out | results/silicon.json silicon.runs['rel:1800'].sequential_cells |
| synthArea | 493.8 µm² | cell area after synthesis, before placement | Yosys and ABC, flattened | laid out | results/silicon.json silicon.runs['rel:1800'].synth_cell_area_um2 |
| coreArea | 982.6 µm² | core area of the floorplan | 50% target utilization, square core | laid out | results/silicon.json silicon.runs['rel:1800'].core_area_um2 |
| dieSide | 33.4 µm | side of the square die outline | sqrt(die_area_um2) | arithmetic | results/silicon.json silicon.runs['rel:1800'].die_area_um2 |
| utilPct | 52 % | cell area over core area | cellArea / coreArea; the flow's own figure also counts the tap cells | arithmetic | results/silicon.json silicon.runs['rel:1800'] |
| areaComb | 359.9 µm² | area of combinational cells | ASAP7 RVT, typical corner, 0.70 V, 25 C, 1,800 ps clock, after routing | laid out | results/silicon.json silicon.runs['rel:1800'].area_by_class_um2.multi_input_combinational_cell |
| areaCombPct | 70 % | share of cell area, combinational cells | over cellArea | arithmetic | same |
| areaFlops | 79.6 µm² | area of flip-flops | ASAP7 RVT, typical corner, 0.70 V, 25 C, 1,800 ps clock, after routing | laid out | results/silicon.json silicon.runs['rel:1800'].area_by_class_um2.sequential_cell |
| areaFlopsPct | 16 % | share of cell area, flip-flops | over cellArea | arithmetic | same |
| areaInv | 39.2 µm² | area of inverters | ASAP7 RVT, typical corner, 0.70 V, 25 C, 1,800 ps clock, after routing | laid out | results/silicon.json silicon.runs['rel:1800'].area_by_class_um2.inverter |
| areaInvPct | 8 % | share of cell area, inverters | over cellArea | arithmetic | same |
| areaRepair | 23.9 µm² | area of buffers added for timing repair | ASAP7 RVT, typical corner, 0.70 V, 25 C, 1,800 ps clock, after routing | laid out | results/silicon.json silicon.runs['rel:1800'].area_by_class_um2.timing_repair_buffer |
| areaRepairPct | 5 % | share of cell area, buffers added for timing repair | over cellArea | arithmetic | same |
| areaClock | 8.5 µm² | area of the clock tree (buffers and inverters) | ASAP7 RVT, typical corner, 0.70 V, 25 C, 1,800 ps clock, after routing | laid out | results/silicon.json silicon.runs['rel:1800'].area_by_class_um2.clock_buffer + clock_inverter |
| areaClockPct | 2 % | share of cell area, clock tree | over cellArea | arithmetic | same |
| repairBuffers | 267 cells | buffers added for timing repair | ASAP7 RVT, typical corner, 0.70 V, 25 C, 1,800 ps clock, after routing | laid out | results/silicon.json silicon.runs['rel:1800'].count_by_class.timing_repair_buffer |
| tapCells | 291 cells | tap cells (not in the cell count) | ASAP7 RVT, typical corner, 0.70 V, 25 C, 1,800 ps clock, after routing | laid out | results/silicon.json silicon.runs['rel:1800'].count_by_class.tap_cell |
| areaLane | 31.9 µm² | cell area per multiply-accumulate lane | cellArea / 16 | arithmetic | silicon.headline.reloadable.area_per_mac_lane_um2 |
| powerDyn | 1.092 mW | dynamic power | OpenSTA report_power, routed netlist, SPEF parasitics, VCD of a zero-delay gate-level simulation; real weight row (w13 row 0, layer 0, expert 0) against 8 token rows, a block on every clock | simulated | results/silicon.json silicon.runs['rel:1800'].workloads.w13_row0.power_w.dynamic |
| powerInternal | 0.623 mW | internal (cell) power | OpenSTA report_power, routed netlist, SPEF parasitics, VCD of a zero-delay gate-level simulation; real weight row (w13 row 0, layer 0, expert 0) against 8 token rows, a block on every clock | simulated | same .internal |
| powerSwitching | 0.469 mW | switching (net) power | OpenSTA report_power, routed netlist, SPEF parasitics, VCD of a zero-delay gate-level simulation; real weight row (w13 row 0, layer 0, expert 0) against 8 token rows, a block on every clock | simulated | same .switching |
| powerLeak | 0.41 µW | leakage power | library, typical corner | simulated | same .leakage |
| powerComb | 0.666 mW | power in combinational cells | OpenSTA report_power, routed netlist, SPEF parasitics, VCD of a zero-delay gate-level simulation; real weight row (w13 row 0, layer 0, expert 0) against 8 token rows, a block on every clock | simulated | results/silicon.json silicon.runs['rel:1800'].workloads.w13_row0.power_by_group_w.combinational |
| powerSeq | 0.276 mW | power in flip-flops | OpenSTA report_power, routed netlist, SPEF parasitics, VCD of a zero-delay gate-level simulation; real weight row (w13 row 0, layer 0, expert 0) against 8 token rows, a block on every clock | simulated | same .sequential |
| powerClock | 0.150 mW | power in the clock tree | OpenSTA report_power, routed netlist, SPEF parasitics, VCD of a zero-delay gate-level simulation; real weight row (w13 row 0, layer 0, expert 0) against 8 token rows, a block on every clock | simulated | same .clock |
| energyBlock | 1.97 pJ | energy per 16-value block | total power x clock period, a block on every clock | arithmetic | results/silicon.json silicon.runs['rel:1800'].workloads.w13_row0.energy_per_block_pj |
| energyMacFj | 123 fJ | energy per multiply-accumulate | energyBlock / 16 | arithmetic | same .energy_per_mac_pj |
| powerTokens | 8 token rows | token rows in the power workload |  | simulated | results/silicon.json silicon.runs['rel:1800'].workloads.w13_row0.description |
| idleSeq | 0.180 mW | idle power in flip-flops | in_valid low | simulated | results/silicon.json silicon.runs['rel:1800'].workloads.idle.power_by_group_w.sequential |
| idleClock | 0.150 mW | idle power in the clock tree | in_valid low | simulated | results/silicon.json silicon.runs['rel:1800'].workloads.idle.power_by_group_w.clock |
| energyRowsLow | 1.94 pJ | lowest energy per block over five weight rows of the first projection | OpenSTA report_power, routed netlist, SPEF parasitics, VCD of a zero-delay gate-level simulation; real weight row (w13 row 0, layer 0, expert 0) against 8 token rows, a block on every clock | simulated | results/silicon.json silicon.runs['rel:1800'].workloads.w13_row* |
| energyRowsHigh | 1.97 pJ | highest of the five | OpenSTA report_power, routed netlist, SPEF parasitics, VCD of a zero-delay gate-level simulation; real weight row (w13 row 0, layer 0, expert 0) against 8 token rows, a block on every clock | simulated | same |
| energyW2 | 1.93 pJ | energy per block on a row of the second projection | w2 row 0, 144 blocks | simulated | results/silicon.json silicon.runs['rel:1800'].workloads.w2_row0 |
| idlePower | 0.331 mW | power with in_valid low and the data pins at zero | same flow; there is no clock gating in the block | simulated | results/silicon.json silicon.runs['rel:1800'].workloads.idle.power_w.total |
| idlePct | 30 % | idle power over active power | idlePower / total active power | arithmetic | same |
| pinsAnnotated | 15,433 pins | pins that took their activity from the VCD |  | simulated | results/silicon.json silicon.runs['rel:1800'].workloads.w13_row0.vcd_pins_annotated |
| pinsNot | 2 pins | pins that did not |  | simulated | same .pins_unannotated |
| s0Flops | 146 flip-flops | registers of stage S0 | reloadable variant, one row | laid out | routed DEF of the rerun, by register name (scripts/layout_figure.py); equal to rtl/nvfp4/SPEC.md section 4 |
| s1Flops | 30 flip-flops | registers of stage S1 | reloadable variant, one row | laid out | routed DEF of the rerun, by register name (scripts/layout_figure.py); equal to rtl/nvfp4/SPEC.md section 4 |
| s2Flops | 28 flip-flops | registers of stage S2 | reloadable variant, one row | laid out | routed DEF of the rerun, by register name (scripts/layout_figure.py); equal to rtl/nvfp4/SPEC.md section 4 |
| s3Flops | 31 flip-flops | registers of stage S3 | reloadable variant, one row | laid out | routed DEF of the rerun, by register name (scripts/layout_figure.py); equal to rtl/nvfp4/SPEC.md section 4 |
| s4Flops | 38 flip-flops | registers of stage S4 | reloadable variant, one row | laid out | routed DEF of the rerun, by register name (scripts/layout_figure.py); equal to rtl/nvfp4/SPEC.md section 4 |
| blockValues | 16 values | values in one block |  | spec | rtl/nvfp4/SPEC.md |
| codeBits | 4 bits | width of a value code (E2M1) |  | spec | rtl/nvfp4/SPEC.md |
| scaleBits | 8 bits | width of a block scale (E4M3) |  | spec | rtl/nvfp4/SPEC.md |
| codeWord | 64 bits | codes of one block | 16 x 4 | spec | rtl/nvfp4/SPEC.md |
| magnitudes | 0, 0.5, 1, 1.5, 2, 3, 4 and 6 | the eight magnitudes of a value code |  | spec | rtl/nvfp4/SPEC.md |
| qMax | 12 | largest magnitude of a doubled value | 2 x 6 | spec | rtl/nvfp4/SPEC.md |
| sigBits | 24 bits | significand of binary32 |  | spec | rtl/nvfp4/SPEC.md |
| mantBits | 4 bits | width of a scale mantissa with its leading one | M in 0..15 | spec | rtl/nvfp4/SPEC.md |
| mmBits | 8 bits | width of the product of the two scale significands | 4 + 4 | spec | rtl/nvfp4/SPEC.md |
| expSumBits | 5 bits | width of the sum of the two scale exponents | two values of 1 to 15 | arithmetic | rtl/nvfp4/SPEC.md section 2; register es_q in the routed netlist |
| termExpBits | 8 bits | width of the exponent of the normalised block term | the binary32 exponent field | laid out | register t_e in the routed netlist (scripts/layout_figure.py) |
| inBits | 144 bits | data bits entering the block per clock | 64 + 64 + 8 + 8 | arithmetic | rtl/nvfp4/SPEC.md section 3 |
| outBits | 32 bits | width of the result | binary32 | spec | rtl/nvfp4/SPEC.md |
| rowBlocks | 288 blocks | blocks in a row of the first expert projection (4,608 inputs) |  | spec | rtl/nvfp4/SPEC.md |
| hidden | 4,608 values | width of the hidden state |  | counted | results/phase0_workload.json config.hidden_size |
| dBits | 13 bits | width of the exact block sum D |  | spec | rtl/nvfp4/SPEC.md |
| dMax | 2,304 | largest magnitude of D | 16 x 12 x 12 | spec | rtl/nvfp4/SPEC.md |
| pBits | 19 bits | width of the exact block term | 2,304 x 15 x 15 < 2^19 | spec | rtl/nvfp4/SPEC.md |
| addBits | 28 bits | width of the accumulator's adder |  | spec | rtl/nvfp4/SPEC.md |
| latency | 5 clocks | clocks from the last block to the result |  | spec | rtl/nvfp4/SPEC.md |
| termExpLow | −20 | smallest exponent of a block term |  | spec | rtl/nvfp4/SPEC.md |
| termBoundExp | 27 | block terms are smaller than 2 to this power | 2,304 x 15 x 15 x 2^8 = 132,710,400 < 2^27 | arithmetic | rtl/nvfp4/SPEC.md section 2 |
| termMax | 1.33e8 | largest magnitude of a block term |  | spec | rtl/nvfp4/SPEC.md |
| fixedPointBits | 57 bits | width of a fixed-point accumulator that holds every partial sum of a 288-block row exactly | terms are multiples of 2^-20 and below 2^27; 288 terms stay below 2^36; one sign bit. Not built | arithmetic | rtl/nvfp4/SPEC.md section 2 |
| vectors | 1,589,011 rows | dot products compared with the NumPy reference, no mismatch | Verilator, RTL; a vector is one row streamed in and one binary32 compared | simulated | results/silicon.json verification.vectors.vectors_total |
| vectorsRel | 1,165,348 rows | of which on the reloadable block |  | simulated | verification.vectors.vectors_reloadable |
| vecReal0 | 655,360 rows | real weight rows of layer 0 expert 0 against its token rows | both projections | simulated | results/silicon.json verification.vectors.sections |
| vecRealOther | 198,144 rows | real weight rows sampled from six other experts | layers 0, 13, 26 and 39; Gaussian, heavy-tailed, sparse and embedding-derived token rows | simulated | results/silicon.json verification.vectors.sections |
| vecRandom | 228,000 rows | random codes and scale bytes, rows of 1 to 288 blocks | all 256 scale bytes including NaN in part of them | simulated | results/silicon.json verification.vectors.sections |
| vecDirected | 1,076 rows | directed cases | NaN, zero, subnormal and maximum scales, cancellation, saturation, sign patterns, rounding orders | simulated | results/silicon.json verification.vectors.sections |
| vecStress | 50,000 rows | rounding stress | terms built to give ties, long cancellations and sticky-only alignments | simulated | results/silicon.json verification.vectors.sections |
| vecX8 | 32,768 rows | eight-row variant | eight weight streams sharing one activation stream | simulated | results/silicon.json verification.vectors.sections |
| vecFixed | 423,663 rows | fixed-weight modules | 24 modules | simulated | results/silicon.json verification.vectors.sections |
| covInexact | 4,436 additions | additions with an inexact result in a sample of the directed and stress rows | counted by an integer model of the adder | simulated | results/silicon.json verification.vectors.adder_case_coverage_* |
| covTies | 720 additions | ties, rounded to even either way | same sample | simulated | results/silicon.json verification.vectors.adder_case_coverage_* |
| covExpCarry | 304 additions | roundings that carried into the exponent | same sample | simulated | results/silicon.json verification.vectors.adder_case_coverage_* |
| covCancel | 1,120 additions | cancellations with a left shift of 2 or more | same sample | simulated | results/silicon.json verification.vectors.adder_case_coverage_* |
| covRealAdds | 12,006 additions | additions in a sample of the real rows, none of them inexact | counted by the same model | simulated | results/silicon.json verification.vectors.adder_case_coverage_* |
| icarusVectors | 104 rows | rows repeated in a second simulator | Icarus Verilog, RTL | simulated | verification.icarus |
| torchOutputs | 2,359,296 outputs | outputs of one real expert projection where the reference equals torch.nn.functional.linear bit for bit | 4,608 x 4,608 weights of layer 0 expert 0 against 512 token rows; CPU, float32 | simulated | verification.torch.sets[w13 x 512 tokens] |
| torchTokens | 512 token rows | token rows in that comparison |  | simulated | verification.torch.sets (name of the set) |
| exactBound | 24 | exponent of the bound under which no addition rounds: partial sums stayed below 2^24 units | the real expert and its token rows | simulated | rtl/nvfp4/REPORT.md section 6 |
| sampleLayers | 0, 13, 26 and 39 | layers the other experts were sampled from |  | counted | rtl/nvfp4/REPORT.md section 5 |
| sampleRows | 64 rows | rows sampled from each of the other experts |  | counted | rtl/nvfp4/REPORT.md section 9 |
| sampleExperts | 6 experts | other experts sampled |  | counted | rtl/nvfp4/REPORT.md section 9 |
| torchRandomPct | 9 % | share of outputs bit-identical to torch when codes and scales are random | 512 rows x 64 tokens; additions round | simulated | verification.torch.sets[random codes] |
| nibbleR | 0.61 | correlation of per-column weight RMS with the layer's norm weights, low nibble first | 4,608 columns of one expert | counted | results/silicon.json layout.evidence.nibble_order_from_data |
| nibbleRSwapped | 0.00 | the same with the nibbles swapped |  | counted | same |
| routedDesigns | 13 designs | designs placed and routed, each re-simulated at gate level and bit-exact | Icarus with the ASAP7 cell models, on the power workloads | simulated | results/silicon.json silicon.runs[*].post_route_netlist_bit_exact |
| fixedArea | 878.1 µm² | cell area of the fixed-weight block, median of 5 rows | one hard-wired row of 288 blocks, same clock, tokens and throughput as the reloadable block; each of 5 real rows placed and routed | laid out | silicon.hardwired_row_length_sweep.blocks_per_row['288'].fixed_cell_area_um2 |
| fixedAreaLow | 871.0 µm² | smallest of the 5 | one hard-wired row of 288 blocks, same clock, tokens and throughput as the reloadable block; each of 5 real rows placed and routed | laid out | same |
| fixedAreaHigh | 882.8 µm² | largest of the 5 | one hard-wired row of 288 blocks, same clock, tokens and throughput as the reloadable block; each of 5 real rows placed and routed | laid out | same |
| fixedRows | 5 rows | real rows built as fixed blocks |  | laid out | same |
| oneDesigns | 2 designs | repeated-block rows built |  | laid out | silicon.hardwired_row_length_sweep.blocks_per_row['1'] |
| fixedCells | 8,234 cells | cells of the fixed-weight block, median | one hard-wired row of 288 blocks, same clock, tokens and throughput as the reloadable block; each of 5 real rows placed and routed | laid out | same .fixed_cells |
| fixedFlops | 279 flip-flops | flip-flops of the fixed-weight block | one hard-wired row of 288 blocks, same clock, tokens and throughput as the reloadable block; each of 5 real rows placed and routed | laid out | same .fixed_flip_flops |
| fixedEnergy | 3.284 pJ | energy per block of the fixed-weight block, median | one hard-wired row of 288 blocks, same clock, tokens and throughput as the reloadable block; each of 5 real rows placed and routed; power as for the reloadable block | simulated | same .fixed_energy_per_block_pj |
| relEnergySame | 1.958 pJ | energy per block of the reloadable block on the same 5 rows, median | one hard-wired row of 288 blocks, same clock, tokens and throughput as the reloadable block; each of 5 real rows placed and routed | simulated | same .reloadable_energy_per_block_pj_same_workload |
| fixedAreaX | 1.72 x | fixed area over reloadable area | 1 / median of the 5 per-row ratios | arithmetic | same .area_ratio_reloadable_over_fixed |
| fixedEnergyX | 1.68 x | fixed energy over reloadable energy | 1 / median of the 5 per-row ratios | arithmetic | same .energy_ratio_reloadable_over_fixed |
| rowWeights | 4,608 weights | weights hard-wired in the fixed block | 288 blocks of 16 | spec | silicon.headline.fixed.weights_in_module |
| tableBits | 72 bits | width of one entry of the weight table | 16 codes of 4 bits and one scale of 8 bits | arithmetic | rtl/nvfp4/SPEC.md section 1 |
| oneArea | 453.7 µm² | cell area of the fixed block when a lane always sees the same weight | 288-block row whose 16 weights are the same in every block; 2 designs placed and routed; the reloadable block does the same work with its weight pins held | laid out | silicon.hardwired_row_length_sweep.blocks_per_row['1'].fixed_cell_area_um2 |
| oneFlops | 204 flip-flops | its flip-flops, median of 2 | 288-block row whose 16 weights are the same in every block; 2 designs placed and routed; the reloadable block does the same work with its weight pins held | laid out | same .fixed_flip_flops |
| oneEnergy | 1.636 pJ | its energy per block | 288-block row whose 16 weights are the same in every block; 2 designs placed and routed; the reloadable block does the same work with its weight pins held | simulated | same .fixed_energy_per_block_pj |
| oneRelEnergy | 1.831 pJ | energy per block of the reloadable block on that work | 288-block row whose 16 weights are the same in every block; 2 designs placed and routed; the reloadable block does the same work with its weight pins held | simulated | same .reloadable_energy_per_block_pj_same_workload |
| oneAreaSavePct | 11 % | area saved by fixing the weights in that case | 1 - fixed / reloadable | arithmetic | same .area_ratio_reloadable_over_fixed |
| oneEnergySavePct | 11 % | energy saved in that case | 1 - fixed / reloadable | arithmetic | same .energy_ratio_reloadable_over_fixed |
| perWeight | 0.092 µm² | cell area added per hard-wired weight between 1 and 288 distinct blocks | (fixedArea - oneArea) / (4,608 - 16) | arithmetic | silicon.hardwired_row_length_sweep.marginal_area_per_hardwired_weight_um2_1_to_288_blocks |
| bitsPerWeight | 4.5 bits | stored bits per NVFP4 weight, scale included |  | counted | results/phase0_workload.json bytes.bits_per_param.nvfp4 |
| romBitUm2 | 0.021 µm² | cell area per stored bit of the weight table | perWeight / bitsPerWeight | arithmetic |  |
| romBitCoreUm2 | 0.039 µm² | the same at the utilization of this block | romBitUm2 / utilization | arithmetic |  |
| sramBitUm2 | 0.060 µm² | the SRAM allowance used further down, per bit | 0.5 mm² per MiB | arithmetic |  |
| trapAreaX | 1.56 x | reloadable over fixed area for rows only one block long | 3 designs; synthesis removes the accumulator from the fixed block, so the ratio is not usable | laid out | silicon.hardwired_row_length_sweep.rows_of_one_block_accumulator_optimized_away |
| trapDesigns | 3 designs | one-block rows built |  | laid out | same |
| trapArea | 328 µm² | cell area of those fixed blocks, median of 3 |  | laid out | same |
| params | 594,050,672,720 parameters | parameters, counted from the tensor headers | equals the count the Hub reports | counted | results/phase0_workload.json params.total_hub_convention |
| paramsBillion | 594 billion | the same, rounded |  | counted | results/phase0_workload.json params.total_hub_convention |
| fileGb | 393.5 GB | size of the weight files | 1,854 tensors in 25 files | counted | results/phase0_workload.json source.file_bytes |
| nvfp4ParamPct | 82 % | share of parameters stored as NVFP4 |  | counted | results/phase0_workload.json params.share_by_class.nvfp4 |
| experts | 384 experts | experts per layer |  | counted | results/phase0_workload.json config.experts |
| topK | 8 experts | experts a token is routed to in each layer |  | counted | results/phase0_workload.json config.top_k |
| nvLayers | 40 layers | layers whose experts are NVFP4 | layers 0 to 39 of 48 | counted | rtl/nvfp4/REPORT.md; results/phase0_workload.json |
| layers | 48 layers | decoder layers |  | counted | results/phase0_workload.json config.layers |
| macsToken | 17.96 billion | weight multiply-accumulates per decoded token | one request | counted | results/phase0_workload.json touched_per_token.weight_macs |
| nvMacsToken | 10.19 billion | of which NVFP4 by NVFP4 | top_k x NVFP4 layers x parameters of one expert | arithmetic | results/phase0_workload.json |
| nvMacsPct | 57 % | their share |  | arithmetic | results/phase0_workload.json |
| bytesToken | 19.3 GB | weight bytes read per decoded token | one request | counted | results/phase0_workload.json touched_per_token.total |
| nvBytesToken | 5.73 GB | of which NVFP4 expert weights |  | counted | results/phase0_workload.json touched_per_token.bytes.routed_experts_nvfp4 |
| laneRate | 8.89 billion per second | multiply-accumulates per second of one block | 16 per clock at the closed clock | arithmetic | 16 x 1e12 / clockPs |
| blocksToken | 637 million | 16-value blocks per decoded token | nvMacsToken / 16 | arithmetic |  |
| laneSeconds | 1.15 s | time one block needs for the NVFP4 work of one token | blocksToken x clock period | arithmetic |  |
| energyToken | 1.25 mJ | energy of the NVFP4 work of one token in such blocks | blocksToken x energyBlock; the block only | arithmetic |  |
| kTok | 1,000 tokens/s | rate used to put the arithmetic in familiar units | a unit, not a design point | arithmetic |  |
| kBlocks | 1,147 blocks | blocks busy at that rate | kTok x laneSeconds | arithmetic |  |
| kAreaMm2 | 0.59 mm² | their cell area | kBlocks x cellArea | arithmetic |  |
| kPowerW | 1.25 W | their power | kTok x energyToken | arithmetic |  |
| kTraffic | 19.3 TB/s | weight bytes per second at that rate | kTok x bytesToken | arithmetic |  |
| kNvTraffic | 5.73 TB/s | NVFP4 weight bytes per second at that rate | kTok x nvBytesToken | arithmetic |  |
| kNvTrafficX | 1.2 x | that over the memory bandwidth of one H200 | kNvTraffic / h200Bw | arithmetic |  |
| kTrafficX | 4.0 x | that traffic over the memory bandwidth of one H200 | kTraffic / h200Bw | arithmetic |  |
| h200Bw | 4.8 TB/s | memory bandwidth of one H200 | vendor figure | cited | results/calibration_llama.json rows.h200.memory (docs/sources.md: nim-bench) |
| sramDensity | 0.5 mm² per MiB | SRAM area the study assumed, with error correction, routing and power grid | a shipped 7 nm cache die holds 64 MB in 36 mm² | assumed | docs/03_architecture.md; docs/sources.md: vcache, sram-density |
| sramBasis | 0.56 mm² per MiB | density of a shipped 7 nm cache die | 64 MB in 36 mm² | cited | docs/sources.md: vcache |
| storageRound | 188,000 mm² | area of the weights alone in SRAM, rounded | fileGb at sramDensity | arithmetic | results/inkling.json designs.s.area.storage |
| storageMm2 | 187,626 mm² | area of the weights alone in SRAM | fileGb at sramDensity | arithmetic | results/inkling.json designs.s.area.storage |
| dieMm2 | 800 mm² | die size used for counting | near the reticle limit | assumed | results/inkling.json designs.s.area.die_mm2 |
| storageDies | 235 dies | dies for the weights alone | storageMm2 / dieMm2, rounded up | arithmetic |  |
| waferMm | 300 mm | wafer diameter used for the wafer count |  | assumed |  |
| storageWafers | 2.7 wafers | the same in gross 300 mm wafer area | pi x 150^2 mm² per wafer | arithmetic |  |
| dies | 521 dies | dies of the machine the study modelled | weights in SRAM with a multiplier lane on every 512-row bank, plus attention, links and control | modelled | results/inkling.json designs.s.area.dies |
| bankRows | 512 rows | depth of an SRAM bank in that machine |  | modelled | results/inkling.json designs.s.scan_rows |
| laneMm2 | 134,274 mm² | area of the multiplier lanes in that machine |  | modelled | results/inkling.json designs.s.area.mac_lanes |
| lanesBillion | 2.4 billion | multiplier lanes in that machine |  | modelled | results/inkling.json designs.s.area.lanes |
| unbuiltLanePct | 52 % | share of that lane area in FP8 and BF16 lanes, which were not built | lanes x assumed area per lane | arithmetic | results/silicon_scaled.json s.um2_per_mac; results/inkling.json designs.s.area.lanes |
| laneScaled | 33.9 µm² | area per NVFP4 lane the machine model used | areaLane x 0.69 (cell pitches, ASAP7 to an N4P-class node) / 0.65 (placement utilization); both factors are the study's assumptions | assumed | results/silicon_scaled.json |
| guessArea | 12 µm² | area per lane assumed before the layout existed | N4P-class | assumed | results/silicon_scaled.json assumed_before.um2_per_mac_fp4 |
| guessAreaX | 2.8 x | scaled layout over that guess |  | arithmetic | results/silicon_scaled.json measured_over_assumed.area |
| guessClock | 1.5 GHz | clock assumed before the layout existed | N4P-class | assumed | results/silicon_scaled.json assumed_before.f_clk |
| pubH200 | 235 tokens/s | published speed, NVIDIA H200, one GPU (HBM) | Llama 3.1 8B; NVIDIA NIM, FP8 weights, 1 request, 1,000 in / 1,000 out, inter-token latency 4.24 ms | cited | results/calibration_llama.json rows.h200.published |
| modH200 | 389 tokens/s | the study's simulator on the same machine | GPU engine; HBM efficiency and per-launch latency as fitted on Blackwell engines serving Inkling 2 Small | modelled | results/calibration_llama.json rows.h200.predicted |
| errH200 | +65 % | model over published, minus one |  | arithmetic | results/calibration_llama.json rows.h200.error |
| pubCerebras | 1,981 tokens/s | published speed, Cerebras WSE-3 (SRAM, one wafer) | Llama 3.1 8B; Artificial Analysis as charted on 19 Feb 2026, about 1k input tokens | cited | results/calibration_llama.json rows.cerebras.published |
| modCerebras | 3,694 tokens/s | the study's simulator on the same machine | weight-stationary engine; one pass = 44 GB / 21 PB/s; nothing else on the clock, so this is an upper bound | modelled | results/calibration_llama.json rows.cerebras.predicted |
| errCerebras | +86 % | model over published, minus one |  | arithmetic | results/calibration_llama.json rows.cerebras.error |
| pubGroq | 594 tokens/s | published speed, Groq LPU (SRAM, many chips) | Llama 3.1 8B; Artificial Analysis as charted on 19 Feb 2026, about 1k input tokens | cited | results/calibration_llama.json rows.groq.published |
| modGroq | 2,693 tokens/s | the study's simulator on the same machine | weight-stationary engine; one pass = 230 MB / 80 TB/s; chip-to-chip latency and chips per model are not published, so this is an upper bound | modelled | results/calibration_llama.json rows.groq.predicted |
| errGroq | +353 % | model over published, minus one |  | arithmetic | results/calibration_llama.json rows.groq.error |
| pubHc1 | 16,960 tokens/s | published speed, Taalas HC1 (fixed weights, one die) | Llama 3.1 8B; vendor's labs, batch 1, 1k in / 1k out | cited | results/calibration_llama.json rows.hc1.published |
| modHc1 | 9,960 tokens/s | the study's simulator on the same machine | weight-stationary engine with this study's own timing constants (bank depth, clock, hand-off), one die, KV on-die; HC1's internals are not published | modelled | results/calibration_llama.json rows.hc1.predicted |
| errHc1 | −41 % | model over published, minus one |  | arithmetic | results/calibration_llama.json rows.hc1.error |
| calBounds | 2 machines | machines for which the simulator's figure is an upper bound by construction | Cerebras and Groq: nothing but the pass over the weights is on the clock | modelled | results/calibration_llama.json rows.*.how |
| calN | 4 machines | machines in that check |  | modelled | results/calibration_llama.json n |
| calTol | 30 % | tolerance of that check |  | modelled | results/calibration_llama.json tolerance |
| borrowedUs | 3.91 µs | time per weight operation the machine model took from a published wafer-scale machine | Llama 3.1 8B | modelled | results/calibration_llama.json op_total_s_us; docs/04_results.md |
| oldSpeedLow | 3 x | speed for one user against a GPU system, low end, as the earlier post gave it | simulator output | modelled | src/app/blog/inkling-2-small-on-purpose-built-silicon/data.ts (the earlier post) |
| oldSpeedHigh | 4 x | the same, high end | simulator output | modelled | src/app/blog/inkling-2-small-on-purpose-built-silicon/data.ts (the earlier post) |
| oldTpj | 4 x | tokens per joule against a GPU system, as the earlier post gave it | simulator output | modelled | src/app/blog/inkling-2-small-on-purpose-built-silicon/data.ts (the earlier post) |
| rbPs | 1,620 ps | clock constraint, float block at its smallest closing clock | same flow, corner, constraints and power workload as the other block | laid out | results/silicon.json exact_sum_variant.silicon.comparison.rel_at_best_clock.clock_period_ps |
| rbMhz | 617 MHz | the same as a frequency |  | arithmetic | results/silicon.json exact_sum_variant.silicon.comparison.rel_at_best_clock.clock_mhz |
| rbSlack | 6.7 ps | worst setup slack, float block at its smallest closing clock | after routing | laid out | results/silicon.json exact_sum_variant.silicon.comparison.rel_at_best_clock.setup_wns_ps |
| rbArea | 536.9 µm² | cell area, float block at its smallest closing clock | fill and tap cells not counted | laid out | results/silicon.json exact_sum_variant.silicon.comparison.rel_at_best_clock.cell_area_um2 |
| rbCells | 4,819 cells | cells, float block at its smallest closing clock |  | laid out | results/silicon.json exact_sum_variant.silicon.comparison.rel_at_best_clock.cells |
| rbFlops | 273 flip-flops | flip-flops, float block at its smallest closing clock |  | laid out | results/silicon.json exact_sum_variant.silicon.comparison.rel_at_best_clock.sequential_cells |
| rbPower | 1.245 mW | dynamic power, float block at its smallest closing clock | OpenSTA report_power, routed netlist, SPEF parasitics, VCD of a zero-delay gate-level simulation; real weight row (w13 row 0, layer 0, expert 0) against 8 token rows, a block on every clock | simulated | results/silicon.json exact_sum_variant.silicon.comparison.rel_at_best_clock.power_w.dynamic |
| rbEnergy | 2.02 pJ | energy per block, float block at its smallest closing clock | total power x clock period | arithmetic | results/silicon.json exact_sum_variant.silicon.comparison.rel_at_best_clock.energy_per_block_pj |
| rbDepth | 53 cells | cells on the worst setup path, float block at its smallest closing clock | counted in the flow's final timing report | laid out | results/silicon.json exact_sum_variant.silicon.comparison.rel_at_best_clock.worst_setup_path.cells_on_path |
| f18Ps | 1,800 ps | clock constraint, exact-sum block at 1,800 ps | same flow, corner, constraints and power workload as the other block | laid out | results/silicon.json exact_sum_variant.silicon.comparison.fx_at_common_clock.clock_period_ps |
| f18Mhz | 556 MHz | the same as a frequency |  | arithmetic | results/silicon.json exact_sum_variant.silicon.comparison.fx_at_common_clock.clock_mhz |
| f18Slack | 475.7 ps | worst setup slack, exact-sum block at 1,800 ps | after routing | laid out | results/silicon.json exact_sum_variant.silicon.comparison.fx_at_common_clock.setup_wns_ps |
| f18Area | 575.0 µm² | cell area, exact-sum block at 1,800 ps | fill and tap cells not counted | laid out | results/silicon.json exact_sum_variant.silicon.comparison.fx_at_common_clock.cell_area_um2 |
| f18Cells | 4,928 cells | cells, exact-sum block at 1,800 ps |  | laid out | results/silicon.json exact_sum_variant.silicon.comparison.fx_at_common_clock.cells |
| f18Flops | 414 flip-flops | flip-flops, exact-sum block at 1,800 ps |  | laid out | results/silicon.json exact_sum_variant.silicon.comparison.fx_at_common_clock.sequential_cells |
| f18Power | 1.100 mW | dynamic power, exact-sum block at 1,800 ps | OpenSTA report_power, routed netlist, SPEF parasitics, VCD of a zero-delay gate-level simulation; real weight row (w13 row 0, layer 0, expert 0) against 8 token rows, a block on every clock | simulated | results/silicon.json exact_sum_variant.silicon.comparison.fx_at_common_clock.power_w.dynamic |
| f18Energy | 1.98 pJ | energy per block, exact-sum block at 1,800 ps | total power x clock period | arithmetic | results/silicon.json exact_sum_variant.silicon.comparison.fx_at_common_clock.energy_per_block_pj |
| f18Depth | 40 cells | cells on the worst setup path, exact-sum block at 1,800 ps | counted in the flow's final timing report | laid out | results/silicon.json exact_sum_variant.silicon.comparison.fx_at_common_clock.worst_setup_path.cells_on_path |
| fbPs | 1,150 ps | clock constraint, exact-sum block at its smallest closing clock | same flow, corner, constraints and power workload as the other block | laid out | results/silicon.json exact_sum_variant.silicon.comparison.fx_at_best_clock.clock_period_ps |
| fbMhz | 870 MHz | the same as a frequency |  | arithmetic | results/silicon.json exact_sum_variant.silicon.comparison.fx_at_best_clock.clock_mhz |
| fbSlack | 1.9 ps | worst setup slack, exact-sum block at its smallest closing clock | after routing | laid out | results/silicon.json exact_sum_variant.silicon.comparison.fx_at_best_clock.setup_wns_ps |
| fbArea | 586.4 µm² | cell area, exact-sum block at its smallest closing clock | fill and tap cells not counted | laid out | results/silicon.json exact_sum_variant.silicon.comparison.fx_at_best_clock.cell_area_um2 |
| fbCells | 5,028 cells | cells, exact-sum block at its smallest closing clock |  | laid out | results/silicon.json exact_sum_variant.silicon.comparison.fx_at_best_clock.cells |
| fbFlops | 414 flip-flops | flip-flops, exact-sum block at its smallest closing clock |  | laid out | results/silicon.json exact_sum_variant.silicon.comparison.fx_at_best_clock.sequential_cells |
| fbPower | 1.792 mW | dynamic power, exact-sum block at its smallest closing clock | OpenSTA report_power, routed netlist, SPEF parasitics, VCD of a zero-delay gate-level simulation; real weight row (w13 row 0, layer 0, expert 0) against 8 token rows, a block on every clock | simulated | results/silicon.json exact_sum_variant.silicon.comparison.fx_at_best_clock.power_w.dynamic |
| fbEnergy | 2.06 pJ | energy per block, exact-sum block at its smallest closing clock | total power x clock period | arithmetic | results/silicon.json exact_sum_variant.silicon.comparison.fx_at_best_clock.energy_per_block_pj |
| fbDepth | 38 cells | cells on the worst setup path, exact-sum block at its smallest closing clock | counted in the flow's final timing report | laid out | results/silicon.json exact_sum_variant.silicon.comparison.fx_at_best_clock.worst_setup_path.cells_on_path |
| r18Depth | 56 cells | cells on the worst setup path, float block at 1,800 ps | counted in the final timing report of the rerun | laid out | results/silicon.json exact_sum_variant.silicon.comparison.rel_at_common_clock.worst_setup_path |
| fxLoopSlack | 427 ps | setup slack of the exact-sum accumulator loop at the block's smallest closing clock | worst slack into the S4 registers | laid out | results/silicon.json exact_sum_variant.silicon.comparison.fx_at_best_clock.worst_setup_slack_by_stage_ps.S4 |
| relFailPs | 1,610 ps | largest constraint the float block failed | 10 ps below its smallest closing constraint | laid out | exact_sum_variant.silicon.clock_search.rel |
| relFailBy | 0.8 ps | setup violation at that constraint | after routing | laid out | same |
| fxFailPs | 1,140 ps | largest constraint the exact-sum block failed | 10 ps below its smallest closing constraint | laid out | exact_sum_variant.silicon.clock_search.fx |
| fxFailBy | 8.4 ps | setup violation at that constraint | after routing | laid out | same |
| relSearchRuns | 7 runs | clock constraints tried for the float block | one placement seed each | laid out | exact_sum_variant.silicon.clock_search.rel.all_points |
| fxSearchRuns | 10 runs | clock constraints tried for the exact-sum block | one placement seed each | laid out | exact_sum_variant.silicon.clock_search.fx.all_points |
| relSlack1730 | 29.8 ps | setup slack of the float block at a 1,730 ps constraint | larger than at 1,800 ps: the slack is not monotonic | laid out | same |
| relPs1730 | 1,730 ps | that constraint |  | laid out | same |
| fxPeriodCutPct | 29 % | shorter clock period of the exact-sum block, each at its smallest closing clock | 1 - fbPs / rbPs | arithmetic | exact_sum_variant.silicon.ratios |
| fxRatePct | 41 % | more blocks per second, each at its smallest closing clock | rbPs / fbPs - 1 | arithmetic | same |
| fxAreaBestPct | 9 % | more cell area, each at its smallest closing clock | fbArea / rbArea - 1 | arithmetic | same |
| fxArea18Pct | 13 % | more cell area at the same 1,800 ps constraint | f18Area / cellArea - 1 | arithmetic | same |
| fxAreaTimePct | 22 % | less cell area for the same number of blocks per second | 1 - (fbArea x fbPs) / (rbArea x rbPs) | arithmetic | same |
| fxAccBits | 57 bits | width of the exact-sum accumulator | two's complement, unit 2^-20 | spec | exact_sum_variant.design; rtl/nvfp4/SPEC.md section 5 |
| fxTermBits | 48 bits | width of an aligned block term | 19 bits and sign, shifted by 0 to 28 | spec | same |
| fxShiftMax | 28 places | largest alignment shift |  | spec | same |
| fxMaxBlocks | 512 blocks | longest row the accumulator holds exactly | 512 x 2^47 = 2^56 | spec | same |
| fxLatency | 7 clocks | clocks from the last block to the result, exact-sum block |  | spec | same |
| fxStagesAfter | 2 stages | stages after the loop of the exact-sum block |  | spec | same |
| fxVectors | 1,181,020 rows | rows of the exact-sum block compared with its reference, no mismatch | Verilator; the reference sums with Python integers | simulated | exact_sum_variant.verification.vectors |
| fxIcarus | 478 rows | rows repeated in Icarus |  | simulated | exact_sum_variant.verification.icarus |
| mutRows | 11,651 rows | rows each deliberately broken copy of the RTL was run on |  | simulated | exact_sum_variant.verification.mutation |
| mutRound | 484 rows | rows wrong with round-half-up in place of ties-to-even |  | simulated | same |
| mutShift | 11,024 rows | rows wrong with the alignment shift off by one |  | simulated | same |
| defsReal | 3,538,944 outputs | outputs of the real expert on which the two definitions give the same bits (all of them) | both projections of layer 0 expert 0 against their token rows; references only | simulated | exact_sum_variant.definitions_compared |
| defsRandomPct | 13 % | share with the same bits for random codes and random finite scales | 512 rows x 64 tokens | simulated | same |
| defsRangePct | 99.5 % | share with the same bits for random codes and scale bytes in the range real weights use | 512 rows x 64 tokens | simulated | same |
| fxLaneSeconds | 0.73 s | time one exact-sum block needs for the NVFP4 work of one token | blocksToken x fbPs | arithmetic |  |
| fxEnergyToken | 1.31 mJ | energy of that work | blocksToken x fbEnergy; the block only | arithmetic |  |
| fxKBlocks | 733 blocks | exact-sum blocks busy at 1,000 tokens per second | kTok x fxLaneSeconds | arithmetic |  |
| fxKAreaMm2 | 0.43 mm² | their cell area | fxKBlocks x fbArea | arithmetic |  |
| fxKPowerW | 1.31 W | their power | kTok x fxEnergyToken | arithmetic |  |
| picUmPerPx | 0.02 µm per pixel | resolution the layout picture was saved at | OpenROAD save_image | laid out | rerun of rel:1800 on 2026-11-25 |
| picSlack | 23.3 ps | setup slack of the rerun | equals the committed run | laid out | rerun: finish__timing__setup__ws 23.3443 |
| picCells | 4,548 cells | cells of the rerun without tap cells | the flow reports 4,839 with the 291 tap cells; equals the committed run | laid out | rerun: finish__design__instance__count__stdcell |
| picFrameUm | 36.76 µm | side of the layout picture | 1,838 pixels at 0.02 µm | laid out | rerun: save_image |
| strapM6 | 0.288 µm | width of the horizontal power straps, on M6 | pitch 5.4 µm | laid out | rerun: 6_final.def SPECIALNETS |
| strapM5 | 0.12 µm | width of the vertical power straps, on M5 |  | laid out | rerun: 6_final.def SPECIALNETS |
