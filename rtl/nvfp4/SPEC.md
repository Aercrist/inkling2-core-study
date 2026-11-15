# NVFP4 block-scaled dot product: arithmetic and interface

This is the contract between the RTL (`gen_nvfp4_dot.py`), the NumPy reference
(`ref/nvfp4_ref.py`) and the testbenches. Where the formats come from, and how each was checked
against the checkpoint and against a serving engine, is in [REPORT.md](REPORT.md).

## 1. Operands

One **block** is `B = 16` values that share one scale. A **row** is a sequence of blocks
(288 for `w13`, whose input is the 4,608-wide hidden state; 144 for `w2`).

| operand | bits | encoding |
|---|---|---|
| value code (E2M1) | 4 | `{sign, e[1:0], m}`. `e = 0`: magnitude `m/2` (0 or 0.5). `e >= 1`: `(1 + m/2) * 2^(e-1)`. Magnitudes by `code[2:0]`: 0, 0.5, 1, 1.5, 2, 3, 4, 6. Codes `0x0` and `0x8` are +0 and -0; both are zero. |
| block scale (E4M3, `float8_e4m3fn`) | 8 | `{sign, e[3:0], m[2:0]}`. `e = 0`: `m * 2^-9` (zero and the seven subnormals). `e >= 1`: `(8 + m) * 2^(e-10)`. `0x7F` and `0xFF` are NaN. There is no infinity. Largest finite value `0x7E` = 448; smallest non-zero `0x01` = 2^-9. |

On the ports, element `j` of a block is bits `[4j+3 : 4j]` of the 64-bit code word. That is the
checkpoint's packed bytes read little-endian: byte `k` holds element `2k` in its low nibble and
element `2k+1` in its high nibble.

The second-level scales (per expert for weights, per token for activations) are not part of this
block. They multiply the result once per output: `y = out * (amax_token * amax_expert) /
(1536 * outer_scale_denominator)`.

## 2. Result

With `q(c) = 2 * value(c)` (an integer in {0, +-1, +-2, +-3, +-4, +-6, +-8, +-12}) and a scale
written as `(-1)^s * M * 2^(E-10)` (`M` in 0..15, `E` in 1..15):

```
D_b = sum_{j=0..15} q(w_bj) * q(x_bj)                        exact integer, |D_b| <= 2304
T_b = (-1)^(sw_b xor sx_b) * D_b * Mw_b * Mx_b * 2^(Ew_b + Ex_b - 22)     exact real number
A_0 = +0
A_(b+1) = RN_even( A_b + T_b )      IEEE-754 binary32, round to nearest, ties to even
out = A_NB
```

* `T_b` needs no rounding: `|D * Mw * Mx| <= 2304 * 15 * 15 < 2^19` and its exponent lies in
  [-20, 8], so every block term is a binary32 number (zero, or normal with magnitude between
  2^-20 and 1.33e8). The **only** rounding is one binary32 addition per block.
* **Order.** Blocks are added in arrival order, `b = 0, 1, 2, ...`. Nothing else is reordered:
  the 16 products of a block are summed exactly, so their order cannot matter.
* **Zero.** A term is zero when `D_b = 0` or either scale is `0x00`/`0x80`. Adding it leaves the
  accumulator unchanged. A sum that cancels exactly is `+0`. The result is never `-0`.
* **Subnormal scales** (`0x01..0x07`, `0x81..0x87`) are exact: `M = m`, `E = 1`.
* **Negative scales** negate the term. (Real checkpoints do not use them; the format allows them.)
* **NaN.** If any block of the row has a NaN scale (`0x7F` or `0xFF`, weight or activation), the
  result is the canonical quiet NaN `0x7FC00000`, whatever the codes are (as `0 * NaN` is NaN).
* **binary32 subnormals** cannot occur: every term is a multiple of 2^-20, so every partial sum
  is zero or at least 2^-20 in magnitude.
* **Overflow.** IEEE-754: if a rounded sum exceeds the largest finite binary32, the accumulator
  becomes infinity with the sign of that sum and stays there for the rest of the row
  (`inf + finite = inf`); the result is `0x7F800000` or `0xFF800000`. A NaN scale later in the row
  still gives NaN. `inf - inf` cannot occur because terms are finite. Overflow cannot be reached
  from the ports: the largest term is 1.33e8, so a row would need more than 2^100 blocks. The RTL
  implements it; the testbench reaches it by overwriting the accumulator register and
  injecting one large term in front of the adder.

Equivalent C: `float acc = 0; for (b...) acc += (float)D[b] * sw[b] * sx[b];` with `float`
products evaluated exactly (they are) and no fused or reordered operations.

## 3. Interface and timing

```
nvfp4_dot        (clk, rst, in_valid, in_last, w_codes[64R], w_scale[8R], x_codes[64], x_scale[8],
                  out_valid, out_data[32R])
nvfp4_dot_const* (clk, rst, in_valid,                                  x_codes[64], x_scale[8],
                  out_valid, out_data[32R])
```

* One block per clock while `in_valid` is high; no back-pressure. Idle clocks may appear anywhere
  and their data pins are ignored.
* `nvfp4_dot`: `in_last` marks the last block of a row; rows may be any length >= 1 and the next
  row may start on the next clock. `R` rows (generator option, default 1) share the activation
  stream and each has its own weight stream.
* `nvfp4_dot_const*`: the weights and weight block scales of `R` rows are constants selected by an
  internal block counter, so a row is always the hard-wired number of blocks.
* `out_valid` is high for one clock, 5 clocks after the last block was on the pins; `out_data`
  (row `r` in bits `[32r+31 : 32r]`) is valid in that clock only.
* `rst` is synchronous, active high, held for at least 3 clocks. Only control state is reset.

## 4. Micro-architecture

Five register stages, one block per clock in both variants:

| stage | work | registers (R = 1) |
|---|---|---|
| S0 | input registers. Reloadable: weights and activations. Fixed: activations and the block index; the weights are a function of the block index | 146 / 92 |
| S1 | 16 E2M1 x E2M1 products, signed adder tree to `D` (13 bits); decode both scales; `Mw * Mx` (8 bits); `Ew + Ex` | 30 |
| S2 | `|D|`, then `P = |D| * Mw*Mx` (19 bits) | 28 |
| S3 | leading-zero count and left shift of `P`: a normalised binary32 operand | 31 |
| S4 | accumulator: magnitude compare and swap, alignment shift with sticky, 28-bit add/subtract, leading-zero count and normalisation, round to nearest even; side flags for NaN and infinity | 38 |

The accumulation is a recurrence (each sum is rounded before the next term is added), so S4 must
finish in one clock; it is the critical path of both variants and sets the clock. The stages before
it are feed-forward. Throughput is one block (16 multiply-accumulates) per clock for every row in
the module, whatever the variant; a 4,608-input row takes 288 clocks.

The generator derives every width from the block size (`gen_nvfp4_dot.py params`): for `B = 16`
the block sum is 13 bits and the block term 19 bits. It refuses a block size whose term would not
fit the 24-bit significand, because the term would stop being exact.
