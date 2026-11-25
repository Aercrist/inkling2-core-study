#!/usr/bin/env python3
"""Generate the NVFP4 block-scaled dot-product engine (SystemVerilog) from the model config.

Two tops share one datapath (`nvfp4_row_dp`, in nvfp4_common.sv):

* `nvfp4_dot`        reloadable: weight codes and weight block scales arrive on ports.
* `nvfp4_dot_const*` fixed: the weights and weight block scales of real expert rows are constants
                     selected by a block counter; only activations arrive on ports.

A third top, `nvfp4_dot_fx`, has the ports of `nvfp4_dot` and its own datapath (`nvfp4_row_dp_fx`,
in nvfp4_dot_fx.sv): the same stages S0 to S2, then an exact fixed-point row sum that is rounded to
binary32 once per row (SPEC.md section 5). It is the baseline the binary32 accumulator is compared with.

Both take one block (B codes + one E4M3 scale per operand) per clock and return one binary32 per
row. Arithmetic: rtl/nvfp4/SPEC.md. Everything that depends on the model comes from
data/inkling2_small/{config.json,hf_quant_config.json}: block size, code and scale formats, row
length. A different block size or row length regenerates the RTL; widths are derived, not typed.

    python rtl/nvfp4/gen_nvfp4_dot.py common     --out DIR
    python rtl/nvfp4/gen_nvfp4_dot.py reloadable --out DIR [--rows N]
    python rtl/nvfp4/gen_nvfp4_dot.py fx         --out DIR
    python rtl/nvfp4/gen_nvfp4_dot.py const      --out DIR --name NAME --fixture NPZ --leaf w13 \
                                                 --rows 0,2,4 [--blocks 0:288]
    python rtl/nvfp4/gen_nvfp4_dot.py tb         --out DIR --top NAME --rows N [--const]
    python rtl/nvfp4/gen_nvfp4_dot.py committed  # rewrite rtl/nvfp4/generated/
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CFG_DIR = ROOT / "data" / "inkling2_small"
PRIMARY = {"fixture": "data/fixtures/inkling2_small_L00_E000.npz", "leaf": "w13", "row": 0}

E2M1_Q = (0, 1, 2, 3, 4, 6, 8, 12)   # twice the E2M1 magnitudes
E4M3_MMAX = 15                       # largest E4M3 significand, in units of 2^-3
E4M3_EMAX = 15                       # largest E4M3 exponent field
FX_ROW_LOG2 = 9                      # exact-sum variant: rows of up to 2^9 = 512 blocks
FP_MANT, FP_BIAS = 24, 127           # binary32


def model_params(cfg_dir: pathlib.Path = CFG_DIR) -> dict:
    """Read the block geometry and formats the RTL depends on, and derive every width."""
    cfg = json.loads((cfg_dir / "config.json").read_text())["text_config"]
    quant = json.loads((cfg_dir / "hf_quant_config.json").read_text())
    grp = next(g for g in quant["model"]["parameter_groups"] if g["compute_dtype"] == "nvfp4")
    act = grp["required_activation_quantization"]
    if grp["weight_transform"] != "nvfp4_nearest" or act["format"] != "nvfp4_e2m1" \
            or act["scale_dtype"] != "float8_e4m3fn" or grp["weight_scale_format"] != "amax_denominator":
        raise ValueError("this generator implements E2M1 codes with float8_e4m3fn block scales only")
    B = int(act["block_size"])
    if grp["weight_block_shape"] != [1, B]:
        raise ValueError("weight and activation block sizes differ")
    return derive(B, {"w13": cfg["hidden_size"] // B, "w2": cfg["intermediate_size"] // B})


def derive(B: int, nblk: dict) -> dict:
    qmax = E2M1_Q[-1]
    magw = (qmax * qmax).bit_length()                 # 144 -> 8 bits
    dmax = B * qmax * qmax                            # largest |block sum| in units of 1/4
    aw = dmax.bit_length()
    pmax = dmax * E4M3_MMAX * E4M3_MMAX               # |D| * Mw * Mx
    pw = pmax.bit_length()
    if pw > FP_MANT:
        raise ValueError(f"block of {B}: a block term needs {pw} bits and is no longer exact in binary32")
    return {"B": B, "MAGW": magw, "DW": aw + 1, "AW": aw, "PW": pw, "MMW": (E4M3_MMAX * E4M3_MMAX).bit_length(),
            # value of a term = P * 2^(Ew + Ex - 22); biased exponent = EXP_BASE - lz + (Ew + Ex)
            "EXP_BASE": FP_BIAS - 22 + pw - 1, "nblk": nblk}


def derive_fx(p: dict) -> dict:
    """Widths of the exact-sum variant. A term is P * 2^(Ew + Ex - 2) in units of 2^-20."""
    shmax = 2 * E4M3_EMAX - 2                         # largest alignment shift: 28
    tw = p["PW"] + shmax                              # |aligned term| < 2^tw: 47 bits
    mw = tw + FX_ROW_LOG2                             # |row sum| < 2^mw for rows of up to 512 blocks: 56 bits
    if not 32 < mw <= 64:
        raise ValueError(f"exact-sum variant: a {mw}-bit magnitude does not fit the two 32-bit leading-zero counters")
    return {"SHMAX": shmax, "TW": tw, "MW": mw, "ACCW": mw + 1, "MAX_BLOCKS": 1 << FX_ROW_LOG2,
            # biased exponent of a sum whose leading one is bit mw-1: value 2^(mw - 1 - 20)
            "EXP_TOP": FP_BIAS + mw - 1 - 20, "LATENCY": 7}


# --------------------------------------------------------------------------------------- common
def lzc32() -> str:
    lines = ["// Leading-zero count of a 32-bit word (5 bits) and an all-zero flag: a balanced tree.",
             "module nvfp4_lzc32 (input logic [31:0] x, output logic [4:0] lz, output logic zero);"]
    lines.append("  logic [15:0] v0; logic [15:0] c0;")
    lines.append("  genvar i;")
    lines.append("  generate for (i = 0; i < 16; i++) begin : l0")
    lines.append("    assign v0[i] = x[2*i+1] | x[2*i];")
    lines.append("    assign c0[i] = ~x[2*i+1];")
    lines.append("  end endgenerate")
    n, w = 8, 1
    prev_v, prev_c = "v0", "c0"
    for lvl in range(1, 5):
        v, c = f"v{lvl}", f"c{lvl}"
        lines.append(f"  logic [{n - 1}:0] {v}; logic [{n * (w + 1) - 1}:0] {c};")
        lines.append(f"  generate for (i = 0; i < {n}; i++) begin : l{lvl}")
        lines.append(f"    assign {v}[i] = {prev_v}[2*i+1] | {prev_v}[2*i];")
        lines.append(f"    assign {c}[{w + 1}*i +: {w + 1}] = {prev_v}[2*i+1] ? {{1'b0, {prev_c}[{w}*(2*i+1) +: {w}]}}"
                     f" : {{1'b1, {prev_c}[{w}*(2*i) +: {w}]}};")
        lines.append("  end endgenerate")
        prev_v, prev_c, n, w = v, c, n // 2, w + 1
    lines.append(f"  assign lz = {prev_c};")
    lines.append(f"  assign zero = ~{prev_v}[0];")
    lines.append("endmodule")
    return "\n".join(lines)


def row_ports_sv(p: dict) -> str:
    """Port list of a row datapath (the same for nvfp4_row_dp and nvfp4_row_dp_fx)."""
    B = p["B"]
    return f"""  input  logic        clk,
  input  logic        rst,
  input  logic        s0_v,
  input  logic        s0_last,
  input  logic [{4 * B - 1}:0] w_codes,
  input  logic [7:0]  w_scale,
  input  logic [{4 * B - 1}:0] x_codes,
  input  logic [7:0]  x_scale,
  output logic        out_valid,
  output logic [31:0] out_data
);
"""


def front_sv(p: dict) -> str:
    """Stages S1 and S2 of a row datapath: products, exact block sum, P = |D| * Mw * Mx.
    Shared verbatim by nvfp4_row_dp and nvfp4_row_dp_fx."""
    B, DW, AW, PW, MMW = p["B"], p["DW"], p["AW"], p["PW"], p["MMW"]
    terms = " + ".join(f"t{j}" for j in range(B)) + " + " + " + ".join(f"c{j}" for j in range(B))
    lane_wires = "\n".join(
        f"  wire signed [{DW - 1}:0] t{j} = {{{{{DW - 8}{{neg[{j}]}}}}, mag[{8 * j}+:8] ^ {{8{{neg[{j}]}}}}}};\n"
        f"  wire signed [{DW - 1}:0] c{j} = {{{DW - 1}'b0, neg[{j}]}};" for j in range(B))
    return f"""  // ---------------------------------------------------------------- S1
  logic [{8 * B - 1}:0] mag;
  logic [{B - 1}:0]  neg;
  genvar g;
  generate for (g = 0; g < {B}; g++) begin : lane
    nvfp4_e2m1_mul u_mul (.a(w_codes[4*g +: 4]), .b(x_codes[4*g +: 4]), .mag(mag[8*g +: 8]), .neg(neg[g]));
  end endgenerate
  // two's complement of a negative product = ~mag + 1; the +1 terms are the c wires
{lane_wires}
  wire signed [{DW - 1}:0] d_sum = {terms};

  logic       sw_sg, sx_sg, sw_nan, sx_nan;
  logic [3:0] sw_m, sx_m, sw_e, sx_e;
  nvfp4_e4m3_dec u_ws (.s(w_scale), .sign(sw_sg), .m(sw_m), .e(sw_e), .nan(sw_nan));
  nvfp4_e4m3_dec u_xs (.s(x_scale), .sign(sx_sg), .m(sx_m), .e(sx_e), .nan(sx_nan));

  logic signed [{DW - 1}:0] d_q;
  logic [{MMW - 1}:0] mm_q;
  logic [4:0] es_q;
  logic       sg_q, nan1_q, v1_q, last1_q;
  always_ff @(posedge clk) begin
    d_q     <= d_sum;
    mm_q    <= sw_m * sx_m;
    es_q    <= {{1'b0, sw_e}} + {{1'b0, sx_e}};
    sg_q    <= sw_sg ^ sx_sg;
    nan1_q  <= sw_nan | sx_nan;
    last1_q <= s0_last;
    v1_q    <= rst ? 1'b0 : s0_v;
  end

  // ---------------------------------------------------------------- S2
  wire [{AW - 1}:0] d_abs = d_q[{DW - 1}] ? (~d_q[{AW - 1}:0] + {AW}'d1) : d_q[{AW - 1}:0];
  wire [{PW - 1}:0] p_c = d_abs * mm_q;
  logic [{PW - 1}:0] p_q;
  logic [4:0] es2_q;
  logic       sg2_q, nan2_q, v2_q, last2_q;
  always_ff @(posedge clk) begin
    p_q     <= p_c;
    es2_q   <= es_q;
    sg2_q   <= d_q[{DW - 1}] ^ sg_q;
    nan2_q  <= nan1_q;
    last2_q <= last1_q;
    v2_q    <= rst ? 1'b0 : v1_q;
  end
"""


def common_sv(p: dict) -> str:
    B, DW, PW = p["B"], p["DW"], p["PW"]
    shifter = []
    prev = "y_ext"
    for k in range(5):
        n = 1 << k
        shifter.append(f"  wire [26:0] ysh{k} = d[{k}] ? {{{n}'b0, {prev}[26:{n}]}} : {prev};")
        shifter.append(f"  wire        yst{k} = d[{k}] & (|{prev}[{n - 1}:0]);")
        prev = f"ysh{k}"
    shifter = "\n".join(shifter)
    return f"""// GENERATED by rtl/nvfp4/gen_nvfp4_dot.py from data/inkling2_small/*.json. Do not edit.
// NVFP4 block-scaled dot product, shared datapath. Arithmetic: rtl/nvfp4/SPEC.md.
// Block size {B}; block sum {DW} bits signed; block term {PW} bits; accumulator IEEE-754 binary32.

// One E2M1 x E2M1 product. Code = {{sign, e[1:0], m}}; value = m/2 (e = 0) or (1 + m/2) * 2^(e-1).
// Output: |2a * 2b| (0..144) and the sign of the product (0 for a zero product).
module nvfp4_e2m1_mul (input logic [3:0] a, input logic [3:0] b,
                       output logic [7:0] mag, output logic neg);
  wire       nz = (|a[2:0]) & (|b[2:0]);
  wire       ma = a[0] & (|a[2:1]);            // fraction bit of a normal number
  wire       mb = b[0] & (|b[2:1]);
  wire [3:0] mm = (ma & mb) ? 4'd9 : (ma | mb) ? 4'd6 : 4'd4;   // (2 + ma) * (2 + mb)
  wire [2:0] sh = {{1'b0, a[2:1]}} + {{1'b0, b[2:1]}};
  wire [9:0] full = {{6'b0, mm}} << sh;        // 4 * |2a * 2b|
  assign mag = nz ? full[9:2] : 8'd0;
  assign neg = nz & (a[3] ^ b[3]);
endmodule

// E4M3 (float8_e4m3fn) block scale: value = (-1)^sign * m * 2^(e - 10); 0x7F and 0xFF are NaN.
module nvfp4_e4m3_dec (input logic [7:0] s, output logic sign, output logic [3:0] m,
                       output logic [3:0] e, output logic nan);
  wire nz = |s[6:3];
  assign sign = s[7];
  assign m    = {{nz, s[2:0]}};
  assign e    = nz ? s[6:3] : 4'd1;
  assign nan  = &s[6:0];
endmodule

{lzc32()}

// One output row. Stage 0 operands come from the top (registers or constants).
//   S1  {B} products, exact signed block sum D, product of the two scale significands
//   S2  P = |D| * Mw * Mx                       (exact, < 2^{PW})
//   S3  normalise P into a binary32 operand     (exact)
//   S4  acc <= RNE(acc + term), one add per block, in arrival order
module nvfp4_row_dp (
{row_ports_sv(p)}{front_sv(p)}
  // ---------------------------------------------------------------- S3
  logic [4:0] p_lz;
  logic       p_zero;
  nvfp4_lzc32 u_plz (.x({{p_q, {32 - PW}'b0}}), .lz(p_lz), .zero(p_zero));
  wire [{PW - 1}:0] p_n = p_q << p_lz;
  wire [7:0]  t_e_c = 8'd{p['EXP_BASE']} + {{3'b0, es2_q}} - {{3'b0, p_lz}};
  wire        t_live = v2_q & ~p_zero;
  logic       t_s;
  logic [7:0] t_e;
  logic [{PW - 1}:0] t_m;
  logic       t_nan, t_v, t_last;
  always_ff @(posedge clk) begin
    t_s    <= t_live ? sg2_q : 1'b0;
    t_e    <= t_live ? t_e_c : 8'd0;          // an idle cycle or a zero term adds +0
    t_m    <= t_live ? p_n : {PW}'d0;
    t_nan  <= v2_q & nan2_q;
    t_last <= last2_q;
    t_v    <= rst ? 1'b0 : v2_q;
  end

  // ---------------------------------------------------------------- S4: binary32 accumulate, RNE
  logic        acc_s_q;
  logic [7:0]  acc_e_q;
  logic [23:0] acc_m_q;                        // explicit leading one; zero is e = 0, m = 0
  logic        first_q, nan_q, inf_q, inf_s_q;

  wire        a_s = first_q ? 1'b0  : acc_s_q;
  wire [7:0]  a_e = first_q ? 8'd0  : acc_e_q;
  wire [23:0] a_m = first_q ? 24'd0 : acc_m_q;
  wire [23:0] b_m = {{t_m, {FP_MANT - PW}'b0}};
  wire        a_ge = {{a_e, a_m[22:0]}} >= {{t_e, b_m[22:0]}};
  wire        x_s = a_ge ? a_s : t_s;
  wire        y_s = a_ge ? t_s : a_s;
  wire [7:0]  x_e = a_ge ? a_e : t_e;
  wire [23:0] x_m = a_ge ? a_m : b_m;
  wire [23:0] y_m = a_ge ? b_m : a_m;
  wire [7:0]  d   = a_ge ? (a_e - t_e) : (t_e - a_e);
  // align the smaller operand: 24 bits + guard + round + sticky
  wire [26:0] y_ext = {{y_m, 3'b000}};
{shifter}
  wire        d_big  = |d[7:5];
  wire        sticky = d_big ? (|y_m) : (yst0 | yst1 | yst2 | yst3 | yst4);
  wire [26:0] y_al   = d_big ? {{26'b0, sticky}} : {{ysh4[26:1], ysh4[0] | sticky}};
  wire        sub = x_s ^ y_s;
  wire [27:0] sum = {{1'b0, x_m, 3'b000}} + ({{1'b0, y_al}} ^ {{28{{sub}}}}) + {{27'b0, sub}};
  // normalise
  logic [4:0] s_lz;
  logic       s_zero;
  nvfp4_lzc32 u_slz (.x({{sum[26:0], 5'b0}}), .lz(s_lz), .zero(s_zero));
  wire        carry  = sum[27] & ~sub;
  wire        r_zero = s_zero & ~carry;
  wire [26:0] n_c = carry ? {{sum[27:2], sum[1] | sum[0]}} : (sum[26:0] << s_lz);
  wire [8:0]  e_n = carry ? ({{1'b0, x_e}} + 9'd1) : ({{1'b0, x_e}} - {{4'b0, s_lz}});
  // round to nearest, ties to even
  wire        inc = n_c[2] & (n_c[1] | n_c[0] | n_c[3]);
  wire [24:0] r_c = {{1'b0, n_c[26:3]}} + {{24'b0, inc}};
  wire [8:0]  e_r = e_n + {{8'b0, r_c[24]}};
  wire        ovf = ~r_zero & (e_r >= 9'd255);
  wire        inf_prev = first_q ? 1'b0 : inf_q;

  always_ff @(posedge clk) begin
    acc_s_q <= r_zero ? 1'b0 : x_s;
    acc_e_q <= r_zero ? 8'd0 : e_r[7:0];
    acc_m_q <= r_zero ? 24'd0 : {{1'b1, r_c[22:0]}};
    if (t_v) begin
      nan_q   <= (first_q ? 1'b0 : nan_q) | t_nan;
      inf_q   <= inf_prev | ovf;
      if (!inf_prev) inf_s_q <= x_s;
    end
    if (rst) begin
      first_q   <= 1'b1;
      out_valid <= 1'b0;
    end else begin
      if (t_v) first_q <= t_last;
      out_valid <= t_v & t_last;
    end
  end

  assign out_data = nan_q ? 32'h7FC00000
                  : inf_q ? {{inf_s_q, 8'hFF, 23'h0}}
                  : {{acc_s_q, acc_e_q, acc_m_q[22:0]}};
endmodule
"""


# --------------------------------------------------------------------------------------- tops
def reloadable_sv(p: dict, rows: int = 1, name: str = "nvfp4_dot", dp: str = "nvfp4_row_dp", latency: int = 5,
                  title: str = "Reloadable NVFP4 dot-product engine") -> str:
    """The top with weights on ports: input registers (S0) and one row datapath `dp` per row."""
    B = p["B"]
    cw = 4 * B
    return f"""// GENERATED by rtl/nvfp4/gen_nvfp4_dot.py. Do not edit.
// {title}: {rows} output row(s) sharing one activation stream.
// One block per clock: {B} weight codes + weight block scale per row, {B} activation codes +
// activation block scale. Element j of a block is bits [4j+3:4j], i.e. the packed bytes of the
// checkpoint loaded little-endian. `in_last` marks the last block of a row; the next row may
// start on the next clock. `out_valid` pulses {latency} clocks after the last block; `out_data` (one
// binary32 per row, row r in bits [32r+31:32r]) is valid in that cycle only.
// Synchronous reset, held for at least 3 clocks; only control state is reset.
module {name} (
  input  logic        clk,
  input  logic        rst,
  input  logic        in_valid,
  input  logic        in_last,
  input  logic [{rows * cw - 1}:0] w_codes,
  input  logic [{rows * 8 - 1}:0] w_scale,
  input  logic [{cw - 1}:0] x_codes,
  input  logic [7:0]  x_scale,
  output logic        out_valid,
  output logic [{rows * 32 - 1}:0] out_data
);
  logic        v0_q, last0_q;
  logic [{rows * cw - 1}:0] wc_q;
  logic [{rows * 8 - 1}:0] ws_q;
  logic [{cw - 1}:0] xc_q;
  logic [7:0]  xs_q;
  always_ff @(posedge clk) begin
    wc_q    <= w_codes;
    ws_q    <= w_scale;
    xc_q    <= x_codes;
    xs_q    <= x_scale;
    last0_q <= in_last;
    v0_q    <= rst ? 1'b0 : in_valid;
  end
  logic [{rows - 1}:0] ov;
  genvar r;
  generate for (r = 0; r < {rows}; r++) begin : g_row
    {dp} u_dp (.clk(clk), .rst(rst), .s0_v(v0_q), .s0_last(last0_q),
                       .w_codes(wc_q[{cw}*r +: {cw}]), .w_scale(ws_q[8*r +: 8]),
                       .x_codes(xc_q), .x_scale(xs_q),
                       .out_valid(ov[r]), .out_data(out_data[32*r +: 32]));
  end endgenerate
  assign out_valid = ov[0];
endmodule
"""


def fx_sv(p: dict, name: str = "nvfp4_dot_fx") -> str:
    """The exact-sum variant: `nvfp4_row_dp_fx` and its top. Stages S0 to S2 are the generated
    text of the binary32 block (`reloadable_sv`, `front_sv`); S3 aligns and signs the term, S4 is
    one two's-complement add per block, S5 and S6 round the row sum to binary32 once per row."""
    B, PW = p["B"], p["PW"]
    q = derive_fx(p)
    TW, MW, ACCW = q["TW"], q["MW"], q["ACCW"]
    dp = f"""// GENERATED by rtl/nvfp4/gen_nvfp4_dot.py from data/inkling2_small/*.json. Do not edit.
// NVFP4 block-scaled dot product, exact-sum variant. Arithmetic: rtl/nvfp4/SPEC.md, section 5.
// The block terms of nvfp4_row_dp, summed exactly in fixed point and rounded to binary32 once per
// row. Needs nvfp4_common.sv (nvfp4_e2m1_mul, nvfp4_e4m3_dec, nvfp4_lzc32).
// Block size {B}; block term {PW} bits; aligned term {TW + 1} bits signed, unit 2^-20; accumulator
// {ACCW} bits two's complement, exact for rows of up to {q['MAX_BLOCKS']} blocks.

// One output row. Stage 0 operands come from the top.
//   S1  {B} products, exact signed block sum D, product of the two scale significands
//   S2  P = |D| * Mw * Mx                       (exact, < 2^{PW})
//   S3  term = (-1)^sign * P * 2^(Ew + Ex - 2)   (exact, two's complement)
//   S4  acc <= acc + term, one add per block     (exact; the only loop)
//   S5  |acc| and its leading-zero count         (once per row)
//   S6  normalise, round to nearest even, pack   (once per row)
module nvfp4_row_dp_fx (
{row_ports_sv(p)}{front_sv(p)}
  // ---------------------------------------------------------------- S3: align and sign
  wire [4:0]  sh_c  = es2_q - 5'd2;                 // Ew + Ex - 2, 0..{q['SHMAX']}
  wire [{PW}:0] p_sgn = sg2_q ? (~{{1'b0, p_q}} + {PW + 1}'d1) : {{1'b0, p_q}};
  wire [{TW}:0] t_c   = {{{{{TW - PW}{{p_sgn[{PW}]}}}}, p_sgn}} << sh_c;
  logic [{TW}:0] t_q;                               // two's complement, unit 2^-20
  logic       t_nan, t_v, t_last;
  always_ff @(posedge clk) begin
    t_q    <= v2_q ? t_c : {TW + 1}'d0;             // an idle cycle adds 0
    t_nan  <= v2_q & nan2_q;
    t_last <= last2_q;
    t_v    <= rst ? 1'b0 : v2_q;
  end

  // ---------------------------------------------------------------- S4: exact accumulate
  logic [{ACCW - 1}:0] acc_q;                       // |row sum| < 2^{MW} for rows of up to {q['MAX_BLOCKS']} blocks
  logic        first_q, nan_q, fin_q;
  wire [{ACCW - 1}:0] a_c   = first_q ? {ACCW}'d0 : acc_q;
  wire [{ACCW - 1}:0] sum_c = a_c + {{{{{ACCW - TW - 1}{{t_q[{TW}]}}}}, t_q}};
  always_ff @(posedge clk) begin
    acc_q <= sum_c;
    if (t_v) nan_q <= (first_q ? 1'b0 : nan_q) | t_nan;
    if (rst) begin
      first_q <= 1'b1;
      fin_q   <= 1'b0;
    end else begin
      if (t_v) first_q <= t_last;
      fin_q   <= t_v & t_last;                      // acc_q holds the row sum for this one clock
    end
  end

  // ---------------------------------------------------------------- S5: sign and magnitude, leading zeros
  wire        s_c = acc_q[{ACCW - 1}];
  wire [{MW - 1}:0] m_c = s_c ? (~acc_q[{MW - 1}:0] + {MW}'d1) : acc_q[{MW - 1}:0];
  wire [63:0] m_pad = {{m_c, {64 - MW}'b0}};
  logic [4:0] lz_hi, lz_lo;
  logic       z_hi, z_lo;
  nvfp4_lzc32 u_lzh (.x(m_pad[63:32]), .lz(lz_hi), .zero(z_hi));
  nvfp4_lzc32 u_lzl (.x(m_pad[31:0]),  .lz(lz_lo), .zero(z_lo));
  logic [{MW - 1}:0] m5_q;
  logic [5:0] lz5_q;
  logic       s5_q, z5_q, nan5_q, v5_q;
  always_ff @(posedge clk) begin
    if (fin_q) begin
      m5_q   <= m_c;
      lz5_q  <= z_hi ? {{1'b1, lz_lo}} : {{1'b0, lz_hi}};
      s5_q   <= s_c;
      z5_q   <= z_hi & z_lo;
      nan5_q <= nan_q;
    end
    v5_q <= rst ? 1'b0 : fin_q;
  end

  // ---------------------------------------------------------------- S6: normalise, round to nearest even
  wire [{MW - 1}:0] n_c = m5_q << lz5_q;                 // leading one at bit {MW - 1}
  wire        inc_c = n_c[{MW - 25}] & (n_c[{MW - 24}] | (|n_c[{MW - 26}:0]));
  wire [24:0] r_c = {{1'b0, n_c[{MW - 1}:{MW - 24}]}} + {{24'b0, inc_c}};
  wire [7:0]  e_c = 8'd{q['EXP_TOP']} - {{2'b0, lz5_q}} + {{7'b0, r_c[24]}};
  always_ff @(posedge clk) begin
    out_data  <= nan5_q ? 32'h7FC00000 : z5_q ? 32'h00000000 : {{s5_q, e_c, r_c[22:0]}};
    out_valid <= rst ? 1'b0 : v5_q;
  end
endmodule

"""
    top = reloadable_sv(p, 1, name, dp="nvfp4_row_dp_fx", latency=q["LATENCY"],
                        title="Reloadable NVFP4 dot-product engine, exact-sum variant")
    return dp + top.split("\n", 1)[1]        # one "GENERATED" line per file


def const_sv(p: dict, name: str, codes: np.ndarray, scales: np.ndarray, origin: str, rom_style: str = "case") -> str:
    """codes: [rows, NB, B] 4-bit codes; scales: [rows, NB] E4M3 bytes.
    rom_style "case": one case statement per row (Yosys turns it into a table with registered
    outputs); "index": a constant vector indexed by the block counter (a multiplexer tree)."""
    B = p["B"]
    cw = 4 * B
    rows, nb = scales.shape
    cb = max(1, (nb - 1).bit_length())
    roms = []
    for r in range(rows):
        ent = []
        for b in range(nb):
            word = 0
            for j in range(B):
                word |= int(codes[r, b, j]) << (4 * j)
            ent.append(f"      {cb}'d{b}: begin wc{r} = {cw}'h{word:0{cw // 4}x}; ws{r} = 8'h{int(scales[r, b]):02x}; end")
        if nb == 1:
            roms.append(f"  wire [{cw - 1}:0] wc{r} = {cw}'h{word:0{cw // 4}x};\n  wire [7:0] ws{r} = 8'h{int(scales[r, 0]):02x};")
        elif rom_style == "index":
            big = 0
            for bb in range(nb):
                wd = int(scales[r, bb])
                for j in range(B):
                    wd |= int(codes[r, bb, j]) << (8 + 4 * j)
                big |= wd << (bb * (cw + 8))
            tw = nb * (cw + 8)
            roms.append(f"  wire [{tw - 1}:0] rom{r} = {tw}'h{big:x};\n"
                        f"  wire [{cw + 7}:0] ent{r} = rom{r}[blk_q * {cw + 8} +: {cw + 8}];\n"
                        f"  wire [{cw - 1}:0] wc{r} = ent{r}[{cw + 7}:8];\n  wire [7:0] ws{r} = ent{r}[7:0];")
        else:
            default = "" if nb == (1 << cb) else f"      default: begin wc{r} = {cw}'bx; ws{r} = 8'bx; end\n"
            roms.append(f"  logic [{cw - 1}:0] wc{r};\n  logic [7:0] ws{r};\n  always_comb begin\n    case (blk_q)\n"
                        + "\n".join(ent) + "\n" + default + "    endcase\n  end")
        roms.append(f"  nvfp4_row_dp u_dp{r} (.clk(clk), .rst(rst), .s0_v(v0_q), .s0_last(last0_q),\n"
                    f"                      .w_codes(wc{r}), .w_scale(ws{r}), .x_codes(xc_q), .x_scale(xs_q),\n"
                    f"                      .out_valid(ov[{r}]), .out_data(out_data[{32 * r} +: 32]));")
    if nb == 1:
        counter = "  wire last_c = 1'b1;"
        cnt_ff = ""
        blk_decl = ""
    else:
        counter = f"  logic [{cb - 1}:0] cnt_q, blk_q;\n  wire last_c = (cnt_q == {cb}'d{nb - 1});"
        cnt_ff = (f"    blk_q   <= cnt_q;\n"
                  f"    if (rst) cnt_q <= {cb}'d0;\n"
                  f"    else if (in_valid) cnt_q <= last_c ? {cb}'d0 : cnt_q + {cb}'d1;\n")
        blk_decl = ""
    return f"""// GENERATED by rtl/nvfp4/gen_nvfp4_dot.py. Do not edit.
// Fixed-weight NVFP4 dot-product engine: {rows} row(s) x {nb} block(s) of {B} weights hard-wired.
// Weights: {origin}
// Same function, timing and throughput as nvfp4_dot (one block per clock, out_valid 5 clocks
// after the last block); the block index is an internal counter, so a row is always {nb} blocks.
module {name} (
  input  logic        clk,
  input  logic        rst,
  input  logic        in_valid,
  input  logic [{cw - 1}:0] x_codes,
  input  logic [7:0]  x_scale,
  output logic        out_valid,
  output logic [{rows * 32 - 1}:0] out_data
);
{counter}{blk_decl}
  logic        v0_q, last0_q;
  logic [{cw - 1}:0] xc_q;
  logic [7:0]  xs_q;
  always_ff @(posedge clk) begin
    xc_q    <= x_codes;
    xs_q    <= x_scale;
    last0_q <= last_c;
    v0_q    <= rst ? 1'b0 : in_valid;
{cnt_ff}  end
  logic [{rows - 1}:0] ov;
{chr(10).join(roms)}
  assign out_valid = ov[0];
endmodule
"""


def tb_sv(p: dict, top: str, rows: int, const: bool, poke: bool = True) -> str:
    """File-driven testbench (Icarus). One hex word per clock:
    bit 0 valid, bit 1 last, [9:2] x_scale, then x_codes, then per row w_scale, w_codes."""
    cw = 4 * p["B"]
    sw = 10 + cw + (0 if const else rows * (8 + cw))
    wports = "" if const else (
        "".join(f"  assign w_scale[{8 * r} +: 8] = cur[{10 + cw + r * (8 + cw)} +: 8];\n"
                f"  assign w_codes[{cw * r} +: {cw}] = cur[{10 + cw + r * (8 + cw) + 8} +: {cw}];\n" for r in range(rows)))
    wdecl = "" if const else f"  wire [{rows * cw - 1}:0] w_codes;\n  wire [{rows * 8 - 1}:0] w_scale;\n"
    wconn = "" if const else ".in_last(cur[1]), .w_codes(w_codes), .w_scale(w_scale), "
    poke = "" if const or not poke else f"""`ifdef NVFP4_POKE
  // Test hook: binary32 overflow cannot be reached from the ports (terms are below 2^27), so at
  // clock +poke_cycle the accumulator of row 0 is overwritten with +poke_val and, if +poke_term
  // is given, one term with that binary32 value is placed in the S3 register for one clock.
  integer poke_cycle; reg [31:0] poke_val, poke_term; reg has_term;
  initial begin
    if (!$value$plusargs("poke_cycle=%d", poke_cycle)) poke_cycle = -1;
    if (!$value$plusargs("poke_val=%h", poke_val)) poke_val = 0;
    has_term = $value$plusargs("poke_term=%h", poke_term);
  end
  always @(negedge clk) if (cyc == poke_cycle) begin
    dut.g_row[0].u_dp.acc_s_q = poke_val[31];
    dut.g_row[0].u_dp.acc_e_q = poke_val[30:23];
    dut.g_row[0].u_dp.acc_m_q = {{|poke_val[30:23], poke_val[22:0]}};
    if (has_term) begin
      dut.g_row[0].u_dp.t_s = poke_term[31];
      dut.g_row[0].u_dp.t_e = poke_term[30:23];
      dut.g_row[0].u_dp.t_m = {{1'b1, poke_term[22:{23 - (p['PW'] - 1)}]}};
      dut.g_row[0].u_dp.t_v = 1'b1;
      dut.g_row[0].u_dp.t_nan = 1'b0;
      dut.g_row[0].u_dp.t_last = 1'b0;
    end
  end
`endif
"""
    return f"""// GENERATED by rtl/nvfp4/gen_nvfp4_dot.py. Do not edit.
// File-driven testbench for {top}: +stim=<hex, one word per clock> +n=<clocks> +out=<results>
// [+vcd=<file>]. Inputs change on the falling edge; each out_valid writes one hex line.
`timescale 1ps/1ps
module tb;
  parameter integer MAXN = 1 << 20;
  parameter integer HALF = 500;
  reg clk = 0, rst = 1;
  reg [{sw - 1}:0] stim [0:MAXN-1];
  reg [{sw - 1}:0] cur = 0;
  integer n, cyc, fo;
  reg [1023:0] stim_file, out_file, vcd_file;
  wire out_valid;
  wire [{rows * 32 - 1}:0] out_data;
{wdecl}{wports}
  {top} dut (.clk(clk), .rst(rst), .in_valid(cur[0]), {wconn}
             .x_codes(cur[10 +: {cw}]), .x_scale(cur[9:2]), .out_valid(out_valid), .out_data(out_data));
  always #(HALF) clk = ~clk;
{poke}
  initial begin
    if (!$value$plusargs("stim=%s", stim_file)) $fatal(1, "no +stim");
    if (!$value$plusargs("out=%s", out_file)) $fatal(1, "no +out");
    if (!$value$plusargs("n=%d", n)) $fatal(1, "no +n");
    $readmemh(stim_file, stim, 0, n - 1);
    fo = $fopen(out_file, "w");
    cyc = -1;
    repeat (4) @(negedge clk);
    rst = 0;
    if ($value$plusargs("vcd=%s", vcd_file)) begin
      $dumpfile(vcd_file);
      $dumpvars(0, tb.dut);
    end
    for (cyc = 0; cyc < n; cyc = cyc + 1) begin
      cur = stim[cyc];
      @(negedge clk);
    end
    cur = 0;
    repeat (12) @(negedge clk);
    $fclose(fo);
    $finish;
  end
  always @(posedge clk) if (!rst && out_valid === 1'b1) $fwrite(fo, "%h\\n", out_data);
endmodule
"""


# --------------------------------------------------------------------------------------- fixtures
def fixture_rows(fixture: str, leaf: str, rows: list[int], blocks: tuple[int, int] | None, p: dict,
                 prefix: str = ""):
    """Return (codes [R, NB, B], scales [R, NB], description) for rows of a fixture file."""
    z = np.load(ROOT / fixture if not pathlib.Path(fixture).is_absolute() else fixture)
    w, s = z[f"{prefix}{leaf}"], z[f"{prefix}{leaf}_scale"]
    B = p["B"]
    b0, b1 = blocks if blocks else (0, s.shape[1])
    codes = np.stack((w[rows] & 15, w[rows] >> 4), -1).reshape(len(rows), -1, B)[:, b0:b1]
    return codes, s[rows][:, b0:b1], (f"{pathlib.Path(fixture).name} {prefix}{leaf} rows {rows} blocks {b0}:{b1} "
                                      f"(thinkingmachines/Inkling-2-Small)")


def write_committed() -> list[pathlib.Path]:
    p = model_params()
    out = HERE / "generated"
    out.mkdir(exist_ok=True)
    codes, scales, origin = fixture_rows(PRIMARY["fixture"], PRIMARY["leaf"], [PRIMARY["row"]], None, p)
    files = {
        "nvfp4_common.sv": common_sv(p),
        "nvfp4_dot.sv": reloadable_sv(p),
        "nvfp4_dot_const.sv": const_sv(p, "nvfp4_dot_const", codes, scales, origin),
        "tb_nvfp4_dot.sv": tb_sv(p, "nvfp4_dot", 1, False),
        "tb_nvfp4_dot_const.sv": tb_sv(p, "nvfp4_dot_const", 1, True),
        "nvfp4_dot_fx.sv": fx_sv(p),
        "tb_nvfp4_dot_fx.sv": tb_sv(p, "nvfp4_dot_fx", 1, False, poke=False),
    }
    for name, text in files.items():
        (out / name).write_text(text)
    return [out / n for n in files]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["common", "reloadable", "fx", "const", "tb", "committed", "params"])
    ap.add_argument("--out")
    ap.add_argument("--name")
    ap.add_argument("--top")
    ap.add_argument("--rows", default="1")
    ap.add_argument("--fixture", default=PRIMARY["fixture"])
    ap.add_argument("--leaf", default="w13")
    ap.add_argument("--prefix", default="", help="array-name prefix inside the fixture, e.g. L13_E005_")
    ap.add_argument("--blocks", help="b0:b1, a slice of the row's blocks")
    ap.add_argument("--const", action="store_true")
    ap.add_argument("--block-size", type=int, help="override the config (to show the generator is generic)")
    a = ap.parse_args()
    p = model_params()
    if a.block_size:
        p = derive(a.block_size, p["nblk"])
    if a.what == "params":
        print(json.dumps(p, indent=1))
        return
    if a.what == "committed":
        for f in write_committed():
            print(f)
        return
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.what == "common":
        (out / "nvfp4_common.sv").write_text(common_sv(p))
    elif a.what == "reloadable":
        name = a.name or "nvfp4_dot"
        (out / f"{name}.sv").write_text(reloadable_sv(p, int(a.rows), name))
    elif a.what == "fx":
        name = a.name or "nvfp4_dot_fx"
        (out / f"{name}.sv").write_text(fx_sv(p, name))
    elif a.what == "const":
        rows = [int(r) for r in a.rows.split(",")]
        blocks = tuple(int(v) for v in a.blocks.split(":")) if a.blocks else None
        codes, scales, origin = fixture_rows(a.fixture, a.leaf, rows, blocks, p, a.prefix)
        (out / f"{a.name}.sv").write_text(const_sv(p, a.name, codes, scales, origin))
    elif a.what == "tb":
        (out / f"tb_{a.top}.sv").write_text(tb_sv(p, a.top, int(a.rows), a.const, poke="_fx" not in a.top))


if __name__ == "__main__":
    main()
