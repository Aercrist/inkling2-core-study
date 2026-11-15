// MoE top-k router for DeepSeek-V4.1-Flash (384 routed experts, top-6, noaux_tc bias).
//
// Selection key = sqrtsoftplus(gate logit) + correction bias, computed upstream in the gate
// epilogue and delivered here as a signed fixed-point key. The router picks the K largest
// keys (ties: lower expert id wins) and emits the selected expert ids in key order, so the
// expert-fetch DMA can be issued as soon as the last gate column leaves the MAC array.
//
// Microarchitecture: LANES parallel insertion queues (one K-entry sorted register each)
// absorb LANES candidates per cycle; a merge phase streams the LANES*K survivors into a final
// K-entry queue. Latency = E/LANES fill beats + LANES*K merge beats + a few cycles of control;
// for E=384, K=6, LANES=8 that is 48 + 48 + 4 = ~100 cycles (~67 ns at 1.5 GHz).
// Area is O(LANES*K) key comparators, negligible next to a GEMV array.
//
// Packed vectors are used on ports so the block simulates in Icarus Verilog 12.
// This is a sketch for latency/area estimation, verified against a numpy golden model
// (rtl/tb_moe_topk_router.sv, rtl/check_router.py). It is not a production block.

module topk_queue #(
    parameter int K = 6,
    parameter int KW = 16,  // signed key width
    parameter int IW = 9    // expert-id width
) (
    input  logic                 clk,
    input  logic                 clear,
    input  logic                 in_valid,
    input  logic signed [KW-1:0] in_key,
    input  logic        [IW-1:0] in_idx,
    output logic      [K*KW-1:0] key_o,
    output logic      [K*IW-1:0] idx_o,
    output logic         [K-1:0] vld_o
);
  logic signed [KW-1:0] key [K];
  logic        [IW-1:0] idx [K];
  logic                 vld [K];
  logic                 gt  [K];

  // the new entry beats slot i if the slot is empty, its key is larger, or keys tie and
  // its id is smaller
  always_comb begin
    for (int i = 0; i < K; i++)
      gt[i] = !vld[i] || (in_key > key[i]) || ((in_key == key[i]) && (in_idx < idx[i]));
  end

  always_ff @(posedge clk) begin
    if (clear) begin
      for (int i = 0; i < K; i++) vld[i] <= 1'b0;
    end else if (in_valid) begin
      for (int i = 0; i < K; i++) begin
        if (gt[i]) begin
          if (i == 0) begin
            key[i] <= in_key; idx[i] <= in_idx; vld[i] <= 1'b1;
          end else if (!gt[i-1]) begin  // insertion point
            key[i] <= in_key; idx[i] <= in_idx; vld[i] <= 1'b1;
          end else begin  // shift down
            key[i] <= key[i-1]; idx[i] <= idx[i-1]; vld[i] <= vld[i-1];
          end
        end
      end
    end
  end

  always_comb begin
    for (int i = 0; i < K; i++) begin
      key_o[i*KW +: KW] = key[i];
      idx_o[i*IW +: IW] = idx[i];
      vld_o[i] = vld[i];
    end
  end
endmodule

module moe_topk_router #(
    parameter int E = 384,
    parameter int K = 6,
    parameter int LANES = 8,
    parameter int KW = 16,
    parameter int IW = 9
) (
    input  logic                  clk,
    input  logic                  rst,
    input  logic                  start,     // pulse: a new token's keys follow
    input  logic                  in_valid,  // LANES keys this cycle
    input  logic [LANES*KW-1:0]   in_keys,
    output logic                  done,      // pulse: out_* valid
    output logic [K*IW-1:0]       out_idx,
    output logic [K*KW-1:0]       out_key
);
  localparam int BEATS = E / LANES;
  localparam int MB = LANES * K;
  localparam logic [1:0] IDLE = 2'd0, FILL = 2'd1, MERGE = 2'd2, FLUSH = 2'd3;
  logic [1:0] st;
  logic [15:0] beat, m;

  logic [K*KW-1:0] lk [LANES];
  logic [K*IW-1:0] li [LANES];
  logic [K-1:0]    lv [LANES];
  logic [K*KW-1:0] fk;
  logic [K*IW-1:0] fi;
  logic [K-1:0]    fv;

  genvar g;
  generate
    for (g = 0; g < LANES; g++) begin : lane
      logic [K*KW-1:0] k_w;
      logic [K*IW-1:0] i_w;
      logic [K-1:0]    v_w;
      topk_queue #(.K(K), .KW(KW), .IW(IW)) q (
          .clk(clk), .clear(rst || start), .in_valid(in_valid && st == FILL),
          .in_key(in_keys[g*KW +: KW]), .in_idx(IW'(beat * LANES + g)),
          .key_o(k_w), .idx_o(i_w), .vld_o(v_w));
      assign lk[g] = k_w;
      assign li[g] = i_w;
      assign lv[g] = v_w;
    end
  endgenerate

  // merge: stream lane entries (lane-major) into the final queue
  logic [15:0] ml, me;
  assign ml = m / K;
  assign me = m % K;
  logic [K*KW-1:0] cur_k;
  logic [K*IW-1:0] cur_i;
  logic [K-1:0]    cur_v;
  assign cur_k = lk[ml];
  assign cur_i = li[ml];
  assign cur_v = lv[ml];

  topk_queue #(.K(K), .KW(KW), .IW(IW)) fq (
      .clk(clk), .clear(rst || start), .in_valid(st == MERGE && cur_v[me]),
      .in_key(cur_k[me*KW +: KW]), .in_idx(cur_i[me*IW +: IW]),
      .key_o(fk), .idx_o(fi), .vld_o(fv));

  always_ff @(posedge clk) begin
    done <= 1'b0;
    if (rst) begin
      st <= IDLE; beat <= '0; m <= '0;
    end else begin
      case (st)
        IDLE:  if (start) begin st <= FILL; beat <= '0; end
        FILL:  if (in_valid) begin
                 if (beat == BEATS - 1) begin st <= MERGE; m <= '0; end
                 beat <= beat + 1'b1;
               end
        MERGE: begin
                 if (m == MB - 1) st <= FLUSH;
                 m <= m + 1'b1;
               end
        FLUSH: begin
                 out_idx <= fi;
                 out_key <= fk;
                 done <= 1'b1;
                 st <= IDLE;
               end
      endcase
    end
  end
endmodule
