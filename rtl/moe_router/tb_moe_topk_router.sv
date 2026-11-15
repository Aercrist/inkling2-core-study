// Testbench: drives vectors from rtl/moe_router/build/keys.hex (N_TESTS tokens x 384 16-bit keys),
// writes the selected ids per token to rtl/moe_router/build/rtl_out.txt and the cycle count from the
// first key beat to `done` to rtl/moe_router/build/latency.txt. rtl/check_router.py compares with numpy.
`timescale 1ps/1ps
module tb;
  localparam int E = 384, K = 6, LANES = 8, KW = 16, IW = 9;
  localparam int N_TESTS = 256;
  logic clk = 0, rst = 1, start = 0, in_valid = 0, done;
  logic [LANES*KW-1:0] in_keys;
  logic [K*IW-1:0] out_idx;
  logic [K*KW-1:0] out_key;
  logic [KW-1:0] mem [0:N_TESTS*E-1];
  integer fo, fl, t, b, i, cyc, lat;

  moe_topk_router #(.E(E), .K(K), .LANES(LANES), .KW(KW), .IW(IW)) dut (
      .clk(clk), .rst(rst), .start(start), .in_valid(in_valid), .in_keys(in_keys),
      .done(done), .out_idx(out_idx), .out_key(out_key));

  always #333 clk = ~clk;  // ~1.5 GHz
  initial cyc = 0;
  always @(posedge clk) cyc <= cyc + 1;

  initial begin
    $readmemh("rtl/moe_router/build/keys.hex", mem);
    fo = $fopen("rtl/moe_router/build/rtl_out.txt", "w");
    fl = $fopen("rtl/moe_router/build/latency.txt", "w");
    repeat (4) @(posedge clk);
    rst <= 0;
    for (t = 0; t < N_TESTS; t = t + 1) begin
      @(posedge clk); start <= 1;
      @(posedge clk); start <= 0; lat = cyc;
      for (b = 0; b < E / LANES; b = b + 1) begin
        in_valid <= 1;
        for (i = 0; i < LANES; i = i + 1) in_keys[i*KW +: KW] <= mem[t*E + b*LANES + i];
        @(posedge clk);
      end
      in_valid <= 0;
      while (!done) @(posedge clk);
      lat = cyc - lat;
      for (i = 0; i < K; i = i + 1) $fwrite(fo, "%0d ", out_idx[i*IW +: IW]);
      $fwrite(fo, "\n");
      $fwrite(fl, "%0d\n", lat);
    end
    $fclose(fo); $fclose(fl);
    $finish;
  end
endmodule
