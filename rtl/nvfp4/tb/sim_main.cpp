// Verilator harness for nvfp4_dot / nvfp4_dot_const* (rtl/nvfp4/simlib.py builds and runs it).
//
//   sim <job.bin> <out.bin>
//
// job.bin (little-endian u32 header: magic 'NVD1', NB, RW, A, NVEC, DUT_ROWS, 0, 0) holds a table
// of RW weight rows and A activation rows, each NB blocks of 8 code bytes + 1 scale byte, and a
// schedule of NVEC vectors (first weight row, activation row, gap seed). A vector streams NB
// blocks; a non-zero gap seed inserts pseudo-random idle clocks carrying garbage data between
// blocks. Vectors follow each other with no idle clock unless a gap says so. out.bin receives
// DUT_ROWS binary32 words per vector, in the order out_valid fires. The comparison with the
// reference is done in Python; this file only drives pins and records outputs.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "verilated.h"
#include VTOP_H

#ifndef DUT_ROWS
#define DUT_ROWS 1
#endif

template <std::size_t N> static void set64(VlWide<N>& d, int i, uint64_t v) { d[2 * i] = (uint32_t)v; d[2 * i + 1] = (uint32_t)(v >> 32); }
static void set64(QData& d, int, uint64_t v) { d = v; }
template <std::size_t N> static void set8(VlWide<N>& d, int i, uint8_t v) { d[i / 4] = (d[i / 4] & ~(0xFFu << (8 * (i % 4)))) | ((uint32_t)v << (8 * (i % 4))); }
static void set8(QData& d, int i, uint8_t v) { d = (d & ~(0xFFull << (8 * i))) | ((uint64_t)v << (8 * i)); }
static void set8(IData& d, int i, uint8_t v) { d = (d & ~(0xFFu << (8 * i))) | ((uint32_t)v << (8 * i)); }
static void set8(SData& d, int i, uint8_t v) { d = (SData)((d & ~(0xFFu << (8 * i))) | ((uint32_t)v << (8 * i))); }
static void set8(CData& d, int, uint8_t v) { d = v; }
template <std::size_t N> static uint32_t get32(const VlWide<N>& d, int i) { return d[i]; }
static uint32_t get32(const QData& d, int i) { return (uint32_t)(d >> (32 * i)); }
static uint32_t get32(const IData& d, int) { return d; }

static uint64_t load64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: %s job.bin out.bin\n", argv[0]); return 2; }
  FILE* f = std::fopen(argv[1], "rb");
  if (!f) { std::perror("job"); return 2; }
  uint32_t h[8];
  if (std::fread(h, 4, 8, f) != 8 || h[0] != 0x3144564E) { std::fprintf(stderr, "bad job header\n"); return 2; }
  const uint32_t NB = h[1], RW = h[2], A = h[3], NVEC = h[4], ROWS = h[5];
  if (ROWS != DUT_ROWS) { std::fprintf(stderr, "job is for %u rows, DUT has %d\n", ROWS, DUT_ROWS); return 2; }
  std::vector<uint8_t> wc((size_t)RW * NB * 8), ws((size_t)RW * NB), xc((size_t)A * NB * 8), xs((size_t)A * NB);
  std::vector<uint32_t> sched((size_t)NVEC * 3);
  bool ok = std::fread(wc.data(), 1, wc.size(), f) == wc.size() && std::fread(ws.data(), 1, ws.size(), f) == ws.size()
         && std::fread(xc.data(), 1, xc.size(), f) == xc.size() && std::fread(xs.data(), 1, xs.size(), f) == xs.size()
         && std::fread(sched.data(), 4, sched.size(), f) == sched.size();
  std::fclose(f);
  if (!ok) { std::fprintf(stderr, "short job file\n"); return 2; }

  VerilatedContext ctx;
  VTOP dut{&ctx};
  std::vector<uint32_t> out;
  out.reserve((size_t)NVEC * ROWS);
  uint64_t cycles = 0;
  auto tick = [&]() {
    dut.clk = 1; dut.eval();
    if (dut.out_valid) for (uint32_t r = 0; r < ROWS; ++r) out.push_back(get32(dut.out_data, (int)r));
    dut.clk = 0; dut.eval();
    ++cycles;
  };
  dut.clk = 0; dut.rst = 1; dut.in_valid = 0;
  for (int i = 0; i < 4; ++i) tick();
  dut.rst = 0;
  uint64_t lcg = 0x9E3779B97F4A7C15ull;
  auto rnd = [&]() { lcg = lcg * 6364136223846793005ull + 1442695040888963407ull; return lcg >> 20; };
  for (uint32_t v = 0; v < NVEC; ++v) {
    const uint32_t wr = sched[3 * v], xr = sched[3 * v + 1], gap = sched[3 * v + 2];
    if (gap) lcg ^= (uint64_t)gap * 0x100000001B3ull;
    for (uint32_t b = 0; b < NB; ++b) {
      if (gap) {
        uint64_t g = rnd();
        int idle = (g & 3) == 0 ? (int)((g >> 2) & 3) + 1 : 0;      // a quarter of the blocks wait 1-4 clocks
        for (int i = 0; i < idle; ++i) {
          dut.in_valid = 0;
          set64(dut.x_codes, 0, rnd() ^ (rnd() << 32)); dut.x_scale = (uint8_t)rnd();
#ifndef CONST_VARIANT
          dut.in_last = (uint8_t)(rnd() & 1);
          for (uint32_t r = 0; r < ROWS; ++r) { set64(dut.w_codes, (int)r, rnd() ^ (rnd() << 32)); set8(dut.w_scale, (int)r, (uint8_t)rnd()); }
#endif
          tick();
        }
      }
      dut.in_valid = 1;
      set64(dut.x_codes, 0, load64(&xc[((size_t)xr * NB + b) * 8]));
      dut.x_scale = xs[(size_t)xr * NB + b];
#ifndef CONST_VARIANT
      dut.in_last = (b == NB - 1);
      for (uint32_t r = 0; r < ROWS; ++r) {
        set64(dut.w_codes, (int)r, load64(&wc[((size_t)(wr + r) * NB + b) * 8]));
        set8(dut.w_scale, (int)r, ws[(size_t)(wr + r) * NB + b]);
      }
#else
      (void)wr;
#endif
      tick();
    }
  }
  dut.in_valid = 0;
  for (int i = 0; i < 16; ++i) tick();
  dut.final();
  FILE* fo = std::fopen(argv[2], "wb");
  if (!fo) { std::perror("out"); return 2; }
  std::fwrite(out.data(), 4, out.size(), fo);
  std::fclose(fo);
  std::printf("vectors=%u results=%zu cycles=%llu\n", NVEC, out.size(), (unsigned long long)cycles);
  return out.size() == (size_t)NVEC * ROWS ? 0 : 1;
}
