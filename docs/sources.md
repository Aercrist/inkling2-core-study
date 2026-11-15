# Sources

Every figure taken from outside this repo carries a key that resolves to a row here. Pages were read
on 2026-11-14 unless noted. Confidence: **V** vendor or project page, **P** paper or preprint,
**R** independent write-up, **C** code read at a pinned commit. Quotes are verbatim.

Constants with no public source are marked **assumption** where they are used and are swept.

## The model (primary sources)

| key | what we take from it | source | quote | conf. |
|---|---|---|---|---|
| tml-card | layers, experts, parameters, context, numerics, hardware requirement, license, release date | thinkingmachines.ai/model-card/inkling-2-small/ | "A 48 layer decoder-only transformer with a sparse Mixture-of-Experts (MoE) feed-forward backbone: each token is routed to 8 of 384 experts."; "594B total parameters, 17B active"; "The main NVFP4 checkpoint requires a GPU cluster with at least 500GB of aggregated VRAM"; "The model was pretrained, midtrained and post-trained in NVFP4"; "The MXFP8 has a very marginal 1–2 percentage points gain over the NVFP4 checkpoint over evals"; "Date of release: Nov 5, 2026" | V |
| tml-news | weights release date, API hosts, vision encoder size, draft and verify, KV-cache variant, the ladder | thinkingmachines.ai/news/inkling-2-small/ (read in full) | "Nov 12, 2026 Update: Inkling 2 Small’s weights are now available for download on HuggingFace"; "available via API at NVIDIA, Fireworks, Together, Baseten, Nebius, Modal, and Cloudflare"; "a 292M-parameter vision encoder"; "During sampling, prefill, draft, and verify run fully on CUDA graphs."; "offering a variant of NVFP4 KV cache, which we find to minimally lower evals"; "Inkling 2 Small is part of a ladder of models" | V |
| inkling-hf | `config.json`, `hf_quant_config.json`, 25 safetensors headers (dtype, shape and bytes of 1,854 tensors), the Hub's parameter counts | huggingface.co/thinkingmachines/Inkling-2-Small at `8361e7a3` | Hub API: `"total": 594050672720` | V |
| hf-transformers | reference model code: attention, short convolutions, attention residuals, expert MLP, router | github.com/huggingface/transformers at `3a134bc`, `src/transformers/models/inkling2/modeling_inkling2.py` and `docs/source/en/model_doc/inkling2.md` | "Currently only the `Inkling-2-Small-MXFP8` checkpoint is supported." | C |
| nvfp4-blog | the NVFP4 format | developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ | "4 bits (1 sign, 2 exponent, 1 mantissa) plus 1 shared FP8 scale per 16 value block"; "a second-level FP32 scalar applied per tensor"; "one FP8 scale per 16 values (4.5 bits per value)" | V |

## How the model is served today

| key | what we take from it | source | quote | conf. |
|---|---|---|---|---|
| vllm-recipe | precision by layer, draft width, TP and EP layout, headline throughput, KV needed at full context, LoRA limits | recipes.vllm.ai/thinkingmachines/Inkling-2-Small (recipe file at vllm-project/vllm `5ec9995`) | "Experts are NVFP4 in layers 0-39 and MXFP8 in layers 40-47."; "DSpark speculative decoding (7 tokens per step)"; "TP4 means TP4 attention with EP4 experts"; "~1,100 output tok/s/GPU at 100 tok/s/user (1K in / 1K out) with DSpark on 2× 4 GB200 (TP8)"; "64.47 GiB KV cache is needed"; "Expert adapters need BF16 expert weights; with the NVFP4/MXFP8 experts of this checkpoint, use attention-only adapters." | V |
| vllm-blog | single-user speed, throughput at four context lengths, acceptance length, activation quantization, attention kernel step time | vllm.ai/blog/2026-11-12-inkling2-small | "For a single user at TP4, the model decodes at about 630 tok/s with an acceptance length of 3.5."; "The NVFP4 activations use a per-token global scale plus dual-candidate block scales (Four Over Six)"; "A batch-1 decode step at 100K context drops from 15.1 ms to 4.8 ms." | V |
| sglang-recipe | checkpoint sizes, why EP equals TP, verified rows per step, GPU memory sizes, KV precision options | docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling-2-Small | "stores the routed experts of layers 0–39 in NVFP4 and of layers 40–47 in MXFP8 (about 394 GB)"; "Keep EP equal to TP: the NVFP4 experts cannot be TP-sharded, because their activation scales must cover each expert’s full input width."; "DSpark proposes from the bundled MTP draft head and verifies 8 tokens per request per step."; "B200 and GB200 (192 GB)" | V |
| tokenspeed-blog | measured speeds on 4 × B300, the workload they were measured on, KV layout | lightseek.org/blog/tokenspeed-inkling-2 | "On 4× B300 replaying AgentX agentic coding sessions, a single user gets 476 tokens/s of decode"; "single-user decode is bound by kernel launches and dependency waits rather than math"; "528 bytes per token per target layer"; "the DSPARK draft simply adds five ordinary attention layers to the full-history group" | V |
| tokenspeed-recipe | MXFP8 KV format, cache ratio, launch command | lightseek.org/tokenspeed/recipes/models (section "Inkling 2") | "the target's KV cache is stored as MXFP8, FP8 E4M3 values with one power-of-two scale per 32; the DSpark draft's stays BF16. A token takes 0.70 of its BF16 cache bytes, so four B300 GPUs hold 15.8M tokens instead of 10.9M." | V |
| tokenspeed-src | the Inkling 2 Small implementation: model, attention, MoE, quantization, KV cache, drafter. Read in full; kernel counts, collective regimes, draft schedule and the kernel and prefill timings in its comments are used in the GPU model | github.com/lightseekorg/tokenspeed at `a439df4`, `python/tokenspeed/runtime/models/inkling2*.py`, `layers/moe/inkling2.py`, `layers/quantization/inkling2.py`, `layers/attention/kv_cache/hybrid_inkling.py` | | C |

On 2026-11-14 the main branches of vLLM (`a16bfee`) and SGLang (`7b5fb39`) held the earlier Inkling
model but not Inkling 2 Small, which both projects ship in development images.

## Fine-tuning and determinism (Thinking Machines' own work)

| key | what we take from it | source | quote | conf. |
|---|---|---|---|---|
| tinker-cookbook | exact LoRA adapter sizes per rank for Inkling 2 Small; the shared-outer scheme for expert adapters | github.com/thinking-machines-lab/tinker-cookbook at `1991897`, `tinker_cookbook/hyperparam_utils.py` | `"thinkingmachines/Inkling-2-Small": {"mlp": 128_065_536, "attn": 2_347_008, "unembed": 205_632}` | C |
| lora-regret | when LoRA matches full fine-tuning. Located; to be read for the multi-LoRA experiment | thinkingmachines.ai/blog/lora/ | | V |
| nondeterminism | batch invariance and why reduction order matters. Located; to be read for the determinism check | thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ | | V |

## Hardware constants carried over from the DeepSeek-V4.1 study

These keys are defined, with quotes, in that study's `docs/sources.md`; the rows below repeat the value used here.

| key | value used | where it comes from | conf. |
|---|---|---|---|
| lenovo-b300 | HGX B300: 7.7 TB/s memory bandwidth, 1,100 W TDP per GPU | lenovopress.lenovo.com/lp2264-thinksystem-sr680a-v4-server | V (OEM) |
| nv-bw-ultra | Blackwell Ultra: 288 GB HBM3E | developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/ | V |
| stream-b200 | B200: 8.0 TB/s peak memory bandwidth | arxiv.org/html/2512.02189v1 | P |
| sram-density | N5-class HD SRAM: about 32 Mib/mm² of array | fuse.wikichip.org/news/3398/ | R |
| dsv41-calibration | HBM efficiency 0.8375 of peak, as calibrated on two measured step times | github.com/Aercrist/dsv41-accel-study at `bc1e959`, `results/decode.json` → `calibration.hbm_eff` | own work |
| hbm3e | HBM3E: 36 GB per 12-high stack, more than 1.2 TB/s per stack (run here at 1.0 TB/s) | micron.com/products/memory/hbm/hbm3e | V |
| ar-latency | best measured 4-GPU small all-reduce 2.37 µs; L2 round trip 0.306 µs | arxiv.org/pdf/2607.16100 | P |
| serdes-energy | 112G long-reach SerDes below 5 pJ/bit | synopsys.com (112G bulletin); Hot Chips 2025 tutorial | V / R |
| d2d-energy | die-to-die links 0.25 to 0.5 pJ/bit | uciexpress.org; see ucie-targets | V |
| electricity | US industrial electricity 8.62 c/kWh (2025 average) | eia.gov/energyexplained/electricity/prices-and-factors-affecting-prices.php | V |
| wafer-price | N5 about $16,988 (CSET model); 4NP $17,000 (Epoch); analyst band $16,000 to $21,300 for N5/N4 | cset.georgetown.edu; epoch.ai; siliconanalysts.com/data/wafer-pricing ("5nm TSMC N5/N4 $18,500 $16,000 - $21,300 EUV"; "Low confidence: no current public print of a TSMC N5 wafer price.") | E |
| sram-energy | 7 nm SRAM read: 1 MB array 14 pJ per 64 bits (0.22 pJ/bit) | Jouppi et al., ISCA 2021, Table 2 | P |
| mac-energy | 7 nm operation energies (Int8 multiply 0.070 pJ, BF16 multiply 0.210 pJ, FP32 multiply 1.310 pJ); a measured 5 nm 4-bit accelerator at 10.5 to 25.8 fJ per operation, whole chip | Jouppi et al., ISCA 2021; doi.org/10.1109/jssc.2023.3234893 ("95.6 tera-operations per second per Watt (TOPS/W) at 0.46 V on a 4-bit benchmarking layer") | P |
| cowos-cost | CoWoS-L about $1,100 per B200-class package plus about $1,000 of packaging yield loss; the basis of MDA-1's $2,500 two-die package and of S's per-die tile cost | epoch.ai/data-insights/b200-cost-breakdown (packaging table: "$1,100") | E |
| hbm-price | HBM3E $1.5 to $1.6 per Gb | en.sedaily.com (DigiTimes), 2026-07-12 | R |
| hbm-phy-area | HBM3E PHY about 11 mm². **Unverified**: seen only in search summaries | tomshardware.com | unverified |
| hbm-energy | HBM2 access energy 3.97 pJ/bit | cs.utexas.edu/users/skeckler/pubs/MICRO_2017_Fine_Grained_DRAM.pdf | P |
| gpu-rental | B300 on-demand market median $9.50 per GPU-hour (not used in the owned-cost headline) | getdeploying.com/gpus/nvidia-b300; nebius.com/prices | E / V |
| defect-density | TSMC N5 D0 0.10 to 0.11 per cm² (2020) | anandtech.com/show/16028 (archived) | R |
| nv-hgx, nv-blackwell | HGX B300 dense FP4 108 PFLOPS per 8 GPUs | nvidia.com/en-us/data-center/hgx/ | V |

## Hardware constants sourced for this study

Read on 2026-11-14 and 2026-11-15. **E** marks an analyst or reseller estimate.

| key | value used | source | quote | conf. |
|---|---|---|---|---|
| sram-3nm-macro | SRAM macro density 35.5 Mbit/mm² and standby leakage 27.5 pW/bit at 25 °C (measured, 3 nm; the closest public silicon to N5/N4) | doi.org/10.1109/jssc.2025.3639747 | "The macro achieves 35.5-Mbit/mm2 density, 27.5-µW/Mbit standby leakage at 25 °C" | P |
| sram-retention | leakage at retention voltage against full voltage: 0.31 (4 nm compiler), 0.2 (16 nm source bias), 0.05 (5 nm, array supply lowered by 400 mV) | doi.org/10.1109/vlsitechnologyandcir65830.2026.11577227; doi.org/10.1109/icsict.2016.7998980; doi.org/10.23919/vlsicircuits52068.2021.9492368 | "0.35x in standby and 0.31x in retention mode"; "standby power are improved by up to ... 80%"; "95.2% leakage power reduction by lowering VDDC by 400mV" | P |
| vcache | die-level SRAM density of a shipped stacked cache die: 64 MB in 36 mm² at 7 nm (14.2 Mbit/mm² including ECC and keep-outs) | tomshardware.com/news/amd-shares-new-second-gen-3d-v-cache-chiplet-details-up-to-25-tbs; doi.org/10.1109/isscc42614.2022.9731565 | "AMD made the 7nm SRAM die smaller, so it now measures 36mm2" | R / P |
| ucie-targets | die-to-die links: 0.25 to 0.5 pJ/bit, under 2 ns, at least 85% power saving when idle; long-reach 56/112G links: 20+ ns plus more than 100 ns of error correction | r6.ieee.org/scv-eps/wp-content/uploads/sites/58/2023/02/D2-4-Sharma.pdf | "Latency (Tx + Rx) < 2ns"; "Power savings estimated at >= 85%"; "20+ ns (+ >100 FEC)" | V |
| taalas-hc1 | a shipped fixed-weight chip: 8B parameters of 3 and 6 bits on an 815 mm² N6 die, 53B transistors, about 200 W; a new model changes two metal layers | nextplatform.com/2026/02/19/taalas-etches-ai-models-onto-transistors-to-rocket-boost-inference/4092140; taalas.com/the-path-to-ubiquitous-ai/ | "To etch a new model on an HC inference engine involves changing a two layers of metal in the HC chip design"; "The Silicon Llama is aggressively quantized, combining 3-bit and 6-bit parameters" | R / V |
| rom-leakage | a mask-programmed array holds no state and can be powered off when not read; published retention work treats ROM as the low-leakage option | synopsys.com/articles/using-foundation-ip.html | "Using an ultra-low leakage viaROM and source biasing in the SRAM can reduce leakage by up to 70 percent while maintaining data." | V |
| mask-cost | full mask set at 5 nm class $6M to $12M for about 70 to 80 masks; $300,000 per unique mask used here is an assumption inside that range | semiconductorx.com/semiconductor-photomasks.html; siliconanalysts.com/data/wafer-pricing | "5nm About 70 to 80 $6M to $12M Heavier EUV" | E |
| nre | design cost from scratch at 5 nm $436.3M (IBS); a fixed-weight chip start-up reports about $30M to first product | semiengineering.com/making-chips-at-3nm-and-beyond; taalas.com/the-path-to-ubiquitous-ai/ | "The design cost for a 3nm chip is $650 million, compared to $436.3 million for a 5nm device"; "a total of just $30M spent" | E / V |
| gpu-power-batch | measured GPU power while serving LLMs: 43% of TDP at one request, 86% at 128 (H100; carried to B300 as an assumption) | arxiv.org/pdf/2603.17280 | "Pidle = 300 W at b=1 and Pnom = 600 W at b=128, both within 3% fit error." | P |
| b300-price | HGX B300: $56,000 to $76,000 per GPU with networking | amcompute.com/compare/gb300-vs-b300 | "An HGX B300 node lists at $400,000 to $520,000, or about $56,000 to $76,000 per GPU once networking is" included | E |
| pue-blackwell | facility overhead 1.25 | developer.nvidia.com/blog/scaling-token-factory-revenue-and-ai-efficiency-by-maximizing-performance-per-watt/ | "NVIDIA Blackwell systems reduce cooling overhead, operating around 1.25 PUE" | V |
| routing-skew | share of routing mass on the top 20% of experts: 47% (DeepSeek-V3), 53% (Kimi K2), 62% (Qwen3), 80% (Llama 4 Maverick); uniform would be 20% | arxiv.org/pdf/2510.05497 | "expert candidates cover 47%, 62%, 80%, and 53% of the cumulative conditional probability in DeepSeek-V3, Qwen3, Llama 4, and Kimi K2, respectively" | P |
| nvfp4-mac-28nm | a published NVFP4 multiply-accumulate block at 28 nm: 0.024 mm², 11.27 mW at 524 MHz (compared with our block in docs/07) | arxiv.org/pdf/2603.28765 | "The NVFP4 MAC achieves a latency of 3.82 ns at 524 MHz, corresponding to 16.77 GFLOPS" | P |
| asap7-scaling | ASAP7 to N4P-class area factor 0.69, derived by us from published pitches: N5 2-fin cell 210 nm x 51 nm against ASAP7's 270 nm x 54 nm = 0.73, divided by N4P's 6% density gain. No published ASAP7-to-foundry delay or power factor exists | angstronomics.com/p/the-truth-of-tsmc-5nm; ASAP7 description (Vashishtha, ASU dissertation, 2019) | "We measure an average standard cell height of 210nm and average CGP of 51nm."; "a relatively conservative 54 nm contacted gate pitch"; "we chose a 7.5-track cell library" | R + P (derived) |
| n4p | N4P against N5: 11% faster, 22% more power-efficient, 6% denser | eenewseurope.com/en/tsmc-introduces-n4p-chip-manufacturing-process/ | "N4P will deliver an 11 percent performance boost over N5"; "a 22 percent improvement in power efficiency and a 6 percent improvement in transistor density" | R |
| lpddr5x | LPDDR5X: up to 8,533 Mb/s per pin (used for the LPDDR KV option; 16 data pins per channel, 4 channels per package and a 16-package pool are assumptions) | JEDEC JESD209-5B, as announced: businesswire.com/news/home/20210728005691/en; anandtech.com/show/16851 | "up to 8533 Mbps" | V |

No public source was found for SRAM leakage at operating temperature at N5/N4, for mask-programmed
storage density at N5/N4, or for the idle power of long-reach serial links. Those are assumptions and
are swept.

## Published speeds for the cross-architecture check (Llama 3.1 8B)

| key | what we take from it | source | quote | conf. |
|---|---|---|---|---|
| nim-bench | H200, one request, 1,000 in / 1,000 out: 234.95 tok/s with FP8 weights, 140.14 with BF16 | docs.nvidia.com/nim/benchmarking/llm/1.0.0/performance.html | table rows for "Llama-3.1-8b-instruct", NIM version 1.8.0, concurrency 1 | V |
| h200-specs | H200: 141 GB HBM3e, 4.8 TB/s | nvidia.com/en-us/data-center/h200/ and the NIM page ("H200 141GB HBM3e") | | V |
| taalas-chart | one chart with six machines on Llama 3.1 8B: H200 230, Groq 594, Cerebras 1,981, fixed-weight chip 16,960 tok/s per user; the SRAM figures are attributed to Artificial Analysis, the last is the vendor's own | taalas.com/the-path-to-ubiquitous-ai/ | "Performance data for Llama 3.1 8B, Input sequence length 1k/1k" | V (relayed) |
| aa-llama | Groq on Artificial Analysis' live page: 630.1 tok/s on the 1k-input workload | artificialanalysis.ai/models/llama-3-1-instruct-8b/providers (retrieved 2026-11-15) | | R |
| groq-docs | Groq's own figure "~560 tps"; chip: 230 MB SRAM, up to 80 TB/s | console.groq.com/docs/model/llama-3.1-8b-instant; GroqChip product brief | "TOKEN SPEED ~560 tps"; "230 MB SRAM per chip"; "Up to 80 TB/s on-die memory bandwidth" | V |
| cerebras-launch | 1,800 tok/s at launch with 16-bit weights; wafer: 44 GB SRAM, 21 PB/s | cerebras.ai/blog/introducing-cerebras-inference-ai-at-instant-speed; CS-3 datasheet | "Cerebras inference delivers 1,800 tokens per second for Llama3.1 8B"; "Processor Memory 44 GB total, 21 PB/s SRAM bandwidth" | V |
| hc1-probe | independent probe of the fixed-weight chip's public demo: 14,000 to 14,300 tok/s for generations above about 100 tokens | gist.github.com/instax-dutta/e797a38f1aec3a422a12cf5996e3e403 | "Decode rate is remarkably stable at ~14,000–14,300 tok/s for generations above ~100 tokens" | R |
| hc1-analyst | one query on the public demo: 14,357 tok/s. The analyst discloses the vendor is a client | karlfreund429026.substack.com/p/taalas-launches-hardcore-chip-with | "(0.138 seconds and 14,357 tokens per second)" | R |

The same vendor's figure for a 671B model (12,382 tok/s per user) is labeled by the vendor as
simulated ("DeepSeek R1 (Simulated by Taalas Labs)") and is not used as a measurement anywhere here.
Artificial Analysis lists one provider for Inkling 2 Small, the developer's own endpoint, at about
81 tok/s end to end including reasoning tokens (retrieved 2026-11-15); hardware is not disclosed, so it
is not a calibration point.
