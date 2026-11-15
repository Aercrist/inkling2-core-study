<!-- generated from docs/templates/01_workload.md.j2 by experiments/render_docs.py; edit the template -->
# 01 · Workload

This page is the inference workload of Inkling 2 Small as read from the checkpoint headers, the model
config and the public serving material, **before any simulation**. The simulator expands it into the per-op
workload the simulator runs. Every number is rendered from `results/phase0_workload.json`.

Three labels are used throughout:

* **derived**: exact arithmetic on the checkpoint headers and config.
* **measured (source)**: quoted from a public source. We did not run it.
* **napkin**: a first-order estimate. The simulator must later reproduce it or correct it.

## 1. What was read

| what | where | key in [sources](sources.md) |
|---|---|---|
| Model card and announcement, in full | thinkingmachines.ai | `tml-card`, `tml-news` |
| `config.json`, `hf_quant_config.json`, 25 safetensors headers (1,854 tensors) | `thinkingmachines/Inkling-2-Small` at `8361e7a3` | `inkling-hf` |
| Reference model code | Hugging Face `transformers`, `models/inkling2` | `hf-transformers` |
| Serving recipes and launch write-ups | vLLM, SGLang, TokenSpeed | `vllm-recipe`, `vllm-blog`, `sglang-recipe`, `tokenspeed-blog`, `tokenspeed-recipe` |
| NVFP4 definition | NVIDIA | `nvfp4-blog` |

Only headers were downloaded: 393.5 GB of weights stayed on the Hub.

TokenSpeed's main branch contains its Inkling 2 Small implementation (`tokenspeed-src`), which we
read line by line (section 11); vLLM and SGLang ship theirs in development images, and their main
branches hold only the earlier Inkling model, so those two were used through their published recipes.

## 2. The model in numbers (derived)

| quantity | value |
|---|---|
| decoder layers | 48: 40 sliding-window (window 1,024) and 8 full-attention (layers 5, 11, 17, 23, 29, 35, 41, 47) |
| hidden size | 4,608 |
| attention heads | 48 query, 8 key/value, 128 wide |
| experts | 384 per layer, 8 routed per token, 0 shared; every layer is MoE |
| one expert | 31,850,496 parameters: 4,608 → 2 × 2,304 → 4,608 |
| context | up to 1,048,576 tokens |
| vocabulary | 201,024 |
| vision encoder | 23-layer ViT, 1,024 wide: 292,041,728 parameters, plus a 35,651,584-parameter adapter |
| speculative draft | separate 5-layer model, 1.71B parameters, 8 positions per step |

## 3. 594B and 17B, reproduced from the headers (derived)

| precision | parameters | share | bytes on disk | bits per parameter |
|---|---|---|---|---|
| NVFP4 (experts, layers 0-39) | 489,223,618,560 | 82.4% | 275.2 GB | 4.50 |
| MXFP8 (experts, layers 40-47) | 97,844,723,712 | 16.5% | 100.9 GB | 8.25 |
| BF16 (everything else) | 6,981,765,120 | 1.2% | 14.0 GB | 16 |
| FP32 (attention-residual queries, router bias) | 534,528 | 0.0001% | 2.1 MB | 32 |
| **total** | **594,050,641,920** | | **390.1 GB** | |
| speculative draft (not in the 594B) | 1,711,710,384 | | 3.42 GB | 16 |

* **594B.** The Hub reports 594,050,672,720 parameters. The headers give exactly that number
  under the Hub's convention, which counts each packed NVFP4 byte as two parameters, leaves out the
  per-block scale tensors and the draft, and counts the 30,800 second-level
  FP32 scales as parameters. Without those scales the weights number 594,050,641,920.
* **17B active.** One token uses the attention and router weights of every layer plus
  8 of 384 experts: 17.03B parameters. That matches the
  published 17B. Adding the output projection gives 17.96B; counting
  every non-expert weight, including the embedding table and the vision encoder, gives
  19.21B.
* **292M vision encoder.** The ViT backbone has 292,041,728 parameters.
* The checkpoint on disk is 393.5 GB including the draft.

## 4. Which tensors are which precision, and what the activations are (derived)

From `hf_quant_config.json`, confirmed against the dtype of every tensor:

| what | stored as | computed in |
|---|---|---|
| routed experts, layers 0-39 | NVFP4: two E2M1 codes per byte, one E4M3 scale per 16 values along the input, one FP32 scale per expert | NVFP4 weights × **NVFP4 activations** |
| routed experts, layers 40-47 | MXFP8: E4M3 values, one power-of-two scale per 32 values | MXFP8 weights × **MXFP8 activations** |
| attention projections (query, key, value, relative, gate, sink, output), router, convolutions, norms, embedding, output projection, vision, draft | BF16 | BF16; the output projection returns FP32 |
| attention-residual queries, router bias | FP32 | FP32 |

The activation format is the answer to "FP4 or FP8?": it follows the weights. The NVFP4 expert GEMMs
quantize their inputs to `nvfp4_e2m1` in
16-value blocks with
`float8_e4m3fn` block scales, a
**per-token** second-level scale, and the
`fouroversix_mae` rule: each block's
maximum is scaled to 6 or to 4, whichever reconstructs the block with the lower mean absolute error
(`vllm-blog`). Two consequences for hardware:

* The second-level scale is per token, so a token's quantization does not depend on what else is
  in the batch. That is a precondition for the batch-invariance check in the silicon section.
* The published engines keep expert parallelism equal to tensor parallelism because an NVFP4
  expert cannot be split across devices: its activation scales must cover the expert's full input
  width (`sglang-recipe`).

NVIDIA's definition of NVFP4 matches what the checkpoint stores: E2M1 values, one E4M3 scale per 16
values, one FP32 scale per tensor, 4.5 bits per value (`nvfp4-blog`). In this checkpoint the "tensor"
for the FP32 scale is one expert.

## 5. The four kinds of layer (derived)

| attention | experts | layers | dense weights | one expert | 8 routed experts | weight MACs per token | resident |
|---|---|---|---|---|---|---|---|
| sliding | NVFP4 | 34 | 100.0M, 200 MB | 17.9 MB | 143 MB | 355M | 7.08 GB |
| full | NVFP4 | 6 | 100.3M, 201 MB | 17.9 MB | 143 MB | 355M | 7.08 GB |
| sliding | MXFP8 | 6 | 100.0M, 200 MB | 32.8 MB | 263 MB | 355M | 12.81 GB |
| full | MXFP8 | 2 | 100.3M, 201 MB | 32.8 MB | 263 MB | 355M | 12.81 GB |

"Dense weights" are shared by every token: the attention projections and the router. "8 routed
experts" is what one token reads from the MoE. Full-attention layers carry slightly more dense
weight because their key and value convolutions depend on the input. The per-layer table for all
48 layers is in `results/phase0_workload.json` under `layers`.

Attention itself costs 12,288 MACs per layer for
every token of context it looks at. In the 8 full-attention layers that is
98,304 MACs per context token, so attention
arithmetic overtakes weight arithmetic at about 183,000
tokens of context.

## 6. What one decoded token touches (derived)

Batch 1, no speculation, text only:

| weights read for one token | bytes | share |
|---|---|---|
| BF16 attention projections and router, all 48 layers | 9.60 GB | 50% |
| NVFP4 experts, 8 per layer, 40 layers | 5.73 GB | 30% |
| MXFP8 experts, 8 per layer, 8 layers | 2.10 GB | 11% |
| BF16 output projection | 1.85 GB | 10% |
| **total** | **19.29 GB** | |

That is 17.96 billion weight MACs per token.

**Half of the bytes a token reads are 16-bit attention weights, not 4-bit experts.** The experts are
98.8% of the parameters but
41% of what one token reads. One token
uses 4.9% of the weight bytes and
2.1% of the experts.

## 7. KV cache (derived, checked against three published figures)

Every layer stores keys and values for 8 heads of 128:
2,048 values per token per layer.

| per token | BF16 | MXFP8 | NVFP4-style |
|---|---|---|---|
| one layer | 4,096 B | 2,112 B | 1,152 B |
| the 8 full-attention layers | 32,768 B | 16,896 B | 9,216 B |
| at 1,048,576 tokens | 34.4 GB | 17.7 GB | 9.7 GB |

* **Full-attention ("global") layers** are the only ones whose cache grows with context.
* **Sliding-window ("local") layers** keep the last 1,024 tokens:
  168 MB per request in BF16, whatever the context.
* **The draft** adds 5 full-history layers that stay in BF16
  (`tokenspeed-recipe`): 20,480 B per token,
  21.5 GB at full context. With an MXFP8 cache for the target
  model, the draft's cache is the larger of the two.

Cross-checks against published figures (measured, source):

| check | ours | published | source |
|---|---|---|---|
| MXFP8 cache, bytes per token per full layer, per rank at TP4 | 528 | 528 | `tokenspeed-blog` |
| MXFP8 cache over BF16 cache, draft included | 0.702 | 0.70 | `tokenspeed-recipe` |
| KV for one request at full context, BF16 | 52.2 GiB | 64.47 GiB | `vllm-recipe` |

The first two agree. **The third does not**: vLLM reserves
1.24x what our count gives.
We did not find the reason; see section 12.

The "NVFP4-style" column is hypothetical. The announcement mentions an NVFP4 KV-cache variant
(`tml-news`), but the cache the public engines expose is MXFP8 (`sglang-recipe`, `tokenspeed-recipe`).

## 8. Speculative decoding changes the step (napkin)

The checkpoint ships a draft model, and every published speed uses it. One step drafts 7 tokens and
verifies 8 rows per request (`vllm-recipe`, `sglang-recipe`); 3.3 to
3.5 tokens are accepted (`vllm-blog`).

The 8 rows share the dense weights but route separately. If rows route uniformly and
independently, 64 picks hit 59.5
distinct experts per layer, and one step reads:

| one speculative step | bytes |
|---|---|
| experts | 58.3 GB |
| BF16 attention projections and router | 9.6 GB |
| draft model | 3.4 GB |
| BF16 output projection | 1.9 GB |
| **total** | **73.2 GB** |

At 3.5 accepted tokens that is 20.9 GB per token,
1.08x the bytes of decoding without speculation. **Speculation
does not save memory traffic under independent routing; it saves fixed latency.** Real routing is
correlated between neighbouring tokens, which lowers the expert bytes. We have no Inkling routing
traces, so the routing sweep ([04](04_results.md), section 10) brackets this.

## 9. Published speeds against the bandwidth ceiling (napkin on measured figures)

The ceiling is the speed at which a step's bytes stream through HBM at
83.8% of peak bandwidth (the efficiency calibrated in our DeepSeek-V4.1 study), with
nothing else on the clock.

| published figure (measured, source) | speed | implied step | time to stream the step's weights | speed / ceiling |
|---|---|---|---|---|
| 4x B300, TokenSpeed, no speculation (`tokenspeed-blog`) | 272 tok/s | 3.68 ms | 0.75 ms | 20% |
| 4x B300, TokenSpeed, DSpark (`tokenspeed-blog`) | 476 tok/s | 7.35 ms | 2.84 ms | 39% |
| 4x GB200, vLLM, DSpark (`vllm-blog`) | 630 tok/s | 5.56 ms | 2.73 ms | 49% |

| ceiling (napkin) | no speculation | DSpark |
|---|---|---|
| 4 × B300 | 1,340 tok/s | 1,230 tok/s |
| 2 × B300 | 670 tok/s | 620 tok/s |

Three cautions. The TokenSpeed figures come from long agentic sessions (prompts average 94k tokens),
where attention adds work the ceiling leaves out. TokenSpeed does not publish its acceptance length,
so its DSpark row assumes vLLM's 3.5. And the ceiling assumes independent routing.

Read with those cautions, the published engines run at 20% to
49% of the bandwidth ceiling on four GPUs. The
TokenSpeed team says the same thing in words: "single-user decode is bound by kernel launches and
dependency waits rather than math" (`tokenspeed-blog`). This is the same finding as our DeepSeek-V4.1
study, from the engine's own authors, and it is why the same-GPU software counterfactual (G1) stays in
this study.

All published figures are in [`data/public_measurements.yaml`](../data/public_measurements.yaml).

## 10. What the checkpoint says that our starting notes did not

We began with a list of facts to verify and two rough anchors. Most held. These did not:

| starting note | what the sources say |
|---|---|
| The weights are NVFP4. An all-NVFP4 set would be about 334 GB. | The checkpoint is **mixed**: NVFP4 experts in layers 0-39, MXFP8 experts in layers 40-47, BF16 everywhere else. It is 393.5 GB, 1.18x the anchor. |
| Batch-1 decode touches about 9.6 GB of weights per token. | **19.3 GB**, 2.0x the anchor, because the attention projections are BF16 and the last 8 layers' experts are MXFP8. |
| No throughput or latency figures have been published. | **Several have**, on disclosed hardware: 476 tok/s per user on 4 × B300 (`tokenspeed-blog`), about 630 on 4 × GB200 (`vllm-blog`), and throughput at four context lengths. The GPU model can be calibrated, not assumed. |
| (not in the notes) | The model ships a **speculative draft** (DSpark, 8 rows per step). Every platform in this study must model it. |
| Serving uses an NVFP4 KV-cache variant. | The announcement says so. The cache the public engines expose is **MXFP8**. Unresolved. |
| The recommended deployment is 2 × B300. | The model card gives 2 × B300 as an example that meets the memory minimum. Every published speed is on **4 or 8 GPUs** at TP4 or TP8. |

Confirmed as stated: 48 layers; 8 of 384 experts; mixed local and global
attention; context up to 1,048,576; a 292M-parameter vision encoder; trained in NVFP4 at
every stage, with the MXFP8 checkpoint 1-2 points better on evals; at least 500 GB and 800 GB of
aggregate VRAM for the two checkpoints; open weights under Apache 2.0 since 12 Nov (the model was
announced on 5 Nov with closed weights); and the seven API hosts.

## 11. What the serving engine's source adds (read at a pinned commit)

TokenSpeed's Inkling 2 Small implementation was read line by line at commit `a439df4`
(tokenspeed-src). What it changed in the workload model:

| finding | effect on the model |
|---|---|
| Every layer runs 17 kernel launches in a fixed order when its experts are NVFP4, 16 when MXFP8. | Launch counts in the GPU model are counted, not fitted: 937 per speculative step for one request in our operation graph (the source reading gives about 977, good to about 15). |
| One request with the draft on verifies at most 8 rows and skips both all-reduces in every layer: each producer writes its partial into every rank's mailbox. Two or more requests, or no draft, run two all-reduces per layer. | Three collective regimes in the builder. A single request without the draft runs 96 standalone all-reduces per step; with the draft it runs none inside the layers. |
| Experts are placed whole on one rank (rank r owns experts 96r to 96r+95); inputs are replicated. | Expert parallelism with the load imbalance that uniform or skewed routing produces. |
| The draft reads the output head a second time and runs 7 sequential rounds, each ending in a distributed argmax. | Draft operations in the step: a second head read and 7 serial rounds with a collective each. |
| The KV cache is BF16 or MXFP8. There is no NVFP4 KV cache in the source. | The NVFP4 KV variant the announcement mentions stays a sensitivity. |
| Source comments give measured times: one full-attention kernel at 8K, 32K and 128K keys, and a prefill forward at 2K and 3K tokens on 4x B300. | The prefill times are calibration points; the attention-kernel times are a held-out check, which the model fails at long context ([06](06_simulator.md)). |

## 12. What happened to the open items

1. **Read the engine source.** Done for TokenSpeed (above). The vLLM and SGLang implementations were
   used through their published recipes and blog figures only.
2. **The 1.24x KV gap against vLLM.** Still unexplained. Our bytes per token match
   TokenSpeed's published figure exactly.
3. **An NVFP4 KV cache.** Not found in any public source.
4. **Dequantize one real expert.** Done as part of the circuit block's reference model ([07](07_silicon.md)).
5. **Routing correlation.** Bracketed with published statistics from four open models
   ([04](04_results.md), section 10).
