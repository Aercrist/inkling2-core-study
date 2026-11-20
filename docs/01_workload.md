<!-- generated from docs/templates/01_workload.md.j2 by experiments/render_docs.py; edit the template -->
# 01 · Workload

This page describes the inference workload of Inkling 2 Small. We read it from the checkpoint headers,
the model config and the public serving material, **before any simulation**. The simulator expands it
into the per-op workload that the simulator runs. We render each number from
`results/phase0_workload.json`.

This page uses three labels:

- **derived**: exact arithmetic on the checkpoint headers and config.
- **measured (source)**: a quote from a public source. We did not run it.
- **napkin**: a first-order estimate. Later, the simulator must reproduce it or correct it.

## 1. What was read

| what | where | key in [sources](sources.md) |
|---|---|---|
| Model card and announcement, in full | thinkingmachines.ai | `tml-card`, `tml-news` |
| `config.json`, `hf_quant_config.json`, 25 safetensors headers (1,854 tensors) | `thinkingmachines/Inkling-2-Small` at `8361e7a3` | `inkling-hf` |
| Reference model code | Hugging Face `transformers`, `models/inkling2` | `hf-transformers` |
| Serving recipes and launch write-ups | vLLM, SGLang, TokenSpeed | `vllm-recipe`, `vllm-blog`, `sglang-recipe`, `tokenspeed-blog`, `tokenspeed-recipe` |
| NVFP4 definition | NVIDIA | `nvfp4-blog` |

We downloaded only the headers. 393.5 GB of weights stayed on the Hub.

The main branch of TokenSpeed contains its Inkling 2 Small implementation (`tokenspeed-src`). We read
it line by line (section 11). vLLM and SGLang ship their implementations in development images. Their
main branches hold only the earlier Inkling model. Thus we used those two engines through their
published recipes.

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

- **594B.** The Hub reports 594,050,672,720 parameters. The headers give exactly that number
  under the convention of the Hub. This convention counts each packed NVFP4 byte as two parameters.
  It does not count the per-block scale tensors or the draft, but it counts the
  30,800 second-level FP32 scales as parameters. Without those scales, the
  parameter count of the weights is 594,050,641,920.
- **17B active.** One token uses the attention weights and router weights of each layer and
  8 of 384 experts. This is 17.03B parameters. That
  agrees with the published 17B. With the output projection, the count is
  17.96B. With each non-expert weight (the embedding table and the vision encoder
  are two of these), the count is 19.21B.
- **292M vision encoder.** The ViT backbone has 292,041,728 parameters.
- The checkpoint on disk is 393.5 GB with the draft.

## 4. Which tensors are which precision, and what the activations are (derived)

This table comes from `hf_quant_config.json`. We confirmed it against the dtype of each tensor.

| what | stored as | computed in |
|---|---|---|
| routed experts, layers 0-39 | NVFP4: two E2M1 codes per byte, one E4M3 scale per 16 values along the input, one FP32 scale per expert | NVFP4 weights × **NVFP4 activations** |
| routed experts, layers 40-47 | MXFP8: E4M3 values, one power-of-two scale per 32 values | MXFP8 weights × **MXFP8 activations** |
| attention projections (query, key, value, relative, gate, sink, output), router, convolutions, norms, embedding, output projection, vision, draft | BF16 | BF16; the output projection returns FP32 |
| attention-residual queries, router bias | FP32 | FP32 |

The activation format answers the question "FP4 or FP8?". It follows the weights. The NVFP4 expert GEMMs
quantize their inputs to `nvfp4_e2m1` in
16-value blocks with
`float8_e4m3fn` block scales. They use a
**per-token** second-level scale and the
`fouroversix_mae` rule. This rule
scales the maximum of each block to 6 or to 4. It selects the value that reconstructs the block with
the lower mean absolute error (`vllm-blog`).

There are two results for hardware:

- The second-level scale is per token. Thus the quantization of a token does not depend on the
  other contents of the batch. The batch-invariance check in the silicon section needs this property.
- The published engines keep expert parallelism equal to tensor parallelism. The cause is that an
  engine cannot split an NVFP4 expert across devices. The activation scales of an expert must cover
  its full input width (`sglang-recipe`).

The NVIDIA definition of NVFP4 agrees with the contents of the checkpoint. The definition is E2M1
values, one E4M3 scale per 16 values, one FP32 scale per tensor, and 4.5 bits per value
(`nvfp4-blog`). In this checkpoint, the "tensor" for the FP32 scale is one expert.

## 5. The four kinds of layer (derived)

| attention | experts | layers | dense weights | one expert | 8 routed experts | weight MACs per token | resident |
|---|---|---|---|---|---|---|---|
| sliding | NVFP4 | 34 | 100.0M, 200 MB | 17.9 MB | 143 MB | 355M | 7.08 GB |
| full | NVFP4 | 6 | 100.3M, 201 MB | 17.9 MB | 143 MB | 355M | 7.08 GB |
| sliding | MXFP8 | 6 | 100.0M, 200 MB | 32.8 MB | 263 MB | 355M | 12.81 GB |
| full | MXFP8 | 2 | 100.3M, 201 MB | 32.8 MB | 263 MB | 355M | 12.81 GB |

Each token uses the "dense weights": the attention projections and the router. "8 routed
experts" is the quantity that one token reads from the MoE. Full-attention layers have slightly more
dense weight, because their key and value convolutions depend on the input. The per-layer table for all
48 layers is in `results/phase0_workload.json` under `layers`.

Attention costs 12,288 MACs per layer for
each token of context that it reads. In the 8 full-attention layers, that is
98,304 MACs per context token. Thus attention
arithmetic becomes larger than weight arithmetic at about 183,000
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

**Half of the bytes that a token reads are 16-bit attention weights, not 4-bit experts.** The experts are
98.8% of the parameters but
41% of the bytes that one token reads. One token
uses 4.9% of the weight bytes and
2.1% of the experts.

## 7. KV cache (derived, checked against three published figures)

Each layer stores keys and values for 8 heads of 128:
2,048 values per token per layer.

| per token | BF16 | MXFP8 | NVFP4-style |
|---|---|---|---|
| one layer | 4,096 B | 2,112 B | 1,152 B |
| the 8 full-attention layers | 32,768 B | 16,896 B | 9,216 B |
| at 1,048,576 tokens | 34.4 GB | 17.7 GB | 9.7 GB |

- **Full-attention ("global") layers** are the only layers with a cache that grows with context.
- **Sliding-window ("local") layers** keep the last 1,024 tokens. This is
  168 MB per request in BF16, for any context.
- **The draft** adds 5 full-history layers that stay in BF16
  (`tokenspeed-recipe`). This is 20,480 B per token and
  21.5 GB at full context. With an MXFP8 cache for the target
  model, the cache of the draft is the larger of the two.

Cross-checks against published figures (measured, source):

| check | ours | published | source |
|---|---|---|---|
| MXFP8 cache, bytes per token per full layer, per rank at TP4 | 528 | 528 | `tokenspeed-blog` |
| MXFP8 cache over BF16 cache, draft included | 0.702 | 0.70 | `tokenspeed-recipe` |
| KV for one request at full context, BF16 | 52.2 GiB | 64.47 GiB | `vllm-recipe` |

The first two agree. **The third does not agree.** vLLM reserves
1.24x the value that our count gives.
We did not find the cause. Refer to section 12.

The "NVFP4-style" column is hypothetical. The announcement mentions an NVFP4 KV-cache variant
(`tml-news`). But the cache that the public engines expose is MXFP8 (`sglang-recipe`, `tokenspeed-recipe`).

## 8. Speculative decoding changes the step (napkin)

The checkpoint includes a draft model, and each published speed uses it. One step drafts 7 tokens and
verifies 8 rows per request (`vllm-recipe`, `sglang-recipe`). The step accepts 3.3 to
3.5 tokens (`vllm-blog`).

The 8 rows share the dense weights but route separately. If rows route uniformly and
independently, 64 picks select 59.5
distinct experts per layer. Then one step reads:

| one speculative step | bytes |
|---|---|
| experts | 58.3 GB |
| BF16 attention projections and router | 9.6 GB |
| draft model | 3.4 GB |
| BF16 output projection | 1.9 GB |
| **total** | **73.2 GB** |

At 3.5 accepted tokens, that is 20.9 GB per token.
This is 1.08x the bytes of decode without speculation. **Under
independent routing, speculation does not decrease memory traffic. It decreases fixed latency.**

Real
routing is correlated between adjacent tokens, and this decreases the expert bytes. We have no Inkling
routing traces. Thus the routing sweep ([04](04_results.md), section 10) gives a low limit and a high
limit for this effect.

## 9. Published speeds against the bandwidth ceiling (napkin on measured figures)

The ceiling is the speed at which the bytes of a step stream through HBM at
83.8% of peak bandwidth. The ceiling includes no other time. This efficiency is the
value that we calibrated in our DeepSeek-V4.1 study.

| published figure (measured, source) | speed | implied step | time to stream the step's weights | speed / ceiling |
|---|---|---|---|---|
| 4x B300, TokenSpeed, no speculation (`tokenspeed-blog`) | 272 tok/s | 3.68 ms | 0.75 ms | 20% |
| 4x B300, TokenSpeed, DSpark (`tokenspeed-blog`) | 476 tok/s | 7.35 ms | 2.84 ms | 39% |
| 4x GB200, vLLM, DSpark (`vllm-blog`) | 630 tok/s | 5.56 ms | 2.73 ms | 49% |

| ceiling (napkin) | no speculation | DSpark |
|---|---|---|
| 4 × B300 | 1,340 tok/s | 1,230 tok/s |
| 2 × B300 | 670 tok/s | 620 tok/s |

There are three cautions:

1. The TokenSpeed figures come from long agentic sessions (the average prompt is 94k tokens). In
   such sessions, attention adds work that the ceiling does not include.
2. TokenSpeed does not publish its acceptance length. Thus its DSpark row assumes the vLLM value of
   3.5.
3. The ceiling assumes independent routing.

With those cautions, the published engines run at 20% to
49% of the bandwidth ceiling on four GPUs. The
TokenSpeed team says the same thing in words: "single-user decode is bound by kernel launches and
dependency waits rather than math" (`tokenspeed-blog`). Our DeepSeek-V4.1 study found the same
result, and here the authors of the engine state it. For this reason, the same-GPU software
counterfactual (G1) stays in this study.

All published figures are in [`data/public_measurements.yaml`](../data/public_measurements.yaml).

## 10. What the checkpoint says that our starting notes did not

We started with a list of facts to verify and two approximate anchors. Most were correct. These were
not correct:

| starting note | what the sources say |
|---|---|
| The weights are NVFP4. An all-NVFP4 set is about 334 GB. | The checkpoint is **mixed**: NVFP4 experts in layers 0-39, MXFP8 experts in layers 40-47, BF16 in all other weights. It is 393.5 GB, 1.18x the anchor. |
| Batch-1 decode touches about 9.6 GB of weights per token. | **19.3 GB**, 2.0x the anchor. The causes are that the attention projections are BF16 and the experts of the last 8 layers are MXFP8. |
| No published throughput or latency figures exist. | **Several exist**, on disclosed hardware: 476 tok/s per user on 4 × B300 (`tokenspeed-blog`), about 630 on 4 × GB200 (`vllm-blog`), and throughput at four context lengths. Thus we can calibrate the GPU model and do not assume it. |
| (not in the notes) | The model includes a **speculative draft** (DSpark, 8 rows per step). Each platform in this study must model it. |
| Serving uses an NVFP4 KV-cache variant. | The announcement says so. The cache that the public engines expose is **MXFP8**. Not resolved. |
| The recommended deployment is 2 × B300. | The model card gives 2 × B300 as an example that meets the memory minimum. Each published speed is on **4 or 8 GPUs** at TP4 or TP8. |

The sources confirmed these notes as stated:

- 48 layers
- 8 of 384 experts
- mixed local and global attention
- context up to 1,048,576
- a 292M-parameter vision encoder
- trained in NVFP4 at each stage, with the MXFP8 checkpoint 1-2 points better on evals
- a minimum of 500 GB and 800 GB of aggregate VRAM for the two checkpoints
- open weights under Apache 2.0 since 12 Nov (the announcement of the model was on 5 Nov, with closed weights)
- the seven API hosts

## 11. What the serving engine's source adds (read at a pinned commit)

We read the TokenSpeed implementation of Inkling 2 Small line by line at commit `a439df4`
(tokenspeed-src). It changed these parts of the workload model:

| what the source shows | effect on the model |
|---|---|
| Each layer runs 17 kernel launches in a fixed order when its experts are NVFP4, and 16 when they are MXFP8. | We count the launches in the GPU model and do not fit them. Our operation graph has 937 per speculative step for one request. Our count from the source is about 977, accurate to about 15. |
| One request with the draft on verifies a maximum of 8 rows and skips the two all-reduces in each layer. Each producer writes its partial into the mailbox of each rank. Two or more requests, or no draft, run two all-reduces per layer. | The builder has three collective regimes. A single request without the draft runs 96 standalone all-reduces per step. With the draft, it runs none in the layers. |
| The engine puts each expert whole on one rank (rank r owns experts 96r to 96r+95). It replicates the inputs. | Expert parallelism with the load imbalance that uniform or skewed routing causes. |
| The draft reads the output head a second time and runs 7 sequential rounds. Each round ends in a distributed argmax. | Draft operations in the step: a second head read and 7 serial rounds, each with a collective. |
| The KV cache is BF16 or MXFP8. There is no NVFP4 KV cache in the source. | The NVFP4 KV variant that the announcement mentions stays a sensitivity. |
| Source comments give measured times: one full-attention kernel at 8K, 32K and 128K keys, and a prefill forward at 2K and 3K tokens on 4x B300. | The prefill times are calibration points. The attention-kernel times are a held-out check. The model fails this check at long context ([06](06_simulator.md)). |

## 12. What happened to the open items

1. **Read the engine source.** Done for TokenSpeed (above). We used the vLLM and SGLang
   implementations only through their published recipes and blog figures.
2. **The 1.24x KV gap against vLLM.** We did not find the cause. Our bytes per token agree
   exactly with the published TokenSpeed figure.
3. **An NVFP4 KV cache.** We did not find one in a public source.
4. **Dequantize one real expert.** Done as part of the reference model of the circuit block ([07](07_silicon.md)).
5. **Routing correlation.** We set a low limit and a high limit with published statistics from four
   open models ([04](04_results.md), section 10).
