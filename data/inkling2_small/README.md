# Inkling 2 Small: checkpoint metadata

These files describe the checkpoint; none of them holds weights.

| file | what it is | origin |
|---|---|---|
| `config.json` | model config | copied unchanged from `thinkingmachines/Inkling-2-Small` at revision `8361e7a31ae7e9966a49a91d3994fde618936c2b` (Apache 2.0) |
| `hf_quant_config.json` | which tensors are NVFP4, MXFP8, BF16 or FP32, and the activation formats | copied unchanged from the same revision (Apache 2.0) |
| `checkpoint_tensors.csv` | file, name, dtype, shape, stored elements and bytes of every tensor | built from the safetensors headers by `tools/fetch_checkpoint_manifest.py` |
| `checkpoint_source.json` | repo, revision and size of every weight file | written by the same tool |
| `hub_model_info.json` | the Hub's own parameter counts, used as a cross-check | Hub API, read 2026-11-14 |

Inkling 2 Small is the work of Thinking Machines Lab. This study is independent and is not affiliated
with or endorsed by them.
