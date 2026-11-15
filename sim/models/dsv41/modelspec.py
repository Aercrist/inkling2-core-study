"""DeepSeek-V4.1-Flash model description.

Shapes come from the released `inference/config.json`; bytes come from the safetensors
headers of the released checkpoint (data/checkpoint_manifest.csv), so weight traffic in the
simulator is the on-disk format the serving engine loads (FP4 experts with E8M0 scales per
32 elements, FP8 dense with E8M0 scales per 32x32 block, BF16 head/gate/compressor).
"""
from __future__ import annotations

import csv
import json
import pathlib
from dataclasses import dataclass, field
from functools import cached_property

ROOT = pathlib.Path(__file__).resolve().parents[3]


@dataclass
class Tensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    count: int
    total_bytes: int

    @property
    def bytes_each(self) -> float:
        return self.total_bytes / self.count

    @property
    def params_each(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        # I8-packed FP4: two values per byte
        return 2 * n if self.dtype == "I8" else n


@dataclass
class ModelSpec:
    cfg: dict
    tensors: dict[str, Tensor] = field(repr=False)

    @classmethod
    def load(cls, config_path: str | None = None, manifest_path: str | None = None) -> "ModelSpec":
        cp = pathlib.Path(config_path or ROOT / "data/dsv41/deepseek_v41_flash_inference_config.json")
        mp = pathlib.Path(manifest_path or ROOT / "data/dsv41/checkpoint_manifest.csv")
        cfg = json.loads(cp.read_text())
        tensors: dict[str, Tensor] = {}
        with open(mp) as f:
            for r in csv.DictReader(f):
                shape = tuple(int(x) for x in r["shape"].split("x")) if r["shape"] else ()
                key = r["tensor"]
                if key in tensors:  # the two engram tables differ by a few rows; merge
                    t = tensors[key]
                    t.count += int(r["count"])
                    t.total_bytes += int(r["total_bytes"])
                    continue
                tensors[key] = Tensor(key, r["dtype"], shape, int(r["count"]), int(r["total_bytes"]))
        return cls(cfg, tensors)

    # ---- raw accessors -------------------------------------------------------------------
    def __getattr__(self, k):  # config passthrough: spec.dim, spec.n_layers, ...
        cfg = self.__dict__.get("cfg")
        if cfg is not None and k in cfg:
            return cfg[k]
        raise AttributeError(k)

    def wbytes(self, prefix: str) -> float:
        """Per-instance bytes of `<prefix>.weight` plus `<prefix>.scale` (if any)."""
        b = self.tensors[prefix + ".weight"].bytes_each
        if prefix + ".scale" in self.tensors:
            b += self.tensors[prefix + ".scale"].bytes_each
        return b

    def params(self, prefix: str) -> int:
        return self.tensors[prefix + ".weight"].params_each

    # ---- derived per-instance byte counts --------------------------------------------------
    @cached_property
    def expert_bytes(self) -> float:
        """One routed expert (w1, w2, w3; FP4 + E8M0 scales)."""
        p = "layers.*.ffn.experts.*"
        return sum(self.wbytes(f"{p}.{w}") for w in ("w1", "w2", "w3"))

    def expert_layout_factor(self, tp: int) -> float:
        """Bytes the engine actually streams for one routed-expert shard / on-disk shard bytes.

        deepseek-ultrafast's `sgl` layout (default on sm_103) pads each rank's slice, but the
        routed-expert tensor maps (model.rs::sgl_init) read w13 over only `2*inter` rows and w2
        rows of `ceil64(inter)/2` bytes, the rest being out-of-bounds fill. So only w2's row
        padding is streamed: TP4 576 -> 576 (+0%), TP8 288 -> 320 on w2 (about +3.5% overall)."""
        if tp <= 0:
            return 1.0
        dim, inter = self.dim, self.moe_inter_dim // tp
        w = 2 * inter * dim // 2 + dim * (-(-inter // 64) * 64) // 2
        scales = 3 * dim * inter / 32  # E8M0 per 32, as on disk
        return (w + scales) / (self.expert_bytes / tp)

    def shared_layout_factor(self, tp: int) -> float:
        """The shared expert's FP8 copy is read over the padded width (tensor maps span 2*pad
        and pad): TP4 576 -> 640 (+11%), TP8 288 -> 384 (+33%) of its bytes."""
        if tp <= 0:
            return 1.0
        inter = self.moe_inter_dim // tp
        return (-(-inter // 128) * 128) / inter

    @cached_property
    def mtp_expert_bytes(self) -> float:
        p = "mtp.*.ffn.experts.*"
        return sum(self.wbytes(f"{p}.{w}") for w in ("w1", "w2", "w3"))

    @cached_property
    def shared_expert_bytes(self) -> float:
        p = "layers.*.ffn.shared_experts"
        return sum(self.wbytes(f"{p}.{w}") for w in ("w1", "w2", "w3"))

    def attn_bytes(self, prefix: str = "layers.*") -> dict[str, float]:
        p = prefix + ".attn"
        return {k: self.wbytes(f"{p}.{k}") for k in ("wq_a", "wkv", "wq_b", "wo_a", "wo_b")}

    @cached_property
    def kv_entry_bytes(self) -> dict[str, float]:
        """Bytes per cached entry (tech report 2.4.4 and the reference model):
        main (compressed) KV: 512 ch FP4 E2M1 + one E4M3 scale / 16 ch -> 288 B
        SWA KV: 512 ch FP8 + E8M0 scale / 32 ch -> 528 B
        indexer K: 128 ch FP4 + E8M0 scale / 32 ch -> 68 B
        """
        hd, ihd = self.head_dim, self.index_head_dim
        return {"main": hd / 2 + hd / 16, "swa": hd + hd / 32, "index": ihd / 2 + ihd / 32}

    @cached_property
    def global_kv_bytes_per_token(self) -> float:
        """Global (context-length-proportional) KV per token: one main-KV + one index-K stream
        per KV-source layer, each divided by that layer's compression ratio."""
        e = self.kv_entry_bytes
        tot = 0.0
        for l in self.kv_source_layers:
            r = self.compress_ratios[l]
            tot += (e["main"] + e["index"]) / r
        return tot

    # ---- parameter counts (for validation against the paper) -------------------------------
    @cached_property
    def param_counts(self) -> dict[str, float]:
        L = self.n_layers
        attn = sum(self.params(f"layers.*.attn.{k}") for k in ("wq_a", "wkv", "wq_b", "wo_a", "wo_b")) * L
        expert_each = sum(self.params(f"layers.*.ffn.experts.*.{w}") for w in ("w1", "w2", "w3"))
        experts = expert_each * self.n_routed_experts * L
        shared = sum(self.params(f"layers.*.ffn.shared_experts.{w}") for w in ("w1", "w2", "w3")) * L
        emb = self.params("embed") + self.params("head")
        gate = self.params("layers.*.ffn.gate") * L
        idx = sum(t.params_each * t.count for k, t in self.tensors.items() if ".indexer." in k and k.endswith(".weight"))
        comp = sum(t.params_each * t.count for k, t in self.tensors.items() if ".compressor." in k and k.endswith(".weight"))
        hc = sum(t.params_each * t.count for k, t in self.tensors.items() if k.startswith("layers.*.hc_"))
        eng_tables = sum(t.params_each * t.count for k, t in self.tensors.items() if k == "layers.*.engram.embed.weight")
        eng_proj = self.params("layers.*.engram.wkv") * len(self.engram_layer_ids)
        backbone = attn + experts + shared + emb + gate + idx + comp + hc + eng_proj
        act_decode = (attn + shared + gate + idx + comp + eng_proj) + expert_each * self.n_activated_experts * L + self.params("head")
        enc_layers = L // 2
        act_prefill = (attn + shared + gate) * enc_layers / L + expert_each * self.n_activated_experts * enc_layers + eng_proj
        return {
            "attention": attn, "routed_experts": experts, "shared_experts": shared, "embed_head": emb,
            "gate": gate, "indexer": idx, "compressor": comp, "mhc": hc, "engram_proj": eng_proj,
            "backbone_total": backbone, "engram_tables": eng_tables,
            "activated_decode": act_decode, "activated_prefill": act_prefill,
        }

    # ---- layer roles (tech report 4.2.1) ---------------------------------------------------
    def layer_role(self, l: int) -> str:
        """'swa' | 'full' | 'reindex' | 'reuse' for backbone layer l."""
        if self.compress_ratios[l] == 0:
            return "swa"
        if l in self.kv_source_layers:
            return "full"
        if l in self.index_source_layers:
            return "reindex"
        return "reuse"

    def index_scan_entries(self, l: int, ctx: int) -> int:
        """Entries a layer's indexer scores per query at context length ctx."""
        role = self.layer_role(l)
        if role not in ("full", "reindex"):
            return 0
        r = self.compress_ratios[l]
        n = ctx // r
        if role == "reindex" and self.candidate_source_layer < l:
            n = min(n, self.candidate_topk_blocks * self.candidate_block_size)
        return n

    def attended_entries(self, l: int, ctx: int) -> int:
        """KV entries one query attends to in layer l: SWA window + top-k compressed."""
        win = min(self.window_size, ctx)
        if self.compress_ratios[l] == 0:
            return win
        return win + min(self.index_topk, ctx // self.compress_ratios[l])
