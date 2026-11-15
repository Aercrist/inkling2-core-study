#!/usr/bin/env python3
"""Fetch a few real NVFP4 expert slabs of Inkling 2 Small with HTTP range requests.

No shard is downloaded. A safetensors file is an 8-byte little-endian header length N, N bytes
of JSON header, then tensor data; `data_offsets` in the header are relative to byte 8 + N. The
headers are already cached in data/headers_cache/ (tools/fetch_checkpoint_manifest.py), so each
slab below is one or a few `Range:` requests against the pinned revision on the Hub.

What is fetched (defaults; 27.7 MB of range requests in 55 slabs, 22 MB committed):

* one WHOLE expert of one NVFP4 layer: packed E2M1 codes, E4M3 block scales, FP32 amax and the
  FP32 outer-scale denominator, for both projections (w13: 4608 x 4608, w2: 4608 x 2304);
* row samples (64 consecutive rows of w13 and of w2, with their scales) from other experts and
  other NVFP4 layers, so that results do not rest on one expert;
* the amax of all 384 experts of each touched layer (1.5 kB each);
* small real tensors used as activation sources and for the layout check: that layer's router
  weight reduced to a per-input-column RMS, its `mlp_norm.weight`, `embed_norm.weight`, and a few
  hundred rows of the token embedding (BF16).

Outputs, under data/fixtures/:

    inkling2_small_L00_E000.npz   the whole expert
    inkling2_small_rows.npz       row samples
    inkling2_small_aux.npz        embedding rows, norms, router column RMS, per-layer amax
    manifest.json                 URL, revision, byte ranges and SHA-256 of every slab fetched

    python tools/fetch_expert_fixture.py                    # the committed fixture set
    python tools/fetch_expert_fixture.py --cache-dir /tmp/x # keep the raw slabs for re-runs
    python tools/fetch_expert_fixture.py --verify           # re-fetch and compare SHA-256 only

Network only; not part of `make reproduce`. Nothing here costs money.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import sys
import time
import urllib.error
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
HUB = "https://huggingface.co"
REPO = "thinkingmachines/Inkling-2-Small"
REVISION = "8361e7a31ae7e9966a49a91d3994fde618936c2b"
UA = "expert-fixture-fetch/1.0"
CHUNK = 8 << 20  # bytes per range request


class Shard:
    """One safetensors file on the Hub, read with range requests."""

    def __init__(self, name: str, repo: str, revision: str, header_dir: pathlib.Path,
                 cache_dir: pathlib.Path | None, log: list):
        self.name, self.repo, self.revision = name, repo, revision
        self.url = f"{HUB}/{repo}/resolve/{revision}/{name}"
        self.cache_dir, self.log = cache_dir, log
        cached = header_dir / f"{name}.json"
        if cached.exists():
            raw = cached.read_bytes()
        else:  # the header itself is two small range requests
            n = struct.unpack("<Q", self._get(0, 8))[0]
            raw = self._get(8, 8 + n)
        self.header_bytes = len(raw)
        self.header = json.loads(raw)
        self.data_start = 8 + self.header_bytes

    def _get(self, start: int, stop: int) -> bytes:
        """Bytes [start, stop) of the file, with retries; checks Content-Range and length."""
        last = None
        for attempt in range(6):
            try:
                req = urllib.request.Request(
                    self.url, headers={"Range": f"bytes={start}-{stop - 1}", "User-Agent": UA})
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = r.read()
                    cr = r.headers.get("Content-Range", "")
                if r.status != 206 or not cr.startswith(f"bytes {start}-{stop - 1}/"):
                    raise IOError(f"unexpected range reply: {r.status} {cr!r}")
                if len(data) != stop - start:
                    raise IOError(f"short read: {len(data)} of {stop - start}")
                return data
            except (urllib.error.URLError, IOError, TimeoutError, ConnectionError) as e:
                last = e
                time.sleep(2 + 3 * attempt)
        raise IOError(f"range {start}-{stop} of {self.name} failed: {last}")

    def read(self, tensor: str, elem_start: int, elem_count: int, itemsize: int) -> bytes:
        """`elem_count` elements of `tensor` starting at flat element `elem_start`."""
        meta = self.header[tensor]
        t0, t1 = meta["data_offsets"]
        a = self.data_start + t0 + elem_start * itemsize
        b = a + elem_count * itemsize
        if b > self.data_start + t1:
            raise ValueError(f"{tensor}: range past the end of the tensor")
        key = hashlib.sha256(f"{self.revision}/{self.name}/{a}-{b}".encode()).hexdigest()[:24]
        cpath = self.cache_dir / f"{key}.bin" if self.cache_dir else None
        if cpath and cpath.exists() and cpath.stat().st_size == b - a:
            data = cpath.read_bytes()
        else:
            parts = [self._get(p, min(p + CHUNK, b)) for p in range(a, b, CHUNK)]
            data = b"".join(parts)
            if cpath:
                cpath.write_bytes(data)
        self.log.append({"file": self.name, "tensor": tensor, "dtype": meta["dtype"],
                         "shape": meta["shape"], "elem_start": elem_start, "elem_count": elem_count,
                         "byte_range": [a, b], "sha256": hashlib.sha256(data).hexdigest()})
        return data


def tensor_file(manifest_csv: pathlib.Path, tensor: str) -> str:
    for line in manifest_csv.read_text().splitlines()[1:]:
        f, name = line.split(",")[:2]
        if name == tensor:
            return f
    raise KeyError(tensor)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default=REVISION, help="commit SHA; pinned, never 'main'")
    ap.add_argument("--layer", type=int, default=0, help="NVFP4 layer of the whole expert(s)")
    ap.add_argument("--experts", default="0", help="comma list of whole experts of --layer")
    ap.add_argument("--row-samples", default="0:77,0:211,0:383,13:5,26:100,39:300",
                    help="comma list of layer:expert, each contributing --rows consecutive rows")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--embed-chunks", type=int, default=8, help="runs of consecutive token ids")
    ap.add_argument("--embed-chunk-rows", type=int, default=64)
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "fixtures"))
    ap.add_argument("--header-dir", default=str(ROOT / "data" / "headers_cache"))
    ap.add_argument("--manifest-csv", default=str(ROOT / "data" / "inkling2_small" / "checkpoint_tensors.csv"))
    ap.add_argument("--config", default=str(ROOT / "data" / "inkling2_small" / "config.json"))
    ap.add_argument("--cache-dir", help="keep raw slabs here (re-runs then need no network)")
    ap.add_argument("--verify", action="store_true",
                    help="fetch again and compare with the SHA-256 in manifest.json; write nothing")
    a = ap.parse_args()

    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = pathlib.Path(a.cache_dir) if a.cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(pathlib.Path(a.config).read_text())["text_config"]
    hidden, inter, n_exp = cfg["hidden_size"], cfg["intermediate_size"], cfg["n_routed_experts"]
    vocab = cfg["vocab_size"]
    blk = 16
    # w13: 2*inter output rows over `hidden` inputs; w2: `hidden` output rows over `inter` inputs
    geom = {"w13": (2 * inter, hidden), "w2": (hidden, inter)}

    log: list = []
    shards: dict[str, Shard] = {}

    def shard_of(tensor: str) -> Shard:
        f = tensor_file(pathlib.Path(a.manifest_csv), tensor)
        if f not in shards:
            shards[f] = Shard(f, a.repo, a.revision, pathlib.Path(a.header_dir), cache, log)
        return shards[f]

    def expect(tensor: str, dtype: str, shape: list[int]) -> Shard:
        s = shard_of(tensor)
        meta = s.header[tensor]
        if meta["dtype"] != dtype or meta["shape"] != shape:
            raise ValueError(f"{tensor}: header says {meta['dtype']} {meta['shape']}, expected {dtype} {shape}")
        return s

    def expert_rows(layer: int, expert: int, leaf: str, row0: int, nrows: int) -> dict:
        """Rows [row0, row0+nrows) of one expert's projection: codes, block scales, outer scale."""
        rows, width = geom[leaf]
        p = f"model.llm.layers.{layer}.mlp.experts.{leaf}_weight"
        wb, sb = width // 2, width // blk
        w = expect(p, "U8", [n_exp, rows, wb]).read(p, (expert * rows + row0) * wb, nrows * wb, 1)
        s = expect(p + "_scale", "F8_E4M3", [n_exp, rows, sb]).read(
            p + "_scale", (expert * rows + row0) * sb, nrows * sb, 1)
        am = expect(p + "_amax", "F32", [n_exp]).read(p + "_amax", 0, n_exp, 4)
        dn = expect(p + "_outer_scale_denominator", "F32", [1]).read(p + "_outer_scale_denominator", 0, 1, 4)
        amax = np.frombuffer(am, "<f4")
        return {
            f"{leaf}": np.frombuffer(w, np.uint8).reshape(nrows, wb),
            f"{leaf}_scale": np.frombuffer(s, np.uint8).reshape(nrows, sb),
            f"{leaf}_amax": amax[expert].copy(),
            f"{leaf}_denominator": np.frombuffer(dn, "<f4")[0].copy(),
            f"{leaf}_amax_all_experts": amax.copy(),
        }

    t_start = time.time()
    files: dict[str, dict] = {}

    # 1. whole experts
    whole = [int(e) for e in a.experts.split(",") if e != ""]
    for e in whole:
        d: dict = {}
        for leaf in ("w13", "w2"):
            d.update(expert_rows(a.layer, e, leaf, 0, geom[leaf][0]))
        d["layer"], d["expert"] = np.int32(a.layer), np.int32(e)
        files[f"inkling2_small_L{a.layer:02d}_E{e:03d}.npz"] = d
        print(f"layer {a.layer} expert {e}: whole expert fetched ({time.time() - t_start:.0f} s)", flush=True)

    # 2. row samples from other experts and layers
    rows_d: dict = {}
    samples = []
    for item in [s for s in a.row_samples.split(",") if s]:
        layer, e = (int(v) for v in item.split(":"))
        tag = f"L{layer:02d}_E{e:03d}"
        for leaf in ("w13", "w2"):
            nrows_total = geom[leaf][0]
            row0 = ((e * 37 + layer * 101) % (nrows_total - a.rows)) & ~1  # even: gate/up pairs stay aligned
            r = expert_rows(layer, e, leaf, row0, a.rows)
            for k, v in r.items():
                if not k.endswith("_all_experts"):
                    rows_d[f"{tag}_{k}"] = v
            rows_d[f"{tag}_{leaf}_row0"] = np.int32(row0)
        samples.append([layer, e])
        print(f"layer {layer} expert {e}: {a.rows} rows of w13 and w2 fetched", flush=True)
    rows_d["samples"] = np.array(samples, np.int32).reshape(-1, 2)
    files["inkling2_small_rows.npz"] = rows_d

    # 3. auxiliary real tensors: activation sources and the layout cross-check
    aux: dict = {}
    L = a.layer
    gname = f"model.llm.layers.{L}.mlp.gate.weight"
    g = np.frombuffer(expect(gname, "BF16", [n_exp, hidden]).read(gname, 0, n_exp * hidden, 2), "<u2")
    aux["gate_weight_col_rms"] = np.sqrt((bf16_to_f32(g).reshape(n_exp, hidden).astype(np.float64) ** 2).mean(0)).astype(np.float32)
    for key, name in (("mlp_norm_weight", f"model.llm.layers.{L}.mlp_norm.weight"),
                      ("embed_norm_weight", "model.llm.embed_norm.weight")):
        aux[key + "_bf16"] = np.frombuffer(expect(name, "BF16", [hidden]).read(name, 0, hidden, 2), "<u2").copy()
    ename = "model.llm.embed.weight"
    es = expect(ename, "BF16", [vocab, hidden])
    ids, emb = [], []
    for c in range(a.embed_chunks):
        # runs of consecutive ids spread over the vocabulary, away from the first (special/byte) ids
        t0 = 1000 + (c * (vocab - 2000 - a.embed_chunk_rows)) // max(a.embed_chunks - 1, 1)
        emb.append(np.frombuffer(es.read(ename, t0 * hidden, a.embed_chunk_rows * hidden, 2), "<u2")
                   .reshape(a.embed_chunk_rows, hidden))
        ids.extend(range(t0, t0 + a.embed_chunk_rows))
    aux["embed_rows_bf16"] = np.concatenate(emb, 0)
    aux["embed_token_ids"] = np.array(ids, np.int32)
    aux["layer"] = np.int32(L)
    for layer in sorted({L} | {s[0] for s in samples}):
        for leaf in ("w13", "w2"):
            p = f"model.llm.layers.{layer}.mlp.experts.{leaf}_weight_amax"
            aux[f"L{layer:02d}_{leaf}_amax_all_experts"] = np.frombuffer(
                expect(p, "F32", [n_exp]).read(p, 0, n_exp, 4), "<f4").copy()
    files["inkling2_small_aux.npz"] = aux
    print(f"aux tensors fetched ({time.time() - t_start:.0f} s)", flush=True)

    manifest = {
        "repo": a.repo, "revision": a.revision,
        "url_template": f"{HUB}/{a.repo}/resolve/{a.revision}/<file>",
        "block_size": blk, "hidden_size": hidden, "intermediate_size": inter, "n_routed_experts": n_exp,
        "whole_experts": [[a.layer, e] for e in whole], "row_samples": samples, "rows_per_sample": a.rows,
        "note": ("data_offsets are relative to byte 8 + header length; byte_range is absolute in the file. "
                 "gate_weight_col_rms is derived (RMS over the 384 router rows); everything else is stored bytes."),
        "slabs": dedup(log),
    }
    manifest["bytes_fetched"] = sum(s["byte_range"][1] - s["byte_range"][0] for s in manifest["slabs"])

    if a.verify:
        old = json.loads((out / "manifest.json").read_text())
        same = {(s["file"], *s["byte_range"]): s["sha256"] for s in old["slabs"]} == \
               {(s["file"], *s["byte_range"]): s["sha256"] for s in manifest["slabs"]}
        print("verify:", "identical to manifest.json" if same else "DIFFERENT from manifest.json")
        return 0 if same else 1

    total = 0
    for name, d in files.items():
        np.savez_compressed(out / name, **d)
        size = (out / name).stat().st_size
        total += size
        manifest.setdefault("files", {})[name] = {
            "bytes": size, "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest(),
            "arrays": {k: [str(np.asarray(v).dtype), list(np.asarray(v).shape)] for k, v in d.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{manifest['bytes_fetched']:,} bytes fetched in {len(manifest['slabs'])} slabs; "
          f"{total:,} bytes written to {out} ({time.time() - t_start:.0f} s)")
    return 0


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    """BF16 is the top 16 bits of an IEEE-754 single."""
    return (u16.astype(np.uint32) << 16).view(np.float32)


def dedup(log: list) -> list:
    seen, out = set(), []
    for s in log:
        k = (s["file"], *s["byte_range"])
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


if __name__ == "__main__":
    sys.exit(main())
