#!/usr/bin/env python3
"""Build a per-tensor checkpoint manifest from safetensors headers on the Hugging Face Hub.

Only the JSON header of each weight file is downloaded (two HTTP range requests per file),
never the weights. The manifest records file, tensor name, dtype, shape, stored elements and
on-disk bytes for every tensor. It is the ground truth the workload builder uses.

Generalized from tools/fetch_checkpoint_manifest.py in Aercrist/dsv41-accel-study, which was
specific to one repo and shard count. Not part of `make reproduce` (needs network); the
generated CSV and source record are committed.

    python tools/fetch_checkpoint_manifest.py \
        --repo thinkingmachines/Inkling-2-Small \
        --revision 8361e7a31ae7e9966a49a91d3994fde618936c2b \
        --out-dir data/inkling2_small
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import struct
import urllib.request
from math import prod

ROOT = pathlib.Path(__file__).resolve().parents[1]
HUB = "https://huggingface.co"


def _get(url: str, byte_range: str | None = None) -> bytes:
    headers = {"User-Agent": "checkpoint-manifest/1.0"}
    if byte_range:
        headers["Range"] = f"bytes={byte_range}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
        return r.read()


def fetch_header(url: str) -> dict:
    """Return the safetensors JSON header: 8-byte little-endian length, then that many bytes."""
    n = struct.unpack("<Q", _get(url, "0-7"))[0]
    return json.loads(_get(url, f"8-{7 + n}"))


def list_weight_files(repo: str, revision: str) -> list[dict]:
    tree = json.loads(_get(f"{HUB}/api/models/{repo}/tree/{revision}?recursive=true"))
    return sorted(
        ({"path": e["path"], "size": e["size"]} for e in tree
         if e["type"] == "file" and e["path"].endswith(".safetensors")),
        key=lambda e: e["path"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True, help="commit SHA; pin it, never 'main'")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", help="directory of already-downloaded <file>.json headers")
    ap.add_argument("--files-json", help="cached output of the Hub tree API (list of {path,size})")
    a = ap.parse_args()

    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = pathlib.Path(a.cache_dir) if a.cache_dir else None

    if a.files_json:
        tree = json.loads(pathlib.Path(a.files_json).read_text())
        files = sorted(({"path": e["path"], "size": e["size"]} for e in tree
                        if e.get("type", "file") == "file" and e["path"].endswith(".safetensors")),
                       key=lambda e: e["path"])
    else:
        files = list_weight_files(a.repo, a.revision)

    rows = []
    for f in files:
        cached = cache / f"{f['path']}.json" if cache else None
        if cached and cached.exists():
            raw = cached.read_bytes()
            hdr = json.loads(raw)
            f["header_bytes"] = len(raw)
        else:
            url = f"{HUB}/{a.repo}/resolve/{a.revision}/{f['path']}"
            hdr = fetch_header(url)
            f["header_bytes"] = None
        data_bytes = 0
        for name, v in hdr.items():
            if name == "__metadata__":
                continue
            s, e = v["data_offsets"]
            rows.append((f["path"], name, v["dtype"], "x".join(map(str, v["shape"])),
                         prod(v["shape"]) if v["shape"] else 1, e - s))
            data_bytes += e - s
        f["tensor_bytes"] = data_bytes
        f["n_tensors"] = sum(1 for k in hdr if k != "__metadata__")

    rows.sort(key=lambda r: (r[0] != "mtp.safetensors", r[1]))
    with open(out / "checkpoint_tensors.csv", "w", newline="") as fo:
        w = csv.writer(fo, lineterminator="\n")
        w.writerow(["file", "tensor", "dtype", "shape", "elements", "bytes"])
        w.writerows(rows)

    source = {
        "repo": a.repo,
        "revision": a.revision,
        "hub_url": f"{HUB}/{a.repo}/tree/{a.revision}",
        "n_weight_files": len(files),
        "n_tensors": len(rows),
        "tensor_bytes": sum(r[5] for r in rows),
        "file_bytes": sum(f["size"] for f in files),
        "files": files,
    }
    (out / "checkpoint_source.json").write_text(json.dumps(source, indent=1) + "\n")
    print(f"{len(rows)} tensors in {len(files)} files, {source['tensor_bytes']:,} tensor bytes "
          f"({source['file_bytes']:,} on disk) -> {out}")


if __name__ == "__main__":
    main()
