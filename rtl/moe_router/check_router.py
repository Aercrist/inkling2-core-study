#!/usr/bin/env python3
"""Golden model + checker for rtl/moe_topk_router.sv.

`gen`  writes rtl/moe_router/build/keys.hex: N tokens x 384 signed 16-bit keys. Keys are fixed-point
       sqrtsoftplus(logit) + bias with logits ~ N(0, 1.5) and biases ~ N(0, 0.1) (Q6.10),
       plus a few forced ties to exercise the tie-break rule.
`check` compares rtl/moe_router/build/rtl_out.txt with a stable numpy top-6 (key desc, id asc) and
       reports the measured latency in cycles.
"""
import pathlib
import sys

import numpy as np

E, K, N = 384, 6, 256
B = pathlib.Path(__file__).resolve().parent / "build"


def keys() -> np.ndarray:
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 1.5, (N, E))
    bias = rng.normal(0, 0.1, E)
    score = np.sqrt(np.log1p(np.exp(logits)))
    q = np.round((score + bias) * 1024).astype(np.int64)
    q[:8, 100] = q[:8, 7]  # forced ties: id 7 must win over id 100
    return np.clip(q, -32768, 32767)


def golden(q: np.ndarray) -> np.ndarray:
    order = np.lexsort((np.tile(np.arange(E), (N, 1)), -q), axis=1)
    return order[:, :K]


def main() -> None:
    B.mkdir(exist_ok=True)
    q = keys()
    if sys.argv[1] == "gen":
        with open(B / "keys.hex", "w") as f:
            for v in q.reshape(-1):
                f.write(f"{int(v) & 0xFFFF:04x}\n")
        return
    rtl = np.loadtxt(B / "rtl_out.txt", dtype=np.int64).reshape(N, K)
    ref = golden(q)
    lat = np.loadtxt(B / "latency.txt", dtype=np.int64)
    bad = int((rtl != ref).any(axis=1).sum())
    print(f"router RTL vs golden: {N - bad}/{N} tokens match; latency {lat.min()}-{lat.max()} cycles "
          f"({lat.max() / 1.5:.0f} ns at 1.5 GHz)")
    (B / "router_summary.txt").write_text(f"match={N - bad}/{N}\nlatency_cycles={int(lat.max())}\n")
    import json
    (B.parent.parent.parent / "results" / "dsv41" / "router.json").write_text(json.dumps(
        {"match": f"{N - bad}/{N}", "latency_cycles": int(lat.max()), "latency_ns_at_1p5ghz": round(lat.max() / 1.5, 1),
         "tool": "Icarus Verilog (iverilog -g2012)", "experts": E, "top_k": K, "lanes": 8}, indent=1))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
