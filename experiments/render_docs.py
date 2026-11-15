"""Render README.md and docs/*.md from docs/templates/*.j2, results/*.json and data/*.yaml, so every
number in the prose is the number a script produced. Same approach as the DeepSeek-V4.1 study.

Predictions (data/preregistration.yaml) and published figures (data/public_measurements.yaml) are
inputs typed once, with a source or a stated basis; everything else comes from results/*.json.
"""
from __future__ import annotations

import json
import pathlib

import jinja2
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TEMPLATES = ROOT / "docs" / "templates"
OUTPUTS = {
    "README.md.j2": ROOT / "README.md",
    "00_hypotheses.md.j2": ROOT / "docs" / "00_hypotheses.md",
    "01_workload.md.j2": ROOT / "docs" / "01_workload.md",
    "02_design_rationale.md.j2": ROOT / "docs" / "02_design_rationale.md",
    "03_architecture.md.j2": ROOT / "docs" / "03_architecture.md",
    "04_results.md.j2": ROOT / "docs" / "04_results.md",
    "05_limitations.md.j2": ROOT / "docs" / "05_limitations.md",
    "06_simulator.md.j2": ROOT / "docs" / "06_simulator.md",
    "07_silicon.md.j2": ROOT / "docs" / "07_silicon.md",
    "napkin.md.j2": ROOT / "docs" / "napkin.md",
    "decisions.md.j2": ROOT / "docs" / "decisions.md",
    "CLAIMS.md.j2": ROOT / "CLAIMS.md",
}


def n(x, d=0):
    """Number with thousands separators and d decimals; d < 0 rounds to 10^-d."""
    if x is None:
        return "n/a"
    if d < 0:
        return f"{round(x, d):,.0f}"
    return f"{x:,.{d}f}"


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def gb(x, d=1):
    """Bytes as decimal gigabytes."""
    return f"{x / 1e9:,.{d}f} GB"


def mb(x, d=0):
    return f"{x / 1e6:,.{d}f} MB"


def bn(x, d=1):
    """A count as billions."""
    return f"{x / 1e9:,.{d}f}B"


def x(v, d=1):
    return f"{v:.{d}f}x"


def usd(v, d=0):
    """Dollars with d decimals; d < 0 rounds to 10^-d."""
    if d < 0:
        return f"${round(v, d):,.0f}"
    return f"${v:,.{d}f}"


def sig(v, k=3):
    """k significant digits, with thousands separators above 1,000."""
    if v is None:
        return "pending"
    if v == 0:
        return "0"
    import math
    d = max(0, k - 1 - int(math.floor(math.log10(abs(v)))))
    out = f"{v:,.{d}f}"
    return out.rstrip("0").rstrip(".") if "." in out else out


def rng(p):
    """Render a prediction's central value and range in its own unit."""
    def f(v):
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 100 or float(v).is_integer():
            return f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"
        return f"{v:g}"
    suffix = "x" if p["unit"] == "ratio" else ""
    return f"**{f(p['central'])}{suffix}** ({f(p['low'])}{suffix} to {f(p['high'])}{suffix})"


# One line on why each prediction that missed did so (docs/00_hypotheses.md). Text only; the numbers are rendered.
WHY = {
    "P04": "A is charged the latency a published SRAM machine shows per operation, which we did not anticipate.",
    "P05": "Same reason as P04: A's batch-1 energy edge over good GPU software disappears.",
    "P06": "A is much cheaper than a GPU for the same memory.",
    "P07": "The prediction counted storage only. The multiplier lanes, at their synthesized size, and the sessions' window cache add the rest.",
    "P13": "The predictions named 2x B300, which is weak at 100 tok/s per user.",
    "P14": "The predictions named 2x B300, which is weak at 100 tok/s per user. Against 4x B300 the cost gain is far smaller.",
    "P15": "Fixing a lane's weight folds the multipliers away, but the adder tree and the 32-bit accumulator remain and they are most of the block.",
    "P16": "Same reason as P15: the accumulator dominates and does not shrink.",
    "P17": "The synthesized lane made S larger; F's density comes from a shipped chip and did not change.",
    "P18": "At a busy point both spend most of their energy on the same arithmetic and attention.",
    "P20": "The GPUs' fastest layout for one user is 8x B300, not the 2x the prediction assumed, and both reviews lowered S's speed.",
    "P21": "We reasoned from dense models. A sparse model with a draft reads so little per token at small sizes that the GPU speeds up faster than the chip does.",
    "P22": "Same reason as P21.",
    "P23": "Same reason as P21.",
    "P24": "The window cache for 256 sessions alone needs more than eight dies; no rung in the scan fits.",
    "P25": "Same reason as P21.",
    "P26": "S takes a published machine's whole time per operation, which is slower than we assumed.",
    "P28": "The pooled LPDDR is faster than the 0.5 TB/s we assumed.",
    "P29": "Once prefill is counted in the fit, the GPUs attend about four times faster than we first inferred, so long context costs them less.",
    "P30": "S costs far more per system than we assumed and is not built for prefill.",
    "P31": "At high batch the corrected GPU fit is more efficient than we assumed.",
    "P32": "Public specifications do not predict these machines; see docs/06 for what we did with that.",
    "P34": "The first review alone moved the cost figure by more than the whole predicted range.",
}


def context() -> dict:
    W = json.loads((RESULTS / "phase0_workload.json").read_text())
    REP = json.loads((RESULTS / "phase0_dsv41_reproduction.json").read_text())
    PRE = yaml.safe_load((ROOT / "data" / "preregistration.yaml").read_text())
    PUB = yaml.safe_load((ROOT / "data" / "public_measurements.yaml").read_text())
    CONST = yaml.safe_load((ROOT / "data" / "phase0_constants.yaml").read_text())
    P = {p["id"]: p for p in PRE["predictions"]}
    M = {m["id"]: m for m in PUB["measurements"]}
    groups = {}
    for p in PRE["predictions"]:
        groups.setdefault(p["group"], []).append(p)
    L = W["layers"]
    # collapse the 48 layers into the four kinds that actually differ
    kinds = {}
    for l in L:
        k = (l["attention"], l["expert_precision"])
        e = kinds.setdefault(k, {"attention": l["attention"], "expert_precision": l["expert_precision"], "layers": [], **l})
        e["layers"].append(l["layer"])
    def load(name):
        f = RESULTS / name
        return json.loads(f.read_text()) if f.exists() else None
    I, U, CG, CL, PR, SI, RV = (load(f) for f in ("inkling.json", "uncertainty.json", "calibration_gpu.json", "calibration_llama.json",
                                                  "predictions.json", "silicon.json", "reviews.json"))
    NK = load("napkin.json")
    from sim.hardware import load_yaml
    S_HW, F_HW, G_HW, A_HW = (load_yaml(f"configs/hw/{f}.yaml") for f in ("s_mds1", "f_mdf1", "gpu_b300", "a_mda1"))
    LL = yaml.safe_load((ROOT / "data" / "published_llama31_8b.yaml").read_text())
    return {"W": W, "REP": REP, "PRE": PRE, "P": P, "PG": groups, "PUB": PUB, "M": M, "CONST": CONST,
            "KINDS": list(kinds.values()), "I": I, "H": I["headline"]["rows"], "R": I["headline"]["ratios"], "U": U,
            "MC": U["monte_carlo"], "TB": U["tornado"]["base"], "CG": CG, "CL": CL, "PR": PR,
            "PRD": {r["id"]: r for r in PR["rows"]}, "SI": SI, "RV": RV, "S_HW": S_HW, "F_HW": F_HW, "G_HW": G_HW, "A_HW": A_HW, "LL": LL, "NK": NK, "SE": I.get("synthesis_effect", {"available": False}),
            "SS": load("silicon_scaled.json"), "WHY": WHY,
            "LOO_REST": max(abs(r["error"]) for r in CG["loo"] if r["id"] != "tokenspeed_b300_c1_target_only")}


def main() -> None:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES), undefined=jinja2.StrictUndefined,
                             keep_trailing_newline=True, trim_blocks=True, lstrip_blocks=True)
    helpers = {"n": n, "pct": pct, "gb": gb, "mb": mb, "bn": bn, "x": x, "rng": rng, "usd": usd, "sig": sig}
    env.filters.update(helpers)
    env.globals.update(helpers)
    ctx = context()
    for name, out in OUTPUTS.items():
        if not (TEMPLATES / name).exists():
            continue
        text = env.get_template(name).render(**ctx)
        header = f"<!-- generated from docs/templates/{name} by experiments/render_docs.py; edit the template -->\n"
        out.write_text(header + text)
        print(f"rendered {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
