"""The adversarial reviews: findings and what was done (data/reviews.yaml), and how far the headline
moved between the snapshot taken before the reviews (data/pre_review_headline.json) and now.
Writes results/reviews.json. Does nothing if no review has been logged.
"""
from __future__ import annotations

import json

import yaml

from experiments.common import RESULTS, save_json
from experiments.render_docs import ROOT

METRICS = ("speed_batch1", "tokens_per_joule_batch1", "tokens_per_joule_best", "cost_best")


def _ratios(snap: dict, pair: str) -> dict:
    return snap.get(pair) or snap.get(pair.replace("/g0", "/g0_x4"))


def main() -> dict | None:
    """Snapshots: data/pre_review_headline.json (before review 1), data/after_review_1_headline.json, ...; now is after the last."""
    f = ROOT / "data" / "reviews.yaml"
    if not f.exists():
        return None
    log = yaml.safe_load(f.read_text())
    n = len(log.get("reviews", []))
    snaps = [json.loads((ROOT / "data" / "pre_review_headline.json").read_text())]
    for k in range(1, n):
        snaps.append(json.loads((ROOT / "data" / f"after_review_{k}_headline.json").read_text()))
    now = json.loads((RESULTS / "inkling.json").read_text())["headline"]["ratios"]
    snaps.append(now)
    pair = "s/g0"
    stages = []
    for k in range(n + 1):
        stages.append({"stage": "before the reviews" if k == 0 else f"after review {k}", **{mt: _ratios(snaps[k], pair)[mt] for mt in METRICS}})
    change = {}
    for mt in METRICS:
        a, b = stages[0][mt], stages[-1][mt]
        change[f"s/g0:{mt}"] = {"before": a, "after": b, "factor": max(a / b, b / a)}
    per_review = []
    for k in range(1, n + 1):
        per_review.append({"review": k, **{mt: max(stages[k - 1][mt] / stages[k][mt], stages[k][mt] / stages[k - 1][mt]) for mt in METRICS}})
    worst = max(change, key=lambda k: change[k]["factor"])
    out = {"label": "review log", "reviews": log.get("reviews", []), "n_reviews": n, "findings": log["findings"], "n_findings": len(log["findings"]),
           "stages": stages, "per_review": per_review, "headline_change": change,
           "largest_headline_change": change[worst]["factor"], "largest_headline_change_in": worst}
    save_json(out, RESULTS / "reviews.json")
    print(f"{n} reviews, {len(log['findings'])} findings; largest headline change {change[worst]['factor']:.2f}x in {worst}")
    return out


if __name__ == "__main__":
    main()
