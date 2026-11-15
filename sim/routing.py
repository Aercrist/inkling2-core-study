"""Expert-routing statistics used by the workload builder.

`distinct_experts` is the expected number of distinct experts touched when R token-rows each
select k of E experts. With independent uniform routing (the auxiliary-loss-free balancer
targets uniform long-run load) this is E * (1 - (1 - k/E)^R). Real consecutive tokens share
experts more often than that; `locality` in [0, 1] shrinks the effective number of
independent rows, R_eff = 1 + (R - 1) * (1 - locality). locality = 0 (default) is the
pessimistic case: most bytes moved per step, for every platform alike.
"""
from __future__ import annotations


def distinct_experts(rows: float, k: int, n_experts: int, locality: float = 0.0) -> float:
    if rows <= 0:
        return 0.0
    r_eff = 1.0 + (rows - 1.0) * (1.0 - locality)
    return n_experts * (1.0 - (1.0 - k / n_experts) ** r_eff)


def rows_per_expert(rows: float, k: int, n_experts: int, locality: float = 0.0) -> float:
    """Average token-rows served by each touched expert (sets GEMM M for an expert)."""
    d = distinct_experts(rows, k, n_experts, locality)
    return rows * k / d if d else 0.0


# ---- additions for this study (the functions above are unchanged from the DeepSeek-V4.1 study) ----

def inclusion_probs(k: int, n_experts: int, skew: float = 0.0) -> list[float]:
    """Probability that each expert is among a token's top-k, for Zipf-skewed popularity.

    Expert popularity is proportional to 1 / rank**skew; skew = 0 is uniform routing. The
    probabilities sum to k and none exceeds 1 (the excess of saturated experts is redistributed).
    """
    if skew <= 0:
        return [k / n_experts] * n_experts
    w = [1.0 / (r + 1) ** skew for r in range(n_experts)]
    p = [0.0] * n_experts
    free = list(range(n_experts))
    budget = float(k)
    while free:
        tot = sum(w[i] for i in free)
        over = [i for i in free if budget * w[i] / tot >= 1.0]
        if not over:
            for i in free:
                p[i] = budget * w[i] / tot
            break
        for i in over:
            p[i] = 1.0
        budget -= len(over)
        free = [i for i in free if i not in set(over)]
    return p


def distinct_experts_skewed(rows: float, k: int, n_experts: int, locality: float = 0.0, skew: float = 0.0) -> float:
    """Expected distinct experts touched by `rows` token-rows under skewed popularity and locality.

    Reduces to `distinct_experts` when skew = 0.
    """
    if rows <= 0:
        return 0.0
    r_eff = 1.0 + (rows - 1.0) * (1.0 - locality)
    return sum(1.0 - (1.0 - p) ** r_eff for p in inclusion_probs(k, n_experts, skew))


_EXPECTED_MAX_STD_NORMAL = {1: 0.0, 2: 0.5642, 3: 0.8463, 4: 1.0294, 6: 1.2672, 8: 1.4236, 16: 1.7660}


def ep_imbalance(distinct: float, n_experts: int, n_chips: int) -> float:
    """Busiest chip's share of routed-expert bytes over the average chip's, under expert parallelism.

    With experts stored whole on one chip each, the experts a step touches are spread unevenly:
    the count on one chip is about Binomial(E/n, d/E), and the step waits for the busiest chip.
    Returns E[max] / mean, at least 1.
    """
    if n_chips <= 1 or distinct <= 0:
        return 1.0
    p = min(1.0, distinct / n_experts)
    mean = distinct / n_chips
    sd = ((n_experts / n_chips) * p * (1.0 - p)) ** 0.5
    m = _EXPECTED_MAX_STD_NORMAL.get(n_chips)
    if m is None:  # asymptotic form for other group sizes
        import math
        m = (2.0 * math.log(n_chips)) ** 0.5
    return 1.0 + m * sd / mean
