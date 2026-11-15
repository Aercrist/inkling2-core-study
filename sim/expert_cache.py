"""Expert-cache study: can on-chip SRAM cache routed experts across decode steps?

No public routing traces exist for DeepSeek-V4.1-Flash, so traces here are SYNTHETIC and the
study is a bound, not a prediction. The generator is a sticky-Zipf process per layer:
each of a step's k experts is, with probability `stay`, one of the previous step's experts,
else drawn from a Zipf(s) popularity over a random permutation of the E experts. stay = 0 and
s = 0 is independent uniform routing (what the auxiliary-loss-free balancer targets in
aggregate).

Policies (the classic cache-replacement menu, applied to expert slots):
  * LRU, LFU: realizable online;
  * Belady/MIN: offline optimum (needs the future) -> an upper bound for any policy;
  * Che approximation for LRU under the independent reference model, as a cross-check.
"""
from __future__ import annotations

import heapq
from collections import OrderedDict

import numpy as np


def sticky_zipf_trace(steps: int, E: int, k: int, rows: int, stay: float, s: float, seed: int = 0) -> list[list[int]]:
    """Per step, the distinct experts requested by `rows` token-rows."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(E)
    w = 1.0 / np.arange(1, E + 1) ** s
    w = w / w.sum()
    prev: list[int] = list(rng.choice(E, k, replace=False, p=w[np.argsort(perm)]))
    pop = w[np.argsort(perm)]
    trace = []
    for _ in range(steps):
        need: set[int] = set()
        for _r in range(rows):
            cur: list[int] = []
            for _j in range(k):
                if prev and rng.random() < stay:
                    e = int(prev[rng.integers(len(prev))])
                else:
                    e = int(rng.choice(E, p=pop))
                if e not in cur:
                    cur.append(e)
            while len(cur) < k:
                e = int(rng.choice(E, p=pop))
                if e not in cur:
                    cur.append(e)
            need.update(cur)
            prev = cur
        trace.append(sorted(need))
    return trace


def hit_rate(trace: list[list[int]], capacity: int, policy: str = "lru") -> float:
    if capacity <= 0:
        return 0.0
    flat = [e for step in trace for e in step]
    hits = 0
    if policy == "lru":
        c: OrderedDict[int, None] = OrderedDict()
        for e in flat:
            if e in c:
                hits += 1
                c.move_to_end(e)
            else:
                c[e] = None
                if len(c) > capacity:
                    c.popitem(last=False)
    elif policy == "lfu":
        cnt: dict[int, int] = {}
        cache: set[int] = set()
        for e in flat:
            cnt[e] = cnt.get(e, 0) + 1
            if e in cache:
                hits += 1
            else:
                cache.add(e)
                if len(cache) > capacity:
                    victim = min((x for x in cache if x != e), key=lambda x: cnt[x])
                    cache.remove(victim)
    elif policy == "opt":  # Belady MIN
        nxt = [0] * len(flat)
        last: dict[int, int] = {}
        for i in range(len(flat) - 1, -1, -1):
            nxt[i] = last.get(flat[i], 1 << 60)
            last[flat[i]] = i
        cache = {}
        heap: list[tuple[int, int]] = []
        for i, e in enumerate(flat):
            if e in cache:
                hits += 1
            elif len(cache) >= capacity:
                while True:
                    negn, v = heapq.heappop(heap)
                    if v in cache and cache[v] == -negn:
                        del cache[v]
                        break
            cache[e] = nxt[i]
            heapq.heappush(heap, (-nxt[i], e))
    else:
        raise ValueError(policy)
    return hits / len(flat)


def che_lru_hit(popularity: np.ndarray, capacity: int) -> float:
    """Che's approximation: LRU hit rate under IRM with request probabilities `popularity`."""
    p = popularity / popularity.sum()
    if capacity >= len(p):
        return 1.0
    lo, hi = 0.0, 1e9
    for _ in range(200):  # solve sum(1 - exp(-p_i T)) = C for the characteristic time T
        t = (lo + hi) / 2
        if np.sum(1 - np.exp(-p * t)) < capacity:
            lo = t
        else:
            hi = t
    return float(np.sum(p * (1 - np.exp(-p * lo))))
