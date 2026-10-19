# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Stage 6 — sort, divide distance, divide order.

Semantics reproduce the modified MATLAB files exactly:

sort (sort.m): peel segments from the divide endpoints inward
(Scherler & Schwanghart 2020, Fig. 4). A junction becomes an active
endpoint once all but one of its incident segments are consumed. When no
active endpoints remain but segments do (closed rings, cyclic
components), an arbitrary remaining segment's start seeds a new wave.
Each consumed segment is directed from its active endpoint toward its
far terminus.

divdist (divdist.m, modified): distance along each directed segment is
base + k*cellsize, where base is the maximum over distances inherited
from segments consumed earlier that end where this one starts (0 at true
endpoints and at arbitrary component seeds). Segments whose *both*
termini are endpoints get the pointwise minimum of the forward and
reverse profiles (distance from the nearer endpoint). Segments shorter
than two nodes get NaN.

divorder (divorder_robust.m): per-segment order propagated over the
directed segment graph in topological passes; joining orders {w} combine
as (paper Eqs. 1-3):

    strahler: max(min{w} + 1, max{w})
    shreve:   sum{w}
    topo:     max{w} + 1

Endpoints and in-degree-0 segments seed order 1. Segments left
unassigned by a cycle are forced to 1 with a counted warning
(tolerate cycles, warn with a count, never silently).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .network import DivideNetwork


@dataclass
class SortedNetwork:
    """Directed, sorted segment list (flat).

    seg_nodes/seg_ptr as in DivideNetwork but segments are stored in
    consumption (peeling) order and directed start -> end.
    """
    seg_nodes: np.ndarray
    seg_ptr: np.ndarray
    is_ring: np.ndarray
    endpoints: np.ndarray
    junctions: np.ndarray

    @property
    def n_segments(self) -> int:
        return len(self.seg_ptr) - 1

    def segment(self, k: int) -> np.ndarray:
        return self.seg_nodes[self.seg_ptr[k]:self.seg_ptr[k + 1]]

    def st(self) -> np.ndarray:
        """(n_seg, 2) directed termini (start, end)."""
        return np.stack([self.seg_nodes[self.seg_ptr[:-1]],
                         self.seg_nodes[self.seg_ptr[1:] - 1]], axis=1)


def sort_network(nw: DivideNetwork,
                 jct_deg: np.ndarray | None = None) -> SortedNetwork:
    """Peel segments from endpoints inward (MATLAB sort.m, modified)."""
    n = nw.n_segments
    segs = [nw.segment(k).tolist() for k in range(n)]
    st1 = np.array([s[0] for s in segs], dtype=np.int64)
    st2 = np.array([s[-1] for s in segs], dtype=np.int64)

    jct = nw.junctions
    if jct_deg is None:
        jct_deg = nw.jct_deg
    jctcnt = {int(j): int(d) - 1 for j, d in zip(jct, jct_deg)}
    jcthit = {int(j): 0 for j in jct}

    # terminus -> segment indices
    touch: dict[int, list[int]] = {}
    for k in range(n):
        touch.setdefault(int(st1[k]), []).append(k)
        if st2[k] != st1[k]:
            touch.setdefault(int(st2[k]), []).append(k)

    remaining = np.ones(n, dtype=bool)
    order_out: list[int] = []
    flipped = np.zeros(n, dtype=bool)

    active: list[int] = sorted(set(nw.endpoints.tolist()))
    ptr_next = 0   # scan pointer for arbitrary re-seeding, MATLAB-style

    while True:
        if not active:
            while ptr_next < n and not remaining[ptr_next]:
                ptr_next += 1
            if ptr_next == n:
                break
            active = [int(st1[ptr_next])]

        wave = []
        for ep in active:
            for k in touch.get(ep, []):
                if remaining[k]:
                    if st2[k] == ep and st1[k] != ep:
                        segs[k].reverse()
                        st1[k], st2[k] = st2[k], st1[k]
                        flipped[k] = True
                    wave.append(k)

        promoted: list[int] = []
        hit_now: set[int] = set()
        for k in wave:
            if not remaining[k]:
                continue
            remaining[k] = False
            order_out.append(k)
            far = int(st2[k])
            if far in jcthit:
                jcthit[far] += 1
                hit_now.add(far)
        # only a junction hit in this wave can have become ready, so the
        # check is over the wave's junctions, not over every junction left
        for j in hit_now:
            if j in jctcnt and jcthit[j] >= jctcnt[j]:
                promoted.append(j)
        for j in promoted:
            del jctcnt[j]
            del jcthit[j]
        active = sorted(promoted)

    seg_ptr = np.zeros(len(order_out) + 1, dtype=np.int64)
    for i, k in enumerate(order_out):
        seg_ptr[i + 1] = seg_ptr[i] + len(segs[k])
    seg_nodes = np.array([v for k in order_out for v in segs[k]],
                         dtype=np.int64)
    return SortedNetwork(
        seg_nodes=seg_nodes,
        seg_ptr=seg_ptr,
        is_ring=nw.is_ring[np.array(order_out, dtype=np.int64)]
        if n else np.zeros(0, dtype=bool),
        endpoints=nw.endpoints,
        junctions=nw.junctions,
    )


def divdist(sn: SortedNetwork, cellsize: float) -> np.ndarray:
    """Divide distance per node instance, aligned with sn.seg_nodes."""
    n = sn.n_segments
    st = sn.st()
    ep = set(sn.endpoints.tolist())

    # inherited seeds appended by earlier segments: start node -> seeds
    seeds: list[list[float]] = [[] for _ in range(n)]
    starts_at: dict[int, list[int]] = {}
    for k in range(n):
        starts_at.setdefault(int(st[k, 0]), []).append(k)

    for k in range(n):
        if int(st[k, 0]) in ep:
            seeds[k].append(0.0)

    is2ep = [int(st[k, 0]) in ep and int(st[k, 1]) in ep for k in range(n)]

    out = np.full(len(sn.seg_nodes), np.nan, dtype=np.float64)
    checked = np.zeros(n, dtype=bool)

    for k in range(n):
        seg = sn.segment(k)
        m = len(seg)
        base = max(seeds[k]) if seeds[k] else 0.0
        if m >= 2:
            prof = base + np.arange(m, dtype=np.float64) * cellsize
            if is2ep[k]:
                prof = np.minimum(prof, prof[::-1])
            out[sn.seg_ptr[k]:sn.seg_ptr[k + 1]] = prof
            nextbase = float(prof.max())
        else:
            nextbase = base
        checked[k] = True
        for k2 in starts_at.get(int(st[k, 1]), []):
            if not checked[k2]:
                seeds[k2].append(nextbase)
    return out


_ORDER_FUNCS = {
    "strahler": lambda w: max(min(w) + 1, max(w)),
    "shreve": lambda w: sum(w),
    "topo": lambda w: max(w) + 1,
}


def divorder(sn: SortedNetwork, scheme: str = "topo") -> np.ndarray:
    """Divide order per node instance, aligned with sn.seg_nodes."""
    if scheme not in _ORDER_FUNCS:
        raise ValueError(f"unknown scheme {scheme!r}")
    fct = _ORDER_FUNCS[scheme]

    n = sn.n_segments
    st = sn.st()
    ep = set(sn.endpoints.tolist())

    ends_at: dict[int, list[int]] = {}
    for k in range(n):
        ends_at.setdefault(int(st[k, 1]), []).append(k)
    in_segs: list[list[int]] = [ends_at.get(int(st[k, 0]), [])
                                for k in range(n)]

    seg_order = np.full(n, -1, dtype=np.int64)
    for k in range(n):
        if int(st[k, 0]) in ep or not in_segs[k]:
            seg_order[k] = 1

    progress = True
    while progress:
        progress = False
        for k in range(n):
            if seg_order[k] >= 0:
                continue
            ups = in_segs[k]
            if all(seg_order[j] >= 0 for j in ups):
                seg_order[k] = fct([int(seg_order[j]) for j in ups])
                progress = True

    n_cyclic = int((seg_order < 0).sum())
    if n_cyclic:
        warnings.warn(
            f"divorder: {n_cyclic} segment(s) in cycles forced to order 1",
            stacklevel=2)
        seg_order[seg_order < 0] = 1

    out = np.empty(len(sn.seg_nodes), dtype=np.float64)
    for k in range(n):
        out[sn.seg_ptr[k]:sn.seg_ptr[k + 1]] = float(seg_order[k])
    return out
