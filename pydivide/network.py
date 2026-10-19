# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Stages 4-5 — diagonal crossings, node splitting, segmentation.

Stage 4: the FX grid marks divide nodes crossed by a *diagonal*
flow edge of the full flow network. At such a node the divide network must
not connect through in the direction the flow crosses.

Computed from cell-index arithmetic (not floating-point midpoints, but
verified equivalent): a diagonal flow edge between cells (r1,c1) and
(r2,c2) crosses divide node (max(r1,r2), max(c1,c2)); it is NW-SE type
(FX=1) when dr*dc == +1 and NE-SW type (FX=2) when dr*dc == -1. MATLAB
assigns 1s first and 2s second, so a node crossed by both diagonals ends
up FX=2; reproduced here.

Stage 5: connectivity through each node:

  FX == 0, degree 2       -> the two edges connect (pass-through)
  FX == 0, degree 1       -> terminus (endpoint)
  FX == 0, degree > 2     -> junction: every incident edge terminates
  FX != 0                 -> the crossing direction splits the four edge
                             directions into two sides:
                               FX=1 (NW-SE): {N, E} vs {S, W}
                               FX=2 (NE-SW): {N, W} vs {S, E}
                             two edges on one side connect; a lone edge
                             on a side terminates; no junctions occur at
                             FX nodes (matches MATLAB divnet's dead-
                             segment merging outcome for degrees 2/3/4).

A segment is a maximal chain of edges connected by pass-throughs. Closed
rings (no terminus anywhere) are walked from an arbitrary edge and
flagged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gridspec import GridSpec, assert_int64

# direction codes for the four possible divide edges at a node
_N, _S, _W, _E = 0, 1, 2, 3
# same-side pairings per FX value: FX=1 -> {N,E},{S,W}; FX=2 -> {N,W},{S,E}
_SIDE_MATE = {
    1: {_N: _E, _E: _N, _S: _W, _W: _S},
    2: {_N: _W, _W: _N, _S: _E, _E: _S},
}


_FX_CHUNK = 50_000_000     # flow edges per chunk (~1.6 GB temporaries)


def fx_grid(grid: GridSpec, fd_src: np.ndarray,
            fd_dst: np.ndarray) -> np.ndarray:
    """FX raster, int8 (nr+1, nc+1).

    Processed in chunks: the full domain has ~5.1e9 flow edges, and a
    single int64 temporary over all of them is 41 GB (38 GiB). Peak extra memory
    is one chunk regardless of domain size.

    For D8 steps, dr and dc are in {-1, 0, 1}, so ``dr * dc == +1``
    identifies NW-SE diagonals and ``== -1`` NE-SW ones without needing
    separate |dr|, |dc| tests. Two passes (1s then 2s) preserve MATLAB's
    assignment order, so a node crossed by both diagonals ends up 2.
    """
    assert_int64(fd_src, fd_dst)
    nc64 = np.int64(grid.nc)
    FX = np.zeros((grid.nr + 1, grid.nc + 1), dtype=np.int8)
    n = fd_src.size

    for prod, value in ((1, 1), (-1, 2)):
        for s in range(0, n, _FX_CHUNK):
            e = min(s + _FX_CHUNK, n)
            src = fd_src[s:e]
            dst = fd_dst[s:e]
            r1 = src // nc64
            r2 = dst // nc64
            c1 = src - r1 * nc64
            c2 = dst - r2 * nc64
            m = ((r2 - r1) * (c2 - c1)) == prod
            if not m.any():
                continue
            FX[np.maximum(r1[m], r2[m]), np.maximum(c1[m], c2[m])] = value
    return FX


@dataclass
class DivideNetwork:
    """Flat segment representation (Stage 5).

    seg_nodes : int64, concatenated node ids of all segments
    seg_ptr   : int64 (n_seg+1,) offsets into seg_nodes
    is_ring   : bool (n_seg,) — closed rings (first node == last node)
    junctions : int64, sorted unique junction node ids (degree>2, FX==0)
    jct_deg   : int64, degree of each junction
    endpoints : int64, sorted unique endpoint node ids (termini that are
                not junctions)
    """
    seg_nodes: np.ndarray
    seg_ptr: np.ndarray
    is_ring: np.ndarray
    junctions: np.ndarray
    jct_deg: np.ndarray
    endpoints: np.ndarray

    @property
    def n_segments(self) -> int:
        return len(self.seg_ptr) - 1

    def segment(self, k: int) -> np.ndarray:
        return self.seg_nodes[self.seg_ptr[k]:self.seg_ptr[k + 1]]

    def termini(self) -> np.ndarray:
        """(n_seg, 2) int64 first/last node of each segment."""
        return np.stack([self.seg_nodes[self.seg_ptr[:-1]],
                         self.seg_nodes[self.seg_ptr[1:] - 1]], axis=1)


def _edge_direction(grid: GridSpec, n_from: np.ndarray,
                    n_to: np.ndarray) -> np.ndarray:
    """Direction code (N/S/W/E) of the edge leaving node ``n_from`` toward
    adjacent node ``n_to``."""
    w = np.int64(grid.nc + 1)
    d = n_to - n_from
    out = np.full(d.shape, -1, dtype=np.int64)
    out[d == -w] = _N
    out[d == w] = _S
    out[d == -1] = _W
    out[d == 1] = _E
    if (out < 0).any():
        raise ValueError("non-adjacent node pair in edge list")
    return out


def segment_network(grid: GridSpec, edges: np.ndarray,
                    FX: np.ndarray,
                    forced_termini: np.ndarray | None = None,
                    forced_pairs: np.ndarray | None = None
                    ) -> DivideNetwork:
    """Stage 5: split the edge set into maximal segments.

    Parameters
    ----------
    edges : (E, 2) int64 node pairs (each row sorted, rows unique).
    FX : int8 (nr+1, nc+1) from :func:`fx_grid`.
    forced_termini : optional (k, 2) int64 of (node id, direction code)
        incidences that must terminate even though the pairing rules
        would connect them. Validation-only: reproduces MATLAB's
        order-dependent failures to merge same-side edge pairs at FX
        nodes. Production runs never set this.
    forced_pairs : optional (k, 3) int64 of (node id, dirA, dirB)
        pass-through pairings imposed at a node, overriding the standard
        rules; remaining incidences at such a node terminate.
        Validation-only: reproduces MATLAB divnet's blind two-termini
        merges at FX nodes, which can connect wrong-side pairs
        in some configurations. Production runs never set this.
    """
    assert_int64(edges)
    E = len(edges)
    fx_flat = FX.ravel()

    # Incidence k in [0, 2E): k < E is the edges[k,0] end, k >= E the
    # edges[k-E,1] end. other(k) = (k + E) mod 2E; edge_of(k) = k mod E.
    inc_node = np.concatenate([edges[:, 0], edges[:, 1]])
    inc_dir = np.concatenate([
        _edge_direction(grid, edges[:, 0], edges[:, 1]),
        _edge_direction(grid, edges[:, 1], edges[:, 0]),
    ])
    n_inc = 2 * E

    # ---- per-node groups (for degrees / junctions / FX==0 pairing) -------
    order = np.argsort(inc_node, kind="stable")
    snode = inc_node[order]
    gstart = np.nonzero(np.r_[True, snode[1:] != snode[:-1]])[0] \
        if n_inc else np.empty(0, dtype=np.int64)
    gend = np.r_[gstart[1:], n_inc] if n_inc else gstart
    degree = (gend - gstart).astype(np.int64)
    uniq_nodes = snode[gstart] if n_inc else np.empty(0, dtype=np.int64)
    ufx = fx_flat[uniq_nodes] if n_inc else np.empty(0, dtype=np.int8)

    is_junction = (degree > 2) & (ufx == 0)
    junctions = uniq_nodes[is_junction]
    jct_deg = degree[is_junction]

    # ---- pairing (vectorised) ---------------------------------------------
    partner = np.full(n_inc, -1, dtype=np.int64)

    # FX == 0, degree 2: connect the two incidences of the group
    g2 = np.nonzero((degree == 2) & (ufx == 0))[0]
    a = order[gstart[g2]]
    b = order[gstart[g2] + 1]
    partner[a] = b
    partner[b] = a

    # FX != 0: same-side mate lookup via (node, direction) keys
    mate_tab = np.full((3, 4), -1, dtype=np.int64)
    for fxv, m in _SIDE_MATE.items():
        for d0, d1 in m.items():
            mate_tab[fxv, d0] = d1
    at_fx = fx_flat[inc_node] != 0
    if at_fx.any():
        keys = inc_node * np.int64(4) + inc_dir
        korder = np.argsort(keys)
        skeys = keys[korder]
        idx = np.nonzero(at_fx)[0]
        mdir = mate_tab[fx_flat[inc_node[idx]], inc_dir[idx]]
        mkeys = inc_node[idx] * np.int64(4) + mdir
        loc = np.searchsorted(skeys, mkeys)
        loc = np.minimum(loc, n_inc - 1)
        found = skeys[loc] == mkeys
        partner[idx[found]] = korder[loc[found]]
        # one-sided links (mate exists but is at an FX==0 node? impossible:
        # same node) are symmetric by construction

    # ---- validation-only overrides (rare; scalar loops are fine) ----------
    if forced_pairs is not None and len(forced_pairs):
        assert_int64(forced_pairs)
        by_node: dict[int, list[tuple[int, int]]] = {}
        for node, da, db in forced_pairs.tolist():
            by_node.setdefault(int(node), []).append((int(da), int(db)))
        touched = np.nonzero(np.isin(inc_node,
                                     np.array(list(by_node), dtype=np.int64)
                                     ))[0]
        slot = {(int(inc_node[k]), int(inc_dir[k])): int(k) for k in touched}
        for k in touched:      # break whatever standard pairing did there
            p = partner[k]
            if p >= 0:
                partner[p] = -1
            partner[k] = -1
        for node, pairs in by_node.items():
            for da, db in pairs:
                ka = slot.get((node, da))
                kb = slot.get((node, db))
                if ka is not None and kb is not None:
                    partner[ka] = kb
                    partner[kb] = ka

    if forced_termini is not None and len(forced_termini):
        assert_int64(forced_termini)
        forced = set(map(tuple, forced_termini.tolist()))
        nodes_f = np.array(sorted({n for n, _ in forced}), dtype=np.int64)
        for k in np.nonzero(np.isin(inc_node, nodes_f))[0]:
            if (int(inc_node[k]), int(inc_dir[k])) in forced:
                p = partner[k]
                if p >= 0:
                    partner[p] = -1
                partner[k] = -1

    # ---- chain walking -----------------------------------------------------
    # step from incidence k: cross the edge to j = other(k); continue via
    # partner[j] (or terminate). succ is a functional graph over incidences.
    succ = partner[(np.arange(n_inc, dtype=np.int64) + E) % n_inc]
    # succ[k] = next incidence after crossing k's edge, or -1 at a chain end

    starts_all = np.nonzero(partner < 0)[0]    # terminus incidences

    # One cursor walk from every terminus, recording each chain's length
    # and its final incidence. Two properties matter at scale:
    #  * the active set is *compacted* each iteration, so total work is
    #    O(total edges) — scanning all chains per iteration instead is
    #    quadratic in the longest chain and hangs on domains with one
    #    very long divide (measured: 3 h without finishing on a 607M-cell
    #    DEM whose main divide is ~10^5 edges);
    #  * the chain's far end comes out of the same walk, so no pointer
    #    doubling is needed — that cost a 14 GB temporary and ~20 passes
    #    over 1.7e9 incidences at full-domain size.
    n_all = len(starts_all)
    len_all = np.zeros(n_all, dtype=np.int64)
    last_all = starts_all.copy()
    act = np.arange(n_all, dtype=np.int64)
    cur = starts_all.copy()
    while act.size:
        len_all[act] += 1
        last_all[act] = cur
        nxt = succ[cur]
        keep = nxt >= 0
        act = act[keep]
        cur = nxt[keep]

    # each chain is walked from both ends; keep one representative
    other_term = (last_all + E) % n_inc
    emit = starts_all <= other_term
    starts = starts_all[emit]
    lengths = len_all[emit]
    n_chain = len(starts)

    seg_ptr = np.zeros(n_chain + 1, dtype=np.int64)
    np.cumsum(lengths + 1, out=seg_ptr[1:])
    seg_nodes = np.empty(int(seg_ptr[-1]), dtype=np.int64)
    visited = np.zeros(E, dtype=bool)

    pos = seg_ptr[:-1].copy()
    seg_nodes[pos] = inc_node[starts]
    pos += 1
    act = np.arange(n_chain, dtype=np.int64)
    cur = starts.copy()
    while act.size:
        visited[cur % E] = True
        j = (cur + E) % n_inc
        seg_nodes[pos[act]] = inc_node[j]
        pos[act] += 1
        nxt = partner[j]
        keep = nxt >= 0
        act = act[keep]
        cur = nxt[keep]

    ring_parts: list[list[int]] = []
    for eidx in np.nonzero(~visited)[0]:       # closed rings: rare
        if visited[eidx]:
            continue
        chain = [int(inc_node[eidx])]
        k = eidx
        while True:
            visited[k % E] = True
            j = (k + E) % n_inc
            chain.append(int(inc_node[j]))
            k = partner[j]
            if k < 0 or visited[k % E]:
                break
        ring_parts.append(chain)

    if ring_parts:
        extra_ptr = np.cumsum([len(c) for c in ring_parts])
        seg_ptr = np.concatenate([seg_ptr, seg_ptr[-1] + extra_ptr])
        seg_nodes = np.concatenate([
            seg_nodes, np.array([v for c in ring_parts for v in c],
                                dtype=np.int64)])
    is_ring = np.concatenate([np.zeros(n_chain, dtype=bool),
                              np.ones(len(ring_parts), dtype=bool)])

    # ---- endpoints ---------------------------------------------------------
    if n_chain:
        real_termini = np.concatenate([
            seg_nodes[seg_ptr[:n_chain]],
            seg_nodes[seg_ptr[1:n_chain + 1] - 1]])
    else:
        real_termini = np.empty(0, dtype=np.int64)
    endpoints = np.unique(real_termini[
        ~np.isin(real_termini, junctions)])

    return DivideNetwork(
        seg_nodes=seg_nodes,
        seg_ptr=seg_ptr,
        is_ring=is_ring,
        junctions=junctions,
        jct_deg=jct_deg,
        endpoints=endpoints,
    )
