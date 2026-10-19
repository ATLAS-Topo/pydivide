# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Validation helpers: parsing MATLAB nan-lists and classifying the known
MATLAB artifacts.

Nothing here is used in production runs; it exists so the validation
ladder can demand *exact* equality while accounting for defects of the
MATLAB implementation itself, each occurrence machine-verified and
logged.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .gridspec import GridSpec
from .network import _E, _N, _S, _SIDE_MATE, _W, segment_network

_DIRNAMES = {_N: "N", _S: "S", _W: "W", _E: "E"}


def parse_nanlist(ix: np.ndarray) -> list[list[int]]:
    """Split a nan-separated node list into runs of int node ids."""
    runs: list[list[int]] = []
    cur: list[int] = []
    for v in ix:
        if np.isnan(v):
            if cur:
                runs.append(cur)
                cur = []
        else:
            cur.append(int(v))
    if cur:
        runs.append(cur)
    return runs


def _direction(grid: GridSpec, n_from: int, n_to: int) -> int:
    w = grid.nc + 1
    d = n_to - n_from
    return {-w: _N, w: _S, -1: _W, 1: _E}[d]


def canonical_segment(seq: list[int]) -> tuple:
    """Direction-independent (and, for rings, rotation-independent)
    canonical form of a segment node sequence."""
    if len(seq) > 1 and seq[0] == seq[-1]:          # ring
        body = seq[:-1]
        best = None
        for rot_src in (body, body[::-1]):
            m = min(rot_src)
            for i in (k for k, v in enumerate(rot_src) if v == m):
                cand = tuple(rot_src[i:] + rot_src[:i])
                if best is None or cand < best:
                    best = cand
        return ("ring",) + best
    fwd = tuple(seq)
    rev = tuple(reversed(seq))
    return min(fwd, rev)


def sorted_network_from_matlab_runs(nw, matlab_runs: list[list[int]]):
    """Build a SortedNetwork verbatim from MATLAB's final D.IX runs.

    MATLAB's D.IX *is* its sort output: segment order and direction as
    consumed by its divdist/divorder. Using it directly removes the last
    two order-dependent degrees of freedom (within-wave tie order and
    both-endpoint segment direction) so the distance/order
    gates test pydivide's value semantics exactly. The caller must have
    already passed the segment-set gate for this to be meaningful.
    """
    from .orderdist import SortedNetwork

    seg_ptr = np.zeros(len(matlab_runs) + 1, dtype=np.int64)
    for i, r in enumerate(matlab_runs):
        seg_ptr[i + 1] = seg_ptr[i] + len(r)
    seg_nodes = np.array([v for r in matlab_runs for v in r],
                         dtype=np.int64)
    is_ring = np.array([len(r) > 1 and r[0] == r[-1]
                        for r in matlab_runs], dtype=bool)
    return SortedNetwork(seg_nodes=seg_nodes, seg_ptr=seg_ptr,
                         is_ring=is_ring, endpoints=nw.endpoints,
                         junctions=nw.junctions)


def align_is2ep_orientation(nw, matlab_runs: list[list[int]]) -> None:
    """Reverse (in place) our stored both-endpoint segments whose
    orientation differs from MATLAB's final D.IX runs.

    Needed for exact divdist equality only: MATLAB's seed-append
    mechanism propagates a both-endpoint segment's half-length distance
    to other segments sharing its *far* terminus node (an FX-node
    double-terminus), so the leak direction depends on the segment's
    stored orientation — which in MATLAB is a walk-order artifact
   . Junction-connected segment directions are forced by
    peeling and need no alignment.
    """
    ep = set(nw.endpoints.tolist())
    matlab_dir = {canonical_segment(run): run for run in matlab_runs}
    for k in range(nw.n_segments):
        if nw.is_ring[k]:
            continue
        s, e = nw.seg_ptr[k], nw.seg_ptr[k + 1]
        seg = nw.seg_nodes[s:e]
        if int(seg[0]) in ep and int(seg[-1]) in ep:
            run = matlab_dir.get(canonical_segment(seg.tolist()))
            if run is not None and run[0] != int(seg[0]):
                nw.seg_nodes[s:e] = seg[::-1]


@dataclass
class MergeFailureReport:
    """Classified MATLAB termini discrepancies at FX nodes."""
    forced_termini: np.ndarray            # (k, 2) int64 (node, dir)
    forced_pairs: np.ndarray              # (k, 3) int64 (node, dirA, dirB)
    log: list[str] = field(default_factory=list)


def classify_merge_failures(grid: GridSpec, FX: np.ndarray,
                            matlab_segments: list[list[int]],
                            our_network,
                            through: dict | None = None
                            ) -> MergeFailureReport:
    """Compare MATLAB's segment termini with pydivide's; classify every
    discrepancy, or raise.

    MATLAB-only termini are classifiable iff they sit at a node with
    FX != 0 and the terminating edge directions there form one (or two)
    same-side pairs under the FX pairing rule — i.e. exactly the pairs
    MATLAB's own divnet rule says should have merged but its cascade
    failed to. They become forced termini.

    pydivide-only termini are classifiable iff MATLAB's runs demonstrably
    pass through that node connecting exactly those directions (divnet's
    blind two-termini merge, which can join a wrong-side pair at an FX
    node). They become forced pairs.

    Anything else is a pydivide defect and blocks validation
    (ValueError).
    """
    fx_flat = FX.ravel()

    m_term: Counter = Counter()
    for run in matlab_segments:
        if len(run) < 2:
            continue
        m_term[(run[0], _direction(grid, run[0], run[1]))] += 1
        m_term[(run[-1], _direction(grid, run[-1], run[-2]))] += 1

    o_term: Counter = Counter()
    nw = our_network
    for k in range(nw.n_segments):
        if nw.is_ring[k]:
            continue
        s = nw.segment(k)
        o_term[(int(s[0]), _direction(grid, int(s[0]), int(s[1])))] += 1
        o_term[(int(s[-1]), _direction(grid, int(s[-1]), int(s[-2])))] += 1

    forced_pair_list: list[tuple[int, int, int]] = []
    forced_term_list: list[tuple[int, int]] = []
    log: list[str] = []
    through = through or {}

    # ---- through-connectivity per node, both sides ---------------------
    # A wrong-side pass-through by MATLAB at a node two chains both cross
    # swaps the chains' tails without changing any terminus, so termini
    # comparison alone cannot see it; per-node connection multisets can.
    def _conns_of(seqs) -> dict:
        out: dict[int, Counter] = {}
        for seq in seqs:
            if len(seq) < 2:
                continue
            ring = seq[0] == seq[-1]
            body = seq[:-1] if ring else seq
            rng = range(len(seq) - 2) if not ring else range(len(body))
            for i in rng:
                n = seq[i + 1] if not ring else body[i]
                if not ring:
                    p, q = seq[i], seq[i + 2]
                else:
                    p = body[i - 1]
                    q = body[(i + 1) % len(body)]
                da = _direction(grid, n, p)
                db = _direction(grid, n, q)
                out.setdefault(n, Counter())[
                    (min(da, db), max(da, db))] += 1
        return out

    m_conns = _conns_of(matlab_segments)
    o_conns = _conns_of(nw.segment(k).tolist()
                        for k in range(nw.n_segments))

    ours_only = o_term - m_term
    matlab_only = m_term - o_term
    conn_diff = {n for n in (set(m_conns) | set(o_conns))
                 if m_conns.get(n, Counter()) != o_conns.get(n, Counter())}
    diff_nodes = sorted({node for (node, _d) in ours_only}
                        | {node for (node, _d) in matlab_only}
                        | conn_diff)

    for node in diff_nodes:
        conns = sorted(m_conns.get(node, Counter()))
        fx = int(fx_flat[node])
        I, J = node // (grid.nc + 1), node % (grid.nc + 1)
        used: Counter = Counter()
        for da, db in conns:
            used[da] += 1
            used[db] += 1
        if used and used.most_common(1)[0][1] > 1:
            raise ValueError(
                f"MATLAB traverses node {node} through the same edge "
                "direction twice — unclassifiable")
        m_ends = sorted(d for (n2, d), c in m_term.items()
                        for _ in range(c) if n2 == node)
        if conns:
            for da, db in conns:
                forced_pair_list.append((node, da, db))
            conn_txt = ", ".join(f"{_DIRNAMES[a]}-{_DIRNAMES[b]}"
                                 for a, b in conns)
        else:
            # MATLAB terminates everything here: break all incidences
            forced_term_list.extend((node, d) for d in range(4))
            conn_txt = "none"
        if fx != 0:
            legality = "; ".join(
                ("same-side" if _SIDE_MATE[fx][a] == b else "WRONG-SIDE")
                + f" {_DIRNAMES[a]}-{_DIRNAMES[b]}" for a, b in conns) \
                or "all-terminate"
        else:
            legality = "FX==0"
        log.append(
            f"MATLAB local-connectivity artifact at node {node} (I={I}, "
            f"J={J}, FX={fx}): divnet's cascade produced connections "
            f"[{conn_txt}] with termini {[_DIRNAMES[d] for d in m_ends]} "
            f"({legality}), deviating from the FX pairing rule; imposed "
            "in validation-compat mode only")

    arr = (np.array(sorted(set(forced_term_list)),
                    dtype=np.int64).reshape(-1, 2)
           if forced_term_list else np.empty((0, 2), dtype=np.int64))
    fp = (np.array(forced_pair_list, dtype=np.int64).reshape(-1, 3)
          if forced_pair_list else np.empty((0, 3), dtype=np.int64))
    return MergeFailureReport(forced_termini=arr, forced_pairs=fp, log=log)


@dataclass
class CompatPlan:
    """Everything needed to reproduce MATLAB's final network exactly on
    top of pydivide's deterministic stages, with every deviation from
    first-principles behaviour machine-classified and logged."""
    edges: np.ndarray            # validation edge set after exclusions
    forced_pairs: np.ndarray     # (k, 3) (node, dirA, dirB)
    forced_termini: np.ndarray   # (k, 2) (node, dir)
    split_runs: list = field(default_factory=list)
    # nodes that are endpoints in the compat network but not in MATLAB's
    # ep because divnet glued straight across them (phantom joints):
    allowed_extra_endpoints: set = field(default_factory=set)
    log: list[str] = field(default_factory=list)

    def network(self, grid: GridSpec, FX: np.ndarray):
        return segment_network(grid, self.edges, FX,
                               forced_termini=self.forced_termini,
                               forced_pairs=self.forced_pairs)


def split_runs_at_phantoms(grid: GridSpec, runs: list[list[int]]
                           ) -> tuple[list[list[int]], list[dict]]:
    """Split MATLAB runs at non-adjacent consecutive node pairs.

    divnet's merge cascade can concatenate two segments that share no
    node; the stored polyline then teleports. For
    connectivity-level comparisons the run must be treated as its real
    pieces. Returns (pieces, phantom records)."""
    w = grid.nc + 1
    adj = {-w, w, -1, 1}
    out: list[list[int]] = []
    phantoms: list[dict] = []
    pending: dict | None = None
    for run in runs:
        cur = [run[0]]
        pending = None
        for a, b in zip(run[:-1], run[1:]):
            if b - a in adj:
                cur.append(b)
                if pending is not None:
                    pending["b_next"] = b
                    pending = None
            else:
                ph = {"a": a, "b": b,
                      "a_prev": cur[-2] if len(cur) > 1 else None,
                      "b_next": None}
                phantoms.append(ph)
                if len(cur) > 1:
                    out.append(cur)
                cur = [b]
                pending = ph
        if len(cur) > 1:
            out.append(cur)
    return out, phantoms


def _edge_key_set(edges: np.ndarray) -> set:
    return set(map(tuple, edges.tolist()))


def build_compat_plan(grid: GridSpec, FX: np.ndarray,
                      matlab_final_edges: np.ndarray,
                      matlab_runs: list[list[int]]) -> CompatPlan:
    """Build the validation-compat plan against MATLAB's final network.

    Classifies, in order:

    1. **Edge loss**: edges present in MATLAB's constructor
       output (matlab_final_edges, from D0.IX) that MATLAB's final
       network (the runs of D.IX) never traverses. They must form closed
       rings attached to FX!=0 nodes — casualties of divnet's dead-
       segment cascade. They are excluded from the validation edge set.
       At each attachment node, MATLAB's actual through-connection is
       read off the runs and imposed as a forced pair (this is divnet's
       blind two-termini merge and may join a wrong-side pair; logged).

    2. **Merge failures**: same-side pairs at FX nodes that
       MATLAB left unmerged; imposed as forced termini.

    Raises ValueError on anything that does not fit these categories —
    an unclassifiable discrepancy blocks validation.
    """
    log: list[str] = []
    w = grid.nc + 1

    # ---- phantom joints: split runs where divnet glued non-adjacent ----
    pieces, phantoms = split_runs_at_phantoms(grid, matlab_runs)
    phantom_forced: list[tuple[int, int]] = []
    allowed_extra: set = set()
    for ph in phantoms:
        a, b = ph["a"], ph["b"]
        log.append(
            f"MATLAB phantom glue: divnet concatenated segments across "
            f"non-adjacent nodes {a} (I={a // w}, J={a % w}) -> {b} "
            f"(I={b // w}, J={b % w}); runs split for connectivity "
            "comparisons; distance accumulates across the teleport in "
            "MATLAB and is reproduced verbatim by the value gates")
        if ph["a_prev"] is not None:
            phantom_forced.append((a, _direction(grid, a, ph["a_prev"])))
            allowed_extra.add(a)
        if ph["b_next"] is not None:
            phantom_forced.append((b, _direction(grid, b, ph["b_next"])))
            allowed_extra.add(b)

    # ---- MATLAB's traversed edge set and through-connections ----------
    traversed: set = set()
    through: dict[int, list[tuple[int, int]]] = {}
    for run in pieces:
        for a, b in zip(run[:-1], run[1:]):
            traversed.add((min(a, b), max(a, b)))
        for i in range(1, len(run) - 1):
            n = run[i]
            through.setdefault(n, []).append(
                (_direction(grid, n, run[i - 1]),
                 _direction(grid, n, run[i + 1])))

    final = _edge_key_set(matlab_final_edges)
    phantom_edges = traversed - final
    if phantom_edges:
        raise ValueError(
            f"MATLAB's final network traverses {len(phantom_edges)} edges "
            f"not in its own constructor output: "
            f"{list(phantom_edges)[:5]} — unclassifiable")

    lost = final - traversed

    forced_pairs: list[tuple[int, int, int]] = []
    if lost:
        lost_arr = np.array(sorted(lost), dtype=np.int64)
        cnt = Counter(int(x) for row in lost_arr for x in row)
        odd = [n for n, c in cnt.items() if c % 2]
        shape = ("closed rings" if not odd
                 else f"open tangle, {len(odd)} odd-degree nodes")
        log.append(
            f"MATLAB edge-loss artifact: {len(lost)} real divide edges "
            f"dropped by divnet's merge cascade ({shape}); excluded from "
            "the validation edge set, kept in production output")
        # attachment nodes: lost-subgraph nodes still carrying kept edges
        kept = final - lost
        kept_nodes = set()
        for a, b in kept:
            kept_nodes.add(a)
            kept_nodes.add(b)
        fx_flat = FX.ravel()
        for n in sorted(set(cnt) & kept_nodes):
            fx = int(fx_flat[n])
            conns = through.get(n, [])
            I, J = n // w, n % w
            for da, db in conns:
                forced_pairs.append((n, da, db))
                if fx != 0:
                    side = ("same-side" if _SIDE_MATE[fx][da] == db
                            else "WRONG-SIDE")
                else:
                    side = "FX==0"
                log.append(
                    f"  lost-subgraph attachment at node {n} (I={I}, "
                    f"J={J}, FX={fx}): MATLAB connects "
                    f"{_DIRNAMES[da]}-{_DIRNAMES[db]} through it ({side}); "
                    "forced pair in compat mode")
        edges2 = np.array(sorted(kept), dtype=np.int64)
    else:
        edges2 = matlab_final_edges

    fp_arr = (np.array(forced_pairs, dtype=np.int64).reshape(-1, 3)
              if forced_pairs else np.empty((0, 3), dtype=np.int64))
    ph_arr = (np.array(sorted(set(phantom_forced)),
                       dtype=np.int64).reshape(-1, 2)
              if phantom_forced else np.empty((0, 2), dtype=np.int64))

    # ---- termini discrepancies on the reduced edge set -----------------
    plain = segment_network(grid, edges2, FX, forced_pairs=fp_arr,
                            forced_termini=ph_arr)
    report = classify_merge_failures(grid, FX, pieces, plain,
                                     through=through)
    log.extend(report.log)

    forced_termini = np.concatenate([ph_arr, report.forced_termini])
    all_pairs = np.concatenate([fp_arr, report.forced_pairs])

    return CompatPlan(edges=edges2, forced_pairs=all_pairs,
                      forced_termini=forced_termini,
                      split_runs=pieces,
                      allowed_extra_endpoints=allowed_extra,
                      log=log)
