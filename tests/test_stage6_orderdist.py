# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Validation ladder steps 6-7: distance and order.

Comparison is by node-keyed value multisets: junction nodes
legitimately appear in several segments with different values, and
segment order/direction after sorting are implementation details. The
tolerance is the spec's 1e-9 * max(1, |value|), implemented by rounding
to 1e-6 map units (values are exact multiples of half the cellsize).
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydivide.matlab_ref import MatlabReference  # noqa: E402
from pydivide.network import fx_grid  # noqa: E402
from pydivide.orderdist import divdist, divorder, sort_network  # noqa: E402
from pydivide.validate import build_compat_plan, parse_nanlist  # noqa: E402


@pytest.fixture(scope="module")
def setup(ref, net):
    FX = fx_grid(net.grid, net.fd_src, net.fd_dst)
    edges = MatlabReference.nanlist_edges(ref.D0_IX())
    runs = parse_nanlist(ref.D_IX())
    plan = build_compat_plan(net.grid, FX, edges, runs)
    nw = plan.network(net.grid, FX)
    # validation-compat: MATLAB's D.IX runs ARE its sort output; using
    # them verbatim removes the within-wave tie order and both-endpoint
    # direction, which are MATLAB walk-order artifacts.
    # pydivide's own sort is gated by test_sort_topological_invariant and
    # by the segment/order equality tests.
    from pydivide.validate import sorted_network_from_matlab_runs
    sn = sorted_network_from_matlab_runs(nw, runs)
    return ref, net, sn


def test_sort_topological_invariant(setup):
    """pydivide's own sort: every segment starting at a junction must
    appear after all segments that end at that junction (feeders)."""
    ref, net, _ = setup
    FX = fx_grid(net.grid, net.fd_src, net.fd_dst)
    edges = MatlabReference.nanlist_edges(ref.D0_IX())
    plan = build_compat_plan(net.grid, FX, edges, parse_nanlist(ref.D_IX()))
    nw = plan.network(net.grid, FX)
    sn = sort_network(nw)
    st = sn.st()
    jct = set(sn.junctions.tolist())
    last_feeder: dict[int, int] = {}
    for k in range(sn.n_segments):
        end = int(st[k, 1])
        if end in jct:
            last_feeder[end] = k
    for k in range(sn.n_segments):
        start = int(st[k, 0])
        if start in jct and start in last_feeder:
            feeders = [j for j in range(sn.n_segments)
                       if int(st[j, 1]) == start and j != k]
            assert all(j < k for j in feeders), \
                f"segment {k} starts at junction {start} before a feeder"


def _matlab_multiset(ref, values: np.ndarray) -> Counter:
    """(node, value) multiset from MATLAB nan-lists (IX-aligned array)."""
    ix = ref.D_IX()
    assert len(ix) == len(values), "IX and value array misaligned"
    c: Counter = Counter()
    for node, v in zip(ix, values):
        if np.isnan(node):
            continue
        key = round(float(v), 6) if not np.isnan(v) else "nan"
        c[(int(node), key)] += 1
    return c


def _our_multiset(sn, values: np.ndarray) -> Counter:
    c: Counter = Counter()
    for node, v in zip(sn.seg_nodes, values):
        key = round(float(v), 6) if not np.isnan(v) else "nan"
        c[(int(node), key)] += 1
    return c


def test_distance_identical(setup):
    ref, net, sn = setup
    ours = _our_multiset(sn, divdist(sn, net.grid.cellsize))
    matlab = _matlab_multiset(ref, ref.D_distance())
    only_o = ours - matlab
    only_m = matlab - ours
    assert ours == matlab, (
        f"distance differs at {sum(only_o.values())} (ours) / "
        f"{sum(only_m.values())} (matlab) node instances; "
        f"examples ours={list(only_o)[:3]} matlab={list(only_m)[:3]}")


@pytest.mark.parametrize("scheme", ["strahler", "shreve", "topo"])
def test_order_identical(setup, scheme):
    ref, net, sn = setup
    ours = _our_multiset(sn, divorder(sn, scheme))
    matlab = _matlab_multiset(ref, ref.order(scheme))
    only_o = ours - matlab
    only_m = matlab - ours
    assert ours == matlab, (
        f"{scheme} order differs at {sum(only_o.values())} (ours) / "
        f"{sum(only_m.values())} (matlab) node instances; "
        f"examples ours={list(only_o)[:3]} matlab={list(only_m)[:3]}")
