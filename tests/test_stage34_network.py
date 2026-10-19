# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Validation ladder steps 3-5.

Step 3: FX diagonal-crossing grid — identical.
Step 4: junction set (+degrees) and endpoint set — identical.
Step 5: segment decomposition — identical as a set of segments (order and
        internal direction are implementation details; ring start points
        are arbitrary and canonicalised before comparison).

Stages 4-6 are validated on MATLAB's final edge set (= pydivide's set
minus the 118 classified single-edge-run drops, see the classifier), so the
gates test the network algorithms on identical geometry.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydivide.matlab_ref import MatlabReference  # noqa: E402
from pydivide.network import fx_grid  # noqa: E402


@pytest.fixture(scope="module")
def FX(ref, net):
    return fx_grid(net.grid, net.fd_src, net.fd_dst)


@pytest.fixture(scope="module")
def plan_and_network(ref, net, FX):
    """Segment the MATLAB-final edge set with pydivide's stage 4-5, in
    validation-compat mode: every MATLAB order-dependent artifact (edge
    loss, blind merges, merge failures, phantom glue) is classified — or
    the classifier raises and blocks — and injected."""
    from pydivide.validate import build_compat_plan, parse_nanlist

    edges = MatlabReference.nanlist_edges(ref.D0_IX())
    plan = build_compat_plan(net.grid, FX, edges, parse_nanlist(ref.D_IX()))
    for line in plan.log:
        print(line)
    return plan, plan.network(net.grid, FX)


@pytest.fixture(scope="module")
def network(plan_and_network):
    return plan_and_network[1]


# ---- step 3 ---------------------------------------------------------------

def test_fx_grid_identical(ref, FX):
    np.testing.assert_array_equal(FX, ref.FX_grid())


# ---- step 4 ---------------------------------------------------------------

def test_junction_set_identical(ref, network):
    ours = set(network.junctions.tolist())
    matlab = set(ref.D_jct().tolist())
    assert ours == matlab, (
        f"junctions differ: {len(ours - matlab)} only ours, "
        f"{len(matlab - ours)} only MATLAB")


def test_junction_degrees_identical(ref, network):
    ours = dict(zip(network.junctions.tolist(), network.jct_deg.tolist()))
    matlab = dict(zip(ref.D_jct().tolist(), ref.D_jctedg().tolist()))
    assert ours == matlab


def test_endpoint_set_identical(ref, plan_and_network):
    plan, network = plan_and_network
    ours = set(network.endpoints.tolist())
    matlab = set(ref.D_ep().tolist())
    surplus = ours - matlab
    unclassified = surplus - plan.allowed_extra_endpoints
    assert not unclassified and not (matlab - ours), (
        f"endpoints differ: {len(unclassified)} unclassified only-ours, "
        f"{len(matlab - ours)} only MATLAB "
        f"(classified phantom endpoints: {len(surplus & plan.allowed_extra_endpoints)})")


# ---- step 5 ---------------------------------------------------------------

def _canonical_segment(seq: list[int]) -> tuple:
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


def test_segment_set_identical(ref, plan_and_network):
    plan, network = plan_and_network
    matlab = set(_canonical_segment(r) for r in plan.split_runs)
    ours = set()
    for k in range(network.n_segments):
        ours.add(_canonical_segment(network.segment(k).tolist()))
    only_ours = ours - matlab
    only_matlab = matlab - ours
    assert ours == matlab, (
        f"segments differ: {len(only_ours)} only ours, "
        f"{len(only_matlab)} only MATLAB; examples ours="
        f"{list(only_ours)[:2]} matlab={list(only_matlab)[:2]}")
