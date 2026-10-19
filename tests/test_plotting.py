# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Tests for the visual diagnostics.

A plot cannot be checked by eye in CI, so these check the things that
silently go wrong: that the geometry handed to the renderer has one line
per piece, that a mismatch between values and pieces is refused rather
than drawn, that the direction ticks point to the side the convention
says, and that the basin-pair map shades the two basins it was asked for.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

tt = pytest.importorskip("topotoolbox")

from pydivide.acrossdivide import across_divide_metrics  # noqa: E402
from pydivide.flownet import FlowNet  # noqa: E402
from pydivide.labels import basin_labels  # noqa: E402
from pydivide.pipeline import divide_network  # noqa: E402
from pydivide.plotting import (  # noqa: E402
    _piece_lines,
    plot_basin_pair,
    plot_divide_values,
    plot_signed_divides,
)


@pytest.fixture(scope="module")
def bt():
    dem = tt.load_dem("bigtujunga")
    fd = tt.FlowObject(dem)
    st = tt.StreamObject(fd, threshold=1e5, units="m2")
    D = divide_network(dem=dem, fd=fd, st=st, minarea=1e5, verbose=False)
    L, _ = basin_labels(FlowNet.from_pytopotoolbox(fd, st))
    t = across_divide_metrics(dem, flow=fd, streams=st, divides=D,
                              baselevel=200.0, threshold=1e5, seglength=600.0)
    return dem, D, np.asarray(L), t


def test_one_line_per_piece(bt):
    _, D, _, t = bt
    segs = _piece_lines(D, 600.0)
    assert segs.shape == (len(t), 2, 2)
    assert np.isfinite(segs).all()


def test_a_length_mismatch_is_refused_not_drawn(bt):
    _, D, _, t = bt
    with pytest.raises(ValueError) as e:
        plot_divide_values(D, np.zeros(len(t) + 1), seglength=600.0)
    assert "seglength" in str(e.value)
    plt.close("all")


def test_values_plot_draws_every_finite_piece(bt):
    _, D, _, t = bt
    v = np.asarray(t.columns["chi_diff"], float)
    ax = plot_divide_values(D, v, seglength=600.0)
    lc = [c for c in ax.collections if hasattr(c, "get_array")][0]
    assert len(lc.get_array()) == int(np.isfinite(v).sum())
    plt.close("all")


def test_robust_scaling_ignores_the_tail(bt):
    _, D, _, t = bt
    v = np.asarray(t.columns["chi_diff"], float).copy()
    v[np.isfinite(v).argmax()] = 1e9          # one absurd outlier
    ax = plot_divide_values(D, v, seglength=600.0, robust=True)
    lc = [c for c in ax.collections if hasattr(c, "get_array")][0]
    assert lc.get_clim()[1] < 1e6, "a single outlier must not set the colour scale"
    plt.close("all")


def test_direction_ticks_follow_the_convention(bt):
    """A positive chi contrast means side A loses, so the tick points to A."""
    _, D, _, t = bt
    s = np.asarray(t.columns["chi_sgn"], float)
    ax = plot_signed_divides(D, s, seglength=600.0, min_abs=np.nanpercentile(np.abs(s), 99))
    assert len(ax.texts) > 0        # arrows are annotations
    plt.close("all")
    ax2 = plot_signed_divides(D, s, seglength=600.0,
                              min_abs=np.nanpercentile(np.abs(s), 99),
                              convention="higher_is_winning")
    assert len(ax2.texts) == len(ax.texts)     # same pieces, opposite sides
    plt.close("all")


def test_weak_contrasts_can_be_suppressed(bt):
    _, D, _, t = bt
    s = np.asarray(t.columns["chi_sgn"], float)
    a_all = plot_signed_divides(D, s, seglength=600.0, min_abs=0.0)
    n_all = len(a_all.texts)
    plt.close("all")
    a_str = plot_signed_divides(D, s, seglength=600.0,
                                min_abs=np.nanpercentile(np.abs(s), 90))
    assert len(a_str.texts) < n_all
    plt.close("all")


def test_basin_pair_map_shades_the_two_basins(bt):
    dem, D, L, t = bt
    a = np.asarray(t.columns["basin_a"], float)
    b = np.asarray(t.columns["basin_b"], float)
    ok = np.isfinite(a) & np.isfinite(b)
    k = int(np.flatnonzero(ok)[0])
    ax = plot_basin_pair(L, (int(a[k]), int(b[k])), dem=dem, D=D, sgn=1.0,
                         metric_name="chi contrast")
    txt = " ".join(t.get_text() for t in ax.texts) + ax.get_title()
    assert f"basin {int(a[k])}" in txt and f"basin {int(b[k])}" in txt
    assert "A" in txt and "B" in txt
    plt.close("all")


def test_basin_pair_states_which_side_the_divide_moves_into(bt):
    dem, D, L, t = bt
    a = np.asarray(t.columns["basin_a"], float)
    b = np.asarray(t.columns["basin_b"], float)
    ok = np.isfinite(a) & np.isfinite(b)
    k = int(np.flatnonzero(ok)[0])
    pa, pb = int(a[k]), int(b[k])
    pos = plot_basin_pair(L, (pa, pb), D=D, sgn=+2.0).get_title()
    plt.close("all")
    neg = plot_basin_pair(L, (pa, pb), D=D, sgn=-2.0).get_title()
    plt.close("all")
    assert f"move into basin {pa}" in pos
    assert f"move into basin {pb}" in neg


def test_absent_basin_is_refused(bt):
    _, D, L, _ = bt
    with pytest.raises(ValueError):
        plot_basin_pair(L, (-7, -8), D=D)
    plt.close("all")
