# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""The basin-pair profiles, on controlled geometry.

What is tested is the definitions the figures rest on: stations sit
inside a path and never on its ends, the direction of a divide is its
chord, a normal points into the higher-labelled basin, a swath collapses
onto its profile where nothing varies along the divide, the along-divide
elevation is the mean of the two flanking cells, every edge of a pair
is drawn exactly once whatever the shape of its divide, the pair key
reads back, every pair of a partition is listed, and ``save`` writes
what it says it writes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydivide import profiles as P  # noqa: E402
from pydivide.basinpairs import decode_pair_key, pair_key, readable_pair_key  # noqa: E402
from pydivide.gridspec import GridSpec  # noqa: E402
from pydivide.orderdist import SortedNetwork  # noqa: E402
from pydivide.pipeline import DivideNetworkResult  # noqa: E402

# ---- the geometry --------------------------------------------------------------

def test_stations_sit_between_the_path_ends():
    s = P.station_positions(10_000.0, 2_000.0)
    assert np.allclose(s, [1000, 3000, 5000, 7000, 9000])
    assert np.allclose(P.station_positions(1_500.0, 2_000.0), [750.0])   # shorter than a spacing
    assert P.station_positions(0.0, 2_000.0).size == 0
    with pytest.raises(ValueError):
        P.station_positions(10.0, 0.0)


def test_a_straight_divide_gives_a_perpendicular_normal():
    xy = np.column_stack([np.linspace(0, 1000, 11), np.zeros(11)])
    s, p, t, n = P.station_frames(xy, 500.0)
    assert np.allclose(s, [250, 750]) and np.allclose(t, [[1, 0], [1, 0]])
    assert np.allclose(n, [[0, 1], [0, 1]])


def test_a_staircase_divide_has_the_direction_of_its_trend():
    """Cell faces alternate east and north; the chord over a window is
    the diagonal, not any single face."""
    pts = [(0.0, 0.0)]
    for _ in range(40):
        x, y = pts[-1]
        pts.append((x + 30, y))
        pts.append((x + 30, y + 30))
    xy = np.asarray(pts)
    s, p, t, n = P.station_frames(xy, 600.0, tangent_window=600.0)
    assert np.allclose(np.abs(t[:, 0]), np.abs(t[:, 1]), atol=0.05)


def test_normals_are_turned_towards_basin_hi():
    n = np.array([[0.0, 1.0], [0.0, 1.0]])
    p = np.array([[0.0, 0.0], [10.0, 0.0]])
    hi = np.array([[0.0, -5.0], [10.0, np.nan]])          # below the first; unknown at the second
    out = P.orient_normals(n, p, hi)
    assert np.allclose(out, [[0, -1], [0, 1]])


def test_the_side_of_a_normal_comes_from_its_window_not_from_one_face():
    """A divide running east with basin hi to the south and a single step
    north half way. A station on that step, below its middle, lies further
    from the flanking hi cell along the normal (5 m) than beside it (15 m
    east), so that cell alone turns the normal north, into lo; the faces
    over the window turn it south."""
    pts = [(30.0 * k, 0.0) for k in range(11)] + [(300.0 + 30.0 * k, 30.0) for k in range(11)]
    xy = np.asarray(pts)
    s, p, t, n = P.station_frames(xy, 620.0)
    assert np.allclose(s, [310.0]) and np.allclose(p, [[300.0, 10.0]])
    inward = np.array([[0.0, -30.0]] * 10 + [[30.0, 0.0]] + [[0.0, -30.0]] * 10)
    edge_s = (np.arange(21) + 0.5) * 30.0
    good = P.orient_normals(n, p, P.side_points(p, s, edge_s, inward, 620.0))
    assert good[0, 1] < -0.99                                  # south, into hi
    one_face = P.orient_normals(n, p, np.array([[315.0, 15.0]]))
    assert one_face[0, 1] > 0.99                               # one face alone: north, into lo


def test_normals_start_into_basin_hi_along_real_divides():
    """Every section of the 15 longest pairs of a real partition leaves the
    divide into basin hi on its positive side and into lo on its negative
    side, a few cells out, wherever both sides are clear."""
    tt = pytest.importorskip("topotoolbox")
    from pydivide.labels import reach_basin_labels
    from pydivide.pipeline import divide_network
    dem = tt.load_dem("bigtujunga")
    fd = tt.FlowObject(dem)
    st = tt.StreamObject(fd, threshold=1e6, units="m2")
    D = divide_network(fd=fd, st=st, verbose=False)
    L = reach_basin_labels(dem, 1e7, fd=fd)
    g = D.grid

    def label(x, y):
        r, c = int((g.y0 - y) // g.cellsize), int((x - g.x0) // g.cellsize)
        return int(L[r, c]) if 0 <= r < g.nr and 0 <= c < g.nc else -1

    pairs = P.basin_pairs(D, dem, labels=L, threshold=1e7)
    right = wrong = 0
    for lo, hi in zip(pairs.basin_lo[:15], pairs.basin_hi[:15]):
        pp = P.basin_pair_profiles(D, dem, pair=(int(lo), int(hi)), labels=L, threshold=1e7,
                                   spacing=1000.0, half_length=300.0)
        for path in pp.paths:
            r = path.normals
            if r is None:
                continue
            for (x, y), (u, v) in zip(r["station_xy"], r["normal"]):
                pos = {label(x + d * u, y + d * v) for d in (45.0, 90.0)}
                neg = {label(x - d * u, y - d * v) for d in (45.0, 90.0)}
                right += pos == {hi} and neg == {lo}
                wrong += pos == {lo} and neg == {hi}
    assert right > 100 and wrong == 0


def test_bilinear_reads_a_plane_exactly_and_refuses_off_grid_and_nodata():
    z = np.fromfunction(lambda r, c: 100 + 2 * c - 3 * r, (20, 30)).astype(np.float32)
    x0, y0, cs = 1000.0, 5000.0, 10.0
    xs = np.array([1005.0, 1123.0, 900.0, 1295.0])
    ys = np.array([4995.0, 4881.5, 4900.0, 4805.0])
    got = P.bilinear(z, x0, y0, cs, xs, ys)
    col = (xs - x0) / cs - 0.5
    row = (y0 - ys) / cs - 0.5
    assert np.allclose(got[:2], 100 + 2 * col[:2] - 3 * row[:2])
    assert np.isnan(got[2]) and np.isnan(got[3])       # west of the grid; last column
    z[9, 12] = -9999
    assert np.isnan(P.bilinear(z, x0, y0, cs, [1123.0], [4905.0], nodata=-9999)[0])


def test_a_swath_collapses_onto_its_profile_where_nothing_varies_along_the_divide():
    def ramp(xs, ys):                     # elevation depends on y only
        return 500.0 + 0.1 * np.asarray(ys, float)
    p = np.array([[0.0, 0.0]])
    t = np.array([[1.0, 0.0]])
    n = np.array([[0.0, 1.0]])
    line = P.sample_normals(ramp, p, t, n, 1000.0, 100.0)
    sw = P.sample_normals(ramp, p, t, n, 1000.0, 100.0, swath_width=400.0)
    assert sw["swath"] and np.allclose(sw["mean"], line["z"]) and np.allclose(sw["min"], sw["max"])
    assert np.allclose(sw["offset"], np.arange(-500, 501, 100))
    assert (sw["n"] == 5).all()


def test_station_letters():
    assert [P.station_label(k) for k in (0, 1, 25, 26, 27, 52)] == ["a", "b", "z", "aa", "ab", "ba"]


# ---- paths of a pair ----------------------------------------------------------------

def test_every_edge_of_a_forked_divide_is_drawn_exactly_once():
    # a stem 0-1-2, forking at 2 into 2-3-4 and 2-5; plus a separate stretch 7-8
    e = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [2, 5], [7, 8]])
    paths = P.order_paths(e)
    drawn = np.concatenate([p[1] for p in paths])
    assert sorted(drawn.tolist()) == list(range(6))
    comps = {p[2] for p in paths}
    assert len(comps) == 2
    stops = {(p[3], p[4]) for p in paths}
    assert {(P.STOP_ENDPOINT, P.STOP_BRANCHING), (P.STOP_BRANCHING, P.STOP_ENDPOINT)} & stops


def test_a_closed_loop_is_one_cycle():
    e = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    paths = P.order_paths(e)
    assert len(paths) == 1 and paths[0][5] is True and len(paths[0][1]) == 4


# ---- a controlled network: two basins, a two-edge divide -----------------------

def network(labels, grid=None, crs="EPSG:32623"):
    sn = SortedNetwork(np.array([1, 4, 7], np.int64), np.array([0, 3], np.int64),
                       np.zeros(1, bool), np.array([1, 7], np.int64), np.array([], np.int64))
    return DivideNetworkResult(grid or GridSpec(2, 2, 30., 0., 60.), sn,
                               np.array([0., 30., 0.]), np.array([[1, 4], [4, 7]], np.int64),
                               np.zeros(2, bool), crs, np.asarray(labels, np.int32))


def test_along_elevation_is_the_mean_of_the_two_flanking_cells():
    D = network([[3, 9], [3, 9]])
    dem = SimpleNamespace(z=np.array([[30., 10.], [20., 5.]], np.float32))
    with pytest.warns(RuntimeWarning):
        p = P.basin_pair_profiles(D, dem, pair=(9, 3), labels=D.labels, spacing=30,
                                  half_length=30, step=30, threshold=1e6)
    assert len(p.paths) == 1 and p.settings["n_edges"] == 2
    assert np.allclose(sorted(p.along_z), [12.5, 20.0])        # (30+10)/2 and (20+5)/2
    assert np.allclose(p.along_s, [15.0, 45.0])
    assert (p.label_a, p.label_b) == (3, 9) and p.key == pair_key(3, 9, 1e6)
    assert p.readable == "1e6 | 3 | 9"
    # the sections: negative offsets in basin 3 (the west column), positive in 9
    for sec in p.across:
        x_neg, x_pos = sec["xy"][0, 0], sec["xy"][-1, 0]
        assert x_neg < x_pos and len(sec["d"]) == 3
    t = p.along_table()
    assert list(t.columns[:5]) == ["pair_key", "basin_lo", "basin_hi", "component", "path"]
    n = p.normals_table()
    assert set(n.station_label) == {"a", "b"}


def test_pair_key_reads_back_and_refuses_the_wrong_domain():
    assert pair_key(3461, 3460, 1e7) == 700034600003461
    assert decode_pair_key(700034600003461) == (1e7, 3460, 3461)
    assert readable_pair_key(700034600003461) == "1e7 | 3460 | 3461"
    assert pair_key([0, 5], [2, 7], 1e6).tolist() == [-1, 600000050000007]
    with pytest.raises(ValueError):
        pair_key(1, 2, 5e6)
    with pytest.raises(ValueError):
        pair_key(4, 4, 1e6)
    with pytest.raises(ValueError):
        pair_key(1, 10 ** 7, 1e6)


def test_save_writes_the_four_figures_the_tables_and_the_geopackage(tmp_path):
    pytest.importorskip("matplotlib")
    D = network([[3, 9], [3, 9]])
    dem = SimpleNamespace(z=np.array([[30., 10.], [20., 5.]], np.float32))
    with pytest.warns(RuntimeWarning):
        p = P.basin_pair_profiles(D, dem, pair=(3, 9), labels=D.labels, threshold=1e6,
                                  spacing=30, half_length=30, step=30)
    rep = p.save(tmp_path, formats=("png",))
    for f in ("1_along.png", "2_along_stations.png", "3_normals_grouped_0_0.png",
              "3_normals_grouped_0_0_all_stations.pdf", "4_normals_stacked_0_0.png",
              "along.csv", "normals.csv", "pair_600000030000009.gpkg"):
        assert f in rep["files"] and (tmp_path / f).exists(), f
    import pyogrio
    layers = {n for n, _ in pyogrio.list_layers(str(tmp_path / "pair_600000030000009.gpkg"))}
    assert layers == {"pair_paths", "stations", "normals"}


def test_every_pair_of_a_partition_is_listed_longest_first():
    D = network([[3, 9], [1, 9]])          # (3, 9) on one edge, (1, 9) on the other
    dem = SimpleNamespace(z=np.array([[30., 10.], [20., 5.]], np.float32))
    with pytest.warns(RuntimeWarning):
        t = P.basin_pairs(D, dem, labels=D.labels, threshold=1e6)
    assert len(t) == 2 and set(zip(t.basin_lo, t.basin_hi)) == {(3, 9), (1, 9)}
    assert (t.divide_length_m == 30.0).all() and (t.pair_key > 0).all()
    with pytest.warns(RuntimeWarning):
        assert len(P.basin_pairs(D, dem, labels=D.labels, min_length=31.0)) == 0
