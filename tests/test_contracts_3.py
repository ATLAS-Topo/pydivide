# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Contracts on edge cases of the pair, the export and the map:
a short closed divide keeps a direction at its stations; a pair needs
two different positive labels, so a fine divide inside one coarse basin
has no pair rather than an error; an export with no stream cell above
the chi baselevel, or above the export baselevel, is a table and not a
crash; a zero contrast implies no direction and the basin-pair map draws
the shared divide; a flow object of another grid is refused by the
reach-label call; the CRS repair script fails on a missing file and
writes nothing on a dry run; a single-node segment's midpoint is its own
node."""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydivide.basinpairs import NO_PAIR, multiscale_pair_ids, pair_ids, pair_table  # noqa: E402
from pydivide.divmetrics import segment_midpoints  # noqa: E402
from pydivide.gridspec import GridSpec  # noqa: E402
from pydivide.orderdist import SortedNetwork  # noqa: E402
from pydivide.pipeline import DivideNetworkResult  # noqa: E402
from pydivide.profiles import basin_pair_profiles, station_frames  # noqa: E402

CS = 30.0


def loop_network():
    """A 120 m square divide around the one cell of basin 9 inside basin 3,
    on a 5 x 5 grid: the smallest closed divide there is."""
    nodes = np.array([14, 15, 21, 20, 14], np.int64)
    edges = np.sort(np.column_stack([nodes[:-1], nodes[1:]]), axis=1)
    net = SortedNetwork(nodes, np.array([0, 5], np.int64), np.ones(1, bool),
                        np.zeros(0, np.int64), np.zeros(0, np.int64))
    labels = np.full((5, 5), 3, np.int32)
    labels[2, 2] = 9
    D = DivideNetworkResult(GridSpec(5, 5, CS, 0.0, 150.0), net,
                            np.array([0.0, 30.0, 60.0, 30.0, 0.0]), edges, np.zeros(4, bool),
                            "EPSG:32623", labels)
    dem = SimpleNamespace(z=np.fromfunction(lambda r, c: 100 + 10 * r + 5 * c, (5, 5)))
    return D, dem


# ---- profiles: a closed divide shorter than the station spacing -----------

def test_short_closed_loop_has_a_real_normal():
    D, dem = loop_network()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = basin_pair_profiles(D, dem, pair=(3, 9), half_length=30.0, step=30.0)
    nrm = p.paths[0].normals["normal"]
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0), nrm
    assert np.allclose(p.gis_layers()["normals"].geometry.length, 60.0)
    z = p.across[0]["z"]
    assert np.isfinite(z).all() and z.max() > z.min()      # a real cross-section, not a flat one


def test_station_frames_wrap_a_closed_path_and_clip_an_open_one():
    square = np.array([[0, 0], [30, 0], [30, 30], [0, 30], [0, 0]], float)
    s, pt, t, n = station_frames(square, 5000.0)
    assert len(s) == 1 and np.allclose(np.hypot(t[:, 0], t[:, 1]), 1.0)
    assert np.allclose(t[:, 0] * n[:, 0] + t[:, 1] * n[:, 1], 0.0)
    s, pt, t, n = station_frames(square[:3], 5000.0)
    assert len(s) == 1 and np.allclose(np.hypot(t[:, 0], t[:, 1]), 1.0)
    # out and back along one line, treated as open: the clipped chord is
    # zero and the edge under the station gives the direction
    s, pt, t, n = station_frames(np.array([[0, 0], [10, 0], [0, 0]], float), 5000.0,
                                 closed=False)
    assert len(s) == 1 and np.allclose(np.hypot(t[:, 0], t[:, 1]), 1.0)


# ---- basin pairs: two different positive labels -----------------------------

def test_pair_ids_need_two_different_positive_labels():
    pid, lookup = pair_ids([0, 3, 5, 7, np.nan], [9, 3, 5, 8, 2])
    assert pid.tolist() == [-1, -1, -1, 0, -1]
    assert lookup["basin_lo"].tolist() == [7] and lookup["basin_hi"].tolist() == [8]
    assert len(pair_table([3, 3, 4], [3, 5, 5], scale="1e6")) == 2


def test_multiscale_pair_ids_keep_no_pair_where_the_coarse_labels_agree():
    # a divide of the 1e5 network can lie inside one basin of the 1e6 partition
    cols = {"basin_a_1e5": np.array([1.0, 2.0, 3.0, 4.0]),
            "basin_b_1e5": np.array([2.0, 3.0, 4.0, 5.0]),
            "basin_a_1e6": np.array([1.0, 1.0, 1.0, 2.0]),
            "basin_b_1e6": np.array([1.0, 1.0, 2.0, 2.0])}
    tables = multiscale_pair_ids(SimpleNamespace(columns=cols))
    assert cols["pair_id_1e6"].tolist() == [-1, -1, 0, -1]
    assert cols["pair_key_1e6"].tolist() == [NO_PAIR, NO_PAIR, 600000010000002, NO_PAIR]
    assert cols["pair_uid_1e6"].tolist() == ["", "", "1e6:0", ""]
    assert len(tables["1e6"]) == 1 and len(tables["1e5"]) == 4


# ---- the basin-pair map ----------------------------------------------------

def test_zero_contrast_claims_no_direction_and_the_divide_is_drawn():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pydivide.plotting import plot_basin_pair
    D, dem = loop_network()
    ax = plot_basin_pair(D.labels, (3, 9), D=D, sgn=0.0)
    title = ax.get_title()
    assert "higher on basin" not in title and "expected to move" not in title
    assert "no signed contrast" in title
    assert any(len(line.get_xdata()) >= 12 for line in ax.lines)   # four edges, drawn
    assert not any(t.arrow_patch is not None for t in ax.texts)
    plt.close(ax.figure)
    # a straight divide between a left and a right basin: the title names
    # the losing side and an arrow points from the divide into it
    nodes = np.array([3, 9, 15, 21, 27, 33], np.int64)
    edges = np.column_stack([nodes[:-1], nodes[1:]])
    net = SortedNetwork(nodes, np.array([0, 6], np.int64), np.zeros(1, bool),
                        np.array([3, 33], np.int64), np.zeros(0, np.int64))
    labels = np.full((5, 5), 3, np.int32)
    labels[:, 3:] = 9
    S = DivideNetworkResult(GridSpec(5, 5, CS, 0.0, 150.0), net, np.arange(6) * CS, edges,
                            np.zeros(5, bool), "EPSG:32623", labels)
    ax = plot_basin_pair(labels, (3, 9), D=S, sgn=1.5)
    assert "expected to move into basin 3" in ax.get_title()
    assert any(len(line.get_xdata()) >= 15 for line in ax.lines)   # five edges, drawn
    assert any(t.arrow_patch is not None for t in ax.texts)
    plt.close(ax.figure)


def _pair_map_network(labels, grid):
    """A network holding exactly the faces between differently labelled
    cells, as the map needs it."""
    from pydivide.geometry import _faces_from_labels
    edges, la, lb = _faces_from_labels(grid, labels)
    edges = edges[(la > 0) & (lb > 0)]
    net = SortedNetwork(np.zeros(0, np.int64), np.array([0], np.int64), np.zeros(0, bool),
                        np.zeros(0, np.int64), np.zeros(0, np.int64))
    return DivideNetworkResult(grid, net, np.zeros(0), edges, np.zeros(len(edges), bool),
                               None, labels)


def _arrow_of(ax):
    arrows = [t for t in ax.texts if t.arrow_patch is not None]
    assert len(arrows) == 1, len(arrows)
    return np.array(arrows[0].xyann, float), np.array(arrows[0].xy, float)


def _label_at(labels, grid, xy):
    r = int(np.floor((grid.y0 - xy[1]) / grid.cellsize))
    c = int(np.floor((xy[0] - grid.x0) / grid.cellsize))
    return int(labels[r, c])


def test_pair_arrow_starts_on_the_divide_and_ends_inside_the_losing_basin():
    """A wide U-shaped basin: its mean cell lies outside it, and the mean of
    the divide's edge midpoints lies off the divide, so neither may anchor
    the arrow."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shapely

    from pydivide.plotting import plot_basin_pair
    labels = np.full((30, 30), 3, np.int32)
    labels[2:28, 2:28] = 9
    labels[2:26, 4:26] = 3
    grid = GridSpec(30, 30, CS, 0.0, 900.0)
    D = _pair_map_network(labels, grid)
    x, y = grid.node_xy(D.edge_nodes.ravel())
    divide = shapely.MultiLineString(np.column_stack([x, y]).reshape(-1, 2, 2).tolist())
    for sgn, loser in ((-1.0, 9), (1.0, 3)):
        ax = plot_basin_pair(labels, (3, 9), D=D, sgn=sgn)
        start, end = _arrow_of(ax)
        assert divide.distance(shapely.Point(start)) < 1e-8, start
        assert _label_at(labels, grid, end) == loser, (sgn, end)
        assert np.hypot(*(end - start)) >= 0.5 * CS
        # the annotation of each basin sits inside that basin
        for t in ax.texts:
            if t.arrow_patch is None and "basin" in t.get_text():
                lab = int(t.get_text().split("basin ")[1])
                assert _label_at(labels, grid, np.array(t.xy, float)) == lab, t.get_text()
        plt.close(ax.figure)


def test_closed_pair_with_nonzero_contrast_has_an_arrow_into_the_named_basin():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pydivide.plotting import plot_basin_pair
    labels = np.full((9, 9), 3, np.int32)
    labels[4, 4] = 9
    grid = GridSpec(9, 9, CS, 0.0, 270.0)
    D = _pair_map_network(labels, grid)
    for sgn, loser in ((-1.0, 9), (1.0, 3)):
        ax = plot_basin_pair(labels, (3, 9), D=D, sgn=sgn)
        start, end = _arrow_of(ax)
        assert _label_at(labels, grid, end) == loser, (sgn, end)
        plt.close(ax.figure)


def test_station_frames_are_invariant_to_the_map_origin():
    xy = np.array([[0, 0], [60, 0], [60, 90], [30, 90], [30, 30], [0, 30]], float)
    here = station_frames(xy, 5000.0)[2]
    there = station_frames(xy + [500000.0, 8000000.0], 5000.0)[2]
    assert np.allclose(here, there), (here, there)


# ---- the CRS repair script -------------------------------------------------

def test_crs_repair_cli_fails_on_a_missing_file_and_a_dry_run_writes_nothing(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "fix_gpkg_srs.py"
    missing = tmp_path / "missing.gpkg"
    r = subprocess.run([sys.executable, str(script), str(missing), "--dry-run"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and not missing.exists(), (r.returncode, r.stdout, r.stderr)
    r = subprocess.run([sys.executable, str(script), str(missing)], capture_output=True, text=True)
    assert r.returncode != 0 and not missing.exists()
    gpd = pytest.importorskip("geopandas")
    import shapely
    gp = tmp_path / "x.gpkg"
    gpd.GeoDataFrame({"a": [1]}, geometry=[shapely.LineString([(0, 0), (1, 1)])],
                     crs="EPSG:32623").to_file(gp)
    before = gp.read_bytes()
    r = subprocess.run([sys.executable, str(script), str(gp), "--dry-run"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert gp.read_bytes() == before


# ---- a single-node segment -------------------------------------------------

def test_segment_midpoint_of_a_later_single_node_segment_is_its_own_node():
    grid = GridSpec(5, 5, CS, 0.0, 150.0)
    net = SimpleNamespace(seg_nodes=np.array([0, 1, 8], np.int64),
                          seg_ptr=np.array([0, 2, 3], np.int64), n_segments=2)
    xm, ym = segment_midpoints(net, grid)
    x8, y8 = grid.node_xy(np.array([8], np.int64))
    assert np.isclose(xm[1], x8[0]) and np.isclose(ym[1], y8[0])
    x01, y01 = grid.node_xy(np.array([0, 1], np.int64))
    assert np.isclose(xm[0], x01.mean()) and np.isclose(ym[0], y01.mean())


# ---- with routing: stream export and the reach-label call ------------------

tt3 = pytest.importorskip("topotoolbox")


def _surface(seed: int, n: int = 120, x0: float = 0.0):
    """A georeferenced synthetic surface, north-up, its west edge at ``x0``."""
    from affine import Affine
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    z = (600.0 - 0.0009 * ((x - n / 2) ** 2 + (y - n / 2) ** 2)
         + 45.0 * np.sin(x / 9.0 + seed) * np.cos(y / 11.0)
         + 18.0 * np.sin((x + y) / 7.0) + rng.normal(0, 0.5, (n, n)))
    dem = tt3.GridObject()
    dem.z = np.ascontiguousarray(z.astype(np.float32))
    dem.cellsize = CS
    dem.transform = Affine(CS, 0.0, x0, 0.0, -CS, n * CS)
    return dem


@pytest.fixture(scope="module")
def routed():
    dem = _surface(1)
    fd = tt3.FlowObject(dem)
    st = tt3.StreamObject(fd, threshold=200)
    return dem, fd, st


def test_chi_baselevel_above_the_dem_gives_nan_chi_not_a_crash(routed):
    from pydivide.streams import stream_segments_gdf
    dem, fd, st = routed
    top = float(np.nanmax(dem.z))
    gdf = stream_segments_gdf(fd, st, dem, seglength=300, chi_field="chi", baselevel=0.0,
                              chi_baselevel=top + 100.0)
    assert len(gdf) > 0
    assert gdf["chi"].isna().all()


def test_nothing_above_the_export_baselevel_gives_an_empty_table(routed):
    from pydivide.streams import stream_segments_gdf
    dem, fd, st = routed
    top = float(np.nanmax(dem.z))
    gdf = stream_segments_gdf(fd, st, dem, seglength=300, chi_field="chi", baselevel=top + 100.0)
    assert len(gdf) == 0
    assert {"segid", "len_m", "nnode", "upakm2", "elev", "chi"} <= set(gdf.columns)


@pytest.mark.parametrize("formats", [("gpkg",), ("shp",), ("gpkg", "shp")])
def test_empty_stream_export_completes_for_every_format(routed, tmp_path, formats):
    """An export with no stream cell above the baselevel writes an empty
    GeoPackage layer, no Shapefile, and does not abort; a stale Shapefile set at the
    name is removed under overwrite and refused without it."""
    from pydivide.streams import export_stream_segments
    dem, fd, st = routed
    top = float(np.nanmax(dem.z))
    base = tmp_path / "streams"
    if "shp" in formats:
        stale = tmp_path / "streams.shp"
        stale.write_bytes(b"stale")
        with pytest.raises(FileExistsError):
            export_stream_segments(fd, st, dem, base, seglength=300, chi_field="chi",
                                   baselevel=top + 100.0, formats=formats)
    export_stream_segments(fd, st, dem, base, seglength=300, chi_field="chi",
                           baselevel=top + 100.0, formats=formats, overwrite=True)
    assert not list(tmp_path.glob("streams*.shp"))       # the stale set is gone, nothing written
    if "gpkg" in formats:
        import geopandas as gpd
        assert len(gpd.read_file(base.with_suffix(".gpkg"))) == 0


def test_empty_streams_keep_the_schema_of_a_full_export(routed, tmp_path):
    """Same column types as a non-empty export (integers stay integers) and a
    GeoPackage layer that declares LineString."""
    import pyogrio

    from pydivide.streams import stream_segments_gdf, write_stream_gdf
    dem, fd, st = routed
    top = float(np.nanmax(dem.z))
    full = stream_segments_gdf(fd, st, dem, seglength=300, chi_field="chi", baselevel=0.0)
    empty = stream_segments_gdf(fd, st, dem, seglength=300, chi_field="chi", baselevel=top + 100)
    assert len(full) > 0 and len(empty) == 0
    assert full.dtypes.to_dict() == empty.dtypes.to_dict()
    assert empty.crs == full.crs
    write_stream_gdf(empty, tmp_path / "streams", formats=("gpkg",))
    info = pyogrio.read_info(tmp_path / "streams.gpkg")
    assert info["geometry_type"] == "LineString" and info["features"] == 0


def test_shapefile_overwrite_touches_only_its_own_set(routed, tmp_path):
    """Replacing a Shapefile removes the stale parts of the previous export
    at that name and nothing else: a numbered sibling of another kind
    survives."""
    from pydivide.streams import stream_segments_gdf, write_stream_gdf
    from pydivide.vectorio import existing_parts
    dem, fd, st = routed
    top = float(np.nanmax(dem.z))
    full = stream_segments_gdf(fd, st, dem, seglength=300, chi_field="chi", baselevel=0.0)
    empty = stream_segments_gdf(fd, st, dem, seglength=300, chi_field="chi", baselevel=top + 100)
    keep = {tmp_path / n: n.encode() for n in ("streams_part001.tif", "streams_part001.csv",
                                                "streams_part001.json", "streams_notes.txt")}
    for p, content in keep.items():
        p.write_bytes(content)
    stale = [tmp_path / n for n in ("streams_part001.shp", "streams_part001.dbf",
                                    "streams_part001_fields.csv", "streams_part002.shp.xml")]
    for p in stale:
        p.write_bytes(b"stale")
    assert sorted(existing_parts(tmp_path / "streams.shp")) == sorted(stale)
    for gdf in (full, empty):
        for p in stale:
            p.write_bytes(b"stale")
        write_stream_gdf(gdf, tmp_path / "streams", formats=("shp",), overwrite=True)
        assert not any(p.exists() for p in stale)
        for p, content in keep.items():
            assert p.exists() and p.read_bytes() == content, p


def _names_alias(folder) -> bool:
    """Whether this folder's file system treats ``Probe`` and ``probe`` as
    one file (a probe file is written and removed)."""
    p = folder / "Probe_case.txt"
    p.write_bytes(b"")
    try:
        return (folder / "probe_case.txt").exists()
    finally:
        p.unlink()


def test_stale_parts_are_found_when_the_name_differs_only_in_case(routed, tmp_path):
    from pydivide.streams import stream_segments_gdf, write_stream_gdf
    if not _names_alias(tmp_path):
        pytest.skip("the file system compares names with case")
    dem, fd, st = routed
    top = float(np.nanmax(dem.z))
    full = stream_segments_gdf(fd, st, dem, seglength=300, chi_field="chi", baselevel=0.0)
    empty = stream_segments_gdf(fd, st, dem, seglength=300, chi_field="chi", baselevel=top + 100)
    write_stream_gdf(full, tmp_path / "Streams", formats=("shp",), max_features=5, overwrite=True)
    assert list(tmp_path.glob("Streams_part*.shp"))
    write_stream_gdf(empty, tmp_path / "streams", formats=("shp",), overwrite=True)
    assert not list(tmp_path.glob("Streams_part*.shp")) and not list(tmp_path.glob("*.shp"))


def _one_line(crs="EPSG:32623"):
    import geopandas as gpd
    import shapely
    return gpd.GeoDataFrame({"segid": [1]}, geometry=[shapely.LineString([(0, 0), (30, 30)])],
                            crs=crs)


@pytest.mark.parametrize("empty", [False, True])
def test_shapefile_overwrite_keeps_a_distinct_unicode_spelling(tmp_path, empty):
    """A dotted capital I folds to ``i`` in a regular expression but names a
    different file on every file system: the overwrite of ``divides`` must
    leave ``divİdes_part001.shp`` alone."""
    from pydivide.streams import write_stream_gdf
    other = tmp_path / "divİdes_part001.shp"
    stale = tmp_path / "divides_part001.shp"
    other.write_bytes(b"another dataset")
    if stale.exists():
        pytest.skip("the two spellings alias on this file system")
    stale.write_bytes(b"old part")
    gdf = _one_line()
    write_stream_gdf(gdf.iloc[:0].copy() if empty else gdf, tmp_path / "divides",
                     formats=("shp",), overwrite=True)
    assert not stale.exists()
    assert other.exists() and other.read_bytes() == b"another dataset"


@pytest.mark.parametrize("empty", [False, True])
def test_shapefile_overwrite_keeps_a_distinct_case_spelling(tmp_path, empty):
    """On a case-sensitive file system ``Streams_part001.shp`` is another
    dataset next to ``streams``: the overwrite removes the stale lower-case
    part and keeps the other."""
    from pydivide.streams import write_stream_gdf
    if _names_alias(tmp_path):
        pytest.skip("the two spellings alias on this file system")
    other = tmp_path / "Streams_part001.shp"
    stale = tmp_path / "streams_part001.shp"
    other.write_bytes(b"another dataset")
    stale.write_bytes(b"old part")
    assert not other.samefile(stale)
    gdf = _one_line()
    write_stream_gdf(gdf.iloc[:0].copy() if empty else gdf, tmp_path / "streams",
                     formats=("shp",), overwrite=True)
    assert not stale.exists()
    assert other.exists() and other.read_bytes() == b"another dataset"


@pytest.mark.parametrize("name", ["streams_part001.shp\n", "streams_part٠٠١.shp"])
def test_shapefile_overwrite_keeps_names_the_writer_never_produces(tmp_path, name):
    """A final newline (``$`` stops before it) and digits of another script
    (``\\d`` takes them) are not the writer's naming: such a neighbour
    survives."""
    from pydivide.streams import write_stream_gdf
    other = tmp_path / name
    try:
        other.write_bytes(b"outside the writer's naming")
    except OSError as e:
        pytest.skip(f"this file system refuses the name: {e}")
    if (tmp_path / "streams_part001.shp").exists():
        pytest.skip("the spellings alias on this file system")
    write_stream_gdf(_one_line().iloc[:0].copy(), tmp_path / "streams", formats=("shp",),
                     overwrite=True)
    assert other.exists() and other.read_bytes() == b"outside the writer's naming", repr(name)


@pytest.mark.parametrize("suffix,canonical", [(".ſhp", ".shp"), ("_fİelds.csv", "_fields.csv")])
@pytest.mark.parametrize("empty", [False, True])
def test_unicode_suffix_spellings_do_not_abort_an_export(tmp_path, suffix, canonical, empty):
    """A suffix that the case-insensitive match accepts but ``lower()`` would
    not resolve (a long s, a dotted capital I) must neither raise nor be
    deleted unless the file system says it is the writer's own file."""
    from pydivide.streams import write_stream_gdf
    other = tmp_path / f"streams_part001{suffix}"
    other.write_bytes(b"original content")
    own = tmp_path / f"streams_part001{canonical}"
    aliases = own.exists() and other.samefile(own)
    gdf = _one_line()
    write_stream_gdf(gdf.iloc[:0].copy() if empty else gdf, tmp_path / "streams",
                     formats=("shp",), overwrite=True)
    assert other.exists() == (not aliases)
    if not aliases:
        assert other.read_bytes() == b"original content"
    assert (tmp_path / "streams.shp").exists() == (not empty)


def test_arrow_tip_is_a_cell_centre_inside_a_narrow_basin():
    """A one-cell-wide basin in four orientations: the tip is the centre of
    a cell of the losing basin, never a point on its far boundary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shapely

    from pydivide.plotting import plot_basin_pair
    base = np.full((30, 30), 3, np.int32)
    base[2:28, 2:28] = 9
    base[2:27, 3:27] = 3
    grid = GridSpec(30, 30, CS, 500000.0, 8000000.0)
    for rot in range(4):
        labels = np.rot90(base, rot).copy()
        D = _pair_map_network(labels, grid)
        r, c = np.where(labels == 9)
        basin = shapely.union_all(shapely.box(grid.x0 + c * CS, grid.y0 - (r + 1) * CS,
                                              grid.x0 + (c + 1) * CS, grid.y0 - r * CS))
        for sgn, loser in ((-1.0, 9), (1.0, 3)):
            ax = plot_basin_pair(labels, (3, 9), D=D, sgn=sgn)
            start, end = _arrow_of(ax)
            plt.close(ax.figure)
            assert _label_at(labels, grid, end) == loser
            if loser == 9:
                assert basin.contains(shapely.Point(end)), (rot, end)
            # a cell centre: half a cell from the cell's edges in both axes
            fx = ((end[0] - grid.x0) / CS) % 1.0
            fy = ((grid.y0 - end[1]) / CS) % 1.0
            assert abs(fx - 0.5) < 1e-6 and abs(fy - 0.5) < 1e-6, (rot, sgn, end)


def test_run_area_survives_an_empty_stream_export_with_shp(routed, tmp_path):
    """The documented command with ``--shp`` and a stream baselevel above the
    DEM builds the divides and exits 0."""
    pytest.importorskip("psutil")
    import rasterio
    dem, fd, st = routed
    tif = tmp_path / "dem.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=dem.z.shape[0], width=dem.z.shape[1],
                       count=1, dtype="float32", crs="EPSG:32623", transform=dem.transform) as dst:
        dst.write(dem.z, 1)
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_area.py"
    r = subprocess.run([sys.executable, str(script), "--dem", str(tif), "--name", "empty",
                        "--root", str(tmp_path), "--minarea", str(200 * CS * CS),
                        "--chi-baselevels", "5", "--stream-baselevel", "5000", "--shp"],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = tmp_path / "exports" / "empty"
    assert (out / "divides_full.gpkg").exists() and (out / "divides_seg150.shp").exists()
    assert (out / "streams_chi5m_seg300.gpkg").exists()
    assert not list(out.glob("streams_*.shp"))


def test_reach_basin_labels_refuse_a_flow_object_of_another_grid(routed):
    from pydivide import reach_basin_labels
    dem, fd, st = routed
    same = reach_basin_labels(dem, 200 * CS * CS, fd=fd)
    assert same.shape == dem.z.shape and same.max() > 0
    shifted = _surface(1, x0=3000.0)          # the same cells, 3 km further east
    with pytest.raises(ValueError, match="origin"):
        reach_basin_labels(shifted, 200 * CS * CS, fd=fd)
