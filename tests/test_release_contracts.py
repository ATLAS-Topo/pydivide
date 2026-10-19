# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Tests on real routing: checkpoint identity and resume,
the spatial across-divide table with its outlets, basin pairs and
settings document, the multiscale table and its side rule, global piece
ids, CRS handling in masks, direction arrows, and overwrite protection.
Every test builds its routing with pytopotoolbox on a synthetic surface;
``tests/test_contracts.py`` holds the contracts that need no routing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pydivide  # noqa: E402

tt3 = pytest.importorskip("topotoolbox")
CS = 30.0


def _surface(seed: int, n: int = 160):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    z = (600.0 - 0.0009 * ((x - n / 2) ** 2 + (y - n / 2) ** 2)
         + 45.0 * np.sin(x / 9.0 + seed) * np.cos(y / 11.0)
         + 18.0 * np.sin((x + y) / 7.0) + rng.normal(0, 0.5, (n, n)))
    dem = tt3.GridObject()
    dem.z = np.ascontiguousarray(z.astype(np.float32))
    dem.cellsize = CS
    return dem


@pytest.fixture(scope="module")
def dem():
    return _surface(1)


@pytest.fixture(scope="module")
def D(dem):
    fd = tt3.FlowObject(dem)
    st = tt3.StreamObject(fd, threshold=200)
    return pydivide.divide_network(fd=fd, st=st, minarea=200 * CS * CS)


@pytest.fixture(scope="module")
def table(dem):
    from pydivide import across_divide_metrics
    return across_divide_metrics(dem, threshold=200 * CS * CS, metrics=("elev", "hr"),
                                 seglength=300)


# ---- checkpoints bind to the inputs ---------------------------------------

def test_checkpoint_is_not_reused_for_a_different_dem_of_the_same_shape(tmp_path):
    ck = tmp_path / "ck"
    a = _surface(1)
    b = _surface(2)
    fa = tt3.FlowObject(a)
    sa = tt3.StreamObject(fa, threshold=200)
    Da = pydivide.divide_network(fd=fa, st=sa, minarea=200 * CS * CS,
                                 checkpoint_dir=ck, verbose=False)
    fb = tt3.FlowObject(b)
    sb = tt3.StreamObject(fb, threshold=200)
    Db = pydivide.divide_network(fd=fb, st=sb, minarea=200 * CS * CS,
                                 checkpoint_dir=ck, verbose=False)
    assert a.z.shape == b.z.shape
    # a fresh run of b, without any checkpoint, must equal the checkpointed run
    Dc = pydivide.divide_network(fd=fb, st=sb, minarea=200 * CS * CS, verbose=False)
    assert Db.n_segments == Dc.n_segments
    assert np.array_equal(Db.network.seg_nodes, Dc.network.seg_nodes)
    assert not np.array_equal(Da.network.seg_nodes, Db.network.seg_nodes)


def test_checkpoint_resumes_the_network_stage(tmp_path, capsys):
    ck = tmp_path / "ck"
    dem = _surface(3)
    fd = tt3.FlowObject(dem)
    st = tt3.StreamObject(fd, threshold=200)
    D1 = pydivide.divide_network(fd=fd, st=st, minarea=200 * CS * CS, checkpoint_dir=ck)
    capsys.readouterr()
    D2 = pydivide.divide_network(fd=fd, st=st, minarea=200 * CS * CS, checkpoint_dir=ck)
    out = capsys.readouterr().out
    assert "network: loading from checkpoint" in out
    assert np.array_equal(D1.network.seg_nodes, D2.network.seg_nodes)
    assert np.allclose(D1.distance, D2.distance, equal_nan=True)


# ---- the across-divide table ------------------------------------------------

def test_across_divide_table_is_spatial_and_carries_outlets(table, tmp_path):
    tab = table
    gdf = tab.to_geodataframe()
    import geopandas as gpd
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == len(tab) and gdf.geometry.notna().all()
    for c in ("ID", "segid", "order", "distance", "length", "azimuth", "isbnd", "isring",
              "elev_mean", "elev_diff", "elev_min", "elev_max", "elev_sgn", "dai",
              "basin_a", "basin_b", "outlet_a", "outlet_b", "basin_pair_mixed",
              "basin_pair_share"):
        assert c in tab.columns, c
    assert np.array_equal(tab.columns["ID"], np.arange(1, len(tab) + 1))
    ok = np.isfinite(tab.columns["basin_a"])
    assert np.all(np.isfinite(tab.columns["outlet_a"][ok]))
    assert set(np.unique(tab.columns["basin_pair_mixed"])) <= {0, 1}
    assert tab.settings["provenance"]["pydivide_version"]
    assert "computed by this call" in tab.settings["provenance"]["routing"]
    out = tmp_path / "t.gpkg"
    written = tab.to_file(out)
    back = gpd.read_file(out)
    assert len(back) == len(tab)
    doc = pydivide.read_export_settings(out)
    assert doc["settings"]["threshold"] == 200 * CS * CS
    assert (tmp_path / "t_settings.json") in written


def test_reported_pair_occurs_on_an_edge_of_the_piece(table, D):
    """The basin pair a piece carries must be one its edges border, and
    the share says how much of the piece that pair covers."""
    from pydivide.divmetrics import edge_cells
    tab = table
    L = D.labels.ravel()
    gs = D.grid
    _, p_start, p_stop = D._piece_table(300.0)
    nodes = D.network.seg_nodes
    a = np.asarray(tab.columns["basin_a"], float)
    b = np.asarray(tab.columns["basin_b"], float)
    share = np.asarray(tab.columns["basin_pair_share"], float)
    mixed = np.asarray(tab.columns["basin_pair_mixed"])
    checked = 0
    for k in range(len(p_start)):
        pos = np.arange(p_start[k], p_stop[k] - 1)
        if pos.size == 0:
            continue
        e = np.sort(np.column_stack([nodes[pos], nodes[pos + 1]]), axis=1)
        ca, cb = edge_cells(gs, e)
        la = np.where(ca >= 0, L[np.maximum(ca, 0)], -1)
        lb = np.where(cb >= 0, L[np.maximum(cb, 0)], -1)
        good = (la > 0) & (lb > 0) & (la != lb)
        pairs = {(min(x, y), max(x, y)) for x, y in zip(la[good], lb[good])}
        if not pairs:
            assert not np.isfinite(a[k]) and not np.isfinite(b[k])
            continue
        assert (min(a[k], b[k]), max(a[k], b[k])) in pairs, k
        assert 0 < share[k] <= 1
        assert (share[k] < 1) == bool(mixed[k]) or share[k] == 1
        checked += 1
    assert checked > 50


def test_unknown_metric_name_is_rejected(dem):
    from pydivide import across_divide_metrics
    with pytest.raises(ValueError, match="unknown metric"):
        across_divide_metrics(dem, threshold=200 * CS * CS, metrics=("elev", "relief"))


def test_objects_from_another_grid_are_refused(dem):
    from pydivide import across_divide_metrics
    other = _surface(4, n=120)
    fd = tt3.FlowObject(other)
    st = tt3.StreamObject(fd, threshold=200)
    with pytest.raises(ValueError, match="shape"):
        across_divide_metrics(dem, flow=fd, streams=st, threshold=200 * CS * CS,
                              metrics=("elev",))


def test_same_shape_objects_on_another_grid_are_refused(dem, D):
    """A DEM of the same shape shifted on the ground, or a divide network
    of another origin, cell size or CRS, must be refused, not sampled."""
    from dataclasses import replace

    from affine import Affine

    from pydivide import across_divide_metrics
    fd = tt3.FlowObject(dem)
    st = tt3.StreamObject(fd, threshold=200)
    shifted = _surface(1)
    g0 = D.grid                                  # the same cells, 1000 units further east
    shifted.transform = Affine(g0.cellsize, 0, g0.x0 + 1000, 0, -g0.cellsize, g0.y0)
    with pytest.raises(ValueError, match="origin"):
        across_divide_metrics(shifted, flow=fd, streams=st, divides=D,
                              threshold=200 * CS * CS, metrics=("elev",))
    g = D.grid
    for bad, msg in ((replace(D, grid=type(g)(g.nr, g.nc, g.cellsize, g.x0 + 1000, g.y0)),
                      "origin"),
                     (replace(D, grid=type(g)(g.nr, g.nc, g.cellsize * 2, g.x0, g.y0)),
                      "cell size")):
        with pytest.raises(ValueError, match=msg):
            across_divide_metrics(dem, flow=fd, streams=st, divides=bad,
                                  threshold=200 * CS * CS, metrics=("elev",))
    # a CRS is compared when both sides carry one: a georeferenced DEM
    # against a divide network, or a flow object, in another system
    geo = _surface(1)
    geo.georef = "EPSG:32623"
    fd_geo = tt3.FlowObject(geo)
    st_geo = tt3.StreamObject(fd_geo, threshold=200)
    D24 = replace(D, crs="EPSG:32624")
    with pytest.raises(ValueError, match="coordinate reference"):
        across_divide_metrics(geo, flow=fd_geo, streams=st_geo, divides=D24,
                              threshold=200 * CS * CS, metrics=("elev",))
    other = _surface(1)
    other.georef = "EPSG:32624"
    with pytest.raises(ValueError, match="coordinate reference"):
        across_divide_metrics(other, flow=fd_geo, streams=st_geo, divides=D24,
                              threshold=200 * CS * CS, metrics=("elev",))
    ok = across_divide_metrics(geo, flow=fd_geo, streams=st_geo,
                               divides=replace(D, crs="EPSG:32623"),
                               threshold=200 * CS * CS, metrics=("elev",))
    assert len(ok) == len(D.piece_selection(300.0))


def test_profiles_refuse_a_dem_on_another_grid(dem, D):
    from affine import Affine

    from pydivide import basin_pair_profiles
    g0 = D.grid
    shifted = _surface(1)
    shifted.transform = Affine(g0.cellsize, 0, g0.x0 + 1000, 0, -g0.cellsize, g0.y0)
    from pydivide import across_divide_metrics
    tab = across_divide_metrics(dem, threshold=200 * CS * CS, metrics=("elev",), seglength=300)
    a = np.asarray(tab.columns["basin_a"], float)
    b = np.asarray(tab.columns["basin_b"], float)
    k = int(np.flatnonzero(np.isfinite(a) & np.isfinite(b))[0])
    pair = (int(a[k]), int(b[k]))
    with pytest.raises(ValueError, match="origin"):
        basin_pair_profiles(D, shifted, pair=pair, spacing=30, half_length=30)
    p = basin_pair_profiles(D, dem, pair=pair, spacing=30, half_length=30)
    assert len(p.across) >= 1


def test_fortran_ordered_routing_is_refused_on_real_objects():
    z = np.asfortranarray(_surface(5).z)
    dem = tt3.GridObject()
    dem.z = z
    dem.cellsize = CS
    fd = tt3.FlowObject(dem)
    if getattr(fd, "order", "C") != "F":
        pytest.skip("this pytopotoolbox keeps a C-ordered FlowObject for a Fortran array")
    st = tt3.StreamObject(fd, threshold=200)
    with pytest.raises(ValueError, match="C-ordered"):
        pydivide.divide_network(fd=fd, st=st, minarea=200 * CS * CS, verbose=False)


def test_chi_areas_equal_the_accumulation_at_the_stream_nodes(dem):
    """The areas handed to the chi integration are those of the trimmed
    network's own nodes, read by row and column."""
    from pydivide import across_divide_metrics
    fd = tt3.FlowObject(dem)
    st = tt3.StreamObject(fd, threshold=200)
    bl = float(np.nanmin(dem.z)) + 5.0
    seen = {}
    real = tt3.StreamObject.chitransform

    def spy(self, a, **kw):
        seen["areas"] = np.asarray(a, float)
        seen["nodes"] = self.node_indices
        return real(self, a, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tt3.StreamObject, "chitransform", spy)
        across_divide_metrics(dem, flow=fd, streams=st, threshold=200 * CS * CS,
                              metrics=("chi",), baselevel=bl)
    acc = np.asarray(fd.flow_accumulation())
    r, c = seen["nodes"]
    assert np.allclose(seen["areas"], acc[np.asarray(r), np.asarray(c)])


# ---- multiscale -------------------------------------------------------------

def test_multiscale_rejects_duplicate_thresholds_and_raises_by_default(dem):
    from pydivide import across_divide_multiscale
    with pytest.raises(ValueError, match="distinct"):
        across_divide_multiscale(dem, thresholds=(1e5, 1e5), metrics=("hr",))
    with pytest.raises(ValueError, match="baselevel"):
        across_divide_multiscale(dem, thresholds=(1e5, 1e6), metrics=("chi",))
    with pytest.raises(ValueError, match="empty"):
        across_divide_multiscale(dem, thresholds=(), metrics=("hr",))
    with pytest.raises(ValueError, match="finite positive"):
        across_divide_multiscale(dem, thresholds=(1e5, -1.0), metrics=("hr",))


@pytest.fixture(scope="module")
def multiscale(dem):
    from pydivide import across_divide_multiscale
    return across_divide_multiscale(dem, thresholds=(100 * CS * CS, 400 * CS * CS),
                                    metrics=("elev", "hr"), seglength=300)


def test_multiscale_sign_is_a_physical_contrast_not_a_convention(multiscale):
    """Elevation does not depend on the channel threshold, so the signed
    elevation contrast of a piece must be identical at every scale once
    side A means the same physical side at every scale."""
    t = multiscale
    a = np.asarray(t.columns["elev_sgn_90000"], float)
    b = np.asarray(t.columns["elev_sgn_360000"], float)
    ok = np.isfinite(a) & np.isfinite(b)
    assert ok.sum() > 50
    assert np.allclose(a[ok], b[ok]), "side A changed physical side between scales"
    assert "side_rule" in t.settings
    assert bool(np.all(t.columns["elev_sign_consistent"][ok & (a != 0)]))


def test_multiscale_table_reads_its_own_columns_and_settings(multiscale):
    from pydivide import coherence
    t = multiscale
    assert t.settings["tags"] == ["_90000", "_360000"]
    for tag in t.settings["tags"]:
        c = coherence(t.columns, "hr", tag)
        assert c is not None and np.isfinite(c).any()
        d = t.disagreement(families=("hr", "elev"), tag=tag)
        assert d == {} or "hr_coherence" in d       # chi is not in this table
        assert tag in t.settings["per_scale"]
        assert "threshold" in t.settings["per_scale"][tag]
    text = t.describe()
    assert "90000" in text and "360000" in text and "held fixed" in text
    assert "computed by this call" in t.settings["provenance"]["routing"]
    assert t.settings["cellsize"] > 0 and t.settings["threshold_units"] == "m2"


# ---- global piece ids ---------------------------------------------------------

def test_piece_ids_are_global_and_unique_across_chunks(D, tmp_path):
    sel = D.piece_selection(300.0)
    ids = []
    for lo in range(0, len(sel), 1):
        ids.append(D.to_geodataframe("strahler", 300.0, piece_sel=sel[lo:lo + 1])["ID"].to_numpy())
    ids = np.concatenate(ids)
    assert len(np.unique(ids)) == len(ids) == len(sel)
    whole = D.to_geodataframe("strahler", 300.0)["ID"].to_numpy()
    assert np.array_equal(ids, whole)
    out = tmp_path / "d.gpkg"
    D.to_geopackage(out, seglength=300.0, chunk=7)
    import geopandas as gpd
    back = gpd.read_file(out)
    assert back["ID"].is_unique and len(back) == len(sel)


def test_piece_table_by_range_matches_whole(D):
    _, ps, pe = D._piece_table(300.0)
    n = len(ps)
    lo, hi = n // 3, 2 * n // 3
    _, ps2, pe2 = D._piece_table(300.0, np.arange(lo, hi))
    assert np.array_equal(ps[lo:hi], ps2) and np.array_equal(pe[lo:hi], pe2)
    assert D._n_pieces(300.0) == n


# ---- overwrite protection -------------------------------------------------------

def test_existing_geopackage_is_not_replaced_silently(D, tmp_path):
    out = tmp_path / "d.gpkg"
    D.to_geopackage(out)
    with pytest.raises(FileExistsError):
        D.to_geopackage(out)
    D.to_geopackage(out, overwrite=True)


def test_network_shapefile_parts_fit_the_budget(D, tmp_path):
    from pydivide import vectorio as V
    D.to_shapefile(tmp_path / "d.shp", seglength=300.0, max_features=50)
    parts = sorted(tmp_path.glob("d_part*.shp"))
    assert len(parts) > 1
    for p in parts:
        assert p.with_suffix(".dbf").stat().st_size < V.FILE_BUDGET_BYTES
    # the dataset is a file set: a surviving companion of a part refuses
    # the whole export and is left untouched
    parts[0].unlink()
    dbf = parts[0].with_suffix(".dbf")
    before = dbf.read_bytes()
    with pytest.raises(FileExistsError):
        D.to_shapefile(tmp_path / "d.shp", seglength=300.0, max_features=50)
    assert dbf.read_bytes() == before
    D.to_shapefile(tmp_path / "d.shp", seglength=300.0, max_features=50, overwrite=True)
    assert parts[0].exists()


# ---- CRS transformation cannot fall back silently -------------------------------

def test_mask_with_unbuildable_transformation_raises(D, tmp_path):
    from dataclasses import replace

    import geopandas as gpd
    import shapely
    Dc = replace(D, crs="EPSG:32632")
    # the grid sits near the UTM 32N origin, i.e. near 4.5 E, 0 N once
    # transformed; a box in degrees around 100 to 110 E must select nothing,
    # whereas the raw metre coordinates (0 to 160) would fall inside a box
    # that covers them untransformed
    far = gpd.GeoDataFrame(geometry=[shapely.box(100, 10, 110, 20)], crs="EPSG:4326")
    p = tmp_path / "m.gpkg"
    far.to_file(p, driver="GPKG")
    assert len(Dc.piece_selection(300.0, p)) == 0
    near = gpd.GeoDataFrame(geometry=[shapely.box(0, -2, 8, 2)], crs="EPSG:4326")
    p2 = tmp_path / "m2.gpkg"
    near.to_file(p2, driver="GPKG")
    assert len(Dc.piece_selection(300.0, p2)) == D._n_pieces(300.0)
    from pyproj.exceptions import CRSError
    Dbad = replace(D, crs="not a crs")
    with pytest.raises((CRSError, ValueError)):
        Dbad.piece_selection(300.0, p)


# ---- the arrow enters the basin the sign names ------------------------------------

def test_direction_arrows_enter_the_losing_basin(dem, D, table):
    import matplotlib
    matplotlib.use("Agg")
    from pydivide.plotting import plot_signed_divides, side_a_direction
    tab = table
    s = np.asarray(tab.columns["elev_sgn"], float)
    ba = np.asarray(tab.columns["basin_a"], float)
    bb = np.asarray(tab.columns["basin_b"], float)
    mid, unit, ok = side_a_direction(D, 300.0, ba, basin_b=bb)
    L = D.labels
    gs = D.grid
    # step a fraction of a cell into side A: the label there must be
    # basin_a, and a fraction into the other side must be basin_b
    hit = 0
    for i in np.flatnonzero(ok & np.isfinite(s) & (s != 0))[:400]:
        for sign, want in ((1.0, ba[i]), (-1.0, bb[i])):
            x, y = mid[i] + sign * unit[i] * 0.55 * gs.cellsize
            c = int((x - gs.x0) // gs.cellsize)
            r = int((gs.y0 - y) // gs.cellsize)
            assert 0 <= r < gs.nr and 0 <= c < gs.nc
            assert L[r, c] == want, (i, L[r, c], want)
        hit += 1
    assert hit > 100
    ax = plot_signed_divides(D, s, seglength=300.0, basin_a=ba, basin_b=bb,
                             tick=0.55 * gs.cellsize, min_abs=np.nanpercentile(np.abs(s), 50))
    # every drawn arrow tip lies in the basin the sign marks as losing
    checked = 0
    for ann in ax.texts:
        x, y = ann.xy
        c = int((x - gs.x0) // gs.cellsize)
        r = int((gs.y0 - y) // gs.cellsize)
        assert 0 <= r < gs.nr and 0 <= c < gs.nc
        x0, y0 = ann.xyann
        dist = np.hypot(mid[:, 0] - x0, mid[:, 1] - y0)
        dist[~ok] = np.inf                       # undrawn pieces carry NaN
        j = int(np.argmin(dist))
        losing = ba[j] if s[j] > 0 else bb[j]
        assert L[r, c] == losing
        checked += 1
    assert checked > 20


# ---- existing data is never replaced silently -----------------------------------

def test_stream_export_respects_overwrite(dem, tmp_path):
    from pydivide.streams import export_stream_segments
    fd = tt3.FlowObject(dem)
    st = tt3.StreamObject(fd, threshold=200)
    base = tmp_path / "streams"
    bl = float(np.nanmin(dem.z)) + 1.0
    export_stream_segments(fd, st, dem, base, seglength=300, chi_field="chi",
                           baselevel=bl, formats=("gpkg",))
    with pytest.raises(FileExistsError):
        export_stream_segments(fd, st, dem, base, seglength=300, chi_field="chi",
                               baselevel=bl, formats=("gpkg",))
    export_stream_segments(fd, st, dem, base, seglength=300, chi_field="chi",
                           baselevel=bl, formats=("gpkg",), overwrite=True)


def test_shapefile_refusal_writes_nothing(D, tmp_path):
    from pydivide.vectorio import write_shapefile
    g = D.to_geodataframe("strahler", 300.0)
    g["a_very_long_field_name_indeed"] = 1.0
    write_shapefile(g, tmp_path / "d.shp")
    side = tmp_path / "d_fields.csv"
    assert side.exists()
    stamp = side.stat().st_mtime_ns
    g["another_very_long_field_name"] = 2.0
    with pytest.raises(FileExistsError):
        write_shapefile(g, tmp_path / "d.shp")
    assert side.stat().st_mtime_ns == stamp          # the sidecar was left as it was


def test_export_gdb_script_refuses_and_reports(D, tmp_path):
    import subprocess
    gp = tmp_path / "x.gpkg"
    D.to_geopackage(gp, seglength=300.0)
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_gdb.py"
    r = subprocess.run([sys.executable, str(script), str(gp)], capture_output=True, text=True)
    assert r.returncode == 0 and (tmp_path / "x.gdb").exists(), r.stdout + r.stderr
    r = subprocess.run([sys.executable, str(script), str(gp)], capture_output=True, text=True)
    assert r.returncode != 0 and "FileExistsError" in r.stdout
    r = subprocess.run([sys.executable, str(script), str(tmp_path / "missing.gpkg")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and not (tmp_path / "missing.gdb").exists()
