# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Stream-network vector export, STREAMobj2mapstruct-style.

Reproduces the semantics of the MATLAB call

    MS = STREAMobj2mapstruct(S,'seglength',L,'attributes',
         {'chi5m' CHI @mean, 'elev' DEM @mean, 'upakm2' A @mean})

on pytopotoolbox objects: the stream network is walked downstream from
channel heads / confluence starts, split into pieces of roughly
``seglength`` metres, and each attribute is averaged over the piece's
nodes (finite values only).

Chi is computed per baselevel: the network is trimmed to
``upstreamto(DEM > baselevel)`` (+ clean), chi is evaluated on the
trimmed network (a0 = 1, default mn = 0.45, cell-size corrected), and
the trimmed network is what gets exported — one file per baselevel,
matching the established workflow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _finite_mean(vals: np.ndarray) -> float:
    v = vals[np.isfinite(vals)]
    return float(v.mean()) if v.size else float("nan")


def _stream_chains(n_nodes: int, src: np.ndarray, dst: np.ndarray):
    """Split the stream graph into downstream chains starting at channel
    heads and at confluence outputs; yields lists of NAL positions."""
    outdeg = np.bincount(src, minlength=n_nodes)
    indeg = np.bincount(dst, minlength=n_nodes)
    down = np.full(n_nodes, -1, dtype=np.int64)
    down[src] = dst          # d8: one receiver per node

    starts = np.nonzero((outdeg > 0) & (indeg != 1))[0]
    visited = np.zeros(n_nodes, dtype=bool)
    chains = []
    for s0 in starts.tolist():
        if visited[s0]:
            continue
        chain = [s0]
        cur = s0
        while True:
            visited[cur] = True
            nxt = int(down[cur])
            if nxt < 0:
                break
            chain.append(nxt)
            if indeg[nxt] != 1 or down[nxt] < 0:
                break
            if visited[nxt]:
                break
            cur = nxt
        if len(chain) > 1:
            chains.append(chain)
    # anything left (loops should not exist in d8, but be safe)
    for s0 in np.nonzero((outdeg > 0) & ~visited)[0].tolist():
        if visited[s0]:
            continue
        chain = [s0]
        cur = s0
        while True:
            visited[cur] = True
            nxt = int(down[cur])
            if nxt < 0 or visited[nxt]:
                if nxt >= 0:
                    chain.append(nxt)
                break
            chain.append(nxt)
            cur = nxt
        if len(chain) > 1:
            chains.append(chain)
    return chains


def stream_segments_gdf(fd, st, dem, *, seglength: float,
                        chi_field: str, baselevel: float,
                        a0: float = 1.0, mn: float = 0.45, acc=None,
                        chi_baselevel: float | None = None):
    """GeoDataFrame of ~seglength stream pieces with mean attributes.

    Fields: ``segid``, ``len_m``, ``nnode``, ``upakm2`` (mean upstream
    area, km^2), ``elev`` (mean DEM elevation), ``<chi_field>`` (mean
    chi for the given baselevel). The exported network is the baselevel-
    trimmed one.

    Pass ``acc`` (a precomputed flow-accumulation GridObject) to avoid
    recomputing it per baselevel — at full-domain size that is a
    ~49 GB allocation and a long pass over 5.1e9 flow edges.
    """
    import geopandas as gpd
    import shapely

    if getattr(fd, "order", "C") != "C":
        raise ValueError(
            "pydivide works on C-ordered (row-major) grids, the layout pytopotoolbox uses for "
            "every raster it reads; this FlowObject is Fortran-ordered")
    if acc is None:
        acc = fd.flow_accumulation()

    # trim to cells above baselevel (MATLAB: S = upstreamto(S, DEM>b))
    zf = np.asarray(dem.z)
    stb = st.upstreamto(zf > baselevel).clean()
    if len(stb.stream) == 0:
        # nothing to export: the table with its columns and no rows, so a
        # caller can tell an empty result from a failure
        print(f"[pydivide] no stream cells above the baselevel {baselevel:g} m: "
              "the export is empty", flush=True)
        # the same columns with the same types as a non-empty export
        # (integers for the id and the node count, floats for the rest),
        # so a consumer that builds a layer from the schema sees one layer
        crs = getattr(fd, "georef", None)
        empty = {"segid": np.zeros(0, np.int64), "len_m": np.zeros(0, np.float64),
                 "nnode": np.zeros(0, np.int64), "upakm2": np.zeros(0, np.float64),
                 "elev": np.zeros(0, np.float64), chi_field: np.zeros(0, np.float64)}
        return gpd.GeoDataFrame(empty, geometry=gpd.GeoSeries([], dtype="geometry", crs=crs),
                                crs=crs)

    cb = baselevel if chi_baselevel is None else float(chi_baselevel)
    if cb == baselevel:
        chi = np.asarray(stb.chitransform(acc, a0=a0, mn=mn,
                                          correctcellsize=True),
                         dtype=np.float64)
    else:
        # The exported network and the chi integration can use different
        # baselevels: rivers may be drawn down to 1 m while chi is
        # integrated from a strict baselevel higher up. chi is computed on
        # the strictly trimmed network and mapped onto the exported one by
        # grid position; nodes below the strict baselevel get NaN, because
        # chi is simply not defined for them.
        stc = st.upstreamto(zf > cb).clean()
        ncol = int(zf.shape[1])
        rb, cbi = stb.node_indices
        lb = np.asarray(rb, np.int64) * ncol + np.asarray(cbi, np.int64)
        chi = np.full(lb.size, np.nan)
        # which exported nodes received a chi value; none when the chi
        # baselevel removes every node
        hit = np.zeros(lb.size, dtype=bool)
        if len(stc.stream) == 0:
            print(f"[pydivide] no stream cells above the chi baselevel "
                  f"{cb:g} m: chi is undefined on the whole export", flush=True)
        else:
            chic = np.asarray(stc.chitransform(acc, a0=a0, mn=mn,
                                               correctcellsize=True),
                              dtype=np.float64)
            rc, cci = stc.node_indices
            lc = np.asarray(rc, np.int64) * ncol + np.asarray(cci, np.int64)
            o = np.argsort(lc)
            lcs = lc[o]
            pos = np.clip(np.searchsorted(lcs, lb), 0, lcs.size - 1)
            hit = lcs[pos] == lb
            chi[hit] = chic[o][pos[hit]]
        print(f"[pydivide] chi baselevel {cb:g} m on a network exported "
              f"at {baselevel:g} m: {int(hit.sum()):,} of {lb.size:,} "
              f"nodes have chi", flush=True)

    rows, cols = stb.node_indices
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    n = rows.size

    elev = np.asarray(stb.ezgetnal(dem), dtype=np.float64)
    accn = np.asarray(stb.ezgetnal(acc), dtype=np.float64)
    cell_area = float(fd.cellsize) ** 2
    upakm2 = accn * cell_area / 1e6

    tr = fd.transform
    x = tr.c + tr.a * (cols + 0.5) + tr.b * (rows + 0.5)
    y = tr.f + tr.d * (cols + 0.5) + tr.e * (rows + 0.5)

    src = np.asarray(stb.source, dtype=np.int64)
    dst = np.asarray(stb.target, dtype=np.int64)
    if src.size and (src.max() >= n or dst.max() >= n):
        raise ValueError("StreamObject edges are not NAL positions")

    chains = _stream_chains(n, src, dst)

    seg_coords = []
    seg_index = []
    props = {"segid": [], "len_m": [], "nnode": [],
             "upakm2": [], "elev": [], chi_field: []}
    segid = 0

    def emit(piece: list[int], plen: float):
        nonlocal segid
        if len(piece) < 2:
            return
        segid += 1
        for p in piece:
            seg_coords.append((x[p], y[p]))
            seg_index.append(segid - 1)
        props["segid"].append(segid)
        props["len_m"].append(round(plen, 3))
        props["nnode"].append(len(piece))
        idx = np.array(piece, dtype=np.int64)
        props["upakm2"].append(_finite_mean(upakm2[idx]))
        props["elev"].append(_finite_mean(elev[idx]))
        props[chi_field].append(_finite_mean(chi[idx]))

    for chain in chains:
        piece = [chain[0]]
        plen = 0.0
        for a, b in zip(chain[:-1], chain[1:]):
            step = float(np.hypot(x[b] - x[a], y[b] - y[a]))
            piece.append(b)
            plen += step
            if plen >= seglength:
                emit(piece, plen)
                piece = [b]
                plen = 0.0
        emit(piece, plen)

    coords = np.array(seg_coords, dtype=np.float64)
    idx = np.array(seg_index, dtype=np.int64)
    geoms = shapely.linestrings(coords, indices=idx)
    crs = getattr(fd, "georef", None)
    return gpd.GeoDataFrame(props, geometry=geoms, crs=crs)


def write_stream_gdf(gdf, out_base: str | Path, *,
                     formats=("gpkg", "shp"),
                     max_features: int = 2_000_000,
                     overwrite: bool = False) -> None:
    """Write an already-built stream GeoDataFrame to disk.

    Shapefiles are split into numbered parts where either 2 GB limit
    would be exceeded, and never hold more than ``max_features``.
    Existing files are replaced only with ``overwrite=True``.
    """
    from pydivide.vectorio import _guard_export_set, write_geopackage, write_shapefile

    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    if "gpkg" in formats:
        # the shared writer repairs the CRS definition ArcGIS needs; an
        # empty table becomes an empty layer that keeps the schema, its
        # geometry type declared as the lines it would hold
        write_geopackage(gdf, f"{out_base}.gpkg", layer=out_base.name,
                         overwrite=overwrite, geometry_type="LineString")
    if "shp" in formats:
        if len(gdf) == 0:
            # a Shapefile cannot carry an empty export with its schema, so
            # none is written; the previous set at this name is still
            # removed under overwrite (and refused without it), so no
            # stale parts read as current data
            _guard_export_set(Path(f"{out_base}.shp"), [], overwrite)
            print(f"[pydivide] {out_base.name}: no stream piece to export; the GeoPackage "
                  "layer is empty and no Shapefile is written", flush=True)
            return
        # the shared writer sizes parts from the real geometry and
        # attribute bytes, so both 2 GB limits are respected
        write_shapefile(gdf, f"{out_base}.shp", max_features=max_features,
                        overwrite=overwrite)


def export_stream_segments(fd, st, dem, out_base: str | Path, *,
                           seglength: float, chi_field: str,
                           baselevel: float, formats=("gpkg", "shp"),
                           a0: float = 1.0, mn: float = 0.45,
                           acc=None, mask=None, chi_baselevel=None,
                           masked_out_base: str | Path | None = None,
                           max_features: int = 2_000_000,
                           overwrite: bool = False) -> None:
    """Build the stream-piece vectors once and write them. Existing
    files are replaced only with ``overwrite=True``.

    With ``mask`` plus ``masked_out_base``, the masked file is written
    from the *same* computation as the full one — at full-domain size
    rebuilding the stream network for the masked copy would cost tens of
    minutes for nothing. Masking is an export filter only; values are
    always computed over the whole domain.
    """
    gdf = stream_segments_gdf(fd, st, dem, seglength=seglength,
                              chi_field=chi_field, baselevel=baselevel,
                              a0=a0, mn=mn, acc=acc,
                              chi_baselevel=chi_baselevel)
    if mask is not None and masked_out_base is None:
        # masked-only output
        write_stream_gdf(_apply_mask(gdf, mask), out_base,
                         formats=formats, max_features=max_features,
                         overwrite=overwrite)
        return
    write_stream_gdf(gdf, out_base, formats=formats,
                     max_features=max_features, overwrite=overwrite)
    if mask is not None:
        write_stream_gdf(_apply_mask(gdf, mask), masked_out_base,
                         formats=formats, max_features=max_features,
                         overwrite=overwrite)


def _apply_mask(gdf, mask):
    """Keep only features whose representative point lies in the mask
    (polygon file/geometry, or valid cells of a raster file)."""
    import numpy as np
    import shapely

    pts = shapely.get_point(gdf.geometry.values, 0)
    x = shapely.get_x(pts)
    y = shapely.get_y(pts)

    def to_crs(dst):
        # reproject when both CRS are known and differ; a transformation
        # that cannot be built, or that fails on any point, raises rather
        # than selecting in the wrong coordinate space (see
        # DivideNetworkResult.piece_selection for the policy)
        if dst is None or gdf.crs is None:
            if (dst is None) != (gdf.crs is None):
                import warnings
                warnings.warn(
                    "only one of the stream layer and the mask has a coordinate "
                    "reference system; their coordinates are compared untransformed",
                    RuntimeWarning, stacklevel=3)
            return x, y
        from pyproj import CRS, Transformer

        from .pipeline import _checked_transform
        if CRS.from_user_input(dst) == CRS.from_user_input(gdf.crs):
            return x, y
        tf = Transformer.from_crs(gdf.crs, dst, always_xy=True)
        return _checked_transform(tf, x, y, gdf.crs, dst)

    if isinstance(mask, (str, Path)) and \
            str(mask).lower().endswith((".tif", ".tiff")):
        import rasterio
        with rasterio.open(str(mask)) as src:
            mx, my = to_crs(src.crs)
            cols, rows = (~src.transform) * (np.asarray(mx),
                                             np.asarray(my))
            rows = np.floor(rows).astype(np.int64)
            cols = np.floor(cols).astype(np.int64)
            inb = ((rows >= 0) & (rows < src.height)
                   & (cols >= 0) & (cols < src.width))
            valid = src.read_masks(1).astype(bool)
        keep = np.zeros(len(gdf), dtype=bool)
        keep[inb] = valid[rows[inb], cols[inb]]
    else:
        if isinstance(mask, (str, Path)):
            import geopandas as gpd
            mg = gpd.read_file(str(mask))
            geom = shapely.union_all(mg.geometry.values)
            mx, my = to_crs(mg.crs)
        else:
            geom = mask
            mx, my = x, y
        keep = shapely.contains_xy(geom, np.asarray(mx), np.asarray(my))
    return gdf.loc[keep].reset_index(drop=True)
