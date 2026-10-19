# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""The user-facing entry point: :func:`divide_network`.

Plays the role MATLAB's ``DIVIDEobj(FD, ST)`` plays, without its limits:

    import pydivide
    D = pydivide.divide_network("dem.tif", minarea=1e6)          # from a DEM
    D = pydivide.divide_network(fd=fd, st=st)                    # from objects
    D.distance            # per node instance, map units
    D.order("strahler")   # / "shreve" / "topo"
    D.to_geopackage("divides.gpkg")
    D.to_shapefile("divides.shp")     # both 2 GB limits respected

Every heavy stage checkpoints through StageStore when ``checkpoint_dir``
is given, and an interrupted run resumes from the last completed stage.

Export fields follow DIVIDEobj2mapstruct: ``ID``, ``order``,
``distance`` (mean of node distances along the segment), ``length``,
``azimuth`` — plus ``distmin``, ``distmax``, ``isbnd`` (segment touches
the DEM edge or nodata: potentially truncation-spurious divides,
Scherler & Schwanghart section 5.1), and ``isring`` (closed component,
endorheic candidate).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .checkpoint import StageStore, array_fingerprint
from .flownet import FlowNet
from .geometry import divide_edges, pair_isin
from .gridspec import GridSpec
from .labels import basin_labels, check_stream_threshold
from .network import fx_grid, segment_network
from .orderdist import SortedNetwork, divdist, divorder, sort_network


def _checked_transform(tf, x, y, src_crs, dst_crs):
    """``tf.transform`` that raises on any point the transformation
    cannot handle (pyproj's default returns infinity instead) and on any
    non-finite result, so a failed reprojection can never be read as
    points lying outside the mask."""
    out = tf.transform(np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                       errcheck=True)
    mx, my = np.asarray(out[0], dtype=float), np.asarray(out[1], dtype=float)
    if not (np.isfinite(mx).all() and np.isfinite(my).all()):
        raise ValueError(
            f"reprojecting from {src_crs} to {dst_crs} produced non-finite coordinates "
            f"for {int((~np.isfinite(mx) | ~np.isfinite(my)).sum())} points; the two "
            "coordinate reference systems are not compatible for these positions")
    return mx, my


@dataclass
class DivideNetworkResult:
    """Computed divide network with values and export methods."""
    grid: GridSpec
    network: SortedNetwork
    distance: np.ndarray                 # aligned with network.seg_nodes
    edge_nodes: np.ndarray               # (E, 2) divide edges
    edge_is_boundary: np.ndarray         # (E,) bool
    crs: object = None
    labels: np.ndarray | None = None     # (nr, nc) int32 reach-basin labels
    _orders: dict = field(default_factory=dict)

    @property
    def endpoints(self) -> np.ndarray:
        return self.network.endpoints

    @property
    def junctions(self) -> np.ndarray:
        return self.network.junctions

    @property
    def n_segments(self) -> int:
        return self.network.n_segments

    def order(self, scheme: str = "topo") -> np.ndarray:
        if scheme not in self._orders:
            self._orders[scheme] = divorder(self.network, scheme)
        return self._orders[scheme]

    # ---- per-segment boundary flag ------------------------------------

    def _edge_boundary_positions(self) -> np.ndarray:
        """Bool array aligned with seg_nodes[:-1]: True where the pair
        (seg_nodes[i], seg_nodes[i+1]) is a boundary divide edge.
        Positions at segment joints are False."""
        bnd = self.edge_nodes[self.edge_is_boundary]
        sn = self.network
        a = sn.seg_nodes[:-1]
        b = sn.seg_nodes[1:]
        joint = np.zeros(len(sn.seg_nodes), dtype=bool)
        joint[sn.seg_ptr[1:] - 1] = True     # last node of each segment
        ok = ~joint[:-1]
        pairs = np.stack([a[ok], b[ok]], axis=1)
        pairs.sort(axis=1)
        out = np.zeros(len(a), dtype=bool)
        out[ok] = pair_isin(pairs, bnd)
        return out

    def segment_is_boundary(self) -> np.ndarray:
        """True where a segment contains at least one boundary edge."""
        sn = self.network
        seg_of = np.repeat(np.arange(sn.n_segments, dtype=np.int64),
                           np.diff(sn.seg_ptr))
        hitpos = self._edge_boundary_positions()
        out = np.zeros(sn.n_segments, dtype=bool)
        np.maximum.at(out.view(np.uint8), seg_of[:-1][hitpos], 1)
        return out

    # ---- export --------------------------------------------------------

    def _piece_counts(self, seglength: float | None):
        """Per-segment piece counts and their cumulative offsets: one
        int64 per segment, never per piece."""
        sn = self.network
        cs = self.grid.cellsize
        n_edges = sn.seg_ptr[1:] - sn.seg_ptr[:-1] - 1   # edges per segment
        if seglength is None:
            step = None
            n_pieces = np.ones(sn.n_segments, dtype=np.int64)
        else:
            step = max(1, int(round(seglength / cs)))
            n_pieces = np.maximum(1, -(-n_edges // step))   # ceil div
        cum = np.zeros(sn.n_segments + 1, dtype=np.int64)
        np.cumsum(n_pieces, out=cum[1:])
        return step, cum

    def _piece_table(self, seglength: float | None, pieces=None):
        """(seg_of_piece, p_start, p_stop) for the pieces ``pieces``, an
        int64 array of global piece indices, or for every piece when
        ``pieces`` is None.

        Only the requested pieces are materialised, so a chunked export
        of a 10^8-piece network holds one chunk at a time; the
        per-segment offsets it maps through are one array of segment
        length. Fully vectorised."""
        sn = self.network
        step, cum = self._piece_counts(seglength)
        if pieces is None:
            g = np.arange(cum[-1], dtype=np.int64)
        else:
            g = np.asarray(pieces, dtype=np.int64)
        seg_of_piece = np.searchsorted(cum, g, side="right") - 1
        rank = g - cum[seg_of_piece]
        seg_start = sn.seg_ptr[:-1][seg_of_piece]
        seg_stop = sn.seg_ptr[1:][seg_of_piece]
        if step is None:
            p_start, p_stop = seg_start, seg_stop
        else:
            p_start = seg_start + rank * step
            p_stop = np.minimum(p_start + step, seg_stop - 1) + 1
        return seg_of_piece, p_start, p_stop

    def _n_pieces(self, seglength: float | None) -> int:
        return int(self._piece_counts(seglength)[1][-1])

    def piece_selection(self, seglength: float | None,
                        mask=None) -> np.ndarray:
        """Indices of the pieces to export.

        ``mask`` may be

        * a polygon file (.shp/.gpkg/...) or a shapely geometry — a
          piece is kept when its midpoint node falls inside;
        * a raster file (.tif/.tiff) — a piece is kept when its midpoint
          falls on a valid (non-nodata) cell of that raster. Use this to
          restrict a large-domain result to the footprint of a smaller
          DEM.

        Masking is purely an *export* filter: every value was computed on
        the full domain, so masked pieces carry their full-domain
        divide distance and order. Point-in-polygon uses shapely's
        vectorised prepared-geometry test and raster masks use index
        arithmetic — geometric clipping of 10^8 LineStrings would be far
        too slow.

        Coordinates are reprojected to the mask's CRS when both are
        known and differ; a transformation that cannot be built, or one
        that fails on any midpoint, raises rather than yielding an empty
        selection. When only one of the two has a CRS a warning is
        issued and the coordinates are compared as they are; a bare
        shapely geometry is taken to be in the result's CRS.
        """
        n = self._n_pieces(seglength)
        if mask is None:
            return np.arange(n, dtype=np.int64)

        transformer = {}

        def to_crs(dst_crs, x, y):
            """Reproject when both CRS are known and differ; raise when a
            transformation is needed but cannot be built or fails, rather
            than selecting in the wrong coordinate space."""
            if dst_crs is None or self.crs is None:
                if (dst_crs is None) != (self.crs is None):
                    import warnings
                    warnings.warn(
                        "only one of the divide network and the mask has a coordinate "
                        "reference system; their coordinates are compared untransformed",
                        RuntimeWarning, stacklevel=3)
                return x, y
            if "tf" not in transformer:
                from pyproj import CRS, Transformer
                if CRS.from_user_input(dst_crs) == CRS.from_user_input(self.crs):
                    transformer["tf"] = None
                else:
                    transformer["tf"] = Transformer.from_crs(
                        self.crs, dst_crs, always_xy=True)
            tf = transformer["tf"]
            return (x, y) if tf is None else _checked_transform(tf, x, y, self.crs, dst_crs)

        is_raster = (isinstance(mask, (str, Path))
                     and str(mask).lower().endswith((".tif", ".tiff")))
        chunk = 4_000_000
        inside = np.zeros(n, dtype=bool)

        def midpoints(lo, hi):
            _, p_start, p_stop = self._piece_table(
                seglength, np.arange(lo, hi, dtype=np.int64))
            mid = p_start + (p_stop - p_start) // 2
            return self.grid.node_xy(self.network.seg_nodes[mid])

        if is_raster:
            import rasterio
            with rasterio.open(str(mask)) as src:
                valid = src.read_masks(1).astype(bool)
                inv = ~src.transform
                for lo in range(0, n, chunk):
                    hi = min(lo + chunk, n)
                    x, y = midpoints(lo, hi)
                    mx, my = to_crs(src.crs, x, y)
                    cols, rows = inv * (np.asarray(mx), np.asarray(my))
                    rows = np.floor(rows).astype(np.int64)
                    cols = np.floor(cols).astype(np.int64)
                    inb = ((rows >= 0) & (rows < src.height)
                           & (cols >= 0) & (cols < src.width))
                    blk = np.zeros(hi - lo, dtype=bool)
                    blk[inb] = valid[rows[inb], cols[inb]]
                    inside[lo:hi] = blk
        else:
            import shapely
            if isinstance(mask, (str, Path)):
                import geopandas as gpd
                gdf = gpd.read_file(str(mask))
                geom = shapely.union_all(gdf.geometry.values)
                mask_crs = gdf.crs
            else:
                geom, mask_crs = mask, None
            shapely.prepare(geom)
            for lo in range(0, n, chunk):
                hi = min(lo + chunk, n)
                x, y = midpoints(lo, hi)
                mx, my = to_crs(mask_crs, x, y)
                inside[lo:hi] = shapely.contains_xy(geom, np.asarray(mx),
                                                    np.asarray(my))

        return np.nonzero(inside)[0].astype(np.int64)

    def to_geodataframe(self, scheme: str = "strahler",
                        seglength: float | None = None,
                        piece_range: tuple[int, int] | None = None,
                        piece_sel: np.ndarray | None = None):
        """Per-segment features; with ``seglength`` (metres), segments
        are split into ~seglength pieces (attributes aggregated per
        piece, ``segid`` = parent segment).

        ``piece_range`` limits output to pieces [lo, hi) and
        ``piece_sel`` to an explicit array of piece indices — both used
        by the chunked/masked exporters to bound memory on very large
        networks.
        """
        import geopandas as gpd
        import shapely

        sn = self.network
        dist = self.distance
        order = self.order(scheme)
        bndpos = self._edge_boundary_positions()
        cs = self.grid.cellsize

        if piece_sel is not None:
            pieces = np.asarray(piece_sel, dtype=np.int64)
        elif piece_range is not None:
            lo, hi = piece_range
            pieces = np.arange(lo, hi, dtype=np.int64)
        else:
            pieces = np.arange(self._n_pieces(seglength), dtype=np.int64)
        seg_of_piece, p_start, p_stop = self._piece_table(seglength, pieces)
        n = len(p_start)
        p_len = p_stop - p_start                     # nodes per piece

        # ---- coordinates: ragged concatenation of node ranges ---------
        # only the nodes of the requested pieces are located, so a chunk
        # costs its own size and not the whole network's
        total = int(p_len.sum())
        offs = np.zeros(n, dtype=np.int64)
        np.cumsum(p_len[:-1], out=offs[1:])
        idx = (np.repeat(p_start, p_len)
               + np.arange(total, dtype=np.int64)
               - np.repeat(offs, p_len))
        cx, cy = self.grid.node_xy(sn.seg_nodes[idx])
        coords = np.empty((total, 2), dtype=np.float64)
        coords[:, 0] = cx
        coords[:, 1] = cy
        seg_index = np.repeat(np.arange(n, dtype=np.int64), p_len)
        geoms = shapely.linestrings(coords, indices=seg_index)

        # ---- attributes via segmented reductions ----------------------
        dvals = dist[idx]
        fin = np.isfinite(dvals)
        cnt = np.add.reduceat(fin.astype(np.int64), offs)
        ssum = np.add.reduceat(np.where(fin, dvals, 0.0), offs)
        with np.errstate(invalid="ignore", divide="ignore"):
            distance = np.where(cnt > 0, ssum / np.maximum(cnt, 1), np.nan)
        distmin = np.minimum.reduceat(
            np.where(fin, dvals, np.inf), offs)
        distmax = np.maximum.reduceat(
            np.where(fin, dvals, -np.inf), offs)
        distmin[cnt == 0] = np.nan
        distmax[cnt == 0] = np.nan

        x0, y0 = self.grid.node_xy(sn.seg_nodes[p_start])
        x1, y1 = self.grid.node_xy(sn.seg_nodes[p_stop - 1])
        dx = x1 - x0
        dy = y1 - y0
        az = np.degrees(np.arctan2(dx, dy)) % 360.0
        azimuth = np.minimum(az, (az + 180.0) % 360.0)

        # boundary: any boundary edge position inside the piece
        bcum = np.concatenate([[0], np.cumsum(bndpos.astype(np.int64))])
        isbnd = (bcum[p_stop - 1] - bcum[p_start]) > 0

        # ID is the global piece index plus one, so it is the same in a
        # chunked, masked or whole export of the same network
        return gpd.GeoDataFrame({
            "ID": pieces + 1,
            "segid": seg_of_piece + 1,
            "order": order[p_start],
            "distance": distance,
            "distmin": distmin,
            "distmax": distmax,
            "length": (p_len - 1) * cs,
            "azimuth": azimuth,
            "isbnd": isbnd,
            "isring": sn.is_ring[seg_of_piece],
        }, geometry=geoms, crs=self.crs)

    def to_geopackage(self, path: str | Path, scheme: str = "strahler",
                      seglength: float | None = None, mask=None,
                      chunk: int = 4_000_000, overwrite: bool = False) -> None:
        """Write a GeoPackage. Large networks are built and appended in
        chunks so peak memory stays bounded. ``mask`` (polygon file or
        geometry) restricts output to pieces inside it. An existing file
        is replaced only with ``overwrite=True``; a GeoPackage may hold
        layers of other origin, and deleting it silently is not
        acceptable."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
            path.unlink()
        sel = self.piece_selection(seglength, mask)
        n = len(sel)
        if n == 0:
            raise ValueError("no divide pieces fall inside the mask")
        for i, lo in enumerate(range(0, n, chunk)):
            hi = min(lo + chunk, n)
            gdf = self.to_geodataframe(scheme, seglength,
                                       piece_sel=sel[lo:hi])
            gdf.to_file(str(path), driver="GPKG", layer="divides",
                        mode="w" if i == 0 else "a")
        from .vectorio import repair_geopackage_crs
        repair_geopackage_crs(path)

    def to_shapefile(self, path: str | Path, scheme: str = "strahler",
                     seglength: float | None = None, mask=None,
                     max_features: int = 2_000_000,
                     overwrite: bool = False) -> None:
        """Write ESRI Shapefile(s).

        A Shapefile has **two** independent 2 GB limits, the ``.shp``
        geometry and the ``.dbf`` attribute table, and which one binds
        depends on the column count and on the number of vertices. Parts
        are cut where the running geometry size, computed from the real
        vertex count of every piece, or the running attribute size would
        exceed the budget, and never hold more than ``max_features``.

        Parts are named ``name_part001.shp``. ``mask`` (polygon file or
        geometry) restricts output to pieces inside it. Existing files
        of the same name are replaced only with ``overwrite=True``.
        """
        from .vectorio import (
            FILE_BUDGET_BYTES,
            _guard_export_set,
            _guard_shapefile,
            dbf_field_widths,
            dbf_header_bytes,
            dbf_record_bytes,
            shp_bytes_for_vertices,
        )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sel = self.piece_selection(seglength, mask)
        n = len(sel)
        if n == 0:
            raise ValueError("no divide pieces fall inside the mask")

        def cast(gdf):
            for col in ("isbnd", "isring"):
                gdf[col] = gdf[col].astype("int32")   # shp has no bool
            return gdf

        def write(gdf, target: Path):
            # every companion of the dataset counts as occupied
            _guard_shapefile(target, overwrite)
            cast(gdf).to_file(str(target), driver="ESRI Shapefile",
                              layer_options={"2GB_LIMIT": "YES"})

        # the attribute row width from the dtypes the driver will see,
        # measured on one piece, so the budget holds for this schema
        widths = dbf_field_widths(cast(self.to_geodataframe(scheme, seglength,
                                                             piece_sel=sel[:1])))
        dbf_row = dbf_record_bytes(widths)
        dbf_fixed = dbf_header_bytes(len(widths))
        # part boundaries from the real per-piece sizes, computed one
        # chunk of pieces at a time so the whole selection is never held
        bounds = [0]
        run_shp = run_dbf = 0
        step_chunk = 4_000_000
        for lo in range(0, n, step_chunk):
            hi = min(lo + step_chunk, n)
            _, p_start, p_stop = self._piece_table(seglength, sel[lo:hi])
            shp_row = shp_bytes_for_vertices(p_stop - p_start)
            for i in range(lo, hi):
                row = int(shp_row[i - lo])
                if (i - bounds[-1] >= max_features
                        or run_shp + row > FILE_BUDGET_BYTES - 100
                        or run_dbf + dbf_row > FILE_BUDGET_BYTES - dbf_fixed):
                    bounds.append(i)
                    run_shp = run_dbf = 0
                run_shp += row
                run_dbf += dbf_row
        bounds.append(n)
        parts = len(bounds) - 1
        targets = ([path] if parts == 1 else
                   [path.with_name(f"{path.stem}_part{k + 1:03d}{path.suffix}")
                    for k in range(parts)])
        # refuse before anything is written; the parts of an earlier
        # export at this name are part of the set (see vectorio)
        _guard_export_set(path, targets, overwrite)
        if parts == 1:
            write(self.to_geodataframe(scheme, seglength, piece_sel=sel), path)
            return
        print(f"[pydivide] {n:,} features exceed a single Shapefile; writing "
              f"{parts} parts (a GeoPackage or File Geodatabase keeps them in "
              "one file)", flush=True)
        for k in range(parts):
            lo, hi = bounds[k], bounds[k + 1]
            write(self.to_geodataframe(scheme, seglength, piece_sel=sel[lo:hi]), targets[k])

    def to_filegdb(self, path: str | Path, scheme: str = "strahler",
                   seglength: float | None = None, mask=None,
                   layer: str | None = None,
                   chunk: int = 2_000_000, overwrite: bool = False) -> Path:
        """Write an ESRI File Geodatabase of the divide network.

        The format ArcGIS prefers, and the one with neither a size limit
        nor a 10-character field-name limit — so a wide metric table
        needs no themes and no row parts. Written in chunks so a
        full-domain network never has to be resident.

        See :mod:`pydivide.vectorio` for the two things that silently go
        wrong here without care: the missing ``driver=`` argument, which
        writes a Shapefile into a ``.gdb`` directory, and the missing
        ``TARGET_ARCGIS_VERSION``, which demotes int64 to float.
        """
        from .vectorio import write_filegdb

        sel = self.piece_selection(seglength, mask)
        n = len(sel)
        if n == 0:
            raise ValueError("no divide pieces fall inside the mask")

        def gen():
            for lo in range(0, n, chunk):
                yield self.to_geodataframe(
                    scheme, seglength, piece_sel=sel[lo:min(lo + chunk, n)])

        return write_filegdb(gen(), path, layer=layer, overwrite=overwrite)


def divide_network(dem=None, *, fd=None, st=None, minarea: float = 1e6,
                   units: str = "m2", checkpoint_dir=None,
                   verbose: bool = True,
                   release_inputs: bool = False) -> DivideNetworkResult:
    """Compute the divide network of a DEM (or of prebuilt routing).

    Parameters
    ----------
    dem : path or pytopotoolbox GridObject, optional
        Ignored when ``fd`` and ``st`` are both given.
    fd, st : pytopotoolbox FlowObject / StreamObject, optional
    minarea : float
        Stream threshold, in ``units`` ('m2' or 'pixels'): a finite
        positive number. Used only when the stream network is built
        here, that is when ``st`` is not given.
    checkpoint_dir : path, optional
        Enables stage checkpointing and resume.
    release_inputs : bool
        Free the DEM and FlowObject rasters that the divide stages do
        not need (``dem.z``, ``fd.stream``, ``fd.fraction``,
        ``fd.direction``) as soon as the stream network exists, and the
        flow-edge arrays after Stage 4. **This mutates the objects you
        passed in**, so they cannot be reused afterwards. Required at
        full-domain size, where those rasters are ~180 GB and holding
        them causes an out-of-memory failure in Stage 4; harmless but
        unnecessary on small DEMs. Always safe when ``divide_network``
        built the routing itself from a DEM path.
    """
    import topotoolbox as tt

    def log(msg):
        if verbose:
            print(f"[pydivide {time.strftime('%H:%M:%S')}] {msg}",
                  flush=True)

    if fd is None or st is None:
        if dem is None:
            raise ValueError("either dem or (fd, st) must be given")
        if st is None:
            minarea = check_stream_threshold(minarea, "minarea")
        if isinstance(dem, (str, Path)):
            log(f"reading {dem}")
            dem = tt.read_tif(str(dem))
        if fd is None:
            log("flow routing (d8, carve)")
            fd = tt.FlowObject(dem)
        if st is None:
            log(f"stream network (threshold {minarea:g} {units})")
            st = tt.StreamObject(fd, threshold=minarea, units=units)

    net = FlowNet.from_pytopotoolbox(fd, st)

    if release_inputs:
        # Rasters the divide stages never read. At full-domain size
        # fd.stream alone is ~97 GB and dem.z ~49 GB; holding them
        # through Stage 4 exhausts a 512 GB machine.
        import gc
        for obj, attrs in ((dem, ("z",)),
                           (fd, ("stream", "fraction", "direction"))):
            for a in attrs:
                try:
                    setattr(obj, a, None)
                except Exception:
                    pass
        gc.collect()

    store = StageStore(checkpoint_dir) if checkpoint_dir else None
    # the checkpoint identity binds to the inputs actually consumed: the
    # grid (shape, cell size, origin, CRS) and the flow and stream edge
    # arrays, so two DEMs of the same shape and threshold cannot share a
    # checkpoint (see StageStore)
    params = {"minarea": minarea, "units": units,
              "shape": [net.grid.nr, net.grid.nc],
              "cellsize": float(net.grid.cellsize),
              "origin": [float(net.grid.x0), float(net.grid.y0)],
              "crs": None if net.crs is None else str(net.crs),
              "flow_edges": array_fingerprint(net.fd_src, net.fd_dst),
              "stream_edges": array_fingerprint(net.st_src, net.st_dst, net.st_nodes),
              "code_version": 3}

    def staged(name, fn):
        if store is None:
            log(name)
            return fn()
        return store.stage(name, fn, params, verbose=verbose)

    lab = staged("labels", lambda: dict(
        zip(("L", "nal"), basin_labels(net))))
    L, nal = lab["L"], lab["nal"]
    if len(nal) != len(net.st_nodes):
        # A checkpoint made from a different stream network (same
        # threshold, different construction) would otherwise fail
        # later with an opaque IndexError in the outlet-label lookup.
        raise ValueError(
            f"checkpointed labels describe {len(nal):,} stream cells but "
            f"the stream network passed in has {len(net.st_nodes):,}; "
            "the checkpoint was made from a different stream network. "
            "Build the StreamObject exactly as the checkpointed run did "
            "(threshold in the same units) or use another checkpoint_dir.")

    FX = staged("fx", lambda: {"FX": fx_grid(net.grid, net.fd_src,
                                             net.fd_dst)})["FX"]
    # the flow-edge arrays are not needed past FX; release every
    # reference (at full-domain size they are the largest allocation,
    # ~195 GB, and the FlowObject holds the same arrays)
    order_st = np.argsort(net.st_nodes, kind="stable")
    sorted_st = net.st_nodes[order_st]
    out_lab = nal[order_st][np.searchsorted(sorted_st, net.outlets)]
    net.fd_src = None
    net.fd_dst = None
    if release_inputs:
        for attr in ("source", "target"):
            try:
                setattr(fd, attr, None)
            except Exception:
                pass
    import gc
    gc.collect()

    ed = staged("edges", lambda: (lambda de: {
        "nodes": de.nodes, "lab_a": de.lab_a, "lab_b": de.lab_b,
        "is_boundary": de.is_boundary})(
        divide_edges(net.grid, L, net.st_src, net.st_dst,
                     net.outlets, out_lab)))

    if store is not None and store.has("network", params):
        log("network: loading from checkpoint")
        z = store.load("network")
        sn = SortedNetwork(seg_nodes=z["seg_nodes"], seg_ptr=z["seg_ptr"],
                           is_ring=z["is_ring"], endpoints=z["endpoints"],
                           junctions=z["junctions"])
        dist = z["distance"]
    else:
        log("segments")
        nw = segment_network(net.grid, ed["nodes"], FX)
        log(f"  {nw.n_segments} segments, {len(ed['nodes'])} edges, "
            f"{int(nw.is_ring.sum())} rings")
        log("sort")
        sn = sort_network(nw)
        log("distance")
        dist = divdist(sn, net.grid.cellsize)
        if store is not None:
            store.save("network", {
                "seg_nodes": sn.seg_nodes, "seg_ptr": sn.seg_ptr,
                "is_ring": sn.is_ring, "endpoints": sn.endpoints,
                "junctions": sn.junctions, "distance": dist,
                "edge_nodes": ed["nodes"],
                "edge_is_boundary": ed["is_boundary"],
            }, params)

    return DivideNetworkResult(
        grid=net.grid, network=sn, distance=dist,
        edge_nodes=ed["nodes"], edge_is_boundary=ed["is_boundary"],
        crs=net.crs, labels=L)
