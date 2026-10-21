# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Standard across-divide metrics in one call.

The reference implementation leaves the standard workflow to the user.
`DIVIDEobj2mapstruct` samples whatever grids it is handed, so computing a
χ contrast means building the stream network, trimming it to a base
level, running `chitransform` with a chosen concavity, mapping χ from the
channel network onto the hillslopes, and only then handing the result in
(see the TopoToolbox introduction to DIVIDEobj, 2020). Every one of those
steps is a choice, none is recorded in the output, and two of them change
the sign of the answer on a material fraction of divides.

This module does the whole of that in one call and records every choice
it made in the result, so a table can be checked after the fact:

    from pydivide import across_divide_metrics
    tab = across_divide_metrics(dem, divides, baselevel=5.0)
    print(tab.settings)          # every parameter that shaped the numbers

The metrics are the ones of Sect. 4 of the paper: elevation, slope,
hillslope relief, flow distance and χ, each as the pair mean, the
absolute contrast, the lower and higher side, and the SIGNED contrast,
with the two sides identified by their reach basin and by the outlet each
drains to.

**χ is not a property of the topography alone.** Its value depends on the
base level the integration starts from, on the concavity index m/n, on
the reference area A0, and on the drainage area at which a channel is
taken to begin. The first and the last change the sign of the contrast on
a measurable fraction of divides, so both are recorded in ``settings``
and neither has a silently correct default.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pydivide.gridspec import check_same_grid  # noqa: F401  (re-exported)

DEFAULT_MN = 0.45
DEFAULT_A0 = 1.0
DEFAULT_THRESHOLD = 1e6
SUPPORTED_METRICS = ("elev", "slope", "hr", "fd", "chi")
#: geometry and identity columns every table carries first, as the divide
#: export writes them; they do not depend on the threshold
BASE_COLUMNS = ("ID", "segid", "order", "distance", "distmin", "distmax",
                "length", "azimuth", "isbnd", "isring")
#: the five fields of every metric family, as :func:`metric_column` names them
METRIC_FIELDS = ("mean", "diff", "min", "max", "sgn")
#: the settings document written beside every export (see
#: :meth:`AcrossDivideTable.to_file`); bump when its layout changes
SETTINGS_DOCUMENT_VERSION = 1
#: identifies the settings document inside a GeoPackage (``md_standard_uri``)
SETTINGS_METADATA_URI = "https://github.com/ATLAS-team/pydivide#across-divide-settings"


def metric_column(family: str, fld: str, tag: str = "") -> str:
    """The column name of one field of one metric family at one scale:
    ``chi_sgn`` for a single-scale table, ``chi_sgn_1e5`` with the
    :func:`threshold_tag` of a multiscale rung. Every function in this
    module that writes or reads a metric column names it through here,
    so a producer and a consumer cannot disagree on the layout."""
    return f"{family}_{fld}{tag}"


def threshold_tag(t: float) -> str:
    """A column suffix that identifies a threshold exactly: ``_1e6`` for
    10^6, ``_110000`` for 1.1 x 10^5, ``_2500000`` for 2.5 x 10^6,
    ``_0p5`` for 0.5. Built from the shortest decimal that reads back as
    the same float (Python's ``repr``), so two distinct thresholds never
    share a tag. Non-finite and non-positive thresholds are refused."""
    t = float(t)
    if not np.isfinite(t) or t <= 0:
        raise ValueError(f"a channel threshold must be a finite positive number, got {t!r}")
    e = int(round(np.log10(t)))
    if t == 10.0 ** e:
        return f"_1e{e}" if e >= 0 else f"_1em{-e}"
    s = repr(t)
    if s.endswith(".0"):
        s = s[:-2]
    s = s.replace("e+0", "e").replace("e+", "e").replace("e-0", "em").replace("e-", "em")
    return "_" + s.replace(".", "p").replace("-", "m")


def _provenance(dem, D, *, flow_supplied: bool, streams_supplied: bool,
                divides_supplied: bool, threshold=None, units: str = "m2") -> dict:
    """What produced the numbers, beyond the parameters: software
    versions, the grid, and where the routing, the streams and the
    divides came from. When the caller supplied an object its
    preprocessing is unknown here and is recorded as such, not as the
    default the call would otherwise have applied."""
    import importlib.metadata as md

    from pydivide import __version__

    def ver(name):
        try:
            return md.version(name)
        except Exception:
            return None
    src = getattr(dem, "path", None) or getattr(dem, "name", None)
    thr = "" if threshold is None else f" at threshold {float(threshold):g} {units}"
    return {"pydivide_version": __version__, "topotoolbox_version": ver("topotoolbox"),
            "numpy_version": ver("numpy"),
            "dem_source": None if src is None else str(src),
            "grid": {"nr": int(D.grid.nr), "nc": int(D.grid.nc),
                     "cellsize": float(D.grid.cellsize),
                     "x0": float(D.grid.x0), "y0": float(D.grid.y0)},
            "crs": None if D.crs is None else str(D.crs),
            "routing": ("FlowObject supplied by the caller; its preprocessing is not "
                        "known here" if flow_supplied else
                        "pytopotoolbox FlowObject(dem), D8 with carving, computed by this call"),
            "streams": ("StreamObject supplied by the caller; how it was built is not "
                        "known here" if streams_supplied else
                        f"pytopotoolbox StreamObject computed by this call{thr}"),
            "divides": ("divide network supplied by the caller" if divides_supplied else
                        f"pydivide.divide_network computed by this call{thr}"),
            "flow_supplied": bool(flow_supplied),
            "streams_supplied": bool(streams_supplied),
            "divides_supplied": bool(divides_supplied)}


def _check_grids(dem, fd, st, D) -> None:
    """Objects supplied separately must describe the same grid, or the
    per-cell samples would be read from the wrong cells without any
    error being raised. Shape, cell size, origin and CRS are compared
    (:func:`check_same_grid`), the stream nodes are bounds-checked, and
    only C-ordered grids are accepted, since the grid is read flat."""
    if getattr(fd, "order", "C") != "C":
        raise ValueError(
            "pydivide works on C-ordered (row-major) grids, the layout pytopotoolbox uses for "
            "every raster it reads; this FlowObject is Fortran-ordered. Rebuild it from a "
            "C-ordered GridObject (numpy.ascontiguousarray) before calling pydivide.")
    check_same_grid(dem, "the DEM", fd, "the flow object")
    check_same_grid(dem, "the DEM", D, "the divide network")
    if hasattr(st, "transform") or hasattr(st, "shape"):
        check_same_grid(dem, "the DEM", st, "the stream network")
    shape = tuple(int(v) for v in np.asarray(dem.z).shape)
    rows, cols = st.node_indices
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    if rows.size and (int(rows.max()) >= shape[0] or int(cols.max()) >= shape[1]
                      or int(rows.min()) < 0 or int(cols.min()) < 0):
        raise ValueError("stream network node indices fall outside the DEM grid")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


@dataclass
class AcrossDivideTable:
    """Per-piece across-divide metrics, with the choices that made them."""

    columns: dict
    settings: dict
    n_pieces: int
    geometry: Any = None
    _notes: list = field(default_factory=list)

    def __len__(self) -> int:
        return self.n_pieces

    def to_geodataframe(self):
        import pandas as pd
        df = pd.DataFrame(self.columns)
        if self.geometry is None:
            return df
        import geopandas as gpd
        return gpd.GeoDataFrame(df, geometry=self.geometry, crs=self.settings.get("crs"))

    def settings_json(self, files=None) -> str:
        """The settings of this table as a JSON document: every parameter,
        the provenance, the warnings, the column names and the piece
        count, so the numbers in a file can be traced without the Python
        object that made them."""
        doc = {"document": "pydivide across-divide settings",
               "document_version": SETTINGS_DOCUMENT_VERSION,
               "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "files": [str(Path(f).name) for f in (files or [])],
               "n_pieces": int(self.n_pieces),
               "columns": list(self.columns),
               "settings": self.settings}
        return json.dumps(doc, indent=1, default=_json_default)

    def to_file(self, path, *, format: str = "auto", overwrite: bool = False,
                layer: str = "divides", **kw):
        """Write the table as a GIS layer, with its settings beside it.

        A continental divide network is tens of millions of pieces, which
        is past what a Shapefile can hold, its DBF being capped at 2 GB
        and its field names at ten characters. :mod:`pydivide.vectorio`
        defaults to GeoPackage for that reason, writes a File Geodatabase
        for ArcGIS on request, and splits a Shapefile into numbered parts
        when one is asked for anyway. Being able to write the whole domain
        in one pass, rather than tiling it, is the point of computing it
        in one pass.

        The scientific settings travel with the file. Every export is
        accompanied by ``<name>_settings.json``, the document of
        :meth:`settings_json` (thresholds, base level, concavity,
        reference area, side rule, provenance, warnings), and a
        GeoPackage additionally carries the same document inside the
        file, in its ``gpkg_metadata`` table, so a single file remains
        self-describing. :func:`read_export_settings` reads either.
        Existing files, the sidecar included, are replaced only with
        ``overwrite=True``, and nothing is written when any of them
        would be refused. Returns the paths written, the sidecar last.
        """
        from pydivide.vectorio import write_geopackage_metadata, write_vector

        path = Path(path)
        side = path.with_name(path.stem + "_settings.json")
        if side.exists() and not overwrite:
            raise FileExistsError(f"{side} exists; pass overwrite=True to replace it")
        written = [Path(p) for p in write_vector(self.to_geodataframe(), path, format=format,
                                                 layer=layer, overwrite=overwrite, **kw)]
        doc = self.settings_json(written)
        side.write_text(doc, encoding="utf-8")
        for p in written:
            if p.suffix.lower() == ".gpkg":
                write_geopackage_metadata(p, layer, doc, uri=SETTINGS_METADATA_URI)
        return written + [side]

    def describe(self) -> str:
        """The parametrisation in prose, for a methods section.

        chi in particular is not a property of the topography: it is
        defined only once a base level, a concavity index, a reference
        area and a channel-initiation threshold have been chosen. Stating
        them is what makes a chi contrast reproducible, and this returns
        the sentence that does it, for a single-scale table and for the
        table of :func:`across_divide_multiscale` alike.
        """
        s = self.settings
        units = s.get("threshold_units", "m2")
        grid = (f"from a {s['cellsize']:g} m grid" if s.get("cellsize") is not None
                else "from the grid")
        nseg = (f"{s['n_segments']:,} divide segments" if s.get("n_segments") is not None
                else "the divide segments")
        pieces = f"{s['seglength_m']:g} m" if s.get("seglength_m") else "one segment"
        if "thresholds" in s:
            thr = ", ".join(f"{float(t):g}" for t in s["thresholds"])
            net = (f", built at a channel-initiation threshold of "
                   f"{float(s['network_threshold']):g} {units} and held fixed"
                   if s.get("network_threshold") is not None else "")
            out = [f"Divide metrics were computed on {nseg} {grid}{net}, aggregated "
                   f"onto pieces of {pieces}, and measured at channel-initiation "
                   f"thresholds of {thr} {units}, one set of columns per threshold."]
            if s.get("side_rule"):
                out.append(f"At every threshold {s['side_rule']}.")
            if s.get("skipped"):
                out.append(f"Thresholds skipped after an error: "
                           f"{', '.join(sorted(s['skipped']))}.")
        else:
            thr = (f", with a channel-initiation threshold of {float(s['threshold']):g} {units}"
                   if s.get("threshold") is not None else "")
            out = [f"Divide metrics were computed on {nseg} {grid}, aggregated onto "
                   f"pieces of {pieces}{thr}."]
        ran = s.get("metrics_computed", s.get("metrics"))
        if ran:
            out.append("The metric families computed were " + ", ".join(str(m) for m in ran)
                       + ".")
        chi_done = s.get("chi_computed")
        if chi_done is None:
            chi_done = "chi" in (ran or []) and s.get("baselevel_m") is not None
        if chi_done and s.get("baselevel_m") is not None:
            out.append(f"chi was integrated upstream from a base level of "
                       f"{float(s['baselevel_m']):g} m with a concavity index m/n of "
                       f"{float(s.get('mn', DEFAULT_MN)):g} and a reference area A0 of "
                       f"{float(s.get('a0', DEFAULT_A0)):g} m2. "
                       f"Both the base level and the channel threshold change the "
                       f"sign of the chi contrast on a measurable fraction of "
                       f"divides, so neither is a default and both are reported here.")
            records = ([("", s)] if "n_chi_pieces_both_outlets_known" in s else
                       [(tag, rec) for tag, rec in s.get("per_scale", {}).items()
                        if "n_chi_pieces_both_outlets_known" in rec])
            for tag, rec in records:
                at = f" at {tag.lstrip('_')} {units}" if tag else ""
                n_no, n_all = rec["n_chi_pieces_without_common_base"], \
                    rec["n_chi_pieces_both_outlets_known"]
                if n_no:
                    out.append(f"{n_no:,} of the {n_all:,} pieces with a chi contrast{at} "
                               f"have a side draining to a river system that leaves the DEM "
                               f"above the base level ({rec['n_systems_above_baselevel']} "
                               f"such systems), so their contrasts compare two different "
                               f"base levels; they are flagged chi_common_base = 0.")
                else:
                    out.append(f"Both sides of every piece with a chi contrast{at} drain to "
                               f"river systems that reach the base level "
                               f"(chi_common_base = 1 throughout).")
        else:
            out.append("chi was not computed.")
        return " ".join(out)

    def disagreement(self, families=("chi", "hr", "fd"), tag: str = "") -> dict:
        """Where do the indicators disagree about the direction?

        χ marks the losing side with a HIGHER value and hillslope relief
        and flow distance mark it with a LOWER one, so agreement between
        χ and either of the other two means opposite signs. Returns one
        boolean array per pair plus a per-piece count of how many of the
        available indicators side with χ, and the coherence of each
        signed column (see :func:`coherence`).

        A piece where the indicators disagree is not a failure of the
        data. It is a divide whose integral and local evidence differ,
        which is where a recent change of forcing or a contrast in
        erodibility shows up, and it is worth mapping.
        """
        out = {}
        sgn = {}
        for f in families:
            k = metric_column(f, "sgn", tag)
            if k in self.columns:
                sgn[f] = np.asarray(self.columns[k], dtype=float)
        if "chi" not in sgn:
            return out
        c = sgn["chi"]
        votes = np.zeros(len(c), dtype=np.int16)
        avail = np.zeros(len(c), dtype=np.int16)
        for f, v in sgn.items():
            if f == "chi":
                continue
            ok = np.isfinite(c) & np.isfinite(v) & (c != 0) & (v != 0)
            agree = np.zeros(len(c), dtype=bool)
            agree[ok] = np.sign(c[ok]) != np.sign(v[ok])   # opposite = agree
            out[f"chi_agrees_with_{f}"] = agree
            votes += agree.astype(np.int16)
            avail += ok.astype(np.int16)
        out["n_indicators_agreeing_with_chi"] = votes
        out["n_indicators_compared"] = avail
        out["all_agree"] = (avail > 0) & (votes == avail)
        out["all_disagree"] = (avail > 0) & (votes == 0)
        for f in sgn:
            co = coherence(self.columns, f, tag)
            if co is not None:
                out[f"{f}_coherence"] = co
        return out


def coherence(columns: dict, family: str, tag: str = "") -> np.ndarray | None:
    """How consistently does a signed contrast point along one piece?

    The absolute contrast and the magnitude of the signed contrast are the
    same quantity per edge, so they can only separate when the contrast
    reverses inside the piece that averages them. The ratio is therefore 1
    where the contrast points one way along the whole piece and 0 where it
    cancels, and it costs nothing because both columns already exist.

    Measured on the Brazilian network it is 1 on 94.3 % of pieces for χ
    and on 69.3 % for hillslope relief, so a low value is common enough to
    be worth checking before a sign is believed.
    """
    s, d = metric_column(family, "sgn", tag), metric_column(family, "diff", tag)
    if s not in columns or d not in columns:
        return None
    sv = np.abs(np.asarray(columns[s], dtype=float))
    dv = np.asarray(columns[d], dtype=float)
    out = np.full(len(dv), np.nan)
    ok = np.isfinite(sv) & np.isfinite(dv) & (dv > 0)
    out[ok] = np.clip(sv[ok] / dv[ok], 0.0, 1.0)
    return out


def read_export_settings(path) -> dict:
    """The settings document an export of :meth:`AcrossDivideTable.to_file`
    was written with, from the file alone.

    ``path`` may be the sidecar itself (``<name>_settings.json``), any
    file the export wrote, or a GeoPackage. A sidecar answers for a file
    only when it lists that file among the files it describes (a
    document with no file list answers for no file, and is read only by
    its own path), so two exports whose names overlap (``metrics.gpkg``
    and ``metrics_part001.gpkg``, or a GeoPackage named like a Shapefile
    part) can never be confused: the file's own sidecar is consulted
    first, then, for a part of a split Shapefile, the export's sidecar.
    A GeoPackage answers from its embedded copy when no sidecar lists
    it, and when both exist they must agree or a ``ValueError`` names
    the two. Returns the parsed document; its ``"settings"`` entry is
    the ``settings`` dict of the table.
    """
    import re

    path = Path(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    is_shp = path.suffix.lower() in (".shp", ".shx", ".dbf", ".prj", ".cpg")
    name = path.with_suffix(".shp").name if is_shp else path.name

    def listed(sidecar):
        # a sidecar answers for a file only when it lists that file; a
        # document without a file list, or with an empty one, answers for
        # no file (it can still be read directly by its own path)
        if not sidecar.exists():
            return None
        d = json.loads(sidecar.read_text(encoding="utf-8"))
        files = [str(f) for f in d.get("files", []) or []]
        return d if name in files else None

    doc = listed(path.with_name(path.stem + "_settings.json"))
    if doc is None and is_shp:
        m = re.fullmatch(r"(.+)_part\d{3,}", path.stem)
        if m:
            doc = listed(path.with_name(m.group(1) + "_settings.json"))
    if path.suffix.lower() == ".gpkg" and path.exists():
        from pydivide.vectorio import read_geopackage_metadata
        docs = read_geopackage_metadata(path, uri=SETTINGS_METADATA_URI)
        if docs:
            embedded = json.loads(docs[-1])
            if doc is not None and doc.get("settings") != embedded.get("settings"):
                raise ValueError(f"the sidecar and the settings embedded in {path.name} "
                                 "disagree; the file was not written with that sidecar")
            return doc if doc is not None else embedded
    if doc is not None:
        return doc
    raise FileNotFoundError(f"no settings document lists {name}: no sidecar beside it "
                            "describes this file and it carries no embedded settings")


def _representative_pair(lab_a, lab_b, valid, pos, p_start, p_stop, n_pieces):
    """The basin pair each piece reports: the (A, B) pair that occupies
    the most edges of the piece, ties going to the smallest labels, so
    the pair reported is always one that occurs on a contributing edge.
    Returns ``(basin_a, basin_b, share)`` per piece, NaN where no edge of
    the piece has two labelled sides; ``share`` is the fraction of the
    piece's labelled edges that carry the reported pair (1 on a piece
    with a single pair)."""
    ba = np.full(n_pieces, np.nan)
    bb = np.full(n_pieces, np.nan)
    share = np.full(n_pieces, np.nan)
    if len(pos) == 0 or n_pieces == 0:
        return ba, bb, share
    k = np.searchsorted(p_start, pos, side="right") - 1
    inside = (k >= 0) & (pos < p_stop[np.maximum(k, 0)] - 1) & valid
    k = k[inside]
    if k.size == 0:
        return ba, bb, share
    A = np.asarray(lab_a, dtype=np.int64)[inside]
    B = np.asarray(lab_b, dtype=np.int64)[inside]
    if A.max() >= (1 << 31) or B.max() >= (1 << 31):
        raise ValueError("reach-basin labels exceed 2^31; the pair key cannot hold them")
    pk = (A << np.int64(32)) | B
    order = np.lexsort((pk, k))
    ks, pks = k[order], pk[order]
    brk = np.ones(len(ks), dtype=bool)
    brk[1:] = (ks[1:] != ks[:-1]) | (pks[1:] != pks[:-1])
    starts = np.flatnonzero(brk)
    counts = np.diff(np.append(starts, len(ks)))
    gk, gp = ks[starts], pks[starts]
    o2 = np.lexsort((gp, -counts, gk))
    gk, gp, counts = gk[o2], gp[o2], counts[o2]
    first = np.ones(len(gk), dtype=bool)
    first[1:] = gk[1:] != gk[:-1]
    sel_k, sel_p, sel_c = gk[first], gp[first], counts[first]
    total = np.bincount(k, minlength=n_pieces)
    ba[sel_k] = sel_p >> np.int64(32)
    bb[sel_k] = sel_p & np.int64(0xFFFFFFFF)
    share[sel_k] = sel_c / total[sel_k]
    return ba, bb, share


def across_divide_metrics(dem, *, flow=None, streams=None, divides=None,
                          baselevel=None, threshold: float = DEFAULT_THRESHOLD,
                          mn: float = DEFAULT_MN, a0: float = DEFAULT_A0,
                          seglength: float = 300.0,
                          metrics=("elev", "slope", "hr", "fd", "chi"),
                          units: str = "m2", verbose: bool = False,
                          side_reference=None):
    """Every standard across-divide metric for one DEM, in one call.

    Parameters
    ----------
    dem
        A pytopotoolbox ``GridObject``.
    flow, streams, divides
        Optional precomputed ``FlowObject``, ``StreamObject`` and
        :class:`~pydivide.pipeline.DivideNetworkResult`, to avoid
        recomputing them.
    baselevel
        Elevation the chi integration starts from. **Required whenever
        chi is requested**, with no default, because a chi contrast is
        meaningful only where both sides are referenced to the same base
        level and moving that base level reverses the sign of the
        contrast for 6 to 18 % of the divides between different river
        systems. A river system that leaves the DEM above the base level
        never reaches it and integrates chi from its own outlet instead,
        so every piece also carries the elevation at which each side's
        system leaves the network (``outlet_z_a``, ``outlet_z_b``) and
        the flag ``chi_common_base``, 1 where both lie at or below the
        base level, 0 where the contrast compares two base levels, NaN
        where a side has no outlet; ``settings`` counts the affected
        pieces and systems.
    threshold
        Drainage area at which a channel is taken to begin, in ``units``.
        It fixes where hillslope relief, flow distance and chi are
        measured from. Changing it between 10^5 and 10^6 m2 changes the
        sign of the chi contrast on a third of divides overall, although
        the disagreement is confined to weak contrasts.
    mn, a0
        Concavity index and reference area of the chi integral. chi has
        no value independent of them.
    seglength
        Length in metres of the pieces the per-edge values are
        aggregated onto, or None for one row per segment.
    side_reference
        Optional flattened reach-basin label raster that fixes which
        physical side of every divide is side A, instead of the labels
        of this threshold. Used by :func:`across_divide_multiscale` so
        that the sign means the same physical side at every scale; the
        basin and outlet columns still carry this threshold's labels, so
        ``basin_a`` may then be the higher label.

    Returns
    -------
    AcrossDivideTable
        ``columns`` holds the per-piece metrics, ``settings`` records
        every parameter above, and ``describe()`` states them in prose
        for a methods section.
    """
    import topotoolbox as tt

    from pydivide import divmetrics as M
    from pydivide.divmetrics import edge_positions, pair_metric_fields, sample_pair
    from pydivide.flownet import FlowNet
    from pydivide.grids import (
        arcslope_degrees,
        distance_to_stream,
        nal_to_hillslopes,
        vertical_distance_to_stream,
    )
    from pydivide.labels import basin_labels
    from pydivide.pipeline import divide_network

    metrics = tuple(metrics)
    unknown = [m for m in metrics if m not in SUPPORTED_METRICS]
    if unknown:
        raise ValueError(f"unknown metric name(s) {unknown}; supported: "
                         f"{list(SUPPORTED_METRICS)}")
    if "chi" in metrics and baselevel is None:
        raise ValueError(
            "baselevel is required for chi and has no default. A chi contrast "
            "is only meaningful where both sides are referenced to the same "
            "base level, and moving the base level changes the sign of the "
            "contrast on a material fraction of divides. "
            "Pass baselevel=<elevation in m>, or drop 'chi' from metrics.")

    threshold_tag(threshold)            # refuses a non-finite or non-positive threshold
    fd = flow if flow is not None else tt.FlowObject(dem)
    st = streams if streams is not None else tt.StreamObject(
        fd, threshold=float(threshold), units=units)
    D = divides if divides is not None else divide_network(
        dem=dem, fd=fd, st=st, minarea=float(threshold), units=units,
        verbose=verbose)
    _check_grids(dem, fd, st, D)

    settings = {
        "threshold": float(threshold), "threshold_units": units,
        "mn": float(mn), "a0": float(a0),
        "baselevel_m": None if baselevel is None else float(baselevel),
        "seglength_m": None if seglength is None else float(seglength),
        "metrics": list(metrics), "cellsize": float(D.grid.cellsize),
        "n_segments": int(D.network.n_segments),
        "crs": None if D.crs is None else str(D.crs),
        "provenance": _provenance(dem, D, flow_supplied=flow is not None,
                                  streams_supplied=streams is not None,
                                  divides_supplied=divides is not None,
                                  threshold=threshold, units=units),
    }
    if streams is not None:
        settings["threshold_note"] = ("streams were supplied by the caller; "
                                      "threshold records the value passed, not "
                                      "how that network was built")
    # which physical side is side A, stated in every table: the lower
    # reach-basin label at the first edge of each segment, held along
    # the segment (the segment rule), on this threshold's labels or on a
    # caller-supplied reference labelling, identified by its fingerprint
    if side_reference is None:
        settings["side_rule"] = ("side A is the physical side with the lower reach-basin "
                                 "label at the first edge of each segment, held fixed along "
                                 "the segment, on the labels of this threshold")
    else:
        from pydivide.checkpoint import array_fingerprint
        settings["side_rule"] = ("side A is the physical side with the lower label of a "
                                 "caller-supplied reference labelling at the first edge of "
                                 "each segment, held fixed along the segment; the basin and "
                                 "outlet columns carry this threshold's labels on those sides")
        settings["side_reference_fingerprint"] = array_fingerprint(
            np.asarray(side_reference).ravel())

    net = FlowNet.from_pytopotoolbox(fd, st)
    L, _ = basin_labels(net)
    L = np.asarray(L).ravel()

    pos, edges = edge_positions(D.network)
    n_pos = len(D.network.seg_nodes) - 1
    forward = D.network.seg_nodes[pos] == edges[:, 0]
    seg_of_edge = M.segment_of_position(D.network)[pos]
    orient_labels = L if side_reference is None else np.asarray(side_reference).ravel()
    tab = M.build_edge_table(D.grid, edges, np.asarray(dem.z, np.float32).ravel(),
                             orient_labels, D.distance[1:][pos],
                             forward=forward, seg=seg_of_edge)
    if side_reference is not None:
        # the orientation came from the reference labels; the identities
        # exported are this threshold's labels on those same two sides
        la = np.where(tab.cell_a >= 0, L[np.maximum(tab.cell_a, 0)], -1)
        lb = np.where(tab.cell_b >= 0, L[np.maximum(tab.cell_b, 0)], -1)
        bad = (la < 0) | (lb < 0)
        tab.lab_a = np.where(bad, -1, la)
        tab.lab_b = np.where(bad, -1, lb)
        tab.pair_lo = np.where(bad, -1, np.minimum(la, lb))
        tab.pair_hi = np.where(bad, -1, np.maximum(la, lb))

    z = np.asarray(dem.z, dtype=np.float32).ravel()
    grids = {}
    if "elev" in metrics:
        grids["elev"] = z
    if "slope" in metrics:
        grids["slope"] = np.asarray(arcslope_degrees(dem.z, D.grid.cellsize),
                                    dtype=np.float32).ravel()
    if "hr" in metrics:
        grids["hr"] = np.asarray(vertical_distance_to_stream(fd, st, dem),
                                 dtype=np.float32).ravel()
    if "fd" in metrics:
        grids["fd"] = np.asarray(distance_to_stream(net), dtype=np.float32).ravel()
    if "chi" in metrics:
        z2 = np.asarray(dem.z)
        zmin = float(np.nanmin(z2))
        zmax = float(np.nanmax(z2))
        # a base level below the lowest ground trims nothing, so chi would
        # silently be integrated from whatever the lowest cell happens to
        # be rather than from the level asked for. That is the failure this
        # whole paper is about, so it is reported rather than allowed.
        settings["dem_min_m"] = zmin
        settings["dem_max_m"] = zmax
        if float(baselevel) <= zmin:
            settings["baselevel_warning"] = (
                f"baselevel {baselevel:g} m is at or below the lowest cell of this "
                f"DEM ({zmin:.1f} m), so the network was not trimmed and chi starts "
                f"from the domain edge, not from the level requested. On a domain "
                f"that does not reach the true base level this makes the chi "
                f"contrast incomparable with one computed elsewhere.")
        elif float(baselevel) >= zmax:
            raise ValueError(
                f"baselevel {baselevel:g} m is at or above the highest cell "
                f"({zmax:.1f} m); the trimmed network would be empty")
        stb = st.upstreamto(z2 > float(baselevel)).clean()
        if len(stb.stream) == 0:
            raise ValueError(
                f"no stream cells remain above baselevel {baselevel:g} m")
        settings["n_stream_cells_above_baselevel"] = int(len(stb.stream))
        # the upstream area of every stream node, gathered by row and
        # column so the sample does not depend on the grid's memory order
        acc = np.asarray(fd.flow_accumulation())
        r_b, c_b = stb.node_indices
        chi_nal = stb.chitransform(acc[np.asarray(r_b), np.asarray(c_b)].astype(np.float32),
                                   a0=a0, mn=mn, correctcellsize=True)
        netb = FlowNet.from_pytopotoolbox(fd, stb)
        grids["chi"] = np.asarray(nal_to_hillslopes(netb, chi_nal),
                                  dtype=np.float32).ravel()

    seg_of_piece, p_start, p_stop = D._piece_table(seglength)
    # the geometry and identity columns of the divide export come first,
    # so a table row reads like a row of the divide layer plus metrics
    base = D.to_geodataframe("strahler", seglength)
    cols = {k: base[k].to_numpy() for k in BASE_COLUMNS}

    def agg(per_edge, how="mean"):
        full = M.scatter(np.asarray(per_edge, dtype=np.float64), pos, n_pos)
        return M.aggregate_to_pieces(full, p_start, p_stop, how)

    for name, g in grids.items():
        p, q = sample_pair(g, tab.cell_a, tab.cell_b)
        for suf, v in pair_metric_fields(p, q).items():
            cols[metric_column(name, suf)] = agg(v)
    valid = (tab.lab_a > 0) & (tab.lab_b > 0)
    # the pair a piece reports is one its edges actually border: the pair
    # on most of its labelled edges, ties to the smallest labels. Where
    # the edges border more than one pair the piece is flagged mixed and
    # basin_pair_share says how much of it the reported pair covers, so
    # the sign is not read as if one pair held along the whole piece.
    n_pieces_total = len(seg_of_piece)
    cols["basin_a"], cols["basin_b"], share = _representative_pair(
        tab.lab_a, tab.lab_b, valid, pos, p_start, p_stop, n_pieces_total)
    # outlet identity of each side: the reach the basin's water leaves
    # the network through, so "different river systems" can be read off
    outl = M.reach_outlet_labels(L, net.st_src, net.st_dst)
    for nm in ("basin_a", "basin_b"):
        b = np.asarray(cols[nm])
        ok = np.isfinite(b)
        o = np.full(b.shape, np.nan)
        o[ok] = outl[b[ok].astype(np.int64)]
        cols[nm.replace("basin", "outlet")] = o
    # the elevation at which each side's river system leaves the stream
    # network, and whether both lie at or below the chi base level. chi is
    # integrated from the base level only along a river that reaches it;
    # a system that leaves the DEM above the base level starts its chi at
    # its own outlet, so a contrast between such a system and one that
    # reaches the base level compares two different base levels, and the
    # flag says so piece by piece
    z_out = np.full(len(outl), np.nan)
    oc = np.asarray(net.outlets, dtype=np.int64)
    oc = oc[(oc >= 0) & (oc < L.size)]
    oc = oc[L[oc] > 0]
    z_out[L[oc]] = z[oc]
    for nm in ("outlet_a", "outlet_b"):
        o = np.asarray(cols[nm])
        ok = np.isfinite(o)
        v = np.full(o.shape, np.nan)
        v[ok] = z_out[o[ok].astype(np.int64)]
        cols[nm.replace("outlet", "outlet_z")] = v
    if "chi" in grids:
        b = float(baselevel)
        za, zb = cols["outlet_z_a"], cols["outlet_z_b"]
        known = np.isfinite(za) & np.isfinite(zb)
        flag = np.full(za.shape, np.nan)
        flag[known] = ((za[known] <= b) & (zb[known] <= b)).astype(np.float64)
        cols["chi_common_base"] = flag
        with_chi = known & np.isfinite(np.asarray(cols[metric_column("chi", "sgn")], float))
        above = np.unique(np.concatenate([
            np.asarray(cols["outlet_a"])[with_chi & (za > b)],
            np.asarray(cols["outlet_b"])[with_chi & (zb > b)]]))
        settings["n_chi_pieces_both_outlets_known"] = int(with_chi.sum())
        settings["n_chi_pieces_without_common_base"] = int((with_chi & (flag == 0)).sum())
        settings["n_systems_above_baselevel"] = int(len(above))
        if settings["n_chi_pieces_without_common_base"]:
            settings["common_base_warning"] = (
                f"{settings['n_chi_pieces_without_common_base']:,} of the "
                f"{settings['n_chi_pieces_both_outlets_known']:,} pieces with a chi contrast "
                f"have a side that drains to a river system leaving the DEM above the base "
                f"level of {b:g} m ({settings['n_systems_above_baselevel']} such systems); "
                f"chi on that side starts at the system's own outlet, so those contrasts "
                f"compare two different base levels. They are flagged chi_common_base = 0; "
                f"read chi contrasts where chi_common_base = 1, or enlarge the DEM.")
    # mixed: the labels on either side are not constant along the piece.
    # Compared side by side, exactly, not through a combined key.
    lo = np.where(valid, tab.pair_lo.astype(np.float64), np.nan)
    hi = np.where(valid, tab.pair_hi.astype(np.float64), np.nan)
    lo_min, lo_max = agg(lo, "min"), agg(lo, "max")
    hi_min, hi_max = agg(hi, "min"), agg(hi, "max")
    mixed = np.isfinite(lo_min) & ((lo_min != lo_max) | (hi_min != hi_max))
    cols["basin_pair_mixed"] = mixed.astype(np.int8)
    cols["basin_pair_share"] = share
    if "hr_diff" in cols:
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.asarray(cols["hr_diff"]) / (2.0 * np.asarray(cols["hr_mean"]))
        d[~np.isfinite(d)] = np.nan
        cols["dai"] = d
    for fam in grids:
        c = coherence(cols, fam)
        if c is not None:
            cols[f"{fam}_coherence"] = c

    n = len(seg_of_piece)
    # what actually ran, so the description cannot claim more than was done
    settings["metrics_computed"] = list(grids)
    settings["chi_computed"] = "chi" in grids
    # the piece geometry, in the same order as the columns, so the table
    # is a GIS layer and not only a table
    geometry = base.geometry.values
    return AcrossDivideTable(columns=cols, settings=settings, n_pieces=n,
                             geometry=geometry)


def across_divide_multiscale(dem, thresholds=(1e5, 1e6, 1e7), *,
                             baselevel=None, network_threshold=None,
                             mn: float = DEFAULT_MN, a0: float = DEFAULT_A0,
                             seglength: float = 300.0,
                             metrics=("hr", "chi"), units: str = "m2",
                             verbose: bool = False, errors: str = "raise"):
    """The same divides, measured at several channel-initiation scales.

    The threshold does not merely perturb a contrast, it changes what is
    being compared. A reach basin is the area draining to one stream
    reach, so refining the threshold splits reaches and a coarse basin
    pair breaks into several fine ones. Measured on a Brazilian window,
    7,373 reach basins at 10\u2076 m\u00b2 become 68,869 at 10\u2075, 21,452 competing
    pairs become 75,316, only 24 % of coarse pairs survive as a single
    fine pair, and no fine basin straddles a coarse one, so the fine
    partition is an exact subdivision of the coarse one.

    That nesting is what makes a ladder of thresholds meaningful rather
    than merely repetitive. At a coarse threshold the contrast compares
    two rivers; at a fine one it compares two hollows inside them. A
    divide whose contrast keeps its sign all the way up the ladder is
    expressing a competition that holds at every scale. One whose sign
    changes partway up is expressing a competition that belongs to a
    particular scale, and the scale is the interesting part.

    The divide network is built once, at ``network_threshold`` or the
    coarsest threshold given, and held fixed, so only the measurement
    scale varies and the geometry cannot confound the comparison. Side A
    is also held fixed: it is the physical side that carries the lower
    reach-basin label at the network threshold, at every scale, so a
    sign that changes between scales is a change of contrast and never
    a change of convention (``settings["side_rule"]``). The
    ``basin_a_<tag>`` and ``basin_b_<tag>`` columns carry each scale's
    own labels on those two fixed sides.

    ``errors`` is ``"raise"`` (default: a failure at any threshold stops
    the call) or ``"skip"`` (the threshold is left out and recorded in
    ``settings["skipped"]`` with the exception). Column suffixes are the
    :func:`threshold_tag` of each threshold, ``_1e6`` for 10^6 and
    ``_110000`` for 1.1 x 10^5, listed in ``settings["tags"]``;
    thresholds must be distinct, finite and positive.

    Returns an :class:`AcrossDivideTable` whose columns carry the suffix
    of each threshold, plus per-family summaries: ``<fam>_sign_consistent``
    (the sign is the same at every scale where it is defined),
    ``<fam>_n_scales`` and ``<fam>_peak_scale`` (the threshold at which
    the absolute contrast is largest, which is the scale at which the
    competition is most strongly expressed). ``describe()``,
    ``disagreement(tag=...)`` and :func:`coherence` read the table by
    tag. ``settings["per_scale"]`` keeps each threshold's own record
    (its warnings and stream-cell counts); the routing and the network
    are shared, so their provenance is recorded once.
    """
    import topotoolbox as tt

    from pydivide.pipeline import divide_network

    thresholds = [float(t) for t in thresholds]
    if not thresholds:
        raise ValueError("thresholds is empty; give at least one channel threshold")
    if not all(np.isfinite(t) and t > 0 for t in thresholds):
        raise ValueError(f"every channel threshold must be a finite positive number: "
                         f"{thresholds}")
    thresholds = sorted(thresholds)
    if errors not in ("raise", "skip"):
        raise ValueError("errors must be 'raise' or 'skip'")
    tags = [threshold_tag(t) for t in thresholds]
    if len(set(tags)) != len(tags) or len(set(thresholds)) != len(thresholds):
        raise ValueError(f"thresholds must be distinct: {thresholds}")
    unknown = [m for m in metrics if m not in SUPPORTED_METRICS]
    if unknown:
        raise ValueError(f"unknown metric name(s) {unknown}; supported: "
                         f"{list(SUPPORTED_METRICS)}")
    if "chi" in metrics and baselevel is None:
        raise ValueError("baselevel is required for chi; see across_divide_metrics")
    net_thr = float(network_threshold if network_threshold is not None else thresholds[-1])

    fd = tt.FlowObject(dem)
    st_net = tt.StreamObject(fd, threshold=net_thr, units=units)
    D = divide_network(dem=dem, fd=fd, st=st_net, minarea=net_thr, units=units,
                       verbose=verbose)

    cols, per_scale = {}, {}
    settings = {"thresholds": thresholds, "tags": tags, "threshold_units": units,
                "network_threshold": net_thr,
                "mn": float(mn), "a0": float(a0),
                "baselevel_m": None if baselevel is None else float(baselevel),
                "seglength_m": None if seglength is None else float(seglength),
                "metrics": list(metrics), "cellsize": float(D.grid.cellsize),
                "n_segments": int(D.network.n_segments),
                "crs": None if D.crs is None else str(D.crs),
                # the routing and the network are computed here and shared
                # by every scale, so their provenance is recorded once
                "provenance": _provenance(dem, D, flow_supplied=False,
                                          streams_supplied=False, divides_supplied=False,
                                          threshold=net_thr, units=units),
                "per_scale": {}}
    geometry = None
    # one physical side is side A at every scale: the side that carries the
    # lower label at the threshold the network was built with. Without this
    # the lower-label rule can pick the opposite side at a finer threshold
    # and a sign change would be a change of convention, not of contrast.
    side_ref = np.asarray(D.labels).ravel()
    settings["side_rule"] = ("side A is the physical side with the lower reach-basin "
                             f"label at the network threshold {net_thr:g} {units}, "
                             "held fixed at every scale")
    for t, tag in zip(thresholds, tags):
        try:
            sub = across_divide_metrics(
                dem, flow=fd, divides=D,
                streams=tt.StreamObject(fd, threshold=t, units=units),
                baselevel=baselevel, threshold=t, mn=mn, a0=a0,
                seglength=seglength, metrics=metrics, units=units,
                side_reference=side_ref)
        except Exception as exc:
            if errors == "raise":
                raise
            settings.setdefault("skipped", {})[tag] = f"{type(exc).__name__}: {exc}"
            continue
        geometry = sub.geometry
        # every scale's own record (threshold, warnings, stream-cell counts),
        # without the shared provenance repeated
        settings["per_scale"][tag] = {k: v for k, v in sub.settings.items()
                                      if k not in ("provenance", "crs", "cellsize",
                                                   "n_segments", "threshold_note")}
        per_scale[t] = sub.columns
        for k, v in sub.columns.items():
            if k in BASE_COLUMNS:
                cols.setdefault(k, v)        # identical at every scale
            else:
                cols[f"{k}{tag}"] = v

    scales = sorted(per_scale)
    ran = sorted({m for rec in settings["per_scale"].values()
                  for m in rec.get("metrics_computed", [])})
    settings["metrics_computed"] = ran
    settings["chi_computed"] = "chi" in ran
    for fam in metrics:
        S = [np.asarray(per_scale[t].get(metric_column(fam, "sgn")), float)
             for t in scales if metric_column(fam, "sgn") in per_scale[t]]
        if len(S) < 2:
            continue
        A = np.vstack(S)
        ok = np.isfinite(A) & (A != 0)
        n = ok.sum(axis=0)
        pos = ((A > 0) & ok).sum(axis=0)
        cols[f"{fam}_n_scales"] = n.astype(np.int16)
        cols[f"{fam}_sign_consistent"] = (n > 1) & ((pos == n) | (pos == 0))
        mag = np.where(ok, np.abs(A), -np.inf)
        best = np.argmax(mag, axis=0)
        peak = np.array([scales[i] for i in best], dtype=float)
        peak[n == 0] = np.nan
        cols[f"{fam}_peak_scale"] = peak
    n_pieces = len(next(iter(cols.values()))) if cols else 0
    return AcrossDivideTable(columns=cols, settings=settings, n_pieces=n_pieces,
                             geometry=geometry)
