# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""GIS layers of the competing basin pairs behind a signed metric.

A signed contrast is a statement about two named basins. To check it in a
GIS you need to see those basins, know which one the sign calls A, and be
able to select the pair. These functions build the layers that allow it:

``pair_table``       one row per competing pair, with a stable ``pair_id``
                     and every signed value re-expressed as side A minus
                     side B of that pair before it is averaged
``basin_polygons``   the catchment of each reach basin, as polygons
``basin_points``     one point per basin, at its centroid
``pair_side_points`` one point per pair per side, so side A and side B can
                     be styled as two separate layers

``pair_id`` is a dense integer over the unordered pair of reach-basin
labels, so it is stable within one labelling and joins the divide pieces,
the catchments and the points to each other.

A pair exists only relative to a labelling, and the labelling follows the
channel-initiation threshold, so the same stretch of divide belongs to a
different pair at every scale. Pass ``scale=`` to :func:`pair_table` and
the rows carry a ``scale`` column and a ``pair_uid`` of the form
``"1e8:417"``, which is unique across scales. Selecting the whole divide
between two large basins is then a matter of building the labels at a
coarse threshold and selecting that scale's pair, while the same divide
pieces still carry their fine-scale pair for a closer question.

**The pair key.** ``pair_id`` and ``pair_uid`` are stable within one
table. :func:`pair_key` names a pair from its labels and its partition
alone, so the same number identifies it in every product built on the
same network and in every file::

    pair_key = E * 10**14 + lo * 10**7 + hi

with ``E`` the exponent of the channel threshold in m² (10⁶ gives 6),
``lo`` the smaller reach-basin label and ``hi`` the larger. Read left to
right, ``700034600003461`` is partition 10⁷, basins 3460 and 3461. The
largest key, 999999999999999, is below 2⁵³, so it survives exactly in
every program that stores numbers as doubles (ArcGIS, Excel, JSON). The
rule has a stated domain and refuses anything outside it: the threshold
must be an exact power of ten from 10¹ to 10⁹ m² (a key has one decimal
digit for the partition), labels must be positive and below 10⁷, and the
two labels must differ. A key is unique within one network's labelling;
across networks the identity is the network plus the key.
:func:`pair_table` adds a ``pair_key`` column when its ``scale`` names
such a threshold.

Put the divide layer on top of the catchments, select a ``pair_id``, and
the two competing basins and the divide between them are all that is
highlighted.
"""

from __future__ import annotations

import math
import re

import numpy as np

#: labels must be strictly below this, so each fits in seven digits
PAIR_KEY_LABEL_LIMIT = 10 ** 7
#: the multiplier of the partition digit
PAIR_KEY_SCALE_FACTOR = 10 ** 14
#: the value written where there is no pair (no label on one side)
NO_PAIR = -1


def pair_key_exponent(threshold) -> int:
    """The exponent E of a threshold 10**E m², or an error.

    Only exact powers of ten between 10¹ and 10⁹ are accepted: a key
    has one decimal digit for the partition, and a threshold such as
    5 × 10⁶ has no digit of its own, so accepting it would let two
    partitions share keys.
    """
    if isinstance(threshold, str):
        threshold = float(threshold)
    t = float(threshold)
    if not (t > 0 and math.isfinite(t)):
        raise ValueError(f"threshold {threshold!r} is not a positive number")
    e = round(math.log10(t))
    if not (1 <= e <= 9) or abs(t - 10.0 ** e) > 1e-9 * t:
        raise ValueError(
            f"threshold {threshold!r} m2 is not an exact power of ten from 1e1 to 1e9; "
            "the pair key has one decimal digit for the partition")
    return int(e)


def pair_key(lo, hi, threshold):
    """The key of the unordered pair ``(lo, hi)`` at one partition.

    Works on scalars and arrays; the order of the two labels does not
    matter. A pair with a label of 0 or less (no label) is ``NO_PAIR``;
    a label at or above ``PAIR_KEY_LABEL_LIMIT`` or ``lo == hi`` raises,
    because a key for it would be wrong rather than missing.
    """
    e = pair_key_exponent(threshold)
    a = np.asarray(lo, dtype=np.int64)
    b = np.asarray(hi, dtype=np.int64)
    small, large = np.minimum(a, b), np.maximum(a, b)
    valid = small > 0
    if np.any(valid & (large >= PAIR_KEY_LABEL_LIMIT)):
        raise ValueError(f"a label reaches {int(large[valid].max()):,}; the key holds "
                         f"labels below {PAIR_KEY_LABEL_LIMIT:,}")
    if np.any(valid & (small == large)):
        raise ValueError("lo equals hi: a divide separates two different basins")
    key = (np.int64(e) * np.int64(PAIR_KEY_SCALE_FACTOR)
           + small * np.int64(PAIR_KEY_LABEL_LIMIT) + large)
    out = np.where(valid, key, np.int64(NO_PAIR))
    return int(out) if out.ndim == 0 else out


def decode_pair_key(key):
    """``(threshold_m2, lo, hi)`` of a key; ``NO_PAIR`` gives ``None``."""
    k = int(key)
    if k == NO_PAIR:
        return None
    if k <= 0:
        raise ValueError(f"{key!r} is not a pair key")
    e, rest = divmod(k, PAIR_KEY_SCALE_FACTOR)
    lo, hi = divmod(rest, PAIR_KEY_LABEL_LIMIT)
    if not (1 <= e <= 9) or lo <= 0 or hi <= lo:
        raise ValueError(f"{key!r} does not follow the pair-key rule")
    return float(10 ** e), int(lo), int(hi)


def readable_pair_key(key) -> str:
    """``'1e7 | 3460 | 3461'``, for a label or a legend."""
    d = decode_pair_key(key)
    if d is None:
        return "no pair"
    return f"1e{pair_key_exponent(d[0])} | {d[1]} | {d[2]}"


def _scale_threshold(scale):
    """The threshold a scale tag names, if it is an exact power of ten."""
    if scale is None:
        return None
    try:
        t = float(scale)
        pair_key_exponent(t)
        return t
    except (TypeError, ValueError):
        return None


def pair_ids(basin_a, basin_b):
    """Stable dense id for each unordered basin pair.

    Returns ``(pair_id, lookup)`` where ``pair_id`` is one integer per
    input row and ``lookup`` is a structured array of the distinct pairs
    in id order. A row is a pair only where both sides carry a positive
    label and the two labels differ: -1 where a side is missing, has no
    label (0 or less), or where both sides carry the same label, which
    is a divide of a finer network lying inside one basin of this
    partition and so has no competing pair at this scale.
    """
    a = np.asarray(basin_a, dtype=float)
    b = np.asarray(basin_b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0) & (a != b)
    lo = np.where(ok, np.minimum(a, b), -1).astype(np.int64)
    hi = np.where(ok, np.maximum(a, b), -1).astype(np.int64)
    key = np.where(ok, lo * np.int64(1 << 32) + hi, -1)
    uniq = np.unique(key[key >= 0])
    pid = np.full(len(key), -1, dtype=np.int64)
    pid[key >= 0] = np.searchsorted(uniq, key[key >= 0])
    lookup = np.empty(len(uniq), dtype=[("pair_id", "i8"), ("basin_lo", "i8"),
                                        ("basin_hi", "i8")])
    lookup["pair_id"] = np.arange(len(uniq))
    lookup["basin_lo"] = uniq >> np.int64(32)
    lookup["basin_hi"] = uniq & np.int64(0xFFFFFFFF)
    return pid, lookup


_SGN_TOKEN = re.compile(r"(?:^|_)sgn(?:_|$)")


def is_signed_column(name: str) -> bool:
    """Whether a value column is a signed contrast by the naming rule of
    :func:`pydivide.across_divide_metrics`: the name carries the token
    ``sgn`` (``chi_sgn``, ``hr_sgn_1e5``). ``chi_diff`` and ``dai`` are
    unsigned."""
    return _SGN_TOKEN.search(str(name)) is not None


def pair_table(basin_a, basin_b, values: dict | None = None,
               scale: str | None = None, signed="auto"):
    """One row per competing pair, with the summary of any per-piece values.

    ``basin_a`` and ``basin_b`` keep the side convention of the signed
    columns: on every input row a signed value is A minus B. Where a pair
    appears with both orientations, which happens only where a reach
    label changes mid-segment, the more common orientation is reported as
    ``side_a``/``side_b`` (ties go to the lower label) and ``n_flipped``
    counts the rows with the other orientation.

    A signed value of a flipped row describes the same physical contrast
    with the opposite sign, so it is negated before it enters the pair
    mean: ``<name>_mean`` is the mean of the values expressed as
    ``side_a`` minus ``side_b``, and reversing A and B on any row while
    negating its signed values leaves the table unchanged. Which columns
    are signed is stated by ``signed``: an iterable of column names, an
    empty tuple for none, or ``"auto"`` (default), which takes every
    column that :func:`is_signed_column` accepts. Unsigned columns are
    averaged as they are. ``<name>_absmax`` is the largest absolute
    value of the pair and does not depend on orientation.
    """
    import pandas as pd

    pid, lookup = pair_ids(basin_a, basin_b)
    a = np.asarray(basin_a, dtype=float)
    ok = pid >= 0
    n = len(lookup)
    rows = {"pair_id": lookup["pair_id"], "basin_lo": lookup["basin_lo"],
            "basin_hi": lookup["basin_hi"]}
    if scale is not None:
        # a pair is only defined relative to a labelling, and the labelling
        # is a function of the channel threshold, so the same two points on
        # the ground belong to different pairs at different scales. The uid
        # carries the scale so the two can never be confused or joined by
        # accident, which is the whole reason the scale is a first-class
        # column rather than a note in a filename.
        rows["scale"] = np.full(len(lookup), str(scale), dtype=object)
        rows["pair_uid"] = np.array([f"{scale}:{i}" for i in lookup["pair_id"]],
                                    dtype=object)
        # the cross-partition key, when the scale is a power of ten the rule
        # can encode; other scales keep the dense id and the uid only
        t = _scale_threshold(scale)
        if t is not None and (len(lookup) == 0
                              or int(lookup["basin_hi"].max()) < PAIR_KEY_LABEL_LIMIT):
            rows["pair_key"] = (pair_key(lookup["basin_lo"], lookup["basin_hi"], t)
                                if len(lookup) else np.zeros(0, np.int64))
    rows["n_pieces"] = np.bincount(pid[ok], minlength=n)
    a_is_lo = np.zeros(len(pid), bool)
    a_is_lo[ok] = a[ok] == np.minimum(a[ok], np.asarray(basin_b, float)[ok])
    n_lo = np.bincount(pid[ok], weights=a_is_lo[ok].astype(float), minlength=n)
    side_a = np.where(n_lo * 2 >= rows["n_pieces"], lookup["basin_lo"], lookup["basin_hi"])
    rows["side_a"] = side_a
    rows["side_b"] = np.where(side_a == lookup["basin_lo"], lookup["basin_hi"],
                              lookup["basin_lo"])
    rows["n_flipped"] = np.minimum(n_lo, rows["n_pieces"] - n_lo).astype(np.int64)
    # +1 where the row's A is the pair's reported side A, -1 where it is
    # the other basin, so a signed value can be re-expressed as side A
    # minus side B before it is averaged
    orient = np.ones(len(pid), dtype=float)
    orient[ok] = np.where(a[ok] == side_a[pid[ok]], 1.0, -1.0)
    values = values or {}
    if isinstance(signed, str):
        if signed != "auto":
            raise ValueError("signed must be 'auto' or an iterable of column names")
        signed_cols = {k for k in values if is_signed_column(k)}
    else:
        signed_cols = set(signed)
        unknown = signed_cols - set(values)
        if unknown:
            raise ValueError(f"signed names {sorted(unknown)} are not in values")
    for name, v in values.items():
        v = np.asarray(v, dtype=float)
        m = ok & np.isfinite(v)
        aligned = v * orient if name in signed_cols else v
        cnt = np.bincount(pid[m], minlength=n)
        s = np.bincount(pid[m], weights=aligned[m], minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            rows[f"{name}_mean"] = np.where(cnt > 0, s / cnt, np.nan)
        mx = np.full(n, -np.inf)
        np.maximum.at(mx, pid[m], np.abs(v[m]))
        mx[~np.isfinite(mx)] = np.nan
        rows[f"{name}_absmax"] = mx
    return pd.DataFrame(rows)


def _basin_stats(labels, grid, keep=None):
    """(label, n_cells, mean x, mean y) for every reach basin present."""
    L = np.asarray(labels).ravel()
    sel = L > 0
    if keep is not None:
        sel &= np.isin(L, np.asarray(keep, dtype=L.dtype))
    idx = np.flatnonzero(sel)
    if idx.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0), np.empty(0)
    lab = L[idx]
    uniq, inv = np.unique(lab, return_inverse=True)
    rows, cols = np.divmod(idx, grid.nc)
    cnt = np.bincount(inv, minlength=len(uniq))
    x = np.bincount(inv, weights=grid.x0 + (cols + 0.5) * grid.cellsize,
                    minlength=len(uniq)) / cnt
    y = np.bincount(inv, weights=grid.y0 - (rows + 0.5) * grid.cellsize,
                    minlength=len(uniq)) / cnt
    return uniq, cnt, x, y


def basin_points(labels, grid, *, crs=None, keep=None):
    """One point per reach basin, at its centroid, with its area."""
    import geopandas as gpd
    import pandas as pd
    import shapely

    lab, cnt, x, y = _basin_stats(labels, grid, keep)
    df = pd.DataFrame({"basin": lab.astype(np.int64),
                       "n_cells": cnt.astype(np.int64),
                       "area_km2": cnt * grid.cellsize ** 2 / 1e6})
    return gpd.GeoDataFrame(df, geometry=shapely.points(x, y), crs=crs)


def pair_side_points(pairs, labels, grid, *, crs=None, side: str = "a"):
    """One point per pair, at the centroid of that pair's side A or side B.

    Written as two layers so a GIS can style them differently and the
    reader can see at a glance which basin a signed column calls A.
    """
    import geopandas as gpd
    import shapely

    col = "side_a" if side.lower() == "a" else "side_b"
    want = np.unique(np.asarray(pairs[col], dtype=np.int64))
    lab, cnt, x, y = _basin_stats(labels, grid, keep=want)
    pos = {int(l): i for i, l in enumerate(lab)}
    keep_rows, px, py = [], [], []
    for i, l in enumerate(np.asarray(pairs[col], dtype=np.int64)):
        j = pos.get(int(l))
        if j is None:
            continue
        keep_rows.append(i)
        px.append(x[j])
        py.append(y[j])
    out = pairs.iloc[keep_rows].copy()
    out["side"] = side.upper()
    out["basin"] = np.asarray(pairs[col], dtype=np.int64)[keep_rows]
    return gpd.GeoDataFrame(out, geometry=shapely.points(px, py), crs=crs)


def basin_polygons(labels, grid, *, crs=None, keep=None, simplify=None):
    """The catchment of each reach basin, as polygons.

    One polygon per basin, dissolved from the label raster. On a
    continental domain there are millions of basins, so pass ``keep`` to
    restrict it to the pairs being looked at; polygonising everything is
    rarely what is wanted and always slow.
    """
    import geopandas as gpd
    import pandas as pd
    import shapely
    from rasterio.features import shapes
    from rasterio.transform import Affine

    L = np.asarray(labels)
    if L.ndim == 1:
        L = L.reshape(grid.nr, grid.nc)
    L = L.astype(np.int32, copy=True)
    if keep is not None:
        L = np.where(np.isin(L, np.asarray(keep, dtype=L.dtype)), L, 0)
    tr = Affine(grid.cellsize, 0.0, grid.x0, 0.0, -grid.cellsize, grid.y0)
    geoms, vals = [], []
    for geom, val in shapes(L, mask=(L > 0), transform=tr, connectivity=4):
        geoms.append(shapely.geometry.shape(geom))
        vals.append(int(val))
    if not geoms:
        return gpd.GeoDataFrame({"basin": []}, geometry=[], crs=crs)
    gdf = gpd.GeoDataFrame(pd.DataFrame({"basin": vals}), geometry=geoms, crs=crs)
    gdf = gdf.dissolve(by="basin", as_index=False)
    if simplify:
        gdf["geometry"] = gdf.geometry.simplify(float(simplify))
    gdf["area_km2"] = gdf.area / 1e6
    return gdf


def multiscale_pair_ids(table, scales=None):
    """Attach a pair identity per scale to a multiscale table.

    :func:`pydivide.acrossdivide.across_divide_multiscale` measures one
    fixed divide network at several channel thresholds, so each piece
    carries ``basin_a_<scale>`` and ``basin_b_<scale>`` for every rung of
    the ladder. This turns those into ``pair_id_<scale>`` and
    ``pair_uid_<scale>`` columns (and ``pair_key_<scale>``, the
    cross-partition key, where the scale is a power of ten), and returns
    one pair table per scale.

    The point is selection. At a coarse threshold the pair is the whole
    divide between two large basins, which is what you want when the
    question is about two river systems. At a fine one the same pieces
    split into many pairs, which is what you want when the question is
    about a single headwater. Both identities live on the same row, so a
    GIS can switch between them without recomputing anything.
    """
    cols = table.columns
    if scales is None:
        scales = sorted({k.split("basin_a_", 1)[1] for k in cols
                         if k.startswith("basin_a_")})
    tables = {}
    for sc in scales:
        ka, kb = f"basin_a_{sc}", f"basin_b_{sc}"
        if ka not in cols or kb not in cols:
            continue
        pid, _ = pair_ids(cols[ka], cols[kb])
        cols[f"pair_id_{sc}"] = pid
        cols[f"pair_uid_{sc}"] = np.where(
            pid >= 0, np.char.add(f"{sc}:", pid.astype(str)), "")
        t = _scale_threshold(sc)
        if t is not None:
            a = np.asarray(cols[ka], float)
            b = np.asarray(cols[kb], float)
            ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0) & (a != b) & \
                (np.maximum(a, b) < PAIR_KEY_LABEL_LIMIT)
            keys = np.full(len(a), NO_PAIR, np.int64)
            if ok.any():
                keys[ok] = pair_key(a[ok].astype(np.int64), b[ok].astype(np.int64), t)
            cols[f"pair_key_{sc}"] = keys
        tables[sc] = pair_table(cols[ka], cols[kb], scale=sc)
    return tables
