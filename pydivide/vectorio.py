# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Writing divide networks to the formats a GIS actually reads.

One place for every vector write in the package, because three things
go wrong when GIS writes are scattered:

* the GeoPackage CRS repair (:func:`repair_geopackage_crs`) has to
  follow every GeoPackage write, or ArcGIS refuses the layer;
* the File Geodatabase writer has to be importable by every caller, not
  buried in a script;
* a Shapefile has two 2 GB limits, the ``.shp`` and the ``.dbf``, and
  with a wide attribute table the ``.dbf`` is the one that binds first,
  so parts must be sized from both.

Format policy: **GeoPackage by default.** It is the
open OGC format, QGIS's native one, a single file at any size, and it
opens in ArcGIS once the WKT2 coordinate-reference definition below is
present. Shapefile and File Geodatabase are written on request: the
Shapefile with both of its 2 GB limits respected and split into parts
when necessary; the File Geodatabase for ArcGIS users, with neither a
size nor a field-name limit.

The ``.gdb`` trap, measured
--------------------------
``gdf.to_file("out.gdb")`` does **not** write a geodatabase and does not
raise. GeoPandas looks the suffix up in its extension→driver map, finds
no ``.gdb`` entry, and falls back to ``ESRI Shapefile`` — producing
``.shp``/``.dbf``/``.shx`` inside a directory that merely happens to be
named ``out.gdb``. ``driver="OpenFileGDB"`` is not optional here, and
:func:`write_filegdb` always passes it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# limits

#: A single ``.shp`` or ``.dbf`` cannot exceed 2 GB. Both are kept under a
#: margin rather than at the limit, because the true record size depends on
#: vertex counts and on how GDAL widens numeric fields.
FILE_BUDGET_BYTES = 1_800_000_000

#: The per-field cost used when no table is at hand (:func:`export_plan`):
#: the width of a float, the widest numeric field. A real table is sized
#: field by field with :func:`dbf_field_widths`.
BYTES_PER_FIELD = 24

#: DBF record overhead: one deletion-flag byte per record.
DBF_RECORD_OVERHEAD = 1

#: DBF field widths as GDAL's Shapefile driver writes them, measured on
#: the files it produces: a float is a 24-character numeric field
#: whatever its value; an integer field starts at 18 characters (64-bit)
#: or 9 (narrower) and is widened to its longest value, sign included
#: (20 for the most negative 64-bit integer, 11 for the most negative
#: 32-bit one); a boolean is a 1-character logical field, a date 8
#: characters, and a string the longer of the driver's default 80 and
#: the longest value, up to the format's 254.
DBF_WIDTH_FLOAT = 24
DBF_WIDTH_INT64 = 18
DBF_WIDTH_INT32 = 9
DBF_WIDTH_BOOL = 1
DBF_WIDTH_DATE = 8
DBF_WIDTH_STRING_DEFAULT = 80
DBF_WIDTH_STRING_MAX = 254

#: The companions a Shapefile is made of, all of which the guard against
#: overwriting an existing dataset considers occupied.
SHAPEFILE_COMPANIONS = (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx",
                        ".shp.xml")

#: DBF file header: 32 bytes, then 32 per field descriptor, then one
#: terminator byte; the file ends with one end-of-file byte.
DBF_HEADER_BYTES = 32
DBF_FIELD_DESCRIPTOR_BYTES = 32
DBF_TRAILER_BYTES = 2

#: Rough per-feature cost in the ``.shp`` geometry file for a short
#: polyline: 8-byte record header + 44-byte polyline header + 16 bytes per
#: vertex. Divide pieces are short, so a small vertex count dominates.
SHP_BYTES_PER_FEATURE = 8 + 44 + 16 * 4

#: Field names longer than this are truncated by the Shapefile driver.
#: GeoPackage and File Geodatabase have no such limit.
SHAPEFILE_NAME_LIMIT = 10

#: Names a File Geodatabase reserves. GDAL silently launders ``order`` to
#: ``order_``; renaming deliberately is better than shipping a mangled name
#: nobody chose.
GDB_RESERVED = {"order": "divorder", "select": "select_", "table": "table_",
                "user": "user_", "where": "where_", "group": "group_",
                "when": "when_", "add": "add_", "alter": "alter_",
                "drop": "drop_", "index": "index_", "view": "view_"}

#: Passed to the OpenFileGDB driver. Without it, int64 fields such as ``ID``
#: and ``segid`` are demoted to Float64 and lose exactness.
DEFAULT_ARCGIS_VERSION = "ARCGIS_PRO_3_2_OR_LATER"

#: Shapefile has no boolean type; these are written as int32.
_BOOL_COLUMNS = ("isbnd", "isring")


# ---------------------------------------------------------------------------
# size arithmetic


def dbf_bytes(n_features: int, n_fields: int) -> int:
    """Estimated ``.dbf`` size for ``n_features`` rows of ``n_fields``
    numeric fields, for planning before a table exists."""
    return (dbf_header_bytes(n_fields)
            + n_features * (DBF_RECORD_OVERHEAD + n_fields * BYTES_PER_FIELD))


def dbf_header_bytes(n_fields: int) -> int:
    """Size of the ``.dbf`` header plus the end-of-file byte."""
    return DBF_HEADER_BYTES + DBF_FIELD_DESCRIPTOR_BYTES * n_fields + DBF_TRAILER_BYTES


def dbf_field_widths(gdf) -> dict:
    """The DBF width of every attribute column of ``gdf``, from its dtype
    and, for a string column, from its longest value in UTF-8 bytes (the
    driver widens a string field to the longest value it writes). Boolean
    columns are sized as the driver writes them; cast them first if the
    writer does. Returns ``{column: width}`` in column order."""
    widths = {}
    geom = getattr(gdf, "geometry", None)
    geom_name = getattr(geom, "name", "geometry")
    for col in gdf.columns:
        if col == geom_name:
            continue
        s = gdf[col]
        kind = s.dtype.kind
        if kind == "f":
            w = DBF_WIDTH_FLOAT
        elif kind == "b":
            w = DBF_WIDTH_BOOL
        elif kind in "iu":
            # the driver widens an integer field to its longest value
            w = DBF_WIDTH_INT64 if s.dtype.itemsize > 4 else DBF_WIDTH_INT32
            if len(s):
                w = max(w, len(str(int(s.min()))), len(str(int(s.max()))))
        elif kind == "M":
            w = DBF_WIDTH_DATE
        else:
            vals = s.dropna()
            longest = int(vals.astype(str).str.encode("utf-8").str.len().max()) if len(vals) else 0
            w = min(DBF_WIDTH_STRING_MAX, max(DBF_WIDTH_STRING_DEFAULT, longest))
        widths[col] = int(w)
    return widths


def dbf_record_bytes(widths) -> int:
    """Bytes per ``.dbf`` record for the field widths of
    :func:`dbf_field_widths`."""
    return DBF_RECORD_OVERHEAD + int(sum(widths.values()))


def shp_bytes(n_features: int) -> int:
    """Estimated ``.shp`` geometry size for ``n_features`` short polylines."""
    return 100 + n_features * SHP_BYTES_PER_FEATURE


def shp_bytes_for_vertices(n_vertices) -> np.ndarray:
    """Exact ``.shp`` record size of a single-part polyline per feature:
    8-byte record header, 4-byte shape type, 32-byte box, 4 + 4 bytes of
    part and point counts, one 4-byte part index, 16 bytes per vertex."""
    n_vertices = np.asarray(n_vertices, dtype=np.int64)
    return 8 + 4 + 32 + 4 + 4 + 4 + 16 * n_vertices


def shp_record_bytes(geometry, *, force_z: bool = False) -> np.ndarray:
    """Exact ``.shp`` record size per feature for any geometry the format
    holds: a point is 28 bytes; a multipoint 48 plus 16 per point; a
    line, polygon or multipart of either is 52 bytes plus 4 per part
    (each line, each polygon ring) plus 16 per vertex. A geometry with
    a Z coordinate costs 8 more per vertex, plus 16 for the Z range of a
    line, polygon or multipoint, as the driver writes them (no measure
    block). A layer is written in one dimension, so when any geometry
    has a Z coordinate every record of the layer is a Z record; pass
    ``force_z=True`` to size a layer that way. Geometries with an M
    coordinate are not sized and are refused by :func:`write_shapefile`."""
    import shapely
    from shapely import GeometryType as G

    g = np.asarray(geometry, dtype=object)
    t = shapely.get_type_id(g)
    nv = shapely.get_num_coordinates(g).astype(np.int64)
    parts = np.ones(len(g), dtype=np.int64)
    poly = t == G.POLYGON
    if poly.any():
        parts[poly] = 1 + shapely.get_num_interior_rings(g[poly])
    multi = np.isin(t, (G.MULTILINESTRING, G.MULTIPOLYGON, G.GEOMETRYCOLLECTION))
    for i in np.flatnonzero(multi):
        n = 0
        for m in shapely.get_parts(g[i]):
            n += 1 + shapely.get_num_interior_rings(m) if m.geom_type == "Polygon" else 1
        parts[i] = n
    rec = 8 + 4 + 32 + 4 + 4 + 4 * parts + 16 * nv
    rec = np.where(t == G.POINT, 8 + 4 + 16, rec)
    rec = np.where(t == G.MULTIPOINT, 8 + 4 + 32 + 4 + 16 * nv, rec)
    z = shapely.has_z(g) | bool(force_z)
    if z.any():
        extra = np.where(t == G.POINT, 8, 16 + 8 * nv)
        rec = np.where(z, rec + extra, rec)
    return rec.astype(np.int64)


def has_measure(geometry) -> np.ndarray:
    """Whether each geometry carries a measure (M) coordinate. Shapely
    2.1 with GEOS 3.12 introduced measures and their detector; an older
    Shapely cannot represent a measure at all, so there every geometry
    is reported as having none, which is then true."""
    import shapely

    g = np.asarray(geometry, dtype=object)
    detector = getattr(shapely, "has_m", None)
    if detector is None:
        return np.zeros(len(g), dtype=bool)
    return np.asarray(detector(g), dtype=bool)


def shapefile_targets(path) -> list:
    """Every file a Shapefile written at ``path`` occupies: the
    companions of the dataset plus the field-name sidecar."""
    path = Path(path)
    base = path.with_suffix("")
    out = [Path(str(base) + ext) for ext in SHAPEFILE_COMPANIONS]
    out.append(path.with_name(path.stem + "_fields.csv"))
    return out


def _same_file(a, b) -> bool:
    """Whether two paths name the same existing file, as the file system
    itself reports it; False when either does not exist."""
    import os

    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def existing_parts(path) -> list:
    """The files of every numbered part written under ``path``'s name by
    an earlier export: ``<stem>_partNNN`` followed by a Shapefile
    companion suffix or by the part's field-name sidecar, which is this
    writer's own naming and nothing else. A numbered sibling of another
    kind (``<stem>_part001.tif``, ``.csv``, ``.json``) is not a part and
    is never touched.

    A file whose name differs from the export's own spelling only in
    case belongs to the set exactly when the file system reports it to
    be the same file as that spelling: on Windows a set written as
    ``Streams`` is found when ``streams`` is overwritten, and on a
    case-sensitive file system ``Streams_part001.shp`` is another
    dataset and survives. Regular-expression case folding is no
    substitute for that (a dotted capital I folds to ``i`` in Python and
    is a different file everywhere), and neither is a look-up of the
    folder under another spelling (``a`` and ``A`` can be two folders on
    Linux)."""
    import re

    path = Path(path)
    folder = path.parent
    if not folder.is_dir():
        return []
    tails = (*SHAPEFILE_COMPANIONS, "_fields.csv")
    # the writer's own naming, literally: ASCII digits (``\d`` also takes
    # other scripts' digits, which the writer never emits) and the whole
    # name (``$`` alone also stops before a final newline)
    body = rf"{re.escape(path.stem)}_part([0-9]{{3,}})({'|'.join(re.escape(t) for t in tails)})"
    exact = re.compile(body)
    loose = re.compile(body, re.IGNORECASE)
    out = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if exact.fullmatch(p.name):
            out.append(p)
            continue
        m = loose.fullmatch(p.name)
        if m is None:
            continue
        # the export's own spelling of this part: its stem and the
        # writer's own suffix, found with the same folding that matched
        # it (``lower()`` folds differently and could find none); the
        # candidate is a member only if the file system says it is that
        # very file
        tail = next((t for t in tails
                     if re.fullmatch(re.escape(t), m.group(2), re.IGNORECASE)), None)
        if tail is not None and _same_file(p, folder / f"{path.stem}_part{m.group(1)}{tail}"):
            out.append(p)
    return sorted(out)


def _guard_shapefile(path, overwrite: bool) -> None:
    """Refuse when any file of the dataset exists, unless ``overwrite``;
    with it, remove the whole set so no stale companion survives."""
    for p in shapefile_targets(path):
        if p.exists():
            if not overwrite:
                raise FileExistsError(f"{p} exists; pass overwrite=True to replace the dataset")
            p.unlink()


def _guard_export_set(path, targets, overwrite: bool) -> None:
    """One policy for the whole export at a name: every companion of
    every planned part, the companions of the base name, the field-name
    sidecar and the numbered parts of any earlier export at this name
    count as occupied. Without ``overwrite`` the first occupied file is
    named and nothing is written; with it the whole previous set is
    removed first, so a shorter export cannot leave earlier parts beside
    it that read as current data."""
    path = Path(path)
    occupied = []
    seen = set()
    for t in [path, *targets]:
        for p in shapefile_targets(t):
            if p.exists() and p not in seen:
                occupied.append(p)
                seen.add(p)
    for p in existing_parts(path):
        if p not in seen:
            occupied.append(p)
            seen.add(p)
    if occupied and not overwrite:
        raise FileExistsError(f"{occupied[0]} exists; pass overwrite=True to replace the dataset")
    for p in occupied:
        # the same file can be listed twice under two spellings of its
        # case on a case-insensitive file system
        p.unlink(missing_ok=True)


def shapefile_capacity(n_fields: int, row_bytes: int | None = None) -> int:
    """How many features fit in one Shapefile part with ``n_fields``.

    Returns the smaller of the two independent limits. With 4 fields the
    geometry binds; from roughly 10 fields upward the attribute table
    binds, and with the ~106 columns of a full metric table the capacity
    collapses to well under a million features, which is why splitting
    on feature count alone is not enough. ``row_bytes`` is the exact
    record width of :func:`dbf_record_bytes` when the table is known;
    without it every field is costed as a float.
    """
    if n_fields < 0:
        raise ValueError("n_fields must be >= 0")
    row = row_bytes if row_bytes is not None else (
        DBF_RECORD_OVERHEAD + n_fields * BYTES_PER_FIELD)
    by_dbf = (FILE_BUDGET_BYTES - dbf_header_bytes(n_fields)) // max(1, int(row))
    by_shp = (FILE_BUDGET_BYTES - 100) // SHP_BYTES_PER_FEATURE
    return int(max(1, min(by_dbf, by_shp)))


def shapefile_parts(n_features: int, n_fields: int) -> int:
    """Number of Shapefile parts ``n_features`` would need."""
    cap = shapefile_capacity(n_fields)
    return int(-(-n_features // cap)) if n_features else 0


# ---------------------------------------------------------------------------
# GeoPackage


def repair_geopackage_crs(path) -> bool:
    """Give a GeoPackage's CRS a WKT2 definition. Returns True if written.

    A custom projection — the South America Albers used throughout this
    project, for instance — is written by GDAL with ``organization='NONE'``
    and a WKT1 ``definition`` only. ArcGIS then refuses the layer with
    "The coordinate system description is invalid". GeoPackage 1.2 added
    the ``definition_12_063`` column for WKT2 and strict readers fall back
    to it, so it is populated here.

    Purely additive: no geometry, attribute or index is touched. Safe to
    call twice. Returns False if pyproj is unavailable or the file has no
    spatial-reference table, rather than raising — a missing WKT2 makes a
    file awkward in ArcGIS, not invalid.
    """
    import sqlite3

    try:
        from pyproj import CRS
    except Exception:
        return False

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    con = sqlite3.connect(str(path))
    try:
        cur = con.cursor()
        tables = {r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "gpkg_spatial_ref_sys" not in tables:
            return False
        cols = [r[1] for r in
                cur.execute("PRAGMA table_info(gpkg_spatial_ref_sys)")]
        if "definition_12_063" not in cols:
            cur.execute("ALTER TABLE gpkg_spatial_ref_sys "
                        "ADD COLUMN definition_12_063 TEXT")
        for srs_id, defn, have in cur.execute(
                "SELECT srs_id, definition, definition_12_063 FROM gpkg_spatial_ref_sys"
        ).fetchall():
            if have and have not in ("", "undefined"):
                continue                      # a WKT2 definition is already there; keep it
            wkt2 = "undefined"
            if defn and defn != "undefined":
                try:
                    wkt2 = CRS.from_wkt(defn).to_wkt("WKT2_2019")
                except Exception as exc:
                    import warnings
                    warnings.warn(
                        f"{path.name}: the CRS of srs_id {srs_id} could not be "
                        f"parsed ({exc}); its WKT2 definition is left undefined "
                        "and ArcGIS may refuse the layer", RuntimeWarning, stacklevel=2)
                    wkt2 = "undefined"
            cur.execute("UPDATE gpkg_spatial_ref_sys "
                        "SET definition_12_063 = ? WHERE srs_id = ?",
                        (wkt2, srs_id))
        cur.execute("""CREATE TABLE IF NOT EXISTS gpkg_extensions (
            table_name TEXT, column_name TEXT, extension_name TEXT NOT NULL,
            definition TEXT NOT NULL, scope TEXT NOT NULL,
            CONSTRAINT ge_tce UNIQUE (table_name, column_name,
                                      extension_name))""")
        cur.execute("""INSERT OR IGNORE INTO gpkg_extensions
            (table_name, column_name, extension_name, definition, scope)
            VALUES ('gpkg_spatial_ref_sys', 'definition_12_063',
                    'gpkg_crs_wkt',
                    'http://www.geopackage.org/spec/#extension_crs_wkt',
                    'read-write')""")
        con.commit()
    finally:
        con.close()
    return True


_GPKG_METADATA_EXTENSION = "http://www.geopackage.org/spec120/#extension_metadata"


def write_geopackage_metadata(path, layer: str, text: str, *, uri: str,
                              mime_type: str = "application/json") -> None:
    """Store a metadata document inside a GeoPackage, attached to
    ``layer``, through the format's own metadata extension
    (``gpkg_metadata`` and ``gpkg_metadata_reference``), so a single file
    carries the record of how its numbers were made. ``uri`` names the
    kind of document and is what :func:`read_geopackage_metadata` selects
    on. Additive: no geometry, attribute or index is touched."""
    import sqlite3

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    con = sqlite3.connect(str(path))
    try:
        cur = con.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS gpkg_metadata (
            id INTEGER CONSTRAINT m_pk PRIMARY KEY ASC NOT NULL,
            md_scope TEXT NOT NULL DEFAULT 'dataset',
            md_standard_uri TEXT NOT NULL,
            mime_type TEXT NOT NULL DEFAULT 'text/xml',
            metadata TEXT NOT NULL DEFAULT '')""")
        cur.execute("""CREATE TABLE IF NOT EXISTS gpkg_metadata_reference (
            reference_scope TEXT NOT NULL,
            table_name TEXT,
            column_name TEXT,
            row_id_value INTEGER,
            timestamp DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            md_file_id INTEGER NOT NULL,
            md_parent_id INTEGER,
            CONSTRAINT crmr_mfi_fk FOREIGN KEY (md_file_id) REFERENCES gpkg_metadata(id),
            CONSTRAINT crmr_mpi_fk FOREIGN KEY (md_parent_id) REFERENCES gpkg_metadata(id))""")
        cur.execute("""CREATE TABLE IF NOT EXISTS gpkg_extensions (
            table_name TEXT, column_name TEXT, extension_name TEXT NOT NULL,
            definition TEXT NOT NULL, scope TEXT NOT NULL,
            CONSTRAINT ge_tce UNIQUE (table_name, column_name, extension_name))""")
        for table in ("gpkg_metadata", "gpkg_metadata_reference"):
            have = cur.execute("SELECT 1 FROM gpkg_extensions WHERE table_name = ? AND "
                               "column_name IS NULL AND extension_name = 'gpkg_metadata'",
                               (table,)).fetchone()
            if not have:
                cur.execute("INSERT INTO gpkg_extensions (table_name, column_name, "
                            "extension_name, definition, scope) VALUES (?, NULL, "
                            "'gpkg_metadata', ?, 'read-write')",
                            (table, _GPKG_METADATA_EXTENSION))
        cur.execute("INSERT INTO gpkg_metadata (md_scope, md_standard_uri, mime_type, "
                    "metadata) VALUES ('dataset', ?, ?, ?)", (uri, mime_type, text))
        md_id = cur.lastrowid
        cur.execute("INSERT INTO gpkg_metadata_reference (reference_scope, table_name, "
                    "md_file_id) VALUES ('table', ?, ?)", (layer, md_id))
        con.commit()
    finally:
        con.close()


def read_geopackage_metadata(path, *, uri: str | None = None,
                             layer: str | None = None) -> list:
    """The metadata documents stored in a GeoPackage by
    :func:`write_geopackage_metadata` (or by any writer of the metadata
    extension), oldest first, optionally restricted to one ``uri`` and
    one ``layer``. Returns ``[]`` when the file has no metadata table."""
    import sqlite3

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    con = sqlite3.connect(str(path))
    try:
        cur = con.cursor()
        tables = {r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "gpkg_metadata" not in tables:
            return []
        sql = "SELECT m.metadata FROM gpkg_metadata m"
        where, args = [], []
        if layer is not None and "gpkg_metadata_reference" in tables:
            sql += " JOIN gpkg_metadata_reference r ON r.md_file_id = m.id"
            where.append("r.table_name = ?")
            args.append(layer)
        if uri is not None:
            where.append("m.md_standard_uri = ?")
            args.append(uri)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY m.id"
        return [r[0] for r in cur.execute(sql, args).fetchall()]
    finally:
        con.close()


def _chunks(data):
    """Accept a GeoDataFrame or an iterable of them; yield GeoDataFrames."""
    if hasattr(data, "geometry") and hasattr(data, "columns"):
        yield data
    else:
        yield from data


def write_geopackage(data, path, *, layer: str = "divides",
                     repair_crs: bool = True, overwrite: bool = False,
                     geometry_type: str | None = None) -> Path:
    """Write a GeoPackage and repair its CRS definition.

    ``data`` may be a single GeoDataFrame or an iterable of them, so a
    network too large to hold in memory can be streamed chunk by chunk.
    An existing file is replaced only with ``overwrite=True`` (a
    GeoPackage may hold layers of other origin); it is never appended
    to, because a leftover partial from an interrupted run would
    silently gain duplicate rows. ``geometry_type`` (``"LineString"``,
    ``"Point"``, ...) declares the layer's geometry type, which matters
    when the frame is empty and the driver would otherwise record
    ``Unknown``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
        path.unlink()

    wrote = False
    for gdf in _chunks(data):
        if len(gdf) == 0 and wrote:
            continue
        gdf.to_file(str(path), driver="GPKG", layer=layer,
                    mode="w" if not wrote else "a",
                    **({"geometry_type": geometry_type} if geometry_type else {}))
        wrote = True
    if not wrote:
        raise ValueError("nothing to write: no features")
    if repair_crs:
        repair_geopackage_crs(path)
    return path


# ---------------------------------------------------------------------------
# Shapefile


def _shapefile_safe(gdf, name_map: dict | None = None):
    """Cast booleans to int32 and apply a field-name mapping."""
    out = gdf
    for col in _BOOL_COLUMNS:
        if col in out.columns:
            if out is gdf:
                out = out.copy()
            out[col] = out[col].astype("int32")
    if name_map:
        out = out.rename(columns=name_map)
    return out


def truncate_field_names(columns) -> dict:
    """Map over-long column names to unique <=10-character names.

    GeoPandas leaves truncation to GDAL, which silently collides two long
    names onto the same short one and drops a column. Truncating here,
    deterministically and with a collision counter, means the loss is
    visible and recorded rather than silent.

    Returns only the names that actually change, so an already-short
    table with distinct names maps to ``{}`` and is written untouched.
    Names are compared without regard to case, since DBF readers do not
    distinguish it: every name that is already short is reserved before
    any long name is shortened, a short name that repeats an earlier
    one in another case is given a suffix (GDAL would otherwise rename
    it on its own, and the mapping here records what was written), and
    a shortened name can collide neither with a short name that comes
    later in the table nor with another shortened one.
    """
    columns = list(columns)
    used = set()
    mapping = {}
    for c in columns:
        if c == "geometry":
            continue
        if len(c) <= SHAPEFILE_NAME_LIMIT and c.lower() not in used:
            used.add(c.lower())
    seen = set()
    for c in columns:
        if c == "geometry":
            continue
        if len(c) <= SHAPEFILE_NAME_LIMIT and c.lower() not in seen:
            seen.add(c.lower())
            continue
        base = c[:SHAPEFILE_NAME_LIMIT]
        cand, k = base, 1
        while cand.lower() in used or cand.lower() in seen:
            suffix = f"_{k}" if len(c) <= SHAPEFILE_NAME_LIMIT else str(k)
            cand = base[:SHAPEFILE_NAME_LIMIT - len(suffix)] + suffix
            k += 1
        used.add(cand.lower())
        seen.add(cand.lower())
        mapping[c] = cand
    return mapping


def write_shapefile(gdf, path, *, max_features: int | None = None,
                    write_name_map: bool = True,
                    overwrite: bool = False) -> list[Path]:
    """Write ESRI Shapefile(s), respecting **both** 2 GB limits.

    Parts are cut where the running geometry size, computed from the
    real type, dimension, part count and vertex count of every feature,
    or the running attribute size, computed from the DBF width of every
    field (:func:`dbf_field_widths`: a string field is as wide as its
    longest value, an integer field as wide as its longest number),
    would exceed the budget, and never hold more than ``max_features``
    (default: the capacity for that row width). A layer holding any Z
    geometry is written, and sized, as a Z layer throughout. Parts are
    named ``name_part001.shp``. A dataset is a set of files, so the
    write is refused when any companion of any planned part, the
    field-name sidecar, or a numbered part of an earlier export at this
    name exists, unless ``overwrite=True``, which removes that whole
    previous set first. A single feature wider than the budget, and a
    geometry with a measure (M) coordinate, are refused. The driver's
    own 2 GB limit is switched on as a backstop.

    Returns the list of files written. If any field name had to be
    changed, a ``<name>_fields.csv`` sidecar records the full mapping,
    because a 10-character name is not self-explanatory.
    """
    import shapely

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(gdf)
    if n == 0:
        raise ValueError("nothing to write: no features")
    if has_measure(gdf.geometry.values).any():
        raise ValueError("geometries with a measure (M) coordinate are not sized here; drop "
                         "the measure before writing a Shapefile")

    name_map = truncate_field_names(gdf.columns)
    safe = _shapefile_safe(gdf, name_map)            # the dtypes the driver sees
    widths = dbf_field_widths(safe)
    n_fields = len(widths)
    dbf_row = dbf_record_bytes(widths)
    dbf_fixed = dbf_header_bytes(n_fields)
    # the driver writes the layer in one dimension: one Z geometry makes
    # every record a Z record, so the layer is sized that way
    layer_z = bool(shapely.has_z(gdf.geometry.values).any())
    shp_row = shp_record_bytes(gdf.geometry.values, force_z=layer_z)
    if dbf_fixed + dbf_row > FILE_BUDGET_BYTES or 100 + int(shp_row.max()) > FILE_BUDGET_BYTES:
        raise ValueError(
            f"a single feature ({dbf_row} attribute bytes, {int(shp_row.max())} geometry "
            f"bytes) exceeds the Shapefile budget of {FILE_BUDGET_BYTES} bytes; write a "
            "GeoPackage or File Geodatabase instead")
    cap = max_features or shapefile_capacity(n_fields, row_bytes=dbf_row)
    bounds = [0]
    run_shp = run_dbf = 0
    for i in range(n):
        if (i - bounds[-1] >= cap or run_shp + int(shp_row[i]) > FILE_BUDGET_BYTES - 100
                or run_dbf + dbf_row > FILE_BUDGET_BYTES - dbf_fixed):
            bounds.append(i)
            run_shp = run_dbf = 0
        run_shp += int(shp_row[i])
        run_dbf += dbf_row
    bounds.append(n)

    # refuse before anything is written: every companion of every planned
    # part, the field-name sidecar and the parts of any earlier export at
    # this name count as occupied; with overwrite the whole previous set
    # goes, so neither a stale companion nor a stale part survives
    parts = len(bounds) - 1
    targets = ([path] if parts == 1 else
               [path.with_name(f"{path.stem}_part{k + 1:03d}{path.suffix}")
                for k in range(parts)])
    _guard_export_set(path, targets, overwrite)

    if name_map and write_name_map:
        import csv
        side = path.with_name(path.stem + "_fields.csv")
        with side.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["original", "shapefile"])
            w.writerows(sorted(name_map.items()))
        print(f"[pydivide] {len(name_map)} field name(s) changed for the Shapefile; "
              f"mapping written to {side.name}", flush=True)

    written = []
    if parts == 1:
        safe.to_file(str(path), driver="ESRI Shapefile", layer_options={"2GB_LIMIT": "YES"})
        return [path]
    print(f"[pydivide] {n:,} features x {n_fields} fields exceed a single "
          f"Shapefile; writing {parts} parts. A GeoPackage or File "
          "Geodatabase would keep this in one file.", flush=True)
    for k in range(parts):
        lo, hi = bounds[k], bounds[k + 1]
        target = targets[k]
        safe.iloc[lo:hi].to_file(str(target), driver="ESRI Shapefile",
                                 layer_options={"2GB_LIMIT": "YES"})
        written.append(target)
    return written


# ---------------------------------------------------------------------------
# File Geodatabase


def write_filegdb(data, path, *, layer: str | None = None,
                  arcgis_version: str = DEFAULT_ARCGIS_VERSION,
                  rename_reserved: bool = True, overwrite: bool = False) -> Path:
    """Write an ESRI File Geodatabase.

    ``data`` may be a GeoDataFrame or an iterable of them. A geodatabase
    is a **directory**, not a file; an existing one is removed first so an
    interrupted run cannot leave rows behind.

    Three things that are not optional and are easy to get wrong:

    * ``driver="OpenFileGDB"`` — without it GeoPandas silently writes a
      Shapefile into a directory named ``.gdb`` (see the module docstring);
    * ``TARGET_ARCGIS_VERSION`` — without it ``ID`` and ``segid`` are
      demoted from int64 to Float64;
    * reserved names — ``order`` is reserved and GDAL launders it to
      ``order_``; it is renamed to ``divorder`` deliberately instead.
    """
    path = Path(path)
    if path.suffix.lower() != ".gdb":
        path = path.with_suffix(".gdb")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
        shutil.rmtree(path)
    layer = layer or path.stem

    wrote = False
    for gdf in _chunks(data):
        if len(gdf) == 0 and wrote:
            continue
        if rename_reserved:
            ren = {c: GDB_RESERVED[c] for c in gdf.columns
                   if c in GDB_RESERVED}
            if ren:
                gdf = gdf.rename(columns=ren)
        gdf.to_file(path, driver="OpenFileGDB", layer=layer,
                    mode="w" if not wrote else "a",
                    layer_options={"TARGET_ARCGIS_VERSION": arcgis_version})
        wrote = True
    if not wrote:
        raise ValueError("nothing to write: no features")
    return path


# ---------------------------------------------------------------------------
# the policy


def choose_format(n_features: int, n_fields: int, *,
                  max_parts: int = 10) -> str:
    """The default format is GeoPackage; this reports whether a
    Shapefile would also have been viable.

    GeoPackage is the open OGC format, QGIS's native one, holds any
    number of features and fields in a single file, and — with the WKT2
    definition :func:`repair_geopackage_crs` adds — opens in ArcGIS as
    well. It is therefore the default regardless of size; most users of
    an open-source tool work in QGIS. Shapefile and File Geodatabase
    remain available on request.

    Returns ``"gpkg"`` always; call :func:`export_plan` to learn whether
    a Shapefile would fit in one file or in ``max_parts`` parts.
    """
    return "gpkg"


def export_plan(n_features: int, n_fields: int, *,
                theme_fields: int = 16, max_parts: int = 10) -> dict:
    """Can this table be a Shapefile? Report before writing, not after.

    ``DIVIDEobj``'s only export is ``DIVIDEobj2mapstruct``, which builds
    an in-memory MATLAB geographic data structure; writing it to disk
    needs ``shapewrite`` from the Mapping Toolbox, and neither step
    knows anything about the Shapefile size limits. At the scales
    ``pydivide`` exists to reach that is not a viable path, so the
    limits are computed up front here and a File Geodatabase is always
    offered as the answer.

    ``theme_fields`` is the typical field count of one themed Shapefile
    (the metric table split into families sharing a geometry), for which
    capacity is far higher than for one flat table.

    Returns a dict with, for each candidate format, whether it fits and
    in how many parts, plus a plain-language ``recommendation``.
    """
    flat_cap = shapefile_capacity(n_fields)
    flat_parts = shapefile_parts(n_features, n_fields)
    theme_cap = shapefile_capacity(theme_fields)
    theme_parts = shapefile_parts(n_features, theme_fields)

    if flat_parts <= 1:
        rec = "shapefile (single file)"
    elif theme_parts <= 1:
        rec = ("themed shapefiles (one part each) or a File Geodatabase; "
               "a single flat Shapefile would need "
               f"{flat_parts} parts")
    elif flat_parts <= max_parts:
        rec = (f"File Geodatabase, or a Shapefile in {flat_parts} parts "
               f"({theme_parts} parts per theme)")
    else:
        rec = ("File Geodatabase — a Shapefile would need "
               f"{flat_parts} parts, which is not a usable product")

    return {
        "n_features": int(n_features),
        "n_fields": int(n_fields),
        "shapefile_fits_single_file": bool(flat_parts <= 1),
        "shapefile_capacity": int(flat_cap),
        "shapefile_parts": int(flat_parts),
        "estimated_dbf_bytes": int(dbf_bytes(n_features, n_fields)),
        "estimated_shp_bytes": int(shp_bytes(n_features)),
        "themed_shapefile_capacity": int(theme_cap),
        "themed_shapefile_parts": int(theme_parts),
        "geodatabase_fits": True,   # no practical size or name limit
        "recommendation": rec,
    }


def describe_export(n_features: int, n_fields: int, **kw) -> str:
    """One-paragraph human summary of :func:`export_plan`."""
    p = export_plan(n_features, n_fields, **kw)
    return (
        f"{p['n_features']:,} features x {p['n_fields']} fields.\n"
        f"  flat Shapefile : {p['shapefile_parts']} part(s), "
        f"capacity {p['shapefile_capacity']:,}/part "
        f"(.dbf ~{p['estimated_dbf_bytes'] / 1e9:.2f} GB, "
        f".shp ~{p['estimated_shp_bytes'] / 1e9:.2f} GB)\n"
        f"  themed Shapefiles: {p['themed_shapefile_parts']} part(s) "
        f"per theme, capacity {p['themed_shapefile_capacity']:,}/part\n"
        f"  File Geodatabase : one layer, no size or field-name limit\n"
        f"  -> {p['recommendation']}")


def write_vector(gdf, path, *, format: str = "auto", layer: str = "divides",
                 max_parts: int = 10, **kw):
    """Write ``gdf`` in ``format``: ``shp``, ``gpkg``, ``gdb`` or ``auto``.

    ``auto`` applies :func:`choose_format`. ``path`` may carry any
    extension; the correct one is substituted for the format chosen, so
    callers do not have to guess which they will get. Returns the list of
    paths written.
    """
    path = Path(path)
    fmt = format.lower()
    if fmt == "auto":
        fmt = choose_format(len(gdf), max(0, len(gdf.columns) - 1),
                            max_parts=max_parts)
    if fmt in ("shp", "shapefile"):
        return write_shapefile(gdf, path.with_suffix(".shp"), **kw)
    if fmt in ("gpkg", "geopackage"):
        return [write_geopackage(gdf, path.with_suffix(".gpkg"),
                                 layer=layer, **kw)]
    if fmt in ("gdb", "filegdb", "openfilegdb"):
        return [write_filegdb(gdf, path.with_suffix(".gdb"),
                              layer=layer, **kw)]
    raise ValueError(f"unknown format {format!r}: "
                     "expected 'shp', 'gpkg', 'gdb' or 'auto'")
