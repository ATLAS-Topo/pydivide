# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Add the WKT2 CRS column to GeoPackages that lack it.

Every GeoPackage written here declares ``user_version`` 10400
(GeoPackage 1.4) but its ``gpkg_spatial_ref_sys`` table carries only the
WKT1 ``definition`` column. GeoPackage 1.2 added ``definition_12_063``
for WKT2, and strict readers fall back to it when the WKT1 string is not
a recognised EPSG code -- which is exactly the Brazil case, where the
Albers projection is custom and is stored as ``organization='NONE'``,
``srs_id=100000``.

The change is additive: a new column is created and populated from the
existing definition via pyproj. No geometry, attribute or index is
touched, and a GeoPackage that already has the column is left alone.

    python -u scripts/fix_gpkg_srs.py exports/myarea/divides_full.gpkg
    python -u scripts/fix_gpkg_srs.py --all exports
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def patch(path: Path, dry: bool = False) -> str:
    # sqlite3.connect(path) would create a missing file; a dry run opens
    # read-only and a real run opens an existing file only
    if not path.is_file():
        raise FileNotFoundError(f"{path} is not a file")
    con = sqlite3.connect(f"{path.resolve().as_uri()}?mode={'ro' if dry else 'rw'}", uri=True)
    cur = con.cursor()
    cols = [r[1] for r in cur.execute("PRAGMA table_info(gpkg_spatial_ref_sys)")]
    if "definition_12_063" in cols:
        con.close()
        return "already has definition_12_063"
    rows = cur.execute(
        "SELECT srs_id, definition FROM gpkg_spatial_ref_sys").fetchall()
    if dry:
        con.close()
        return f"would add WKT2 for {len(rows)} SRS entries"

    from pyproj import CRS
    cur.execute("ALTER TABLE gpkg_spatial_ref_sys "
                "ADD COLUMN definition_12_063 TEXT")
    n_ok = 0
    for srs_id, defn in rows:
        if not defn or defn == "undefined":
            wkt2 = "undefined"
        else:
            try:
                wkt2 = CRS.from_wkt(defn).to_wkt("WKT2_2019")
                n_ok += 1
            except Exception:
                wkt2 = "undefined"
        cur.execute("UPDATE gpkg_spatial_ref_sys SET definition_12_063 = ? "
                    "WHERE srs_id = ?", (wkt2, srs_id))
    # the WKT2 column is delivered by a registered extension
    cur.execute("""CREATE TABLE IF NOT EXISTS gpkg_extensions (
        table_name TEXT, column_name TEXT, extension_name TEXT NOT NULL,
        definition TEXT NOT NULL, scope TEXT NOT NULL,
        CONSTRAINT ge_tce UNIQUE (table_name, column_name, extension_name))""")
    cur.execute("""INSERT OR IGNORE INTO gpkg_extensions
        (table_name, column_name, extension_name, definition, scope)
        VALUES ('gpkg_spatial_ref_sys', 'definition_12_063', 'gpkg_crs_wkt',
                'http://www.geopackage.org/spec/#extension_crs_wkt',
                'read-write')""")
    con.commit()
    con.close()
    return f"added WKT2 for {n_ok}/{len(rows)} SRS entries"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--all", action="store_true",
                    help="treat target as a directory tree of .gpkg")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    t = Path(a.target)
    files = sorted(t.rglob("*.gpkg")) if a.all else [t]
    if not files:
        sys.exit(f"no GeoPackages under {t}")
    failed = 0
    for f in files:
        try:
            print(f"{f}: {patch(f, a.dry_run)}", flush=True)
        except Exception as e:
            print(f"{f}: FAILED {type(e).__name__}: {e}", flush=True)
            failed += 1
    if failed:
        sys.exit(f"{failed} of {len(files)} GeoPackages failed")


if __name__ == "__main__":
    main()
