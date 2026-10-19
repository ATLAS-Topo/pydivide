# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Write divide metrics as an ESRI File Geodatabase.

A File Geodatabase is the format an ArcGIS workflow reads directly. GDAL 3.11's
``OpenFileGDB`` driver is read-write, so the same product can be built
here without ArcGIS Pro or arcpy.

This removes both reasons the Shapefile exports are awkward:

* a File Geodatabase table has no 2 GB limit, so all ~106 metric columns
  live in **one** feature class -- no themes, no row parts, no joining
  ``_bpd`` back to ``_core`` to recover ``distance``;
* field names are not truncated to 10 characters.

The GeoPackage stays as the interchange copy; this is the ArcGIS one.

    python -u scripts/export_gdb.py exports/myarea/divides_seg300.gpkg
    python -u scripts/export_gdb.py --all exports/myarea      # every .gpkg there
    python -u scripts/export_gdb.py --overwrite ...            # replace an existing .gdb

An existing geodatabase is never replaced without ``--overwrite``, and the
exit status is non-zero when any conversion fails.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import pyogrio


def write_gdb(gpkg: Path, chunk: int = 2_000_000,
              out_dir: Path | None = None, overwrite: bool = False) -> Path:
    gpkg = Path(gpkg)
    if not gpkg.is_file():
        raise FileNotFoundError(gpkg)
    n = pyogrio.read_info(gpkg)["features"]      # the input is readable before anything is touched
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        gdb = out_dir / (gpkg.stem + ".gdb")
    else:
        gdb = gpkg.with_suffix(".gdb")
    if gdb.exists():
        if not overwrite:
            raise FileExistsError(f"{gdb} exists; pass --overwrite to replace it")
        # a leftover partial from an interrupted run would otherwise be
        # appended to rather than replaced
        import shutil
        shutil.rmtree(gdb)
    layer = gpkg.stem
    print(f"[gdb] {gpkg.name}: {n:,} features -> {gdb.name}", flush=True)
    t0 = time.time()
    # written in chunks so a full-domain table never has to be resident;
    # the first chunk creates the layer, the rest append to it
    first = True
    for lo in range(0, n, chunk):
        g = gpd.read_file(gpkg, rows=slice(lo, min(lo + chunk, n)))
        # "order" is reserved in a File Geodatabase and GDAL silently
        # launders it to "order_". Renaming it deliberately is better
        # than shipping a mangled name nobody chose.
        if "order" in g.columns:
            g = g.rename(columns={"order": "divorder"})
        g.to_file(gdb, driver="OpenFileGDB", layer=layer,
                  mode="w" if first else "a",
                  # without this ID and segid are demoted to Float64
                  TARGET_ARCGIS_VERSION="ARCGIS_PRO_3_2_OR_LATER")
        first = False
        print(f"[gdb]   {min(lo + chunk, n):,}/{n:,} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        del g
    print(f"[gdb] {gdb.name} done in {(time.time()-t0)/60:.1f} min",
          flush=True)
    return gdb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--all", action="store_true",
                    help="treat target as a directory of *_metrics*.gpkg")
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--out-dir", default=None,
                    help="write the geodatabase here instead of next to "
                         "the GeoPackage")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing geodatabase of the same name")
    a = ap.parse_args()
    t = Path(a.target)
    files = sorted(t.glob("*.gpkg")) if a.all else [t]
    if not files:
        sys.exit(f"no GeoPackages found in {t}")
    od = Path(a.out_dir) if a.out_dir else None
    failed = 0
    for f in files:
        try:
            write_gdb(f, a.chunk, od, overwrite=a.overwrite)
        except Exception as e:
            failed += 1
            print(f"[gdb] {f.name} FAILED: {type(e).__name__}: {e}",
                  flush=True)
    if failed:
        sys.exit(f"{failed} of {len(files)} conversion(s) failed")


if __name__ == "__main__":
    main()
