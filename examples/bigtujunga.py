# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""The README example, runnable: the divide network of the Big Tujunga
catchment (the TopoToolbox example DEM), its distance and orders, the
exports, and a signed across-divide contrast of elevation with the two
sides identified by basin.

    python examples/bigtujunga.py path/to/srtm_bigtujunga30m_utm11.tif [outdir]

The DEM ships with TopoToolbox (MATLAB: DEMs/srtm_bigtujunga30m_utm11.tif;
pytopotoolbox: topotoolbox.load_dem("bigtujunga")). Runs in a few seconds
and writes divides.gpkg (and, if the drivers are present, divides.shp and
divides.gdb) into outdir (default: the current directory).
"""

import sys
from pathlib import Path

import numpy as np

import pydivide


def main(dem_path: str, outdir: str = ".") -> int:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. the divide network: D8 routing with carving, streams above 1e6 m2
    D = pydivide.divide_network(dem_path, minarea=1e6)
    print(f"{D.n_segments:,} segments, {len(D.edge_nodes):,} divide edges, "
          f"max divide distance {np.nanmax(D.distance) / 1000:.1f} km")

    # 2. hierarchical orders (computed on demand, cached on the result)
    for scheme in ("strahler", "shreve", "topo"):
        print(f"  max {scheme} order: {int(np.nanmax(D.order(scheme)))}")

    # 3. exports; every writer applies the GeoPackage CRS repair ArcGIS needs
    D.to_geopackage(out / "divides.gpkg", overwrite=True)
    try:
        D.to_shapefile(out / "divides.shp", overwrite=True)
        D.to_filegdb(out / "divides.gdb", overwrite=True)
    except Exception as e:  # a missing OpenFileGDB writer is not a failure of the example
        print(f"  (shapefile or file geodatabase not written: {e})")

    # 4. a signed across-divide contrast, here of elevation itself, with
    #    side A fixed per segment by reach-basin label (D.labels), so the
    #    sign of A - B means "the lower-labelled basin minus the other"
    import topotoolbox as tt
    z = np.asarray(tt.read_tif(dem_path).z, np.float32).ravel()
    labels = D.labels.ravel()
    p, q = pydivide.getvalue(D.network, D.grid, z, label_flat=labels)
    a = pydivide.asymmetry(D.network, D.grid, z, label_flat=labels)
    valid = np.isfinite(a["sgn"])
    print(f"  elevation contrast A - B per segment: median {np.nanmedian(a['sgn']):.1f} m, "
          f"{int(np.sum(a['sgn'][valid] > 0)):,} positive of {int(valid.sum()):,}")
    both = np.isfinite(p) & np.isfinite(q)
    print(f"  per-position values on both sides at {int(both.sum()):,} of {len(p):,} positions "
          f"(NaN at segment joints and where one side is outside every basin)")
    print(f"  side identities available: {sorted(k for k in a if k.startswith('side_'))}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "."))
