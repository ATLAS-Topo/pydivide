# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Run the complete pydivide framework on one study area.

Produces, under exports/<name>/:
  - divides_seg150.gpkg       divide network split into ~150 m pieces
  - divides_seg300.gpkg       divide network split into ~300 m pieces
    (Shapefile copies only with --shp; off since 22 September 2026)
  - divides_full.gpkg          un-split divide segments (reference)
  - streams_<chifield>_seg300.shp/.gpkg   per chi baselevel: the
    baselevel-trimmed stream network in ~300 m pieces with mean chi,
    mean elevation, mean upstream area (km^2)

Everything checkpoints under checkpoints/<name>; rerunning resumes.

Example (from the repository root, dependencies installed):

    python scripts/run_area.py --dem path/to/dem.tif --name myarea \
        --chi-baselevels 5 1

Options are listed by --help and explained in docs/user_guide.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pydivide  # noqa: E402
from pydivide.streams import export_stream_segments  # noqa: E402


def log(msg):
    rss = psutil.Process().memory_info().rss / 1e9
    print(f"[{time.strftime('%H:%M:%S')}] [rss {rss:5.1f} GB] {msg}",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--checkpoint-name", default=None,
                    help="checkpoint folder name (default: --name). Use "
                         "to reuse checkpoints written under another "
                         "name for the same DEM and --minarea.")
    ap.add_argument("--minarea", type=float, default=1e6,
                    help="stream threshold for divides AND streams, m^2")
    ap.add_argument("--divide-baselevel", type=float, default=None,
                    help="metres: trim the stream network to cells above "
                         "this elevation BEFORE building the divides. "
                         "Without it the network runs down into the "
                         "coastal flats -- on the Brazil DEM 0.59%% of "
                         "cells sit at or below 3 m and the minimum is "
                         "0.0 m -- so the divide network starts at sea "
                         "level, breaks near the coast and its distances "
                         "are measured from the wrong seeds.")
    ap.add_argument("--chi-baselevels", nargs="+", type=float,
                    required=True, metavar="METRES")
    ap.add_argument("--divide-seglengths", nargs="+", type=float,
                    default=[150.0, 300.0],
                    help="seglengths (m) exported over the whole domain")
    ap.add_argument("--shp", action="store_true",
                    help="also write Shapefile copies of the divide exports "
                         "(off by default: the GeoPackage is the product)")
    ap.add_argument("--mask", default=None,
                    help="polygon file; additionally writes exports "
                         "clipped to it (suffix _masked), at every "
                         "--mask-seglengths value")
    ap.add_argument("--mask-seglengths", nargs="+", type=float,
                    default=None,
                    help="seglengths (m) for the masked exports "
                         "(default: same as --divide-seglengths)")
    ap.add_argument("--stream-seglength", type=float, default=300.0)
    ap.add_argument("--stream-baselevel", type=float, default=None,
                    help="metres: export the river geometry down to this "
                         "elevation while chi is still integrated from "
                         "--chi-baselevels. For example: rivers to 1 m, "
                         "chi from the strict divide baselevel.")
    ap.add_argument("--mn", type=float, default=0.45)
    ap.add_argument("--root", default=None,
                    help="folder that receives exports/<name> and checkpoints/<name> "
                         "(default: the repository holding this script)")
    args = ap.parse_args()

    root = Path(args.root) if args.root else ROOT
    out = root / "exports" / args.name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    import topotoolbox as tt
    log(f"reading {args.dem}")
    dem = tt.read_tif(args.dem)
    log("flow routing (d8, carve)")
    fd = tt.FlowObject(dem)
    log(f"stream network (threshold {args.minarea:g} m2)")
    st = tt.StreamObject(fd, threshold=args.minarea, units="m2")

    # ---- streams first: chi needs flow accumulation + the DEM ----------
    # accumulation computed once and shared across baselevels (at
    # full-domain size it is a ~49 GB array and a long pass)
    log("flow accumulation")
    acc = fd.flow_accumulation()
    sl_s = args.stream_seglength
    for b in args.chi_baselevels:
        chifield = f"chi{b:g}m"
        log(f"stream export, baselevel {b:g} m -> {chifield}"
            + (" (+ masked, from the same build)" if args.mask else ""))
        export_stream_segments(
            fd, st, dem, out / f"streams_{chifield}_seg{sl_s:g}",
            seglength=sl_s, chi_field=chifield,
            baselevel=(b if args.stream_baselevel is None
                       else args.stream_baselevel),
            mn=args.mn, acc=acc, mask=args.mask,
            # Shapefile copies of the streams follow the same switch as
            # the divides (the writer's own default would add them)
            formats=("gpkg", "shp") if args.shp else ("gpkg",),
            chi_baselevel=b,   # chi always integrates from --chi-baselevels
            masked_out_base=(out / f"streams_{chifield}_seg{sl_s:g}_masked"
                             if args.mask else None),
            overwrite=True)
    del acc
    import gc
    gc.collect()

    # ---- divides -------------------------------------------------------
    st_div = st
    if args.divide_baselevel is not None:
        b = float(args.divide_baselevel)
        log(f"trimming the stream network to elevations above {b:g} m "
            "for the divides")
        st_div = st.upstreamto(np.asarray(dem.z) > b).clean()
        log(f"  divide stream network trimmed "
            f"(baselevel {b:g} m); chi for the divides uses the same "
            "baselevel")
    log("divide network")
    # release_inputs: the stream exports above are the last users of the
    # DEM/FlowObject rasters, so the divide stages may free them. This is
    # required at full-domain size (~180 GB of rasters).
    D = pydivide.divide_network(
        fd=fd, st=st_div, minarea=args.minarea,
        checkpoint_dir=root / "checkpoints" /
        (args.checkpoint_name or args.name),
        release_inputs=True)
    log(f"  {D.n_segments} segments, {len(D.endpoints)} endpoints, "
        f"{len(D.junctions)} junctions, "
        f"{int(D.network.is_ring.sum())} rings; max distance "
        f"{float(np.nanmax(D.distance))/1000:.1f} km")

    D.to_geopackage(out / "divides_full.gpkg", overwrite=True)
    for sl in args.divide_seglengths:
        n = D._n_pieces(sl)
        log(f"divide export, seglength {sl:g} m ({n:,} pieces)")
        if args.shp:
            D.to_shapefile(out / f"divides_seg{sl:g}.shp", seglength=sl, overwrite=True)
        D.to_geopackage(out / f"divides_seg{sl:g}.gpkg", seglength=sl, overwrite=True)

    if args.mask:
        for sl in (args.mask_seglengths or args.divide_seglengths):
            sel = D.piece_selection(sl, args.mask)
            log(f"masked divide export, seglength {sl:g} m "
                f"({len(sel):,} of {D._n_pieces(sl):,} pieces inside "
                f"{Path(args.mask).name})")
            if args.shp:
                D.to_shapefile(out / f"divides_seg{sl:g}_masked.shp",
                               seglength=sl, mask=args.mask, overwrite=True)
            D.to_geopackage(out / f"divides_seg{sl:g}_masked.gpkg",
                            seglength=sl, mask=args.mask, overwrite=True)

    log(f"DONE {args.name} in {(time.time()-t0)/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
