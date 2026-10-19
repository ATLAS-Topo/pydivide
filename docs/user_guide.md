# pydivide user guide

Every function a user calls, with its options, in the order a typical
analysis uses them. Plain language throughout. Commands are run from
the repository root after `pip install -e ".[test]"`, or from anywhere
after `pip install git+https://github.com/ATLAS-Topo/pydivide`.

Contents

1. The divide network: `divide_network`
2. The result object
3. Exports: GeoPackage, Shapefile, File Geodatabase
4. Reading a grid across the divides: `getvalue` and `asymmetry`
5. Every standard across-divide metric in one call
6. Basin pairs, profiles and plots
7. The other `DIVIDEobj` operations
8. Stream exports
9. The one-command script: `scripts/run_area.py`
10. Checkpointing, memory and time
11. Conventions worth knowing
12. Low-memory grid helpers

---

## 1. The divide network: `divide_network`

```python
import pydivide

# from a DEM file: pydivide does the flow routing itself
D = pydivide.divide_network("dem.tif", minarea=1e6)

# or from pytopotoolbox objects you already built
import topotoolbox as tt
dem = tt.read_tif("dem.tif")
fd = tt.FlowObject(dem)                                  # D8, carve
st = tt.StreamObject(fd, threshold=1e6, units="m2")
D = pydivide.divide_network(fd=fd, st=st)
```

`divide_network(dem=None, *, fd=None, st=None, minarea=1e6, units="m2",
checkpoint_dir=None, verbose=True, release_inputs=False)`

| argument | meaning | default |
|---|---|---|
| `dem` | path to a GeoTIFF, or a pytopotoolbox `GridObject`; ignored when `fd` and `st` are both given | |
| `fd`, `st` | pytopotoolbox `FlowObject` and `StreamObject` | built from `dem` |
| `minarea` | drainage area at which a channel begins, in `units` | `1e6` |
| `units` | `"m2"` or `"pixels"` | `"m2"` |
| `checkpoint_dir` | folder for stage checkpoints; an interrupted run resumes from the last completed stage (section 10) | none |
| `verbose` | print each stage with a time stamp | `True` |
| `release_inputs` | free the rasters the divide stages do not need as soon as the stream network exists; required on domains of more than a few billion cells, harmless elsewhere; the objects passed in cannot be reused afterwards | `False` |

The DEM must be in a projected coordinate system in metres. Nodata is
respected: faces against nodata are divide edges, flagged `isbnd`
(section 11).

## 2. The result object

`divide_network` returns a `DivideNetworkResult`, the counterpart of a
`DIVIDEobj`:

```python
D.distance            # divide distance per node instance, in map units (m)
D.order("strahler")   # or "shreve" / "topo"; computed once and cached
D.endpoints           # node ids of divide endpoints
D.junctions           # node ids of divide junctions
D.n_segments
D.labels              # (nr, nc) int32 reach-basin label raster; -1 where unlabelled
D.network.seg_nodes   # flat node list of every segment
D.network.seg_ptr     # offsets: segment k is seg_nodes[seg_ptr[k]:seg_ptr[k+1]]
D.edge_nodes          # (E, 2) divide edges as node-id pairs
D.edge_is_boundary    # (E,) True for edges on the DEM perimeter or against nodata
D.grid                # GridSpec: nr, nc, cellsize, x0, y0
D.grid.node_xy(ids)   # node ids -> map coordinates
D.crs                 # coordinate reference system of the DEM
D.segment_is_boundary()   # per segment: touches the perimeter or nodata
```

`distance` and every order are aligned with `seg_nodes`: one value per
node instance, so a junction node appears once per segment that meets
there, with the same value each time.

## 3. Exports: GeoPackage, Shapefile, File Geodatabase

```python
D.to_geopackage("divides.gpkg")                    # one feature per segment
D.to_geopackage("divides_300m.gpkg", seglength=300)   # ~300 m pieces
D.to_shapefile("divides.shp", seglength=150)
D.to_filegdb("divides.gdb", seglength=300)
gdf = D.to_geodataframe(seglength=300)             # in memory, as GeoDataFrame
```

All four take `scheme` (`"strahler"`, `"shreve"` or `"topo"`, the order
written to the `order` field; default Strahler), `seglength` (metres;
`None` writes whole segments) and `mask`. Attribute fields:

| field | meaning |
|---|---|
| `ID` | feature number |
| `segid` | the full divide segment the piece belongs to |
| `order` | divide order in the chosen scheme |
| `distance` | mean divide distance of the piece's nodes (m) |
| `distmin`, `distmax` | minimum and maximum divide distance in the piece |
| `length` | piece length (m) |
| `azimuth` | orientation, 0 to 180 degrees |
| `isbnd` | 1 if the piece touches the DEM edge or nodata |
| `isring` | 1 if the piece belongs to a closed ring |

**Formats.** GeoPackage is the default: one file, native to QGIS, and
written in chunks so that only one chunk of pieces is held at a time
(the per-segment bookkeeping is one array of segment length). A
Shapefile has two independent 2 GB limits, the `.shp` geometry and the
`.dbf` attribute table; `to_shapefile` sizes its numbered parts from
the real geometry bytes of every piece (its type, dimension, parts and
vertices) and the DBF width of every field as the driver writes them
(24 bytes for a float; for an integer the longer of the driver's
default, 18 for a 64-bit and 9 for a narrower one, and its longest
value; for a string the longer of 80 and its longest value), so
neither limit is exceeded (`max_features` caps the part size further;
a single feature wider than the budget, and a geometry with a measure
coordinate, are refused; a layer holding any Z geometry is written and
sized as a Z layer throughout), and the driver's own 2 GB limit is
switched on as a backstop. A Shapefile is a set of files, so the
writers refuse when any companion of any planned part (`.shp`, `.shx`,
`.dbf`, `.prj`, `.cpg` and the rest), the field-name sidecar, or a
numbered part of an earlier export under the same name exists, unless
`overwrite=True`, which removes that whole previous set first, so a
shorter export never leaves earlier parts beside it. Existing
files are never replaced unless `overwrite=True` is passed, because a
GeoPackage or File Geodatabase may hold layers of other origin. A File Geodatabase is what
ArcGIS Pro prefers and has no size limit; `to_filegdb` needs GDAL's
OpenFileGDB writer, present in every recent GDAL.

**The GeoPackage projection repair.** ArcGIS Pro refuses every layer of
a GeoPackage whose custom projection (a local Albers or similar) carries
only GDAL's WKT1 definition; the symptom is "One or more items failed
to add". EPSG-coded systems such as UTM are not affected. Every writer
in `pydivide` therefore calls `pydivide.vectorio.repair_geopackage_crs`
after writing, which adds the WKT2 definition and the `gpkg_crs_wkt`
extension row without touching geometry or attributes. Calling it twice
is harmless. A GeoPackage written by other software can be repaired in
place:

```bash
python scripts/fix_gpkg_srs.py file.gpkg          # one file
python scripts/fix_gpkg_srs.py --all exports/     # every .gpkg in a folder
```

and a GeoPackage can be converted to a File Geodatabase with
`python scripts/export_gdb.py file.gpkg` (`--overwrite` to replace an
existing geodatabase; the exit status is non-zero when a conversion
fails).

**Masks.** `mask` restricts the export to pieces inside a polygon file,
a shapely geometry, or the valid (non-nodata) area of a raster. Masking
is purely an export filter: every value was computed on the whole
domain, so a masked piece carries the divide distance and order it has
in the whole network. That is the point. Clipping the DEM instead would
truncate the divide tree and change every distance. When the mask and
the network have different coordinate reference systems the piece
midpoints are reprojected; a transformation that cannot be built, or
that fails on any midpoint, raises rather than selecting nothing. When
only one of the two has a CRS a warning is issued and the coordinates
are compared as they are; a bare shapely geometry is taken to be in the
network's CRS.

`pydivide.vectorio` also exposes `write_vector`, `export_plan` and
`describe_export` for any GeoDataFrame, which report before writing
whether a table fits a Shapefile and in how many parts.

## 4. Reading a grid across the divides: `getvalue` and `asymmetry`

Both take the network, the grid specification and a flattened raster
aligned with the DEM (`values.ravel()`), as `DIVIDEobj` takes a
`GRIDobj`.

```python
labels = D.labels.ravel()

# the two values adjacent to every divide edge, per position (NaN at joints)
p, q = pydivide.getvalue(D.network, D.grid, relief.ravel(), label_flat=labels)
# or one number per edge
d = pydivide.getvalue(D.network, D.grid, relief.ravel(), "diff", label_flat=labels)

# one record per segment
a = pydivide.asymmetry(D.network, D.grid, relief.ravel(), label_flat=labels)
```

`getvalue(network, grid, values_flat, fct=None, *, label_flat=None)`:
without `fct` returns `(p, q)`; with `fct` in `min`, `max`, `mean`,
`diff`, `normdiff` returns one value per edge with MATLAB's
missing-value semantics.

`asymmetry(network, grid, values_flat, *, label_flat=None,
method="basin", aggfun="mean", clip_negative=True, order=None,
distance=None)` returns a dict of per-segment arrays: `x`, `y`
(segment midpoint), `rho` (magnitude), `theta` (azimuth from north
towards the lower side, as in MATLAB), `dai` (divide asymmetry index),
and, with `method="basin"`, the extension that makes the sign usable:
`side_lo` and `side_hi` (the two reach-basin labels), `value_lo` and
`value_hi` (their aggregated values) and `sgn = value_lo − value_hi`.
`n_basins` counts the basins a segment borders; segments bordering
more than two are skipped, as in MATLAB. `method="pixel_pairs"` gives
MATLAB's default per-edge vector sum with `u` and `v` components.

**Which side is A.** With `label_flat` given, side A of every edge is
the side with the lower reach-basin label, held fixed along the whole
segment. Without it, `p` is MATLAB's `pix` side (the top cell of a
horizontal face, the right cell of a vertical one), which is only a
geometric convention and changes basin along most segments that trend
north-west or south-east. Pass the labels whenever the sign of a
difference matters.

## 5. Every standard across-divide metric in one call

```python
import topotoolbox as tt
from pydivide import across_divide_metrics

dem = tt.read_tif("dem.tif")
tab = across_divide_metrics(dem, threshold=1e6, baselevel=5.0, seglength=300)
print(tab.describe())            # every parameter, in prose
tab.columns["chi_sgn"]           # one array per column, one row per piece
tab.settings                     # the parameters that shaped the numbers
tab.to_file("divides_metrics.gpkg")     # or .gdb, or .shp in parts
gdf = tab.to_geodataframe()
```

`across_divide_metrics(dem, *, flow=None, streams=None, divides=None,
baselevel=None, threshold=1e6, mn=0.45, a0=1.0, seglength=300.0,
metrics=("elev", "slope", "hr", "fd", "chi"), units="m2")`

Every row is one divide piece and starts with the columns of the divide
export (`ID`, `segid`, `order`, `distance`, `distmin`, `distmax`,
`length`, `azimuth`, `isbnd`, `isring`; section 3). Then, for each
metric family requested (`elev` elevation, `slope`, `hr` hillslope
relief, `fd` flow distance to the channel, `chi`), five columns: the
pair mean `<family>_mean` ((A + B) / 2), the absolute contrast
`<family>_diff` (|A − B|), the lower and the higher side `<family>_min`
and `<family>_max`, and the signed contrast `<family>_sgn` (A − B),
each the mean over the piece's edges of the per-edge value. Then the
divide asymmetry index `dai` (`hr_diff / (2 hr_mean)`), the basin and
outlet identity of both sides (`basin_a`, `basin_b`, `outlet_a`,
`outlet_b`), a flag `basin_pair_mixed` for pieces whose edges border
more than one basin pair, and a coherence column per family. On a
mixed piece the pair reported is the one on most of its edges, ties
going to the smallest labels, so it is always a pair one of its edges
borders, and `basin_pair_share` is the fraction of the piece's
labelled edges that carry it (1 on a piece with a single pair). Filter
on `isbnd = 0` before any analysis, since boundary pieces are artefacts
of where the DEM stops. `settings["provenance"]` records the software
versions, the grid, the CRS, the DEM source when known, and for the
routing, the streams and the divides whether this call computed them or
the caller supplied them; a supplied object's preprocessing is not
known to the call and is recorded as such. Unknown metric names raise
`ValueError`. Pass `flow`, `streams` or `divides` to reuse objects you
already have; they must describe the same grid as the DEM (shape, cell
size, origin and coordinate reference system, the last compared as
systems, not as strings), or the call refuses them with a message that
names the difference. An object that carries no transform or no CRS is
compared on what it does carry, and a warning says what could not be
checked. `settings["side_rule"]` states which physical side is side A
(the lower reach-basin label at the first edge of each segment, held
along the segment), and `settings["metrics_computed"]` lists the
families that actually ran, which is what `describe()` reports.

**The settings travel with the file.** `to_file` writes
`<name>_settings.json` beside every export, the document of
`tab.settings_json()`: thresholds, base level, concavity, reference
area, side rule, provenance, warnings, the column names and the piece
count. A GeoPackage additionally carries the same document inside the
file, in its `gpkg_metadata` table, so a single file stays
self-describing. `pydivide.read_export_settings(path)` reads either,
from the sidecar, from any file of the export or from the GeoPackage
itself, so the parameters behind a table can be recovered without the
Python object that made it. A sidecar answers only for a file it lists
among the files it describes, so two exports with overlapping names
(`metrics.gpkg` beside `metrics_part001.gpkg`) can never be confused,
and a sidecar that disagrees with the settings embedded in its
GeoPackage is reported rather than trusted. The sidecar follows the
same overwrite rule as the layer.

**chi needs a base level.** chi is integrated upstream from a base
level, so two chi values are comparable only when both sides are
referenced to the same one, and `across_divide_metrics` refuses to
compute chi without `baselevel`. The stream network is trimmed to the
cells above that elevation and the integration starts there. If the
value is at or below the lowest cell of the DEM nothing is trimmed and
`tab.settings["baselevel_warning"]` says so. The concavity `mn` and
reference area `a0` are recorded in `settings` for the same reason.

**Where the indicators disagree.**

```python
d = tab.disagreement()
d["all_agree"]                       # chi, relief and flow distance concur
d["n_indicators_agreeing_with_chi"]
```

chi marks the losing side with a higher value; hillslope relief and
flow distance mark it with a lower one, so agreement means opposite
signs. A piece where they disagree is not bad data; it is a divide
whose integral and local evidence differ, which is worth mapping.

**Several channel thresholds at once.**

```python
from pydivide import across_divide_multiscale, coherence, metric_column
tab = across_divide_multiscale(dem, thresholds=(1e5, 1e6, 1e7), baselevel=5.0)
tab.settings["tags"]                 # ["_1e5", "_1e6", "_1e7"], the column suffixes
tab.columns["chi_sgn_1e6"]           # one column per threshold
tab.columns["chi_sign_consistent"]   # the sign holds at every scale
tab.settings["side_rule"]            # which physical side is side A, at every scale
tab.columns["chi_peak_scale"]        # where the contrast is largest
tab.describe()                       # the parametrisation, every threshold named
tab.disagreement(tag="_1e6")         # the indicators at one threshold
coherence(tab.columns, "chi", "_1e6")
metric_column("chi", "sgn", "_1e6")  # "chi_sgn_1e6": how every column is named
```

The divide network is built once, at `network_threshold` or the
coarsest threshold, and held fixed; only the measurement scale varies.
Side A is held fixed too: it is the physical side that carries the
lower reach-basin label at the network threshold, at every scale, so a
sign that changes between scales is a change of contrast and never a
change of side convention. The `basin_a_<tag>` columns carry each
scale's own labels on that fixed side, and may be the higher label.
Column suffixes identify their threshold exactly (`_1e6`, `_110000`,
`_2500000`; `settings["tags"]` lists them), thresholds must be
distinct, finite and positive, and a failure at any threshold stops
the call unless `errors="skip"` is passed, in which case the threshold
is left out and recorded in `settings["skipped"]`.
`settings["per_scale"]` keeps each threshold's own record (its
warnings and stream-cell counts); the routing and the network are
shared, so their provenance is recorded once. A reach basin is the
area draining to one stream reach, so a finer threshold splits every
coarse basin pair into several fine ones, and the partitions are
exactly nested.

## 6. Basin pairs, profiles and plots

A signed contrast is a statement about two named basins. These functions
let a GIS show them.

```python
from pydivide import pair_table, basin_polygons, basin_points, pair_side_points
from pydivide import multiscale_pair_ids

pairs = pair_table(tab.columns["basin_a"], tab.columns["basin_b"],
                   values={"chi_sgn": tab.columns["chi_sgn"]})
polys = basin_polygons(D.labels, D.grid, crs=D.crs, keep=pairs["side_a"])
pts = basin_points(D.labels, D.grid, crs=D.crs)
a_pts = pair_side_points(pairs, D.labels, D.grid, crs=D.crs, side="a")
per_scale = multiscale_pair_ids(tab)      # for a multiscale table
```

`pair_id` is a dense integer over the unordered pair of labels, stable
within one labelling, so the same two basins carry the same id in every
layer. In `pair_table` every row's signed value is A minus B for that
row's `basin_a` and `basin_b`; the table reports the more common
orientation of each pair as `side_a` and `side_b`, counts the rest in
`n_flipped`, and re-expresses every signed value as `side_a` minus
`side_b` before averaging it, so two rows that describe the same
contrast with opposite orientations add up instead of cancelling.
Which value columns are signed is read from their names (`chi_sgn`,
`hr_sgn_1e5`) or stated with `signed=`; unsigned columns are averaged
as they are. On a continental domain there are millions of basins; pass
`keep` to `basin_polygons` rather than polygonising everything.

### The pair key

`pair_id` and `pair_uid` are stable inside one table. `pair_key` names a
pair from its labels and its partition alone, so the same number
identifies it in every file built on the same network:

```
pair_key = E * 10**14 + lo * 10**7 + hi        # E: 10^E m² threshold, lo < hi the labels
```

`700034600003461` reads, left to right, as partition 10⁷, basins 3460
and 3461 (`readable_pair_key`); `decode_pair_key` gives the threshold
and the two labels back. The largest key is below 2⁵³, so it survives
in every program that stores numbers as doubles. The rule refuses what
it cannot encode without collision: the threshold must be an exact
power of ten from 10¹ to 10⁹ m², labels must be positive and below 10⁷,
and the two labels must differ. `pair_table(..., scale="1e7")` adds the
column when its scale is such a threshold, and `multiscale_pair_ids`
adds `pair_key_<scale>` per rung.

### The profiles of a pair: the whole divide, along and across

A signed contrast names two basins. `basin_pair_profiles` draws the
divide those two basins share, in full, and the ground on either side
of it. The pair is two reach basins of one partition of the stream
network (the channel threshold: 10⁶, 10⁷, 10⁸ or 10⁹ m²); its divide is
every native divide edge whose two flanking cells carry the two labels,
which can be one line, several separate stretches, or a line that
forks. All of it is assembled into paths and all of it is drawn. Nothing
in `DIVIDEobj` does this.

```python
from pydivide import basin_pair_profiles, basin_pairs, reach_basin_labels

L7 = reach_basin_labels(dem, 1e7)                # labels of the 10⁷ partition (fd= to reuse routing)
p = basin_pair_profiles(D, dem, pair=(18, 19), labels=L7, threshold=1e7,
                        spacing=5000, half_length=2500)          # defaults: 5 km, 5 km sections
p.paths                      # one PairPath per path: nodes, xy, s, z, edges, normals
p.length, p.n_stations, p.readable, p.key
p.plot_along()               # figure 1: the along-divide profile of every path
p.plot_along(stations=True)  # figure 2: the same, with the stations lettered a, b, c, ...
p.plot_normals("grid")       # figure 3: one panel per station
p.plot_normals("stacked")    # figure 4: every station on one axis, coloured by position
p.save("pair_18_19")         # the four figures (png + pdf), along.csv, normals.csv, pair_<key>.gpkg
p.along_table(), p.normals_table(), p.gis_layers()

sw = basin_pair_profiles(D, dem, pair=(18, 19), labels=L7, threshold=1e7,
                         swath_width=1000)       # swaths 1 km wide: mean, median, quartiles, min, max
basin_pairs(D, dem, labels=L7, threshold=1e7)   # every pair of the partition, longest divide first
```

The conventions, stated because a reader needs them:

- The elevation of a native edge is the mean of its two flanking cells,
  and every edge counts one cell size of distance, the staircase length
  of the crest on the grid.
- Stations sit at `spacing/2 + k*spacing` along each path, never on a
  path end, where the direction of a divide is undefined; a path shorter
  than a spacing gets one station at its middle. `min_path` leaves the
  shortest paths without stations (they are still drawn).
- The direction of the divide at a station is the chord between the
  points `tangent_window/2` before and after it along the path (default:
  the spacing), because a divide on a grid has a direction only over a
  length.
- A section is `2 * half_length` long, centred on the divide, sampled
  bilinearly every `step` (the cell size); a sample off the grid or next
  to NoData is missing, never filled. With `swath_width` every section is
  a swath of lines `step` apart across that width, and the statistics
  across the width are reported at every distance.
- Every normal is turned so that negative distances lie in the basin with
  the lower label (`label_a`) and positive distances in the other
  (`label_b`), decided by the labels of the two cells of the edge under
  the station: the side convention of the signed metrics, so the picture
  and the sign cannot disagree. A section is a straight line in map space
  and can leave both basins within its length.
- The main path is the longest; every other path is drawn in its own
  panel, named as a branch (with the kilometre where it leaves the main
  path), a branch of a branch, or a separate stretch. `ylim` fixes the
  elevation axis so pairs compare by eye; `same_y` shares one range
  across the panels of figure 3.

The DEM must describe the same grid as the network (shape, cell size,
origin and CRS, checked as for `across_divide_metrics`), or the call
refuses it. Without `labels` the network's own labels are used; without
`threshold` the key is -1. With `segid=` instead of `pair=`, the pair
profiled is the one on most of that segment's edges, ties to the
smallest labels, so it is always a pair one of its edges borders.
`along_s`, `along_z`, `along_xy` and `across` are the main path's
profile and sections, for callers that expect one line.

Plots (`pip install "pydivide[plot]"`):

```python
from pydivide.plotting import plot_divide_values, plot_signed_divides, plot_basin_pair
plot_divide_values(D, tab.columns["chi_diff"], seglength=300)
plot_signed_divides(D, tab.columns["chi_sgn"], basin_a=tab.columns["basin_a"],
                    basin_b=tab.columns["basin_b"], min_abs=2.0)   # arrows into the losing basin
plot_basin_pair(D.labels, (1692, 1701), dem=dem, D=D, sgn=+2.1)  # which side is A
```

Pass the table's `basin_a` and `basin_b` columns to
`plot_signed_divides`, and the arrow of every piece starts on an edge
of that piece which borders exactly those two basins and enters the
one the sign names (the physical side is read from the basin labels,
never from the drawing direction of the line); a piece with no such
edge, because every edge of it lies on the DEM boundary or on another
pair, is not drawn, so no arrow points into no-data. Without them the
lower label at the chosen edge is taken as side A, which is the same
side except where a reach label changes inside the piece. The arrow is
`tick` map units long; a long tick can cross a further divide even
though the direction it starts in is right.

## 7. The other `DIVIDEobj` operations

`pydivide.divideops` reproduces `cleanedges`, `removeshortdivides`,
`shrink`, `dist2node`, `jctangle` and `jctcon`. Like the MATLAB
originals, the three filters need the diagonal-crossing grid `FX`,
which is built from the flow edges:

```python
import numpy as np
from pydivide.network import fx_grid
from pydivide import divideops as ops

FX = fx_grid(D.grid, np.asarray(fd.source, np.int64), np.asarray(fd.target, np.int64))

D2 = ops.clean_edges(D, FX)                    # drop perimeter divides, renumber
D3 = ops.remove_short_divides(D, FX, 1000.0)   # drop first-order divides shorter than 1 km
D4 = ops.shrink(D, FX, "distance", 500.0)      # keep the interior beyond 500 m
dist = ops.dist_to_node(D, nodes)              # along-network distance from given nodes
ang = ops.junction_angles(D)                   # angles at degree-3 junctions
cj = ops.junction_connectivity(D, maxdist=5000.0, junctions=some_ids)
```

Each filter returns a new result with the network re-segmented,
re-sorted, and distance and orders recomputed. `junction_connectivity`
is one wavefront expansion per junction; pass `junctions` to restrict
it on large networks.

## 8. Stream exports

```python
from pydivide.streams import export_stream_segments
export_stream_segments(fd, st, dem, "exports/myarea/streams_chi5m_seg300",
                       seglength=300, chi_field="chi5m", baselevel=5)
```

writes the stream network trimmed to the cells above `baselevel`, split
into pieces of about `seglength` metres, with per-piece mean chi
(`chi_field`), mean elevation (`elev`), mean upstream area in km²
(`upakm2`), length (`len_m`) and node count (`nnode`), as GeoPackage
and Shapefile (`formats`). `mn` and `a0` set the chi integral; `mask`
with `masked_out_base` writes a clipped copy from the same computation;
existing files are replaced only with `overwrite=True`. An export with
no stream cell above the base level writes an empty GeoPackage layer
that keeps the schema and no Shapefile, and says so.

## 9. The one-command script: `scripts/run_area.py`

```bash
python scripts/run_area.py --dem path/to/dem.tif --name myarea --chi-baselevels 5
```

runs flow routing, the stream network, the divide network and every
export for one DEM, writing to `exports/myarea/` and checkpointing under
`checkpoints/myarea/`. Rerunning the same command after an interruption
resumes.

| option | meaning | default |
|---|---|---|
| `--dem` | GeoTIFF in a projected CRS in metres | required |
| `--name` | output and checkpoint folder name | required |
| `--minarea` | stream threshold in m², for divides and streams alike | `1e6` |
| `--chi-baselevels` | one or more elevations (m); one stream export per base level | required |
| `--divide-baselevel` | trim the stream network to cells above this elevation before building the divides, so the network does not run into coastal flats | none |
| `--divide-seglengths` | piece lengths (m) for the divide exports | `150 300` |
| `--mask` | polygon file; additionally writes exports restricted to it, suffixed `_masked` | none |
| `--mask-seglengths` | piece lengths for the masked exports | as above |
| `--stream-seglength` | piece length for streams | `300` |
| `--stream-baselevel` | export the river geometry down to this elevation while chi is integrated from `--chi-baselevels` | none |
| `--mn` | concavity index of chi | `0.45` |
| `--checkpoint-name` | reuse checkpoints written under another name for the same DEM and threshold | `--name` |
| `--shp` | also write Shapefile copies of the divide and stream layers, in numbered parts where a 2 GB limit would be exceeded | off |
| `--root` | the folder that receives `exports/<name>` and `checkpoints/<name>` | the repository |

Outputs: `divides_seg150.gpkg`, `divides_seg300.gpkg`, `divides_full.gpkg`
(whole segments) and `streams_chi<b>m_seg300.gpkg` per base level, with
`.shp` copies of the divide and stream layers when `--shp` is given. A
File Geodatabase is made from any of these GeoPackages with
`scripts/export_gdb.py` (section 3). The script needs `psutil`
(`pip install "pydivide[scripts]"`).

## 10. Checkpointing, memory and time

The labels, crossing-grid, edge and network stages save their outputs
with a fingerprint of what produced them (`pydivide.checkpoint.StageStore`):
the threshold, the grid (shape, cell size, origin, CRS) and the flow and
stream edge arrays actually consumed. Rerun the same call with the same
`checkpoint_dir` and it resumes from the last completed stage; a
different DEM, routing or threshold gives a different fingerprint and
the checkpoints are recomputed, so results are never mixed. Only
north-up grids without rotation are accepted; a south-up or rotated
raster is rejected rather than mis-georeferenced.

Measured on one workstation (32 cores, 512 GB), divide stages only:

| cells | time | peak memory |
|---|---|---|
| 1.6 × 10⁷ | 13 s | under 1 GB |
| 4.0 × 10⁸ | 8.5 min | 15 GB |
| 4.5 × 10⁹ | 31 min | tens of GB |
| 1.2 × 10¹⁰ | 6.8 h | about 460 GB |

Flow routing by pytopotoolbox comes on top and dominates memory on the
largest domains. On domains of more than a few billion cells pass
`release_inputs=True` and a `checkpoint_dir`.

## 11. Conventions worth knowing

- Divide nodes sit on pixel corners; the network lies on the boundaries
  between cells, exactly as in `DIVIDEobj`. Node `(I, J)` is the
  top-left corner of cell `(I, J)`; `nid = I*(nc+1) + J`.
- All indexing is 64-bit. There is no 4.29-billion-cell ceiling.
- `isbnd` marks pieces that touch the DEM edge or nodata. A DEM built
  with a buffer so that complete basins sit inside it lets you discard
  these as truncation fragments.
- Endorheic basins cannot occur on carve-preprocessed DEMs. If a DEM
  ever produces one, its pieces carry `isring=1` and everything still
  runs.
- Divide distance and the orders are the reference implementation's
  definitions, and equal its values on its own sorted network; on
  `pydivide`'s own topological order they differ from the reference's
  where the reference's depend on its walk order, on 0.02 to 2.1 % of
  node instances on the validation DEMs (`docs/validation.md`).
- `pydivide` works on C-ordered (row-major) grids, the layout
  pytopotoolbox uses for every raster it reads from a file. A
  `FlowObject` built from a Fortran-ordered array (`order == "F"`)
  stores its indices column-major and is refused at every entry point
  with a message, rather than supported in part; rebuild it from a
  C-ordered array (`numpy.ascontiguousarray`). Every index is checked
  against the grid, and a stream network that describes another grid
  than its flow object is refused.
- Field names a Shapefile cannot hold (longer than ten characters, or
  repeating another name in a different case) are changed
  deterministically and every change is recorded in the
  `<name>_fields.csv` sidecar.
- `pydivide.gridspec` converts between MATLAB and Python indices
  (`matlab_node_to_py`, `py_node_to_matlab`, `matlab_cell_to_py`,
  `py_cell_to_matlab`) when results have to be compared.

## 12. Low-memory grid helpers

`pydivide.grids` holds versions of a few pytopotoolbox operations that
avoid full-grid temporaries, with bit-identical results:
`flow_accumulation_cells(fd)`, `flow_accumulation_inplace(fd, w)`,
`vertical_distance_to_stream(fd, stream, dem)`, `distance_to_stream(net)`,
`nal_to_hillslopes(net, nal)` (propagate a stream node-attribute list to
every cell), `arcslope_degrees(z, cellsize)` and `upslope_mean(fd, values)`.
They matter only on domains where a single extra copy of the grid is
tens of gigabytes.

## 13. Index of the public names

Everything `import pydivide` exports, with the section that describes it.

| name | what | section |
|---|---|---|
| `divide_network` | the divide network of a DEM or of routing objects | 1 |
| `DivideNetworkResult` | the result: network, distance, orders, labels, exports | 2, 3 |
| `GridSpec` | grid geometry: `nr`, `nc`, `cellsize`, `x0`, `y0`, `node_xy` | 11 |
| `node_id`, `node_rc` | node `(I, J)` to linear id and back, on the `(nr+1) × (nc+1)` node grid | 11 |
| `matlab_node_to_py`, `py_node_to_matlab`, `matlab_cell_to_py`, `py_cell_to_matlab` | index conversions between MATLAB and Python conventions | 11 |
| `assert_int64` | the guard against silent 32-bit downcasts | 11 |
| `StageStore` | stage checkpointing and resume | 10 |
| `MatlabReference` | reader of the MATLAB reference exports used by the validation ladder (`docs/validation.md`) | |
| `getvalue`, `asymmetry` | a grid read on the two sides of every divide | 4 |
| `orient_by_basin` | the per-edge side rule: side A is the lower reach-basin label | 4 |
| `across_divide_metrics`, `across_divide_multiscale` | every standard across-divide metric in one call, at one or several thresholds | 5 |
| `AcrossDivideTable` | their result: `columns`, `settings`, `to_geodataframe`, `to_file`, `settings_json`, `describe`, `disagreement` | 5 |
| `coherence` | how consistently a signed contrast points one way inside a piece | 5 |
| `metric_column` | the name of one field of one family at one threshold, `chi_sgn_1e6` | 5 |
| `read_export_settings` | the settings document of an export, from the file alone | 5 |
| `pair_ids`, `pair_table`, `basin_points`, `basin_polygons`, `pair_side_points`, `multiscale_pair_ids` | basin pairs as GIS objects | 6 |
| `pair_key`, `decode_pair_key`, `readable_pair_key` | the pair key, unique across partitions | 6 |
| `reach_basin_labels` | the reach-basin labels of any partition of the stream network | 6 |
| `basin_pairs` | every competing pair of a partition, longest divide first | 6 |
| `basin_pair_profiles`, `BasinPairProfiles`, `PairPath` | the whole divide of one pair: along-divide profile of every path, normal profiles or swaths at stations, the four figures, tables, GeoPackage | 6 |
