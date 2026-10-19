# Changelog

## 0.1.1

- `flow_accumulation_inplace` refuses weights it cannot index correctly.
  The kernel reads the buffer flat, so a Fortran-ordered array or a
  strided view was summed as if it were C-ordered, and a view was
  written outside its own cells; such arrays, and read-only ones, now
  raise a `ValueError` instead. `upslope_mean` sums a C-ordered copy of
  any input that is not an owned, writable, C-ordered array, so its
  result no longer depends on the input layout. Nothing in the package
  passes such arrays to either function, and grids read with
  `topotoolbox.read_tif` or `rasterio` are C-ordered.
- `divide_network` (`minarea`) and `reach_basin_labels` (`threshold`)
  refuse a stream threshold that is not a finite positive number, which
  gave an empty network (NaN, infinity) or a stream through every cell
  (zero, negative values) without warning.
- The installation commands with an extra name the GitHub source:
  `pip install "pydivide[plot]"` on its own installs an unrelated
  project that holds the name on PyPI.
- `matlab_scripts/check_divide_stages.m` replaces
  `check_divnet_edge_loss.m`. It runs the `DIVIDEobj` constructor,
  `divnet` and `sort` separately, optionally on injected flow edges, and
  reports the edges each stage removes. On the Brazil window of the
  paper's data deposit the 16,346 edges that the full `DIVIDEobj`
  pipeline of TopoToolbox 3 `main` drops are removed by `sort`, not by
  `divnet`; the earlier script compared only the constructor with the
  finished network and could not tell the two apart.
- The test workflow also runs the low-memory flow tests on the Brazil
  window, fetched from the data deposit and checked against the SHA-256
  its manifest publishes.

## 0.1.0 (first public release)

- Divide networks from a DEM or from pytopotoolbox routing objects
  (`divide_network`): reach-basin labels, divide edges, diagonal-crossing
  handling, segments, sorting, divide distance and the three divide orders
  (Strahler, Shreve, topological), with 64-bit indexing throughout.
- Agreement with MATLAB TopoToolbox 3 `DIVIDEobj` verified on nine DEMs
  with the reference implementation's own flow routing injected: labels
  identical, every reference edge reproduced (the surplus being edges
  the reference drops or misses through documented defects of its own),
  and, on the reference's edge set and sorted network, identical
  junctions, endpoints, segments, distance and orders
  (`docs/validation.md` states what each step conditions on).
- Exports to GeoPackage, Shapefile (split into parts sized from the real
  geometry bytes of every feature and the DBF width of every field, so
  neither 2 GB limit is exceeded) and ESRI File Geodatabase, with the
  WKT2 coordinate-reference repair ArcGIS Pro requires for custom
  projections. Every metrics export carries its settings in a JSON
  sidecar and, in a GeoPackage, inside the file as well. Objects
  supplied separately, to the across-divide call, the profiles or the
  routing adapter, must describe one grid (shape, cell size, origin,
  CRS) and be C-ordered, or the call refuses them.
- The remaining analytical `DIVIDEobj` methods: `getvalue`, `asymmetry`,
  `cleanedges`, `removeshortdivides`, `shrink`, `dist2node`, `jctangle`,
  `jctcon`. The MATLAB plotting methods (`plot`, `plotc`) are not
  reproduced; the network is a GeoDataFrame and `pydivide.plotting`
  draws signed contrasts.
- A signed across-divide contrast whose two sides are identified by
  reach basin, with the basin and outlet identity of both sides exported
  (the pair on most of a piece's edges), a flag and a share for pieces
  whose edges border more than one basin pair, and pair summaries in
  which every signed value is expressed as side A minus side B of its
  pair before it is averaged.
- `across_divide_metrics` and `across_divide_multiscale`: elevation,
  slope, hillslope relief, flow distance and chi across every divide in
  one call, at one or several channel-initiation thresholds, with every
  parameter recorded in the result.
- Basin pairs as objects: a key unique across partitions of the stream
  network (`pair_key`), the pair table with every signed value expressed
  as side A minus side B, catchment polygons and side points for a GIS,
  and the list of every pair of a partition (`basin_pairs`).
- The profiles of a competing basin pair (`basin_pair_profiles`): the
  whole finite divide two basins share, assembled into its paths
  (branches and separate stretches included), the along-divide profile
  of each, and normal profiles or swaths across it at stations of a
  chosen spacing and length, every normal turned into the same basin;
  `save` writes the four figures, two tables and a GeoPackage.
  `reach_basin_labels` gives the labels of any partition.
- matplotlib plots of signed contrasts (optional `plot` extra).
- Stage checkpointing and resume for long runs, keyed to the grid, the
  flow edges and the stream edges actually consumed.
