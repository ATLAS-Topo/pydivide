# Changelog

## 0.2.1

- Outlet-corner blanking uses the outlets that DIVIDEobj uses, those of
  TopoToolbox's `streampoi(S, 'outlets')`: stream nodes with no outgoing
  and at least one incoming stream edge (`FlowNet.blanking_outlets`). A
  one-cell stream network, a stream node with no edge at all, was also
  blanked as an outlet, which removed the divide edges that touch the
  corners of its cell. Such a node occurs only where a cell drains
  straight out of the DEM, so only boundary edges were affected, three
  for each one-cell network on the edge of the DEM or beside nodata. On a
  20,000 × 20,000-cell window of the south-eastern Brazil FathomDEM
  mosaic, 36 one-cell networks lie on the DEM edge, and the network has
  108 more boundary edges, 36 fewer segments and 72 fewer endpoints than
  with 0.2.0. With the reference routing injected, the nine
  validation DEMs have no one-cell network and their validation is
  unchanged. With the routing of pytopotoolbox, the seven external
  validation landscapes have 1 to 13 one-cell networks each and gain 3
  to 41 boundary edges. `FlowNet.outlets` still includes the
  one-cell networks, so their reach basins keep an outlet and an outlet
  elevation in the across-divide tables. The checkpoint identity
  changes, so existing checkpoints are rebuilt rather than reused.
  `tests/test_outlet_rule.py` builds a one-cell network on a synthetic
  surface and checks the edges around it.

## 0.2.0

- `across_divide_metrics` records the elevation at which each side's
  river system leaves the stream network (`outlet_z_a`, `outlet_z_b`)
  and, when chi is computed, whether both lie at or below the base level
  (`chi_common_base`: 1, 0, or NaN where a side has no outlet). chi is
  integrated from the base level only along a river that reaches it; a
  system that leaves the DEM above the base level starts its chi at its
  own outlet, so a contrast between it and a system that reaches the
  base level compares two different base levels. `settings` counts the
  affected pieces and systems and carries `common_base_warning`, and
  `describe()` states the count; the multiscale table carries the flag
  per scale. On the Big Tujunga DEM of pytopotoolbox at a base level of
  400 m, 28 of the 30 river systems leave the DEM above it and 2,482 of
  the 4,745 pieces with a chi contrast are flagged; on the Taiwan DEM at
  20 m none is. The user guide (section 5) describes the flag.
- `basin_pair_profiles` decides which way each normal faces from the
  cells flanking the divide over the same stretch that sets the
  direction of the normal (the tangent window), so that negative
  distances lie in basin A at every station. It used the flanking cell
  of the single edge under the station. On a grid the divide is a
  staircase of cell faces, and a station on a face running across the
  trend of the divide can lie further from that cell along the normal
  than beside it, so some sections were drawn with the two basins
  swapped: 11 of the 125 sections whose two ends lie clearly in the two
  basins, on the 15 longest divides of the 10⁷ m² partition of Big
  Tujunga. The along-divide profile, the signed metrics and the arrows
  of `plot_signed_divides` do not use the normals and are unchanged. The
  rule is `profiles.side_points`; a test checks the side of every such
  section on Big Tujunga, and `docs/user_guide.md` describes it.

## 0.1.2

No change in behaviour; documentation and docstrings only.

- `docs/validation.md` states which code computed the MATLAB reference
  routing. The reference exports were made with a TopoToolbox copy whose
  libtopotoolbox bindings were not compiled, so MATLAB ran its own code
  for every routing step; on Big Tujunga the same call with the bindings
  gives identical flow edges. The disagreement with pytopotoolbox is
  described by the classes a screening assigns most of it to, with the
  rest left unclassified, instead of as tie-breaking alone.
- The memory figures in the docstrings and comments of
  `flow_accumulation_inplace`, `vertical_distance_to_stream`,
  `distance_to_stream`, `upslope_mean`, `fx_grid` and the pipeline's
  release of the flow-edge arrays size per-edge arrays by the number of
  flow edges (about 5.1e9 on the Brazil domain) rather than the number
  of cells, mark binary units as GiB, and separate the arrays alive at
  once from the total allocated over a call.

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
