# Changelog

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
