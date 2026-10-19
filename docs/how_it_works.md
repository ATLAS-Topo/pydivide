# How pydivide works

What `pydivide` computes, in what order, how that relates to MATLAB
TopoToolbox 3 `DIVIDEobj`, and where the two deliberately differ. The
scientific definitions of a divide, a junction, an endpoint, divide
distance and divide order are those of Scherler and Schwanghart (2020)
and are unchanged. The paper describing the reimplementation and its
validation gives the measurements behind every statement here.

## 1. Grid and index conventions

Both implementations place divide nodes on pixel corners, so the divide
network lies on the boundaries between cells.

| | MATLAB | pydivide |
|---|---|---|
| cell grid | `nr × nc`, 1-based, column-major | `nr × nc`, 0-based, row-major |
| node grid | `(nr+1) × (nc+1)` | same shape |
| node index | 1-based column-major | 0-based row-major, `nid = I*(nc+1) + J` |
| index type | `uint32` | `int64` everywhere |

Node `(I, J)` sits at the top-left corner of cell `(I, J)`, at world
coordinates `(x0 + J·cellsize, y0 − I·cellsize)`. `GridSpec` holds
`nr`, `nc`, `cellsize`, `x0`, `y0` and converts node ids to coordinates
(`node_xy`). The MATLAB and Python conventions are converted in exactly
one place, `pydivide.gridspec`, and the conversion is tested against
MATLAB's own `ind2coord` and `coord2ind`.

A divide edge is a face between two cells. A vertical face separates
cells `(i, j)` and `(i, j+1)` and joins nodes `(i, j+1)` and `(i+1, j+1)`;
a horizontal face separates `(i, j)` and `(i+1, j)` and joins nodes
`(i+1, j)` and `(i+1, j+1)`. In node ids, a vertical face's two nodes
differ by `nc+1` and a horizontal face's by 1.

Every index array is asserted `int64` at function boundaries
(`assert_int64`). A silent 32-bit downcast is the most likely way a very
large run fails, and it produces plausible wrong answers rather than an
error, so it is never allowed.

## 2. The six stages

`divide_network` runs six stages. Each consumes only the flow edges, the
stream edges and the grid geometry, so the routing can come from
pytopotoolbox (production) or from a MATLAB export (validation).

**Stage 1, reach-basin labels.** Every stream reach (the channel between
two confluences, or between a confluence and an outlet) gets a label,
and the label is propagated to every hillslope cell that drains to it.
MATLAB does this with sequential sweeps over the flow edges; `pydivide`
uses a pointer-doubling fixpoint, in which every cell's pointer jumps to
its pointer's pointer until nothing changes, which converges in
O(log of the longest flow path) vectorised passes. The label raster is
`int32` (reach counts are far below 2³¹ even on a 10¹⁰-cell DEM); cell
and node indices stay `int64`. This raster is `D.labels`.

**Stage 2, divide edges.** A divide is by definition a face between two
cells with different labels, so the edge set is a 4-neighbour comparison
of the label raster, O(N) in cells, with no polygons. MATLAB polygonises
every basin and matches coordinates, which is O(B × M) in basins and
stream pixels and produces every interior edge twice. Three consequences
of the polygon method are reproduced explicitly because the validation
demands equality: faces on the DEM perimeter and against nodata are
divide edges and are flagged `isbnd`; a face crossed by a cardinal
stream flow edge is removed; and the corners of each outlet pixel are
opened so the basin's ring can be sorted.

**Stage 3, the diagonal-crossing grid (FX).** Where a stream flows
diagonally between two cells, its flow edge crosses a divide node. MATLAB
finds these nodes from floating-point flow-edge midpoints; `pydivide`
finds them from integer cell arithmetic (`dr·dc = ±1`), with no
tolerance, and reproduces MATLAB's precedence when a node is crossed by
both diagonals.

**Stages 4 and 5, junctions, endpoints and segments.** At every node the
incident edges are connected by one local rule: with no crossing, degree
1 is an endpoint, degree 2 connects, degree 3 or more is a junction; at
a crossing node the four directions are split into the two sides the
stream separates, two edges on one side connect and a lone edge
terminates. A segment is a maximal chain of connections, walked once.
MATLAB's `divnet` does this with an order-dependent merge cascade over a
struct array; the local rule is where `pydivide` is more correct rather
than merely faster, because the cascade can connect edges across the
stream that separates them.

**Stage 6, sort, distance and order.** Segments are peeled inward from
the endpoints, a junction becoming active once all but one of its
segments are consumed, and a new wave is seeded from any remaining
segment when no endpoint is left (closed rings). Divide distance is
`base + k·cellsize` along each directed segment, the base being the
largest distance inherited at the segment's start; a segment with
endpoints at both ends takes the pointwise minimum of its two
directions. The three orders are one pass in topological order:
Strahler `ω = max(min ω + 1, max ω)`, Shreve `ω = Σ ω`, topological
`ω = max ω + 1`, all starting at 1 at endpoints. MATLAB rebuilds a
struct array once per level, which is quadratic in the number of
segments; `pydivide` keeps the network flat (`seg_nodes` with `seg_ptr`
offsets, plus endpoint and junction arrays); the sort is one peeling
sweep in which every segment is consumed once and every junction is
promoted once, and distance and the three orders are single passes in
that order, so the measured time grows almost linearly with the number
of cells (exponent 1.04 between 1.6 × 10⁷ and 4.0 × 10⁸ cells).

## 3. Reading a grid across the divide

`getvalue` reads any grid on the two cells adjacent to every divide edge
and `asymmetry` summarises the difference per segment, as in
`DIVIDEobj`. `pydivide` adds a signed contrast whose two sides are
identified by reach-basin label: side A is the side with the lower
label, held fixed along the whole segment, and the labels of both sides
are exported so the sign can be read back to two named basins.

The obvious alternative, a sign defined by grid geometry (west cell
minus east, north minus south), is wrong in a way that is easy to miss.
A segment that steps diagonally towards the north-west alternates
between vertical and horizontal faces, and at every corner the west
cell of one face and the north cell of the next lie in opposite basins,
so the sign flips from edge to edge along one segment and the segment
mean is a mixture of the two. Measured on a 4000 × 4000 window, that
rule switches basin along 77 % of the segments. Basin identity is the
only orientation that is constant along a segment, and it is the rule
`asymmetry` itself uses.

## 4. What is identical, and what is different

**Reproduced, by validation against MATLAB on nine DEMs with the
reference routing injected:** basin labels, the divide edge set (every
reference edge present; the surplus edges are the ones the reference
drops or misses through its documented defects), the diagonal-crossing
grid, and, on the reference's edge set and sorted network, junctions
and endpoints, the segment decomposition, divide distance, and all three
divide orders (`docs/validation.md` says what each step conditions on
and how far `pydivide`'s own sort departs from the reference's). The remaining `DIVIDEobj` methods (`cleanedges`,
`removeshortdivides`, `shrink`, `dist2node`, `jctangle`, `jctcon`,
`getvalue`, `asymmetry`) are reproduced with their MATLAB semantics.

**Deliberately different:**

- `int64` indexing throughout, removing the 4.29 × 10⁹-cell ceiling;
- divide geometry from label boundaries rather than polygons, O(N) and
  free of the polygon-tracing defect;
- the crossing grid from integer arithmetic rather than floating-point
  midpoints;
- a deterministic local pairing rule at crossing nodes instead of an
  order-dependent cascade;
- flat arrays instead of struct arrays rebuilt per stage;
- boundary divides flagged (`isbnd`) rather than silently included;
- a signed across-divide contrast with the identity of both sides;
- the competing basin pair as an object, with a key that is unique across
  partitions of the stream network, a pair table of A minus B contrasts,
  and the whole divide a pair shares drawn along the crest and across it
  at stations (user guide, section 6); the reference has no counterpart;
- every heavy stage checkpoints to disk and resumes;
- exports to GeoPackage, Shapefile and File Geodatabase written one
  chunk of pieces at a time, with the WKT2 coordinate-reference
  definition that ArcGIS Pro requires for custom projections.

**Known limitation, reproduced not fixed:** internally drained
(endorheic) basins are not handled, as in the reference. With carve
preprocessing, which is the default, no truly endorheic basin can
exist; if one ever occurs its segments form a closed ring and are
flagged `isring`.

## 5. Scale

The divide network is a tree, and divide distance is the depth of that
tree, so a domain cannot be tiled or clipped without changing the
answer: the continental divide at the root of the tree is the first
thing clipping destroys. `pydivide` therefore computes the whole domain
in one piece. Measured on one workstation (32 cores, 512 GB), the
divide stages take about 13 s on 1.6 × 10⁷ cells, 8.5 min on 4.0 × 10⁸,
31 min on 4.5 × 10⁹ and 6.8 h on 1.2 × 10¹⁰, scaling almost linearly
with the number of cells, and the largest run peaks at about 460 GB of
memory. The memory-relevant choices are in `docs/user_guide.md`.
