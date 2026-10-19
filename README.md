# pydivide

Drainage divide networks in Python. `pydivide` reimplements and extends
the `DIVIDEobj` class of MATLAB TopoToolbox 3 (Scherler and Schwanghart,
2020): it extracts the network of drainage divides from a digital
elevation model, orders it by hierarchical depth (divide distance and
the Strahler, Shreve and topological divide orders), reads any grid
across every divide, and writes the result to the formats a GIS opens
directly.

It exists because the MATLAB implementation cannot process a domain of
more than 4.29 × 10⁹ cells, and because divide distance is the depth of
the whole divide tree, so a large domain cannot be tiled or clipped
without changing the answer. `pydivide` uses 64-bit indexing throughout
and single-pass algorithms in place of the reference's quadratic stages
(measured scaling exponent 1.04 in the number of cells), and has
computed the divide network of a 1.2 × 10¹⁰-cell DEM in one piece.
Where the two implementations can both run, `pydivide` is validated
against the reference on nine test DEMs with the reference's own flow
routing injected (`docs/validation.md`): the reach-basin labels are
identical; every divide edge the reference produces is reproduced, the
only surplus edges being ones the reference drops or misses through
documented defects of its own; on the reference's edge set, with those
defects injected, the junctions, endpoints and segments are identical;
and on the reference's own sorted network every divide distance and
order is identical. `pydivide`'s own sort is a valid topological order
and not the reference's: where the reference's values depend on the
order its walk happens to take, between 0.02 % and 2.1 % of node
instances receive a different divide distance, measured and classified
in `docs/validation.md`.

`pydivide` is the first module of ATLAS Topo, the topographic-analysis
software of the ATLAS lab (Advanced Terrestrial Landscape
Analysis and Simulations). It stands alone and depends only on the
packages listed below.

## Requirements

Python 3.10 or later, with `numpy`, `scipy`, `rasterio`, `geopandas`,
`shapely`, `pyproj` and `topotoolbox` (pytopotoolbox, which does the
flow routing). All are installed by `pip` with the package. Plots need
`matplotlib`, installed with the `plot` extra (see Installation).
Memory scales with the
DEM: a 16-million-cell DEM needs under 1 GB, a 400-million-cell DEM
about 15 GB, and a 12-billion-cell DEM about 460 GB at its peak.

## Installation

From GitHub (the package is not on PyPI; an unrelated project holds the
name `pydivide` there):

```bash
pip install git+https://github.com/ATLAS-Topo/pydivide
```

With an extra (`plot` for the figures, `scripts` for the command-line
scripts), name it before the address. Never `pip install "pydivide[plot]"`
alone, which installs the unrelated PyPI project:

```bash
pip install "pydivide[plot] @ git+https://github.com/ATLAS-Topo/pydivide"
```

Or clone the repository and install it in place, which is the way to
run the tests and the example:

```bash
git clone https://github.com/ATLAS-Topo/pydivide
cd pydivide
pip install -e ".[test]"
pytest tests -q
```

## Getting started

```python
import pydivide

D = pydivide.divide_network("dem.tif", minarea=1e6)   # streams above 1 km²

D.distance              # divide distance per node, in map units
D.order("strahler")     # also "shreve", "topo"
D.endpoints, D.junctions, D.n_segments

D.to_geopackage("divides.gpkg")   # one file, QGIS-native, written in chunks
D.to_shapefile("divides.shp")     # split into parts sized from real geometry and attribute bytes
D.to_filegdb("divides.gdb")       # ESRI File Geodatabase, for ArcGIS Pro
```

Reading a grid across the divides, with the two sides identified by
basin so that the sign of a difference has a meaning:

```python
labels = D.labels.ravel()                       # reach-basin label per cell
p, q = pydivide.getvalue(D.network, D.grid, relief.ravel(), label_flat=labels)
a = pydivide.asymmetry(D.network, D.grid, relief.ravel(), label_flat=labels)
a["sgn"], a["side_lo"], a["side_hi"], a["theta"], a["dai"]
```

Every standard across-divide quantity of a DEM, in one call:

```python
import topotoolbox as tt
from pydivide import across_divide_metrics

dem = tt.read_tif("dem.tif")
tab = across_divide_metrics(dem, threshold=1e6, baselevel=5.0, seglength=300)
print(tab.describe())            # every parameter, in prose
tab.to_file("divides_metrics.gpkg")
```

Every competing basin pair, drawn. The divide two reach basins share is
one object, named by a key that is unique across partitions of the
stream network (10⁶, 10⁷, 10⁸, 10⁹ m²), and pydivide draws it in
full: the along-divide profile of every path of the divide, and normal
profiles or swaths across it at a chosen spacing and length, with the
two basins identified on every figure. `DIVIDEobj` has no counterpart.

```python
from pydivide import basin_pair_profiles, basin_pairs, reach_basin_labels

L7 = reach_basin_labels(dem, 1e7)                          # labels of the 10⁷ partition
p = basin_pair_profiles(D, dem, pair=(18, 19), labels=L7, threshold=1e7,
                        spacing=1000, half_length=750)     # stations every 1 km, 1.5 km long
p.save("pair_18_19")                                       # the four figures, tables, GeoPackage
basin_pairs(D, dem, labels=L7, threshold=1e7)              # every pair of the partition
```

`examples/bigtujunga.py` runs the first two snippets on the TopoToolbox
example DEM in a few seconds.

## Documentation

| file | what |
|---|---|
| [`docs/user_guide.md`](docs/user_guide.md) | every function and option, the export formats, the scripts, checkpointing, memory |
| [`docs/how_it_works.md`](docs/how_it_works.md) | grid conventions, the six stages, what is identical to `DIVIDEobj` and what is deliberately different |
| [`docs/validation.md`](docs/validation.md) | how the agreement with MATLAB is tested, and how to test it on your own DEM |
| [`CHANGELOG.md`](CHANGELOG.md) | what each release contains |

The paper describing the reimplementation, its validation and its
measurements is in preparation for *Earth Surface Dynamics*. The MATLAB
reference exports, benchmark scripts, timings and experiment outputs
behind it are archived in a separate Zenodo data deposit,
[doi:10.5281/zenodo.14748886](https://doi.org/10.5281/zenodo.14748886).
The tests in this repository that need the reference exports skip
without them; `docs/validation.md` says how to run them with the deposit.

## How to cite

Please cite the method itself:

Scherler, D. and Schwanghart, W. (2020): Drainage divide networks,
Part 1: Identification and ordering in digital elevation models. *Earth
Surface Dynamics* 8, 245–259,
[doi:10.5194/esurf-8-245-2020](https://doi.org/10.5194/esurf-8-245-2020).

and the software, as given in [`CITATION.cff`](CITATION.cff) (a Zenodo
DOI is created for every release).

## Licence

GNU GPL v3 or later. `pydivide` reimplements the semantics of
TopoToolbox 3, which is GPL-licensed, and was developed with reference
to its source.

## Funding

The development of `pydivide` was supported by the National Council for
Scientific and Technological Development (CNPq, Brazil), process
no. 446330/2024-8.

## Artificial intelligence disclosure

Claude Code (Fable 5.1, Anthropic) was used as an assistant during the
development and revision of this work: for language refinement and
consistency checks across the manuscript, the supplement and the
software documentation; for the organisation of the documentation and
the reproducibility files of the associated code and data archives; and
for drafting and revising code and tests under the author's direction.
The author designed the method, independently performed the numerical
modelling, the topographic analyses, the validation against the
reference implementation, the figure preparation, the interpretation and
all scientific decision making, reviewed, edited, tested and fact-checked
all AI-assisted code and text, and remains fully responsible for the
work. No generative AI tools were used to create scientific data, model
outputs, figures, maps or images.
