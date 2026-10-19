# Validation against MATLAB DIVIDEobj

`pydivide` is validated by demanding equality with the reference
implementation quantity by quantity, not by agreement statistics, with
the reference's own documented defects as the only admitted exceptions.
This page says what is compared, what each comparison takes from the
reference and what it computes independently, how to run the
comparison, and how to run it on a DEM of your own.

## What is compared, and what each step conditions on

A seven-step ladder, each step demanded before the next is attempted,
with the reference implementation's own flow and stream networks
injected into `pydivide` so that only the divide code is under test.
The third column says what else a step takes from the reference; the
claim of each step is exactly as wide as that column allows.

| step | quantity | taken from the reference | computed by `pydivide` | demanded |
|---|---|---|---|---|
| 0 | grid and index conventions | `ind2coord` and `coord2ind` results | the index conversions | identical |
| 1 | reach-basin label raster | the flow and stream networks | the labels | identical, and the `labelreach` node-attribute list identical element for element |
| 2 | divide edge set, as unordered node pairs | the flow and stream networks | the edge set | every reference edge present; every surplus edge proven a reference artefact (see below) |
| 3 | diagonal-crossing grid `FX` | the flow network | the grid | bit-identical |
| 4 | junction set, junction degrees, endpoint set | the reference's final edge set, with its classified order-dependent artefacts injected (`pydivide.validate.build_compat_plan`) | the crossing rule, the junctions and endpoints | identical |
| 5 | segment decomposition, as a set | the same | the segments | identical |
| 6 | divide distance | the reference's sorted network, its `D.IX` runs used verbatim as the sort output | the distance values | identical as node-keyed value multisets |
| 7 | the three divide orders | the same | the order values | identical |

Steps 1 to 3 condition on nothing but the routing. Steps 4 and 5 run
`pydivide`'s segmentation on the reference's final edge set, with the
reference's order-dependent artefacts (dropped edges, blind merges,
merge failures, phantom joins) classified and injected, so that the two
implementations decompose the same geometry; the classifier raises,
and validation stops, on anything it cannot classify. Steps 6 and 7
evaluate `pydivide`'s distance and order code on the reference's own
sorted network, because the reference's sort leaves two things to the
order its walk happens to take: the direction of a segment whose two
termini become ready in the same wave, and the order in which the
segments of one wave append their seeds. `pydivide`'s own sort fixes
both deterministically and is checked separately to be a valid
topological order (`test_sort_topological_invariant`); how far it
departs from the reference's is measured in the last section.

The edge-set step is the make-or-break test. Two behaviours of the
reference constructor make literal equality with its final output
unattainable in principle (it drops isolated single-edge runs in a
traversal-order-dependent way, and its polygon tracer mis-traces some
diagonal embayments), so the gate is: no MATLAB edge may be missing
from `pydivide`, ever; every surplus `pydivide` edge must be proven by
machine to be one of those two artefacts, either present in MATLAB's
pre-drop edge list (re-derived by `matlab_scripts/export_T0_nodrop.m`)
or genuine against MATLAB's own label raster; and anything
unclassifiable blocks validation. The classifier is
`pydivide.validate.build_compat_plan`.

The ladder passes on nine DEMs: the TopoToolbox example catchment
(Big Tujunga), a 4000 × 4000 window of a 30 m DEM of south-eastern
Brazil, and seven Copernicus GLO-30 DEMs of other escarpments and
plateaus (six used whole, one as a 6000 × 6000 window).

## Why the reference routing is injected

pytopotoolbox and MATLAB TopoToolbox both route flow with D8 and carving,
but not with the same code: at the reference commit MATLAB fills sinks,
finds flats and computes the carving costs with the libtopotoolbox C
library that pytopotoolbox uses for every step, and then computes the
auxiliary distance and chooses the receivers in its own MATLAB code.
On Big Tujunga they disagree on 0.64 % of flow edges, mostly inside
flats of the filled surface and at ties in the steepest descent, with
neither systematically steeper. The divide stages consume only the flow
edges, the stream edges and the grid, so the validation feeds them the
MATLAB routing. Production runs use pytopotoolbox routing.

## Which MATLAB

The reference is TopoToolbox 3 at commit `1bc93e7` (December 2025).
Four of its files carry documented local patches that let the stock code
complete on large projected rasters and keep the segments the stock
`sort` drops when those left in a component form loops through junctions
(on the Brazil window 382 segments, 1.7 % of the node instances; on Big
Tujunga the patches change no result). They are listed as diffs in the
paper's supplement. "Identical to MATLAB" means identical to that
documented configuration.

## Running the ladder

```bash
pytest tests -q
```

Tests that need MATLAB reference exports look for them under
`reference_data/<name>/` (one directory per DEM, each with a
`manifest.json`) and skip cleanly when none is present, which is what
happens on a fresh clone and in continuous integration. The tests that
run everywhere build the whole pipeline on synthetic surfaces (labels,
edges, crossings, segments, sort, distance, orders, the across-divide
metrics, the exports and the `DIVIDEobj` operations) and check the
invariants that hold without a reference; `tests/test_contracts.py`
needs neither a reference nor pytopotoolbox. The ladder steps 1 to 7
above compare against MATLAB and need the exports.

## Running the ladder with the deposited references

The reference exports for the Big Tujunga catchment and the Brazil
4000 × 4000 window are in the paper's Zenodo data deposit,
doi:10.5281/zenodo.14748886 (https://doi.org/10.5281/zenodo.14748886). Unzip
`reference_exports_bigtujunga.zip` and `reference_exports_brazil4000.zip`
so that `reference_data/bigtujunga/manifest.json` and
`reference_data/brazil4000/manifest.json` exist under the repository
root, then run `pytest tests -q`; the ladder tests pick them up
automatically (about 20 minutes for both).

## Validating your own DEM against MATLAB

1. In MATLAB, with TopoToolbox 3 on the path, export the reference
   products for your DEM (`thr_cells` is the stream threshold in cells):

   ```matlab
   addpath('matlab_scripts')
   export_reference('path/to/dem.tif', 'reference_data/mydem', thr_cells)
   export_T0_nodrop('path/to/dem.tif', 'reference_data/mydem', thr_cells)
   ```

   `export_external_refs.m` shows the same two calls looped over several
   DEMs.

2. Run `pytest tests -q`. Every ladder test is parametrised over every
   directory under `reference_data/` that holds a `manifest.json`, so
   the new DEM is picked up without any configuration.

`matlab_scripts/check_divide_stages.m` is a stand-alone check that
runs entirely inside MATLAB: it builds the raw divide geometry with the
`DIVIDEobj` constructor, runs `divnet` and `sort` on it separately, and
reports the edges each stage removes and where, optionally on the flow
edges of a reference export so that two TopoToolbox versions see the
same flow network. It involves no `pydivide` code and is useful for
checking a TopoToolbox version independently.

## What the ladder does not establish, measured

Steps 6 and 7 do not establish that `pydivide`'s own sort reproduces
the reference's distance and order values, and it does not: the
reference's values depend on its walk order where the sort has a
choice, and `pydivide` makes those choices deterministically. The
departure was measured on the nine DEMs on the same network the ladder
uses (`own_sort_vs_reference.py` in the paper's data deposit): the
segments the two sorts direct oppositely (every one either has an
endpoint at both ends or joins two junctions that become ready in the
same wave; no DEM has a closed component), the node instances whose
divide distance or order differs, and how many of the differing values
lie on an oppositely directed segment or on a segment sorted after it,
counted one instance per unit of the multiset difference so that every
column counts the same population. The remaining differences are
attributed to the order in which the segments of one wave append their
seeds; that attribution is an interpretation and was not traced value
by value. Aligning the direction of the both-endpoint segments to the
reference's changes no count.

| landscape | segments | directed oppositely (two endpoints + two junctions) | node instances | distance differing | Strahler / Shreve / topological differing | distance differences on or after an opposite segment | order differences on or after an opposite segment |
|---|---:|---:|---:|---:|---:|---:|---:|
| Big Tujunga | 1,349 | 3 (3 + 0) | 53,066 | 8 (0.02 %) | 0 / 0 / 0 | 100 % | 0 of 0 |
| Brazil 4000 × 4000 | 22,975 | 308 (32 + 276) | 1,018,846 | 21,506 (2.11 %) | 15,803 / 17,654 / 17,466 | 96 % | 50,907 of 50,923 |
| Vermilion Cliffs | 4,905 | 12 (11 + 1) | 244,931 | 314 (0.13 %) | 63 / 63 / 0 | 73 % | 126 of 126 |
| Black Mesa | 9,858 | 51 (49 + 2) | 444,929 | 1,502 (0.34 %) | 0 / 9 / 0 | 75 % | 9 of 9 |
| Vindhya Range | 20,025 | 35 (34 + 1) | 903,355 | 2,150 (0.24 %) | 0 / 0 / 0 | 58 % | 0 of 0 |
| Chapada do Araripe | 43,041 | 107 (101 + 6) | 1,997,936 | 8,382 (0.42 %) | 88 / 230 / 0 | 57 % | 315 of 318 |
| Drakensberg 6000 × 6000 | 54,277 | 82 (82 + 0) | 2,262,509 | 7,682 (0.34 %) | 0 / 0 / 0 | 82 % | 0 of 0 |
| Siwa and El Dabaa | 58,906 | 217 (210 + 7) | 3,297,676 | 29,018 (0.88 %) | 43 / 43 / 0 | 65 % | 84 of 86 |
| Serra Geral de Goiás | 144,795 | 487 (476 + 11) | 6,953,006 | 25,337 (0.36 %) | 68 / 224 / 0 | 68 % | 292 of 292 |
| all nine | 360,131 | 1,302 (998 + 304) | 17,176,254 | 95,899 (0.56 %) | 16,065 / 18,223 / 17,466 | 74 % | 51,733 of 51,754 |

On the reference-conditioned networks used for this comparison,
`pydivide`'s own sort gives the agreement measured above, which
isolates the effect of the sort alone: distance and order are the
reference's definitions computed on `pydivide`'s own topological
order, and where the two differ the reference's own value is the one
that depends on the order of its walk. The comparison does not
establish the same agreement rate for an end-to-end production run,
whose routing and edge set are `pydivide`'s own and differ from the
reference's by the routing disagreement and the reference's documented
edge defects.
