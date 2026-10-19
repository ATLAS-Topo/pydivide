# Third-party material

`pydivide` is an independent Python reimplementation and extension of the
drainage-divide method of TopoToolbox 3 `DIVIDEobj` (Scherler and
Schwanghart, 2020). The Python package was written for this project by
reading the published method and the GPL-licensed MATLAB source; it
contains no MATLAB code and no code copied from TopoToolbox.

One file in `matlab_scripts/` reproduces reference code verbatim, for
validation only:

| file | what it reproduces | source |
|---|---|---|
| `matlab_scripts/export_T0_nodrop.m` | the geometry section of the `DIVIDEobj` constructor, run to the point before the single-edge-run drop, so that the pre-drop edge list can be exported | TopoToolbox 3, `@DIVIDEobj/DIVIDEobj.m`, commit `1bc93e7` (2025-12-10), https://github.com/TopoToolbox/topotoolbox3, Copyright (C) Dirk Scherler and Wolfgang Schwanghart, GNU GPL v3 or later |

The other three MATLAB scripts call TopoToolbox functions and contain no
copied code. TopoToolbox 3 is not redistributed here; the scripts expect
a checkout on the MATLAB path.

Both TopoToolbox 3 and `pydivide` are licensed under the GNU General
Public License v3 or later.
