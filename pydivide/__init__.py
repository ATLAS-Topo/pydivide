# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""pydivide — drainage divide networks in Python.

Reimplements the semantics of MATLAB TopoToolbox 3 DIVIDEobj
(Scherler & Schwanghart 2020, Earth Surf. Dynam. 8, 245-259) with int64
indexing throughout and O(N)/O(N log N) algorithms.
"""

from .acrossdivide import (
    AcrossDivideTable,
    across_divide_metrics,
    across_divide_multiscale,
    coherence,
    metric_column,
    read_export_settings,
)
from .basinpairs import (
    basin_points,
    basin_polygons,
    decode_pair_key,
    multiscale_pair_ids,
    pair_ids,
    pair_key,
    pair_side_points,
    pair_table,
    readable_pair_key,
)
from .checkpoint import StageStore
from .divmetrics import asymmetry, getvalue, orient_by_basin
from .gridspec import (
    GridSpec,
    assert_int64,
    matlab_cell_to_py,
    matlab_node_to_py,
    node_id,
    node_rc,
    py_cell_to_matlab,
    py_node_to_matlab,
)
from .labels import reach_basin_labels
from .matlab_ref import MatlabReference
from .pipeline import DivideNetworkResult, divide_network
from .profiles import BasinPairProfiles, PairPath, basin_pair_profiles, basin_pairs

try:                                   # one source of truth: the package metadata
    from importlib.metadata import version as _version
    __version__ = _version("pydivide")
except Exception:                      # not installed (run from a checkout)
    __version__ = "0.1.1"

__all__ = [
    "GridSpec",
    "node_id",
    "node_rc",
    "matlab_cell_to_py",
    "matlab_node_to_py",
    "py_cell_to_matlab",
    "py_node_to_matlab",
    "assert_int64",
    "StageStore",
    "MatlabReference",
    "divide_network",
    "DivideNetworkResult",
    "getvalue",
    "asymmetry",
    "orient_by_basin",
    "across_divide_metrics",
    "across_divide_multiscale",
    "AcrossDivideTable",
    "coherence",
    "metric_column",
    "read_export_settings",
    "basin_pair_profiles",
    "BasinPairProfiles",
    "PairPath",
    "basin_pairs",
    "reach_basin_labels",
    "pair_ids",
    "pair_key",
    "decode_pair_key",
    "readable_pair_key",
    "pair_table",
    "multiscale_pair_ids",
    "basin_points",
    "basin_polygons",
    "pair_side_points",
]
