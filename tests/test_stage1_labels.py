# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Validation ladder step 1: basin label raster L.

Gate: L identical to MATLAB's, allowing a permutation of label values.
Because pydivide mirrors MATLAB's labelreach numbering exactly, the reach
NAL is additionally compared for strict equality — a stronger check that
costs nothing while it holds.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydivide.labels import basin_labels  # noqa: E402


@pytest.fixture(scope="module")
def computed(ref, net):
    L, nal = basin_labels(net)
    return net, L, nal


def test_labelreach_nal_exact(ref, computed):
    _, _, nal = computed
    np.testing.assert_array_equal(nal, ref.labelreach_nal())


def test_L_nan_pattern_identical(ref, computed):
    """pydivide uses -1 where MATLAB uses NaN (int32 raster, NOTES)."""
    _, L, _ = computed
    Lm = ref.L_grid()
    np.testing.assert_array_equal(L == -1, np.isnan(Lm))


def test_L_identical_up_to_permutation(ref, computed):
    _, L, _ = computed
    Lm = ref.L_grid()
    ok = L != -1
    a = L[ok].astype(np.int64)
    b = Lm[ok].astype(np.int64)
    pairs = np.unique(np.stack([a, b], axis=1), axis=0)
    # bijection: each label on either side appears in exactly one pair
    assert len(np.unique(pairs[:, 0])) == len(pairs), \
        "a pydivide label maps to several MATLAB labels"
    assert len(np.unique(pairs[:, 1])) == len(pairs), \
        "a MATLAB label maps to several pydivide labels"


def test_L_exact_equality_bonus(ref, computed):
    """Stronger than the gate requires: with mirrored numbering the rasters
    should be strictly identical. If this ever fails while the permutation
    test passes, the gate still holds; investigate but do not block."""
    _, L, _ = computed
    Lm = ref.L_grid()
    ok = L != -1
    assert np.array_equal(L[ok].astype(np.float64), Lm[ok])
