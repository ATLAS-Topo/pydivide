# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Shared fixtures: every validation gate runs against every MATLAB
reference export found under reference_data/, and skips when there is
none."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REF_ROOT = Path(__file__).resolve().parents[1] / "reference_data"


def _available() -> list[str]:
    if not REF_ROOT.exists():
        return []
    return sorted(d.name for d in REF_ROOT.iterdir()
                  if (d / "manifest.json").exists())


_datasets = _available()


@pytest.fixture(scope="session", params=_datasets or ["__none__"])
def ref_dir(request):
    if request.param == "__none__":
        pytest.skip("no MATLAB reference exports generated yet")
    return REF_ROOT / request.param


@pytest.fixture(scope="session")
def ref(ref_dir):
    from pydivide.matlab_ref import MatlabReference
    return MatlabReference(ref_dir)


@pytest.fixture(scope="session")
def net(ref):
    from pydivide.flownet import FlowNet
    return FlowNet.from_matlab_reference(ref)
