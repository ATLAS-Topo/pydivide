# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Daniel Peifer
"""Tests for the one-call across-divide API, the diagnostics and the plots.

Three levels. Synthetic surfaces pin the definitions and the sign
conventions, where the right answer is known by construction. Big
Tujunga, the TopoToolbox example DEM, exercises the whole path on real
topography and is fast enough to run in CI. The published Swabian Alb
and south-eastern Brazil products are checked where they exist, and
skipped where they do not, so the same file serves a laptop and this
machine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

tt = pytest.importorskip("topotoolbox")

from pydivide.acrossdivide import (  # noqa: E402
    across_divide_metrics,
    across_divide_multiscale,
    coherence,
)

# --------------------------------------------------------------------------
# definitions and conventions, on surfaces whose answer is known
# --------------------------------------------------------------------------

def test_coherence_is_one_when_the_sign_never_changes():
    cols = {"x_sgn": np.array([3.0, -4.0, 0.5]), "x_diff": np.array([3.0, 4.0, 0.5])}
    c = coherence(cols, "x")
    assert np.allclose(c, 1.0)


def test_coherence_falls_when_the_contrast_cancels():
    # a piece whose signed mean is 0 while its absolute mean is 4 has
    # cancelled completely and must score 0
    cols = {"x_sgn": np.array([0.0, 2.0]), "x_diff": np.array([4.0, 4.0])}
    c = coherence(cols, "x")
    assert c[0] == pytest.approx(0.0)
    assert c[1] == pytest.approx(0.5)


def test_coherence_is_nan_where_there_is_no_contrast():
    cols = {"x_sgn": np.array([0.0]), "x_diff": np.array([0.0])}
    assert np.isnan(coherence(cols, "x")[0])


def test_coherence_absent_family_returns_none():
    assert coherence({"a_sgn": np.zeros(1)}, "b") is None


def test_chi_without_a_baselevel_is_refused():
    """chi has no safe default base level and must say so, not guess."""
    dem = tt.load_dem("bigtujunga")
    with pytest.raises(ValueError) as e:
        across_divide_metrics(dem, metrics=("chi",), threshold=1e5)
    msg = str(e.value).lower()
    assert "baselevel" in msg and "sign" in msg


def test_metrics_without_chi_need_no_baselevel():
    dem = tt.load_dem("bigtujunga")
    t = across_divide_metrics(dem, metrics=("elev",), threshold=1e5, seglength=600.0)
    assert len(t) > 0
    assert t.settings["baselevel_m"] is None


# --------------------------------------------------------------------------
# the whole path, on real topography
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bigtujunga_table():
    dem = tt.load_dem("bigtujunga")
    return across_divide_metrics(dem, threshold=1e5, baselevel=200.0,
                                 seglength=300.0)


def test_every_requested_family_is_present(bigtujunga_table):
    t = bigtujunga_table
    for fam in ("elev", "slope", "hr", "fd", "chi"):
        for suf in ("mean", "diff", "min", "max", "sgn"):
            assert f"{fam}_{suf}" in t.columns, f"{fam}_{suf} missing"
    for extra in ("dai", "basin_a", "basin_b", "segid"):
        assert extra in t.columns


def test_columns_all_have_one_value_per_piece(bigtujunga_table):
    t = bigtujunga_table
    for k, v in t.columns.items():
        assert len(v) == len(t), f"{k} has {len(v)} values for {len(t)} pieces"


def test_the_identity_that_defines_the_signed_column(bigtujunga_table):
    """|sgn| == diff per edge, so per piece |sgn| can only be <= diff."""
    t = bigtujunga_table
    for fam in ("elev", "hr", "chi"):
        s = np.abs(np.asarray(t.columns[f"{fam}_sgn"], float))
        d = np.asarray(t.columns[f"{fam}_diff"], float)
        ok = np.isfinite(s) & np.isfinite(d)
        assert np.all(s[ok] <= d[ok] + 1e-9), f"{fam}: |sgn| exceeded diff"


def test_min_is_never_above_max(bigtujunga_table):
    t = bigtujunga_table
    for fam in ("elev", "hr", "chi"):
        lo = np.asarray(t.columns[f"{fam}_min"], float)
        hi = np.asarray(t.columns[f"{fam}_max"], float)
        ok = np.isfinite(lo) & np.isfinite(hi)
        assert np.all(lo[ok] <= hi[ok] + 1e-9)


def test_mean_lies_between_the_two_sides(bigtujunga_table):
    t = bigtujunga_table
    lo = np.asarray(t.columns["elev_min"], float)
    hi = np.asarray(t.columns["elev_max"], float)
    mu = np.asarray(t.columns["elev_mean"], float)
    ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(mu)
    assert np.all(mu[ok] >= lo[ok] - 1e-6)
    assert np.all(mu[ok] <= hi[ok] + 1e-6)


def test_sides_are_distinct_basins_wherever_both_are_set(bigtujunga_table):
    t = bigtujunga_table
    a = np.asarray(t.columns["basin_a"], float)
    b = np.asarray(t.columns["basin_b"], float)
    ok = np.isfinite(a) & np.isfinite(b)
    assert ok.sum() > 0
    assert np.all(a[ok] != b[ok])


def test_dai_is_the_published_formula(bigtujunga_table):
    t = bigtujunga_table
    d = np.asarray(t.columns["dai"], float)
    expect = (np.asarray(t.columns["hr_diff"], float)
              / (2.0 * np.asarray(t.columns["hr_mean"], float)))
    ok = np.isfinite(d) & np.isfinite(expect)
    assert np.allclose(d[ok], expect[ok])
    assert np.all(d[ok] >= -1e-9)


def test_settings_record_every_choice(bigtujunga_table):
    s = bigtujunga_table.settings
    for k in ("threshold", "mn", "a0", "baselevel_m", "seglength_m", "cellsize"):
        assert k in s
    assert s["baselevel_m"] == 200.0
    assert s["mn"] == 0.45


def test_describe_states_the_chi_parameters(bigtujunga_table):
    d = bigtujunga_table.describe()
    for token in ("base level", "concavity", "0.45", "threshold"):
        assert token in d


def test_disagreement_counts_are_consistent(bigtujunga_table):
    d = bigtujunga_table.disagreement()
    votes = d["n_indicators_agreeing_with_chi"]
    avail = d["n_indicators_compared"]
    assert np.all(votes <= avail)
    assert np.all(d["all_agree"] == ((avail > 0) & (votes == avail)))
    assert np.all(d["all_disagree"] == ((avail > 0) & (votes == 0)))
    assert avail.max() >= 1


def test_seglength_none_gives_one_row_per_segment():
    dem = tt.load_dem("bigtujunga")
    t = across_divide_metrics(dem, threshold=1e5, metrics=("elev",), seglength=None)
    assert len(t) == t.settings["n_segments"]


def test_a_shorter_piece_gives_more_pieces():
    dem = tt.load_dem("bigtujunga")
    a = across_divide_metrics(dem, threshold=1e5, metrics=("elev",), seglength=1200.0)
    b = across_divide_metrics(dem, threshold=1e5, metrics=("elev",), seglength=300.0)
    assert len(b) > len(a)


# --------------------------------------------------------------------------
# the multiscale ladder
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def multiscale_table():
    dem = tt.load_dem("bigtujunga")
    return across_divide_multiscale(dem, thresholds=(1e5, 1e6), baselevel=200.0,
                                    metrics=("hr", "chi"), seglength=600.0)


def test_multiscale_carries_a_column_per_scale(multiscale_table):
    t = multiscale_table
    assert any(k.endswith("1e5") for k in t.columns)
    assert any(k.endswith("1e6") for k in t.columns)


def test_multiscale_consistency_flag_matches_the_per_scale_signs(multiscale_table):
    t = multiscale_table
    a = np.asarray(t.columns["chi_sgn_1e5"], float)
    b = np.asarray(t.columns["chi_sgn_1e6"], float)
    ok = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    expect = np.zeros(len(a), bool)
    expect[ok] = np.sign(a[ok]) == np.sign(b[ok])
    got = np.asarray(t.columns["chi_sign_consistent"], bool)
    assert np.array_equal(got[ok], expect[ok])


def test_multiscale_peak_scale_is_one_of_the_scales(multiscale_table):
    pk = np.asarray(multiscale_table.columns["chi_peak_scale"], float)
    fin = pk[np.isfinite(pk)]
    assert set(np.unique(fin)).issubset({1e5, 1e6})


def test_multiscale_network_is_held_fixed(multiscale_table):
    """Every scale must describe the same divides, or nothing is comparable."""
    t = multiscale_table
    assert t.settings["network_threshold"] == 1e6
    n = len(t)
    for k, v in t.columns.items():
        assert len(v) == n


# --------------------------------------------------------------------------
# any metrics table on this machine: set PYDIVIDE_METRICS_TABLES to one or
# more paths (separated by the OS path separator) of tables written by
# across_divide_metrics or by the paper's pipeline, and the invariants
# above are checked on their first 200,000 rows
# --------------------------------------------------------------------------

import os  # noqa: E402

PRODUCTS = {Path(p).name: Path(p) for p in
            os.environ.get("PYDIVIDE_METRICS_TABLES", "").split(os.pathsep) if p}


def _families(columns):
    """(family, prefix) pairs for every signed family in the table: the
    column ``<prefix>_sgn`` with its ``<prefix>_diff``; the prefix is the
    family name, optionally followed by the threshold exponent the
    paper's tables carry (``chi6``)."""
    import re
    out = []
    for c in columns:
        m = re.fullmatch(r"([a-z]+)(\d*)_sgn", c)
        if m and f"{m.group(1)}{m.group(2)}_diff" in columns:
            out.append((m.group(1), f"{m.group(1)}{m.group(2)}"))
    return out


def _read(path, cols=None):
    pd = pytest.importorskip("pandas")
    if path.suffix == ".csv":
        head = pd.read_csv(path, nrows=0)
        use = [c for c in (cols or head.columns) if c in head.columns]
        return next(pd.read_csv(path, usecols=use, chunksize=200_000, low_memory=False))
    pyogrio = pytest.importorskip("pyogrio")
    have = list(pyogrio.read_info(path)["fields"])
    use = [c for c in (cols or have) if c in have]
    return pyogrio.read_dataframe(path, columns=use, max_features=200_000, read_geometry=False)


@pytest.mark.parametrize("name", sorted(PRODUCTS))
def test_metrics_tables_obey_the_same_identities(name):
    """The invariants above must hold on any table written by the package."""
    path = PRODUCTS[name]
    if not path.exists():
        pytest.skip(f"{name} not on this machine")
    df = _read(path)
    fams = _families(set(df.columns))
    assert fams, f"{name}: no <family>_sgn/_diff column pair"
    for _, pre in fams:
        s = df[f"{pre}_sgn"].abs().to_numpy(float)
        d = df[f"{pre}_diff"].to_numpy(float)
        ok = np.isfinite(s) & np.isfinite(d)
        assert np.all(s[ok] <= d[ok] * (1 + 1e-9) + 1e-9), f"{name}: |sgn| > diff"
    if {"elev_min", "elev_max"} <= set(df.columns):
        lo = df["elev_min"].to_numpy(float)
        hi = df["elev_max"].to_numpy(float)
        ok = np.isfinite(lo) & np.isfinite(hi)
        assert np.all(lo[ok] <= hi[ok] + 1e-6)
    if {"basin_a", "basin_b"} <= set(df.columns):
        a = df["basin_a"].to_numpy(float)
        b = df["basin_b"].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        assert np.all(a[ok] != b[ok]), f"{name}: a side equals its own counterpart"
    for fam, pre in fams:
        if fam != "hr":
            continue
        dai = "dai" + pre[2:]
        if {dai, f"{pre}_diff", f"{pre}_mean"} <= set(df.columns):
            d = df[dai].to_numpy(float)
            e = df[f"{pre}_diff"].to_numpy(float) / (2.0 * df[f"{pre}_mean"].to_numpy(float))
            ok = np.isfinite(d) & np.isfinite(e)
            assert np.allclose(d[ok], e[ok], rtol=1e-6, atol=1e-9), f"{name}: DAI formula"


@pytest.mark.parametrize("name", sorted(PRODUCTS))
def test_metrics_tables_coherence_is_within_bounds(name):
    path = PRODUCTS[name]
    if not path.exists():
        pytest.skip(f"{name} not on this machine")
    df = _read(path)
    chi = [pre for fam, pre in _families(set(df.columns)) if fam == "chi"]
    if not chi:
        pytest.skip(f"{name}: no chi family")
    c = coherence({"chi_sgn": df[f"{chi[0]}_sgn"].to_numpy(float),
                   "chi_diff": df[f"{chi[0]}_diff"].to_numpy(float)}, "chi")
    fin = c[np.isfinite(c)]
    assert fin.size > 0
    assert fin.min() >= 0.0 and fin.max() <= 1.0
