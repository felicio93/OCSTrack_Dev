"""Tests for the collocated-output writer helpers in Collocation/output.py."""

import numpy as np
import pytest
import xarray as xr

from ocstrack.Collocation.output import (
    get_max_neighbors,
    pad_arrays_to_max,
    make_collocated_nc_2d,
    make_collocated_nc_3d,
)


# ---------------------------------------------------------------------------
# get_max_neighbors
# ---------------------------------------------------------------------------

class TestGetMaxNeighbors:
    def test_empty_list_returns_one(self):
        assert get_max_neighbors([]) == 1

    def test_no_valid_arrays_returns_one(self):
        # 1-D arrays are not counted as valid (need ndim == 2)
        assert get_max_neighbors([np.array([1, 2, 3])]) == 1

    def test_returns_max_columns(self):
        arrs = [np.zeros((5, 2)), np.zeros((3, 4)), np.zeros((2, 3))]
        assert get_max_neighbors(arrs) == 4


# ---------------------------------------------------------------------------
# pad_arrays_to_max
# ---------------------------------------------------------------------------

class TestPadArraysToMax:
    def test_pads_to_max_cols(self):
        arrs = [np.ones((2, 1)), np.ones((3, 3))]
        out = pad_arrays_to_max(arrs, 3)
        assert out.shape == (5, 3)
        # First two rows padded with NaN in cols 1 and 2
        assert np.isnan(out[0, 1])
        assert np.isnan(out[0, 2])

    def test_reshapes_1d(self):
        out = pad_arrays_to_max([np.array([1.0, 2.0, 3.0])], 2)
        # 1-D (3,) -> (3, 1) -> padded to (3, 2)
        assert out.shape == (3, 2)

    def test_truncates_when_larger(self):
        out = pad_arrays_to_max([np.ones((2, 5))], 3)
        assert out.shape == (2, 3)

    def test_empty_input_returns_empty(self):
        out = pad_arrays_to_max([], 4)
        assert out.shape == (0, 4)


# ---------------------------------------------------------------------------
# make_collocated_nc_2d
# ---------------------------------------------------------------------------

def _build_2d_results(n=4, k=3):
    """Build a minimal 2D results dict as produced by the surface collocator."""
    return {
        "time_obs": [np.arange(n).astype("datetime64[s]")],
        "lat_obs": [np.linspace(30, 40, n)],
        "lon_obs": [np.linspace(-80, -70, n)],
        "time_deltas": [np.zeros(n)],
        "bias_raw": [np.zeros(n)],
        "bias_weighted": [np.zeros(n)],
        "model_sigWaveHeight_weighted": [np.ones(n)],
        "obs_swh": [np.ones(n)],
        "model_sigWaveHeight": [np.ones((n, k))],
        "model_dpt": [np.full((n, k), np.nan)],
        "dist_deltas": [np.ones((n, k))],
        "node_idx": [np.zeros((n, k))],
    }


class TestMakeCollocatedNC2D:
    def test_empty_results_returns_empty_dataset(self):
        ds = make_collocated_nc_2d({"time_obs": []})
        assert isinstance(ds, xr.Dataset)
        assert len(ds.data_vars) == 0

    def test_builds_dataset(self):
        results = _build_2d_results(n=4, k=3)
        ds = make_collocated_nc_2d(
            results, n_nearest=3,
            model_var_name="sigWaveHeight", obs_var_name="swh",
        )
        assert ds.sizes["time"] == 4
        assert ds.sizes["nearest_nodes"] == 3
        assert "model_sigWaveHeight" in ds
        assert "obs_swh" in ds
        assert ds.attrs["Conventions"] == "CF-1.7"

    def test_extra_obs_vars_carried_through(self):
        results = _build_2d_results(n=4, k=3)
        results["obs_distance_to_coast"] = [np.arange(4).astype(float)]
        ds = make_collocated_nc_2d(
            results, n_nearest=3,
            model_var_name="sigWaveHeight", obs_var_name="swh",
        )
        assert "obs_distance_to_coast" in ds

    def test_infers_max_neighbors_when_none(self):
        results = _build_2d_results(n=4, k=2)
        ds = make_collocated_nc_2d(
            results, n_nearest=None,
            model_var_name="sigWaveHeight", obs_var_name="swh",
        )
        assert ds.sizes["nearest_nodes"] == 2


# ---------------------------------------------------------------------------
# make_collocated_nc_3d
# ---------------------------------------------------------------------------

class TestMakeCollocatedNC3D:
    def test_empty_returns_empty_dataset(self):
        results = {"time": np.array([], dtype="datetime64[s]")}
        ds = make_collocated_nc_3d(results, max_levels=10)
        assert isinstance(ds, xr.Dataset)
        assert len(ds.data_vars) == 0

    def test_builds_profile_dataset(self):
        n, levels, k = 3, 5, 4
        results = {
            "time": np.arange(n).astype("datetime64[s]"),
            "lat": np.linspace(30, 35, n),
            "lon": np.linspace(-75, -70, n),
            "time_deltas": np.zeros(n),
            "dist_deltas": np.ones((n, k)),
            "node_idx": np.zeros((n, k)),
            "argo_depth": np.zeros((n, levels)),
            "argo_temp": np.ones((n, levels)),
            "model_temperature": np.ones((n, levels)),
        }
        ds = make_collocated_nc_3d(results, max_levels=levels)
        assert ds.sizes["time"] == n
        assert ds.sizes["n_levels"] == levels
        assert ds.sizes["nearest_nodes"] == k
        assert "argo_temp" in ds
        assert "model_temperature" in ds
