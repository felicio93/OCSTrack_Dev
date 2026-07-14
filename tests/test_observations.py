"""Tests for SatelliteData and ArgoData observation classes."""

import os
import tempfile

import numpy as np
import pytest
import xarray as xr

from ocstrack.Observation.satellite import SatelliteData
from ocstrack.Observation.argofloat import ArgoData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_satellite_nc(path, n=20):
    """Write a minimal satellite altimetry NetCDF file."""
    times = np.arange(
        np.datetime64("2023-04-01"),
        np.datetime64("2023-04-01") + np.timedelta64(n, "h"),
        np.timedelta64(1, "h"),
    ).astype("datetime64[ns]")
    ds = xr.Dataset(
        {
            "swh": ("time", np.random.rand(n).astype(np.float32)),
            "sla": ("time", np.random.rand(n).astype(np.float32)),
            "lat": ("time", np.linspace(30, 40, n).astype(np.float32)),
            "lon": ("time", np.linspace(-80, -70, n).astype(np.float32)),
            "source": ("time", np.full(n, "test_sat", dtype=object)),
        },
        coords={"time": times},
    )
    ds.to_netcdf(path)
    return ds


def _write_argo_nc(path, n_prof=5, n_levels=50):
    """Write a minimal Argo profile NetCDF file."""
    base_time = np.datetime64("2023-04-01", "ns")
    juld = np.array(
        [base_time + np.timedelta64(i * 12, "h") for i in range(n_prof)],
        dtype="datetime64[ns]",
    )
    ds = xr.Dataset(
        {
            "JULD": ("N_PROF", juld),
            "LATITUDE": ("N_PROF", np.linspace(30, 35, n_prof)),
            "LONGITUDE": ("N_PROF", np.linspace(-75, -70, n_prof)),
            "PRES": (
                ("N_PROF", "N_LEVELS"),
                np.tile(np.linspace(0, 2000, n_levels), (n_prof, 1)).astype(np.float32),
            ),
            "TEMP": (
                ("N_PROF", "N_LEVELS"),
                np.random.rand(n_prof, n_levels).astype(np.float32),
            ),
            "PSAL": (
                ("N_PROF", "N_LEVELS"),
                np.random.rand(n_prof, n_levels).astype(np.float32),
            ),
        },
    )
    ds.to_netcdf(path)
    return ds


# ---------------------------------------------------------------------------
# SatelliteData tests
# ---------------------------------------------------------------------------

class TestSatelliteData:
    @pytest.fixture
    def sat_file(self, tmp_path):
        p = str(tmp_path / "sat.nc")
        _write_satellite_nc(p)
        return p

    def test_init_loads_dataset(self, sat_file):
        sd = SatelliteData(sat_file)
        assert sd.ds is not None
        assert sd.ds.sizes["time"] == 20

    def test_properties_return_arrays(self, sat_file):
        sd = SatelliteData(sat_file)
        assert isinstance(sd.time, np.ndarray)
        assert isinstance(sd.lon, np.ndarray)
        assert isinstance(sd.lat, np.ndarray)
        assert isinstance(sd.swh, np.ndarray)
        assert isinstance(sd.sla, np.ndarray)
        assert isinstance(sd.source, np.ndarray)

    def test_filter_by_time_reduces_size(self, sat_file):
        sd = SatelliteData(sat_file)
        original_size = sd.ds.sizes["time"]
        sd.filter_by_time("2023-04-01T05", "2023-04-01T10")
        assert sd.ds.sizes["time"] < original_size
        assert sd.ds.sizes["time"] > 0

    def test_filter_by_time_empty_result(self, sat_file):
        sd = SatelliteData(sat_file)
        sd.filter_by_time("2025-01-01", "2025-01-02")
        assert sd.ds.sizes["time"] == 0

    def test_missing_required_variable_raises(self, tmp_path):
        """A file missing the 'source' variable should raise ValueError."""
        p = str(tmp_path / "sat_incomplete.nc")
        times = np.array(["2023-04-01"], dtype="datetime64[ns]")
        ds = xr.Dataset(
            {
                "swh": ("time", [1.0]),
                "sla": ("time", [0.1]),
                "lat": ("time", [35.0]),
                # 'lon' and 'source' intentionally omitted
            },
            coords={"time": times},
        )
        ds.to_netcdf(p)
        with pytest.raises(ValueError, match="Missing required variables"):
            SatelliteData(p)

    def test_lon_setter_correct_size(self, sat_file):
        sd = SatelliteData(sat_file)
        new_lon = np.linspace(-90, -60, 20)
        sd.lon = new_lon
        np.testing.assert_allclose(sd.lon, new_lon)

    def test_lon_setter_wrong_size_raises(self, sat_file):
        sd = SatelliteData(sat_file)
        with pytest.raises(ValueError, match="must match"):
            sd.lon = np.zeros(999)

    def test_lat_setter_wrong_size_raises(self, sat_file):
        sd = SatelliteData(sat_file)
        with pytest.raises(ValueError, match="must match"):
            sd.lat = np.zeros(999)


# ---------------------------------------------------------------------------
# ArgoData tests
# ---------------------------------------------------------------------------

class TestArgoData:
    @pytest.fixture
    def argo_dir(self, tmp_path):
        for i in range(3):
            p = str(tmp_path / f"argo_{i:04d}.nc")
            _write_argo_nc(p, n_prof=4, n_levels=30)
        return str(tmp_path)

    def test_init_loads_and_concatenates(self, argo_dir):
        ad = ArgoData(argo_dir)
        assert ad.ds is not None
        # 3 files × 4 profiles = 12 (minus any duplicates)
        assert ad.ds.sizes["JULD"] > 0

    def test_properties_return_arrays(self, argo_dir):
        ad = ArgoData(argo_dir)
        assert isinstance(ad.time, np.ndarray)
        assert isinstance(ad.lon, np.ndarray)
        assert isinstance(ad.lat, np.ndarray)
        assert isinstance(ad.pres, np.ndarray)
        assert isinstance(ad.temp, np.ndarray)
        assert isinstance(ad.psal, np.ndarray)

    def test_depth_is_negative_of_pressure(self, argo_dir):
        ad = ArgoData(argo_dir)
        np.testing.assert_allclose(ad.depth, ad.pres * -1.0197)

    def test_filter_by_time_reduces_profiles(self, argo_dir):
        ad = ArgoData(argo_dir)
        original = ad.ds.sizes["JULD"]
        ad.filter_by_time("2023-04-01T00", "2023-04-01T12")
        assert ad.ds.sizes["JULD"] <= original

    def test_empty_directory_raises(self, tmp_path):
        empty_dir = str(tmp_path / "empty")
        os.makedirs(empty_dir)
        with pytest.raises(ValueError, match="No .nc files found"):
            ArgoData(empty_dir)

    def test_dataset_sorted_by_juld(self, argo_dir):
        ad = ArgoData(argo_dir)
        juld = ad.time
        assert np.all(np.diff(juld.astype(np.int64)) >= 0), "JULD is not sorted"

    def test_lon_setter_correct_size(self, argo_dir):
        ad = ArgoData(argo_dir)
        n = ad.ds.sizes["JULD"]
        new_lon = np.linspace(-80, -70, n)
        ad.lon = new_lon
        np.testing.assert_allclose(ad.lon, new_lon)

    def test_lon_setter_wrong_size_raises(self, argo_dir):
        ad = ArgoData(argo_dir)
        with pytest.raises(ValueError, match="must match"):
            ad.lon = np.zeros(999)

    def test_lat_setter_wrong_size_raises(self, argo_dir):
        ad = ArgoData(argo_dir)
        with pytest.raises(ValueError, match="must match"):
            ad.lat = np.zeros(999)

    def test_padding_to_max_levels(self, tmp_path):
        """Files with different N_LEVELS should be padded and concatenated cleanly."""
        for i, n_lev in enumerate([20, 40]):
            _write_argo_nc(str(tmp_path / f"argo_{i}.nc"), n_prof=2, n_levels=n_lev)
        ad = ArgoData(str(tmp_path))
        # All profiles should have N_LEVELS == 40 (the max)
        assert ad.ds.sizes["N_LEVELS"] == 40
