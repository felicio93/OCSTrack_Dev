"""Tests for SatelliteData and ArgoData observation classes."""

import os

import numpy as np
import pytest
import xarray as xr

from ocstrack.Observation.satellite import SatelliteData, _detect_format
from ocstrack.Observation.argofloat import ArgoData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_satellite_nc(path, n=20):
    """Write a minimal CoastWatch-style satellite altimetry NetCDF file."""
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


def _write_cci_satellite_nc(path, n=20, include_8m_correction=True,
                             include_adjusted=True):
    """Write a minimal ESA CCI-style satellite altimetry NetCDF file."""
    times = np.arange(
        np.datetime64("2023-01-16"),
        np.datetime64("2023-01-16") + np.timedelta64(n, "h"),
        np.timedelta64(1, "h"),
    ).astype("datetime64[ns]")
    data_vars = {
        "swh": ("time", np.random.rand(n).astype(np.float64)),
        "lat": ("time", np.linspace(30, 40, n).astype(np.float64)),
        "lon": ("time", np.linspace(-80, -70, n).astype(np.float64)),
        "swh_quality_level": ("time", np.ones(n, dtype=np.int8)),
        "swh_uncertainty": ("time", np.random.rand(n).astype(np.float64) * 0.5),
        "bathymetry": ("time", np.random.rand(n).astype(np.float64) * -1000),
        "distance_to_coast": ("time", np.random.rand(n).astype(np.float64) * 200_000),
    }
    if include_adjusted:
        data_vars["swh_adjusted"] = (
            "time", np.random.rand(n).astype(np.float64)
        )
    if include_8m_correction:
        data_vars["swh_with_8m_offset_correction"] = (
            "time", np.random.rand(n).astype(np.float64) * 15
        )
    ds = xr.Dataset(data_vars, coords={"time": times})
    ds.attrs["title"] = "ESA CCI Sea State L2P from test altimeter"
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
        """A file missing 'lon' should raise ValueError."""
        p = str(tmp_path / "sat_incomplete.nc")
        times = np.array(["2023-04-01"], dtype="datetime64[ns]")
        ds = xr.Dataset(
            {
                "swh": ("time", [1.0]),
                "sla": ("time", [0.1]),
                "lat": ("time", [35.0]),
                # 'lon' intentionally omitted
            },
            coords={"time": times},
        )
        ds.to_netcdf(p)
        with pytest.raises(ValueError, match="Missing required coordinate"):
            SatelliteData(p)

    def test_coastwatch_format_detected(self, sat_file):
        sd = SatelliteData(sat_file)
        assert sd.data_source == 'coastwatch'

    def test_coastwatch_cci_properties_are_none(self, sat_file):
        """CCI-specific properties should return None for CoastWatch files."""
        sd = SatelliteData(sat_file)
        assert sd.swh_adjusted is None
        assert sd.swh_with_8m_offset_correction is None
        assert sd.swh_quality_level is None
        assert sd.bathymetry is None
        assert sd.distance_to_coast is None

    def test_coastwatch_sla_source_present(self, sat_file):
        sd = SatelliteData(sat_file)
        assert isinstance(sd.sla, np.ndarray)
        assert isinstance(sd.source, np.ndarray)


# ---------------------------------------------------------------------------
# CCI SatelliteData tests
# ---------------------------------------------------------------------------

class TestCCISatelliteData:
    @pytest.fixture
    def cci_file(self, tmp_path):
        p = str(tmp_path / "cci_sat.nc")
        _write_cci_satellite_nc(p)
        return p

    @pytest.fixture
    def cci_file_no_8m(self, tmp_path):
        """CCI file without the 8m correction variable."""
        p = str(tmp_path / "cci_sat_no8m.nc")
        _write_cci_satellite_nc(p, include_8m_correction=False)
        return p

    @pytest.fixture
    def cci_file_minimal(self, tmp_path):
        """CCI file with only swh (no optional variables)."""
        p = str(tmp_path / "cci_minimal.nc")
        _write_cci_satellite_nc(p, include_8m_correction=False,
                                include_adjusted=False)
        return p

    def test_cci_format_detected(self, cci_file):
        sd = SatelliteData(cci_file)
        assert sd.data_source == 'cci'

    def test_cci_loads_dataset(self, cci_file):
        sd = SatelliteData(cci_file)
        assert sd.ds is not None
        assert sd.ds.sizes["time"] == 20

    def test_cci_coordinates_return_arrays(self, cci_file):
        sd = SatelliteData(cci_file)
        assert isinstance(sd.time, np.ndarray)
        assert isinstance(sd.lon, np.ndarray)
        assert isinstance(sd.lat, np.ndarray)

    def test_cci_swh_returns_array(self, cci_file):
        sd = SatelliteData(cci_file)
        assert isinstance(sd.swh, np.ndarray)
        assert len(sd.swh) == 20

    def test_cci_optional_vars_present(self, cci_file):
        sd = SatelliteData(cci_file)
        assert isinstance(sd.swh_adjusted, np.ndarray)
        assert isinstance(sd.swh_with_8m_offset_correction, np.ndarray)
        assert isinstance(sd.swh_quality_level, np.ndarray)
        assert isinstance(sd.swh_uncertainty, np.ndarray)
        assert isinstance(sd.bathymetry, np.ndarray)
        assert isinstance(sd.distance_to_coast, np.ndarray)

    def test_cci_8m_correction_none_when_missing(self, cci_file_no_8m):
        """swh_with_8m_offset_correction should be None if not in file."""
        sd = SatelliteData(cci_file_no_8m)
        assert sd.swh_with_8m_offset_correction is None

    def test_cci_adjusted_none_when_missing(self, cci_file_minimal):
        sd = SatelliteData(cci_file_minimal)
        assert sd.swh_adjusted is None

    def test_cci_sla_source_are_none(self, cci_file):
        """CoastWatch-specific properties should return None for CCI files."""
        sd = SatelliteData(cci_file)
        assert sd.sla is None
        assert sd.source is None

    def test_cci_missing_swh_raises(self, tmp_path):
        """A CCI file without 'swh' should raise ValueError."""
        p = str(tmp_path / "cci_noswh.nc")
        times = np.array(["2023-01-16"], dtype="datetime64[ns]")
        ds = xr.Dataset(
            {
                "lat": ("time", [35.0]),
                "lon": ("time", [-75.0]),
                "distance_to_coast": ("time", [50_000.0]),
            },
            coords={"time": times},
        )
        ds.attrs["title"] = "ESA CCI Sea State test"
        ds.to_netcdf(p)
        with pytest.raises(ValueError, match="swh"):
            SatelliteData(p)

    def test_cci_filter_by_time(self, cci_file):
        sd = SatelliteData(cci_file)
        original_size = sd.ds.sizes["time"]
        sd.filter_by_time("2023-01-16T05", "2023-01-16T10")
        assert sd.ds.sizes["time"] < original_size
        assert sd.ds.sizes["time"] > 0

    def test_cci_lon_setter(self, cci_file):
        sd = SatelliteData(cci_file)
        new_lon = np.linspace(-90, -60, 20)
        sd.lon = new_lon
        np.testing.assert_allclose(sd.lon, new_lon)

    def test_cci_lon_setter_wrong_size(self, cci_file):
        sd = SatelliteData(cci_file)
        with pytest.raises(ValueError, match="must match"):
            sd.lon = np.zeros(999)


# ---------------------------------------------------------------------------
# Format detection tests
# ---------------------------------------------------------------------------

class TestDetectFormat:
    def test_detects_coastwatch(self, tmp_path):
        p = str(tmp_path / "cw.nc")
        _write_satellite_nc(p)
        with xr.open_dataset(p) as ds:
            assert _detect_format(ds) == 'coastwatch'

    def test_detects_cci_by_variable(self, tmp_path):
        p = str(tmp_path / "cci.nc")
        _write_cci_satellite_nc(p)
        with xr.open_dataset(p) as ds:
            assert _detect_format(ds) == 'cci'

    def test_detects_cci_by_title(self, tmp_path):
        """A file with ESA CCI in the title but no extra vars should be CCI."""
        times = np.array(["2023-01-16"], dtype="datetime64[ns]")
        ds = xr.Dataset(
            {
                "swh": ("time", [2.5]),
                "lat": ("time", [35.0]),
                "lon": ("time", [-75.0]),
            },
            coords={"time": times},
            attrs={"title": "ESA CCI Sea State L2P test"},
        )
        p = str(tmp_path / "cci_title.nc")
        ds.to_netcdf(p)
        with xr.open_dataset(p) as loaded:
            assert _detect_format(loaded) == 'cci'


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
