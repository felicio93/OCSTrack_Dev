"""Tests for the ESA CCI Sea State (IFREMER) download/crop/merge pipeline.

Network access (FTP download) is always mocked, so these tests run offline.
They exercise the helpers, cropping, concatenation, and the high-level
orchestration functions (``get_per_sat_cci`` / ``get_multi_sat_cci``).
"""

import os
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from ocstrack.Observation import get_sat_cci
from ocstrack.Observation.get_sat_cci import (
    _check_credentials,
    _generate_dates,
    _subset_vars,
    crop_cci_data,
    concat_cci_data,
    get_per_sat_cci,
    get_multi_sat_cci,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cci_pass(path, n=10, lat0=30.0, lat1=40.0, lon0=-80.0, lon1=-70.0,
                   t0="2023-01-16"):
    """Write a minimal CCI-style per-pass NetCDF file."""
    times = np.arange(
        np.datetime64(t0),
        np.datetime64(t0) + np.timedelta64(n, "h"),
        np.timedelta64(1, "h"),
    ).astype("datetime64[ns]")
    ds = xr.Dataset(
        {
            "swh": ("time", np.random.rand(n)),
            "swh_adjusted": ("time", np.random.rand(n)),
            "swh_with_8m_offset_correction": ("time", np.random.rand(n) * 15),
            "swh_quality_level": ("time", np.ones(n, dtype=np.int8)),
            "distance_to_coast": ("time", np.random.rand(n) * 200_000),
            # A variable that should be dropped by _subset_vars
            "some_other_var": ("time", np.random.rand(n)),
        },
        coords={
            "time": times,
            "lat": ("time", np.linspace(lat0, lat1, n)),
            "lon": ("time", np.linspace(lon0, lon1, n)),
        },
    )
    ds.attrs["title"] = "ESA CCI Sea State L2P"
    ds.to_netcdf(path)
    return path


# ---------------------------------------------------------------------------
# Credential checks
# ---------------------------------------------------------------------------

class TestCheckCredentials:
    def test_missing_user_raises(self):
        with pytest.raises(ValueError, match="credentials are required"):
            _check_credentials("", "pass")

    def test_missing_pass_raises(self):
        with pytest.raises(ValueError, match="credentials are required"):
            _check_credentials("user", "")

    def test_both_none_raises(self):
        with pytest.raises(ValueError, match="credentials are required"):
            _check_credentials(None, None)

    def test_valid_credentials_ok(self):
        # Should not raise
        _check_credentials("user", "pass")


# ---------------------------------------------------------------------------
# Date generation
# ---------------------------------------------------------------------------

class TestGenerateDates:
    def test_inclusive_range(self):
        result = _generate_dates("2023-01-16", "2023-01-18")
        assert result == ["20230116", "20230117", "20230118"]

    def test_single_day(self):
        assert _generate_dates("2023-01-16", "2023-01-16") == ["20230116"]

    def test_month_boundary(self):
        result = _generate_dates("2023-01-30", "2023-02-01")
        assert result == ["20230130", "20230131", "20230201"]


# ---------------------------------------------------------------------------
# Variable subsetting
# ---------------------------------------------------------------------------

class TestSubsetVars:
    def test_drops_unlisted_vars(self, tmp_path):
        p = str(tmp_path / "pass.nc")
        _make_cci_pass(p)
        with xr.open_dataset(p) as ds:
            ds = ds.load()
        subset = _subset_vars(ds, ["swh", "swh_adjusted"])
        assert "swh" in subset
        assert "swh_adjusted" in subset
        assert "some_other_var" not in subset
        # Coordinates always retained
        assert "lat" in subset
        assert "lon" in subset

    def test_keeps_all_when_no_extra(self, tmp_path):
        p = str(tmp_path / "pass.nc")
        _make_cci_pass(p)
        with xr.open_dataset(p) as ds:
            ds = ds.load()
        # Keeping a var not present should not error
        subset = _subset_vars(ds, ["swh", "nonexistent_var"])
        assert "swh" in subset


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------

class TestCropCCIData:
    def test_crop_keeps_in_box(self, tmp_path):
        raw = str(tmp_path / "raw.nc")
        _make_cci_pass(raw, lat0=30.0, lat1=40.0, lon0=-80.0, lon1=-70.0)
        cropped_dir = str(tmp_path / "cropped")

        result = crop_cci_data(
            [raw], cropped_dir, lat_min=33, lat_max=37,
            lon_min=-78, lon_max=-72,
        )
        assert len(result) == 1
        ds = result[0]
        assert float(ds["lat"].min()) >= 33
        assert float(ds["lat"].max()) <= 37
        # Subsetting applied (extra var dropped)
        assert "some_other_var" not in ds
        # A cropped file was written
        assert os.path.isdir(cropped_dir)
        assert len(os.listdir(cropped_dir)) == 1

    def test_crop_empty_is_skipped(self, tmp_path):
        raw = str(tmp_path / "raw.nc")
        _make_cci_pass(raw, lat0=30.0, lat1=40.0)
        cropped_dir = str(tmp_path / "cropped")

        # Box far away from the data -> empty crop
        result = crop_cci_data(
            [raw], cropped_dir, lat_min=-10, lat_max=-5,
            lon_min=0, lon_max=10,
        )
        assert result == []

    def test_crop_bad_file_is_skipped(self, tmp_path):
        cropped_dir = str(tmp_path / "cropped")
        result = crop_cci_data(
            [str(tmp_path / "does_not_exist.nc")], cropped_dir,
            lat_min=0, lat_max=1, lon_min=0, lon_max=1,
        )
        assert result == []


# ---------------------------------------------------------------------------
# Concatenation
# ---------------------------------------------------------------------------

class TestConcatCCIData:
    def test_concat_writes_and_sets_source(self, tmp_path):
        ds_list = []
        for i in range(2):
            p = str(tmp_path / f"p{i}.nc")
            _make_cci_pass(p, t0=f"2023-01-1{6 + i}")
            with xr.open_dataset(p) as ds:
                ds_list.append(ds.load())

        out = str(tmp_path / "merged.nc")
        merged = concat_cci_data(ds_list, out, sat="jason-3")
        assert merged is not None
        assert os.path.exists(out)
        assert str(merged.source.values) == "jason-3"
        # Sorted and concatenated
        assert merged.sizes["time"] == 20

    def test_concat_empty_returns_none(self, tmp_path):
        out = str(tmp_path / "merged.nc")
        assert concat_cci_data([], out, sat="jason-3") is None
        assert not os.path.exists(out)


# ---------------------------------------------------------------------------
# High-level orchestration (download mocked)
# ---------------------------------------------------------------------------

class TestGetPerSatCCI:
    def test_missing_credentials_raises(self, tmp_path):
        with pytest.raises(ValueError, match="credentials are required"):
            get_per_sat_cci(
                "2023-01-16", "2023-01-17", "jason-3", str(tmp_path),
                ftp_user="", ftp_pass="",
            )

    def test_box_and_shape_mutually_exclusive(self, tmp_path):
        with pytest.raises(ValueError, match="not both"):
            get_per_sat_cci(
                "2023-01-16", "2023-01-17", "jason-3", str(tmp_path),
                ftp_user="u", ftp_pass="p",
                lat_min=30, lat_max=40, lon_min=-80, lon_max=-70,
                shape={"type": "Polygon", "coordinates": []},
            )

    def test_no_files_downloaded_returns_none(self, tmp_path):
        with patch.object(get_sat_cci, "download_cci_data", return_value=[]):
            result = get_per_sat_cci(
                "2023-01-16", "2023-01-17", "jason-3", str(tmp_path),
                ftp_user="u", ftp_pass="p",
                lat_min=30, lat_max=40, lon_min=-80, lon_max=-70,
            )
        assert result is None

    def test_full_pipeline_with_box_crop(self, tmp_path):
        # Prepare a fake "downloaded" raw file
        raw = str(tmp_path / "raw_pass.nc")
        _make_cci_pass(raw, lat0=30.0, lat1=40.0, lon0=-80.0, lon1=-70.0)

        with patch.object(get_sat_cci, "download_cci_data", return_value=[raw]):
            result = get_per_sat_cci(
                "2023-01-16", "2023-01-16", "jason-3", str(tmp_path),
                ftp_user="u", ftp_pass="p",
                lat_min=32, lat_max=38, lon_min=-78, lon_max=-72,
            )
        assert result is not None
        assert str(result.source.values) == "jason-3"
        # Merged file written with the expected naming convention
        expected = tmp_path / "jason-3" / "concat_cci_cropped_jason-3_2023-01-16_2023-01-16.nc"
        assert expected.exists()

    def test_full_pipeline_no_crop(self, tmp_path):
        raw = str(tmp_path / "raw_pass.nc")
        _make_cci_pass(raw)

        with patch.object(get_sat_cci, "download_cci_data", return_value=[raw]):
            result = get_per_sat_cci(
                "2023-01-16", "2023-01-16", "jason-3", str(tmp_path),
                ftp_user="u", ftp_pass="p",
            )
        assert result is not None
        # No crop tag in the filename
        expected = tmp_path / "jason-3" / "concat_cci_jason-3_2023-01-16_2023-01-16.nc"
        assert expected.exists()

    def test_key_normalization(self, tmp_path):
        """A CoastWatch-style key ('jason3') should resolve to CCI 'jason-3'."""
        raw = str(tmp_path / "raw_pass.nc")
        _make_cci_pass(raw)

        with patch.object(get_sat_cci, "download_cci_data", return_value=[raw]):
            result = get_per_sat_cci(
                "2023-01-16", "2023-01-16", "jason3", str(tmp_path),
                ftp_user="u", ftp_pass="p",
            )
        assert result is not None
        # Directory uses the canonical CCI key
        assert (tmp_path / "jason-3").is_dir()

    def test_unknown_key_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown satellite key"):
            get_per_sat_cci(
                "2023-01-16", "2023-01-16", "not-a-sat", str(tmp_path),
                ftp_user="u", ftp_pass="p",
            )

    def test_clean_raw_removes_raw_dir(self, tmp_path):
        raw_dir = tmp_path / "jason-3" / "raw"
        raw_dir.mkdir(parents=True)
        raw = str(raw_dir / "raw_pass.nc")
        _make_cci_pass(raw)

        with patch.object(get_sat_cci, "download_cci_data", return_value=[raw]):
            get_per_sat_cci(
                "2023-01-16", "2023-01-16", "jason-3", str(tmp_path),
                ftp_user="u", ftp_pass="p",
                clean_raw=True,
            )
        assert not raw_dir.exists()


class TestGetMultiSatCCI:
    def test_missing_credentials_raises(self, tmp_path):
        with pytest.raises(ValueError, match="credentials are required"):
            get_multi_sat_cci(
                "2023-01-16", "2023-01-17", ["jason-3"], str(tmp_path),
                ftp_user="", ftp_pass="",
            )

    def test_no_datasets_returns_none(self, tmp_path):
        with patch.object(get_sat_cci, "download_cci_data", return_value=[]):
            result = get_multi_sat_cci(
                "2023-01-16", "2023-01-17", ["jason-3", "saral"], str(tmp_path),
                ftp_user="u", ftp_pass="p",
                lat_min=30, lat_max=40, lon_min=-80, lon_max=-70,
            )
        assert result is None

    def test_multi_sat_merge(self, tmp_path):
        def fake_download(*args, **kwargs):
            # Write a raw file into the per-sat raw dir being requested
            raw_dir = kwargs["raw_dir"]
            os.makedirs(raw_dir, exist_ok=True)
            p = os.path.join(raw_dir, "raw_pass.nc")
            _make_cci_pass(p)
            return [p]

        with patch.object(get_sat_cci, "download_cci_data", side_effect=fake_download):
            result = get_multi_sat_cci(
                "2023-01-16", "2023-01-16", ["jason-3", "saral"], str(tmp_path),
                ftp_user="u", ftp_pass="p",
                lat_min=30, lat_max=40, lon_min=-80, lon_max=-70,
                clean_raw=False, clean_cropped=False,
            )
        assert result is not None
        # Combined file written
        expected = tmp_path / "multisat_cci_cropped_2023-01-16_2023-01-16.nc"
        assert expected.exists()
        # 2 satellites x 10 obs each (after crop that keeps all in-box)
        assert result.sizes["time"] > 0
