
from ocstrack.Observation.get_argo import generate_monthly_dates

def test_generate_monthly_dates():
    """
    Test the generate_monthly_dates function for a multi-month range.
    """
    start_date = "2023-01-15"
    end_date = "2023-03-20"
    
    expected_months = [
        ("2023", "01"),
        ("2023", "02"),
        ("2023", "03"),
    ]
    
    result = generate_monthly_dates(start_date, end_date)
    
    assert result == expected_months

def test_generate_monthly_dates_single_month():
    """
    Test the function for a date range within a single month.
    """
    start_date = "2023-01-10"
    end_date = "2023-01-20"
    
    expected_months = [("2023", "01")]
    
    result = generate_monthly_dates(start_date, end_date)
    
    assert result == expected_months

def test_generate_monthly_dates_year_boundary():
    """
    Test the function across a year boundary.
    """
    start_date = "2022-12-01"
    end_date = "2023-02-01"
    
    expected_months = [
        ("2022", "12"),
        ("2023", "01"),
        ("2023", "02"),
    ]
    
    result = generate_monthly_dates(start_date, end_date)
    
    assert result == expected_months


from ocstrack.Observation.get_argo import _download_file
from unittest.mock import patch, mock_open, MagicMock
import requests

@patch('requests.get')
def test_download_file_success(mock_get):
    """
    Test the _download_file helper for a successful download.
    """
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.iter_content.return_value = [b'fake data']
    mock_get.return_value.__enter__.return_value = mock_response

    m_open = mock_open()
    with patch('builtins.open', m_open):
        success = _download_file('http://fake.url/file.nc', '/fake/path/file.nc')

        assert success is True
        mock_get.assert_called_once_with('http://fake.url/file.nc', stream=True)
        m_open.assert_called_once_with('/fake/path/file.nc', 'wb')
        m_open().write.assert_called_once_with(b'fake data')

@patch('requests.get')
@patch('os.path.exists', return_value=True)
@patch('os.remove')
def test_download_file_failure(mock_remove, mock_exists, mock_get):
    """
    Test the _download_file helper for a failed download.
    It should return False and clean up partially downloaded files.
    """
    mock_get.side_effect = requests.exceptions.RequestException('Test Error')

    # The function should catch the exception and return False
    success = _download_file('http://fake.url/file.nc', '/fake/path/file.nc')

    assert success is False
    mock_get.assert_called_once_with('http://fake.url/file.nc', stream=True)
    # Check that it attempts to clean up the file
    mock_remove.assert_called_once_with('/fake/path/file.nc')


# ---------------------------------------------------------------------------
# crop_by_shape_argo
# ---------------------------------------------------------------------------

import pytest
import numpy as np
import xarray as xr
from ocstrack.Observation.get_argo import crop_by_shape_argo, get_argo


def _make_argo_ds(n=10):
    """Build a minimal Argo-like dataset with N_PROF profiles."""
    juld = np.array(
        [np.datetime64("2023-01-01") + np.timedelta64(i, "h") for i in range(n)],
        dtype="datetime64[ns]",
    )
    return xr.Dataset(
        {
            "JULD":      ("N_PROF", juld),
            "LATITUDE":  ("N_PROF", np.linspace(0.0, 9.0, n)),
            "LONGITUDE": ("N_PROF", np.linspace(0.0, 9.0, n)),
            "PRES":      (("N_PROF", "N_LEVELS"), np.ones((n, 5))),
        },
        coords={"N_PROF": np.arange(n)},
    )


class TestCropByShapeArgo:
    def test_polygon_keeps_correct_profiles(self):
        """Profiles inside the polygon are kept; outside are dropped."""
        shapely = pytest.importorskip("shapely")
        from shapely.geometry import box

        ds = _make_argo_ds(n=10)
        # Box covering lon/lat in [0, 4.5] -> first 5 profiles should survive
        bbox = box(-0.5, -0.5, 4.5, 4.5)
        cropped = crop_by_shape_argo(ds, bbox)
        assert cropped.sizes["N_PROF"] == 5
        assert float(cropped["LONGITUDE"].max()) <= 4.5

    def test_polygon_drops_all_outside(self):
        """A polygon with no profiles inside returns an empty dataset."""
        shapely = pytest.importorskip("shapely")
        from shapely.geometry import box

        ds = _make_argo_ds(n=10)
        # Box far outside the data range
        empty_box = box(50.0, 50.0, 60.0, 60.0)
        cropped = crop_by_shape_argo(ds, empty_box)
        assert cropped.sizes["N_PROF"] == 0

    def test_geojson_dict(self):
        """Accepts a bare GeoJSON geometry dict."""
        shapely = pytest.importorskip("shapely")

        ds = _make_argo_ds(n=10)
        geojson = {
            "type": "Polygon",
            "coordinates": [[[-0.5, -0.5], [4.5, -0.5], [4.5, 4.5], [-0.5, 4.5], [-0.5, -0.5]]],
        }
        cropped = crop_by_shape_argo(ds, geojson)
        assert cropped.sizes["N_PROF"] == 5

    def test_missing_coords_raises(self):
        """Raises ValueError when LONGITUDE is missing from the dataset."""
        shapely = pytest.importorskip("shapely")
        from shapely.geometry import box

        ds = _make_argo_ds().drop_vars("LONGITUDE")
        with pytest.raises(ValueError, match="does not contain"):
            crop_by_shape_argo(ds, box(0, 0, 5, 5))

    def test_invalid_shape_raises(self):
        """Raises ValueError for an unrecognised shape type."""
        shapely = pytest.importorskip("shapely")

        ds = _make_argo_ds()
        with pytest.raises(ValueError):
            crop_by_shape_argo(ds, {"bad": "dict"})


class TestGetArgoShapeBoxMutuallyExclusive:
    def test_both_box_and_shape_raises(self):
        """Providing both bounding-box and shape arguments raises ValueError."""
        shapely = pytest.importorskip("shapely")
        from shapely.geometry import box

        with pytest.raises(ValueError, match="not both"):
            get_argo(
                start_date="2023-01-01",
                end_date="2023-01-31",
                region="atlantic_ocean",
                output_dir="/tmp/test_argo",
                lat_min=0, lat_max=10, lon_min=0, lon_max=10,
                shape=box(0, 0, 5, 5),
            )

