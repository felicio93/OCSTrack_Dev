
import pytest
from unittest.mock import patch, mock_open, MagicMock
import numpy as np
import xarray as xr
from ocstrack.Observation.get_sat_coastwatch import download_sat_data, crop_by_shape
from ocstrack.Observation.urls import (
    URL_TEMPLATES,
    CCI_ALTIMETERS,
    resolve_sat_key,
    _canonical_sat_key,
)
import requests

@pytest.fixture
def mock_requests_get():
    """Fixture to mock requests.get."""
    with patch('requests.get') as mock_get:
        yield mock_get

@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_download_sat_data_success(mock_makedirs, mock_exists, mock_requests_get):
    """
    Test successful download of a single file.
    """
    # Configure the mock for a successful response
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.iter_content.return_value = [b'fake', b'file', b'content']
    mock_requests_get.return_value.__enter__.return_value = mock_response

    # Use mock_open to simulate file writing
    m_open = mock_open()
    with patch('builtins.open', m_open):
        result = download_sat_data(
            dates_str=['20230101'],
            url_template='http://fake.url/',
            raw_dir='/fake/dir',
            sat='test_sat'
        )

        # Assertions
        mock_makedirs.assert_called_once_with('/fake/dir', exist_ok=True)
        mock_requests_get.assert_called_once_with('http://fake.url/20230101.nc', stream=True)
        m_open.assert_called_once_with('/fake/dir/20230101.nc', 'wb')
        handle = m_open()
        handle.write.assert_any_call(b'fake')
        handle.write.assert_any_call(b'content')
        assert result == ['/fake/dir/20230101.nc']

@patch('os.path.exists', return_value=False)
@patch('os.makedirs')
def test_download_sat_data_failure_and_retry(mock_makedirs, mock_exists, mock_requests_get):
    """
    Test that the function retries on a request failure.
    """
    # Configure the mock to raise an exception
    mock_requests_get.side_effect = requests.exceptions.RequestException("Test Error")

    m_open = mock_open()
    with patch('builtins.open', m_open):
        with patch('time.sleep', return_value=None) as mock_sleep: # Mock time.sleep
            result = download_sat_data(
                dates_str=['20230101'],
                url_template='http://fake.url/',
                raw_dir='/fake/dir',
                sat='test_sat',
                retries=3,
                delay=1
            )

            # Assertions
            mock_makedirs.assert_called_once_with('/fake/dir', exist_ok=True)
            assert mock_requests_get.call_count == 3
            assert mock_sleep.call_count == 3 # Sleeps after each of the 3 failures
            m_open.assert_not_called() # File should not be opened on failure
            assert result == ['/fake/dir/20230101.nc'] # Still returns the expected path


# ---------------------------------------------------------------------------
# Satellite key normalization
# ---------------------------------------------------------------------------

class TestResolveSatKey:
    def test_canonical_strips_punctuation_and_case(self):
        assert _canonical_sat_key("Jason-3") == "jason3"
        assert _canonical_sat_key("JASON_3") == "jason3"
        assert _canonical_sat_key("jason 3") == "jason3"
        assert _canonical_sat_key("sentinel-3_a") == "sentinel3a"

    def test_exact_match_coastwatch(self):
        assert resolve_sat_key("jason3", URL_TEMPLATES) == "jason3"

    def test_variant_resolves_to_coastwatch_key(self):
        # CCI-style 'jason-3' should resolve to CoastWatch 'jason3'
        assert resolve_sat_key("jason-3", URL_TEMPLATES) == "jason3"
        assert resolve_sat_key("Sentinel-3A", URL_TEMPLATES) == "sentinel3a"

    def test_exact_match_cci(self):
        assert resolve_sat_key("jason-3", CCI_ALTIMETERS) == "jason-3"

    def test_variant_resolves_to_cci_key(self):
        # CoastWatch-style 'jason3' should resolve to CCI 'jason-3'
        assert resolve_sat_key("jason3", CCI_ALTIMETERS) == "jason-3"
        assert resolve_sat_key("sentinel3a", CCI_ALTIMETERS) == "sentinel-3a"

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown satellite key"):
            resolve_sat_key("not-a-satellite", URL_TEMPLATES)


# ---------------------------------------------------------------------------
# crop_by_shape (regression test: numpy must be imported)
# ---------------------------------------------------------------------------

class TestCropByShape:
    def _make_ds(self):
        n = 10
        times = np.arange(
            np.datetime64("2023-01-01"),
            np.datetime64("2023-01-01") + np.timedelta64(n, "h"),
            np.timedelta64(1, "h"),
        ).astype("datetime64[ns]")
        return xr.Dataset(
            {
                "swh": ("time", np.random.rand(n)),
                "lat": ("time", np.linspace(0, 9, n)),
                "lon": ("time", np.linspace(0, 9, n)),
            },
            coords={"time": times},
        )

    def test_crop_by_shape_polygon(self):
        shapely = pytest.importorskip("shapely")
        from shapely.geometry import box

        ds = self._make_ds()
        # Box covering lon/lat in [0, 4.5] -> should keep first 5 points
        bbox = box(-0.5, -0.5, 4.5, 4.5)
        cropped = crop_by_shape(ds, bbox)
        assert cropped.sizes["time"] == 5
        assert float(cropped.lon.max()) <= 4.5

    def test_crop_by_shape_missing_coords_raises(self):
        shapely = pytest.importorskip("shapely")
        from shapely.geometry import box

        ds = self._make_ds().drop_vars("lon")
        with pytest.raises(ValueError, match="does not contain"):
            crop_by_shape(ds, box(0, 0, 1, 1))
