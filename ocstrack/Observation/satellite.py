"""Module for handling Satellite altimetry data.

This module provides a unified ``SatelliteData`` class that transparently
handles files from two different data sources:

- **CoastWatch** (NOAA STAR): daily merged files downloaded via
  :mod:`ocstrack.Observation.get_sat`. These files contain ``swh``, ``sla``,
  and ``source`` variables.

- **ESA CCI Sea State v5** (IFREMER): per-pass files downloaded and merged
  via :mod:`ocstrack.Observation.get_sat_cci`. These files contain
  ``swh``, and optionally ``swh_adjusted``,
  ``swh_with_8m_offset_correction``, ``swh_quality_level``,
  ``swh_uncertainty``, ``bathymetry``, and ``distance_to_coast``.

The source format is detected automatically from the file contents.
Both formats share the same coordinate names (``time``, ``lat``, ``lon``),
so the collocation engine works identically for either source.
"""

from typing import Optional, Union

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _detect_format(ds: xr.Dataset) -> str:
    """
    Detect the satellite data format from the dataset contents.

    Returns ``'coastwatch'`` or ``'cci'``.

    Detection heuristic:
    - CoastWatch files always contain a ``sla`` (sea level anomaly) variable
      and a ``source`` variable. These are not present in CCI files.
    - CCI files contain at least one of: ``swh_adjusted``,
      ``swh_with_8m_offset_correction``, ``distance_to_coast``,
      or have a ``title`` attribute containing 'ESA CCI'.
    """
    cw_markers = {'sla', 'source'}
    cci_markers = {
        'swh_adjusted', 'swh_with_8m_offset_correction',
        'distance_to_coast', 'swh_quality_level',
    }

    has_cw = bool(cw_markers & set(ds.variables))
    has_cci = bool(cci_markers & set(ds.variables))
    has_cci_title = 'ESA CCI' in ds.attrs.get('title', '')

    if has_cw and not has_cci:
        return 'coastwatch'
    if has_cci or has_cci_title:
        return 'cci'
    # Fall back: if it has sla it is most likely CoastWatch
    if 'sla' in ds.variables:
        return 'coastwatch'
    return 'cci'


# ---------------------------------------------------------------------------
# Unified SatelliteData class
# ---------------------------------------------------------------------------

class SatelliteData:
    """
    Unified satellite altimetry data handler.

    Loads a NetCDF file produced by either the CoastWatch (NOAA STAR) or the
    ESA CCI Sea State (IFREMER) pipeline and exposes a consistent interface
    for use with the :class:`~ocstrack.Collocation.collocate.Collocate`
    engine.

    The data source format is detected automatically. The following
    coordinates are always available: ``time``, ``lat``, ``lon``.

    Wave height variables (accessed as properties) depend on what is present
    in the file and are never ``None`` only if the variable actually exists:

    - ``swh`` — always required.
    - ``swh_adjusted`` — CCI only (when present).
    - ``swh_with_8m_offset_correction`` — CCI only (when present).
    - ``sla`` — CoastWatch only (when present).
    - ``source`` — CoastWatch only (when present).

    Parameters
    ----------
    filepath : str
        Path to the satellite NetCDF file.

    Raises
    ------
    ValueError
        If required coordinates (``time``, ``lat``, ``lon``) or the primary
        ``swh`` variable are missing from the dataset.

    Examples
    --------
    CoastWatch data::

        sat = SatelliteData('/path/to/multisat_cropped_2019-07-30_2019-08-03.nc')
        print(sat.source)   # array of satellite names

    CCI data::

        sat = SatelliteData('/path/to/concat_cci_cropped_jason-3_2023-01-16_2023-01-31.nc')
        print(sat.swh_with_8m_offset_correction)  # not capped at 8 m
        print(sat.swh_quality_level)              # quality flags
    """

    def __init__(self, filepath: str) -> None:
        self.ds: xr.Dataset = xr.open_dataset(filepath)
        self._format: str = _detect_format(self.ds)

        # Coordinates are required regardless of source
        required_coords = ['time', 'lat', 'lon']
        missing_coords = [c for c in required_coords if c not in self.ds.variables]
        if missing_coords:
            raise ValueError(
                f"Missing required coordinate(s) in dataset: {missing_coords}. "
                f"File: {filepath}"
            )

        # swh is the primary wave variable and must be present
        if 'swh' not in self.ds.variables:
            raise ValueError(
                f"Missing required variable 'swh' in dataset. File: {filepath}"
            )

    # ------------------------------------------------------------------
    # Format info
    # ------------------------------------------------------------------

    @property
    def data_source(self) -> str:
        """Return the detected data source format: ``'coastwatch'`` or ``'cci'``."""
        return self._format

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------

    @property
    def time(self) -> np.ndarray:
        """Return time values as a NumPy array."""
        return self.ds.time.values

    @property
    def lon(self) -> np.ndarray:
        """Return longitude values as a NumPy array."""
        return self.ds.lon.values

    @lon.setter
    def lon(self, new_lon: Union[np.ndarray, list]) -> None:
        """Set longitude values."""
        if len(new_lon) != len(self.ds.lon):
            raise ValueError("New longitude array must match the existing size.")
        self.ds['lon'] = ('time', np.array(new_lon))

    @property
    def lat(self) -> np.ndarray:
        """Return latitude values as a NumPy array."""
        return self.ds.lat.values

    @lat.setter
    def lat(self, new_lat: Union[np.ndarray, list]) -> None:
        """Set latitude values."""
        if len(new_lat) != len(self.ds.lat):
            raise ValueError("New latitude array must match the existing size.")
        self.ds['lat'] = ('time', np.array(new_lat))

    # ------------------------------------------------------------------
    # Primary wave variable (both sources)
    # ------------------------------------------------------------------

    @property
    def swh(self) -> np.ndarray:
        """Return significant wave height (SWH) as a NumPy array."""
        return self.ds.swh.values

    # ------------------------------------------------------------------
    # CCI-specific variables (return None if not present)
    # ------------------------------------------------------------------

    @property
    def swh_adjusted(self) -> Optional[np.ndarray]:
        """
        Return denoised/adjusted SWH as a NumPy array, or ``None`` if not present.

        Available in CCI files only.
        """
        if 'swh_adjusted' in self.ds:
            return self.ds.swh_adjusted.values
        return None

    @property
    def swh_with_8m_offset_correction(self) -> Optional[np.ndarray]:
        """
        Return SWH with 8 m offset correction as a NumPy array, or ``None`` if not present.

        This variable is not capped at 8 m and is recommended for extreme
        wave event analysis. Available in CCI files only.
        """
        if 'swh_with_8m_offset_correction' in self.ds:
            return self.ds.swh_with_8m_offset_correction.values
        return None

    @property
    def swh_quality_level(self) -> Optional[np.ndarray]:
        """
        Return SWH quality level flags as a NumPy array, or ``None`` if not present.

        Available in CCI files only. Higher values generally indicate better quality.
        """
        if 'swh_quality_level' in self.ds:
            return self.ds.swh_quality_level.values
        return None

    @property
    def swh_uncertainty(self) -> Optional[np.ndarray]:
        """
        Return SWH uncertainty as a NumPy array, or ``None`` if not present.

        Available in CCI files only.
        """
        if 'swh_uncertainty' in self.ds:
            return self.ds.swh_uncertainty.values
        return None

    @property
    def bathymetry(self) -> Optional[np.ndarray]:
        """
        Return bathymetry values as a NumPy array, or ``None`` if not present.

        Available in CCI files only.
        """
        if 'bathymetry' in self.ds:
            return self.ds.bathymetry.values
        return None

    @property
    def distance_to_coast(self) -> Optional[np.ndarray]:
        """
        Return distance-to-coast values as a NumPy array, or ``None`` if not present.

        Available in CCI files only.
        """
        if 'distance_to_coast' in self.ds:
            return self.ds.distance_to_coast.values
        return None

    # ------------------------------------------------------------------
    # CoastWatch-specific variables (return None if not present)
    # ------------------------------------------------------------------

    @property
    def sla(self) -> Optional[np.ndarray]:
        """
        Return sea level anomaly (SLA) as a NumPy array, or ``None`` if not present.

        Available in CoastWatch files only.
        """
        if 'sla' in self.ds:
            return self.ds.sla.values
        return None

    @property
    def source(self) -> Optional[np.ndarray]:
        """
        Return the satellite source identifier as a NumPy array, or ``None`` if not present.

        Available in CoastWatch files only.
        """
        if 'source' in self.ds:
            return self.ds.source.values
        return None

    # ------------------------------------------------------------------
    # Time filtering
    # ------------------------------------------------------------------

    def filter_by_time(self, start_date: str, end_date: str) -> None:
        """
        Filter the dataset to a specific time range (in-place).

        Parameters
        ----------
        start_date : str
            ISO 8601 start date string (e.g. ``'2023-01-16'``).
        end_date : str
            ISO 8601 end date string (e.g. ``'2023-01-31'``).
        """
        start = np.datetime64(start_date)
        end = np.datetime64(end_date)

        if not np.issubdtype(self.ds['time'].dtype, np.datetime64):
            self.ds['time'] = xr.decode_cf(self.ds).time

        self.ds = self.ds.sortby('time')
        self.ds = self.ds.sel(time=slice(start, end))
