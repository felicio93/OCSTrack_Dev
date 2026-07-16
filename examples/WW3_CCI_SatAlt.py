"""
Example: Collocate ESA CCI Sea State altimetry with a WW3 model run.

This example demonstrates how to use the ESA CCI Sea State data from IFREMER
as an alternative to the CoastWatch satellite altimetry data. The CCI data has
no SWH cap at 8 m, making it suitable for extreme wave event analysis.

Prerequisites
-------------
- A free IFREMER FTP account. Register at: https://eftp.ifremer.fr
- Credentials stored as environment variables (recommended) or passed directly.

Usage note
----------
The SatelliteData class automatically detects whether the file is from
CoastWatch or CCI - the collocation workflow is identical for both sources.
"""

import os

import numpy as np

from ocstrack.Model.model import WW3
from ocstrack.Observation.satellite import SatelliteData
from ocstrack.Observation.get_sat_cci import get_multi_sat_cci
from ocstrack.Collocation.collocate import Collocate
from ocstrack.utils import convert_longitude

# ---------------------------------------------------------------------------
# 1. Parameters
# ---------------------------------------------------------------------------
SAT_DATA_DIR  = r"Your/Path/to/CCI_satellite_data/"
MODEL_RUN_DIR = r"Your/Path/to/WW3_run_dir/"
OUTPUT_FILE   = r"Your/Path/Here/ww3_cci_collocated.nc"

START_DATE = "2019-07-30"
END_DATE   = "2019-08-04"

LAT_MIN, LAT_MAX =  50.0,  66.0
LON_MIN, LON_MAX = 165.0, -158.0  # crosses the antimeridian

# Satellites to download. Valid altimeter keys:
#   'jason-3', 'sentinel-3a', 'sentinel-3b', 'sentinel-6a',
#   'cryosat-2', 'saral', 'swot', 'cfosat', 'jason-1', 'jason-2', ...
SAT_LIST = ['jason-3', 'sentinel-3a', 'sentinel-3b', 'sentinel-6a',
            'cryosat-2', 'saral', 'swot']
MODEL_VAR = 'HS'

# ---------------------------------------------------------------------------
# 2. Credentials (read from environment variables - never hard-code these)
# ---------------------------------------------------------------------------
# Register at https://eftp.ifremer.fr, then set before running:
#   export CCI_FTP_USER="your_username"
#   export CCI_FTP_PASS="your_password"
CCI_FTP_USER = os.environ.get("CCI_FTP_USER", "")
CCI_FTP_PASS = os.environ.get("CCI_FTP_PASS", "")

# ---------------------------------------------------------------------------
# 3. Download, crop, and merge CCI data
# ---------------------------------------------------------------------------
# The workflow mirrors the CoastWatch pipeline:
#   - Downloads per-pass NetCDF files from the IFREMER FTP server
#   - Crops each file to the bounding box
#   - Merges all files per satellite, then across all satellites
#
# Only these variables are retained in the merged output:
#   swh, swh_adjusted, swh_with_8m_offset_correction, swh_quality_level,
#   swh_uncertainty, bathymetry, distance_to_coast
print("--- Downloading CCI data ---")
get_multi_sat_cci(
    start_date=START_DATE,
    end_date=END_DATE,
    sat_list=SAT_LIST,
    output_dir=SAT_DATA_DIR,
    ftp_user=CCI_FTP_USER,
    ftp_pass=CCI_FTP_PASS,
    lat_min=LAT_MIN,
    lat_max=LAT_MAX,
    lon_min=LON_MIN,
    lon_max=LON_MAX,
    clean_raw=False,
    clean_cropped=False,
)

# ---------------------------------------------------------------------------
# 4. Load the merged CCI satellite data
# ---------------------------------------------------------------------------
# SatelliteData auto-detects the CCI format.
# The merged file path follows the naming convention:
#   <SAT_DATA_DIR>/multisat_cci_cropped_<start>_<end>.nc
sat_file = os.path.join(
    SAT_DATA_DIR, f"multisat_cci_cropped_{START_DATE}_{END_DATE}.nc"
)
print("--- Loading satellite data ---")
sat_data = SatelliteData(sat_file)
print(f"Source      : {sat_data.data_source}")
print(f"N obs       : {len(sat_data.time)}")
print(f"Time range  : {sat_data.ds.time.min().values} to {sat_data.ds.time.max().values}")
print(f"Lat range   : {sat_data.ds.lat.min().values:.2f} to {sat_data.ds.lat.max().values:.2f}")
print(f"Lon range   : {sat_data.ds.lon.min().values:.2f} to {sat_data.ds.lon.max().values:.2f}")
if sat_data.swh_with_8m_offset_correction is not None:
    print("swh_with_8m_offset_correction available (no SWH cap).")
if sat_data.swh_quality_level is not None:
    print("Quality flags available.")

# It's crucial to ensure longitude conventions match between satellite and
# model data. CCI data uses -180 to 180; WW3 here uses 0 to 360, so convert
# with mode=1.
sat_data.lon = convert_longitude(sat_data.lon, mode=1)
print(f"Lon (0-360) : {sat_data.ds.lon.min().values:.2f} to {sat_data.ds.lon.max().values:.2f}")

# ---------------------------------------------------------------------------
# 5. Load WW3 model data
# ---------------------------------------------------------------------------
print("--- Loading WW3 model ---")
model_run = WW3(
    rundir=MODEL_RUN_DIR,
    model_dict={'var': MODEL_VAR},
    start_date=np.datetime64(START_DATE),
    end_date=np.datetime64(END_DATE),
)
print(f"Model time  : {model_run.time.min()} to {model_run.time.max()}")
print(f"Model lat   : {model_run.mesh_y.min():.2f} to {model_run.mesh_y.max():.2f}")
print(f"Model lon   : {model_run.mesh_x.min():.2f} to {model_run.mesh_x.max():.2f}")

# ---------------------------------------------------------------------------
# 6. Perform collocation
# ---------------------------------------------------------------------------
# The Collocate engine works identically for CoastWatch and CCI data.
# Collocation is always based on 'swh' as the primary observation variable.
print("--- Running collocation ---")
coll = Collocate(
    model_run=model_run,
    observation=sat_data,
    n_nearest=3,
    temporal_interp=True,
    time_buffer=np.timedelta64(30, 'm'),
)
ds_coll = coll.run(output_path=OUTPUT_FILE)
print(f"Done. Output: {OUTPUT_FILE}")
