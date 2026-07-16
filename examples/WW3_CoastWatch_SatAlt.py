"""
Example: Collocate CoastWatch (NOAA STAR) satellite altimetry with a WW3 model run.

This mirrors the CCI example (WW3_CCI_SatAlt.py) but uses the CoastWatch
(NOAA STAR) altimetry pipeline. CoastWatch data requires no credentials, but
note that old data may be removed without notice and SWH is capped at 8 m.

IMPORTANT - data availability
-----------------------------
CoastWatch (NOAA STAR) altimetry data prior to 2020 has been DELETED from the
server and is no longer retrievable. Do NOT use this pipeline for dates before
2020. For earlier periods, use the ESA CCI Sea State (IFREMER) source instead
(see WW3_CCI_SatAlt.py), which provides a stable long-term archive.

Usage note
----------
The SatelliteData class automatically detects whether the file is from
CoastWatch or CCI - the collocation workflow is identical for both sources.
The main differences from the CCI example are:
  - the download function: get_multi_sat_coastwatch (no FTP credentials)
  - the satellite keys (e.g. 'jason3' instead of 'jason-3')
  - the merged file naming convention: multisat_cropped_<start>_<end>.nc
"""

import os

import numpy as np

from ocstrack.Model.model import WW3
from ocstrack.Observation.satellite import SatelliteData
from ocstrack.Observation.get_sat_coastwatch import get_multi_sat_coastwatch
from ocstrack.Collocation.collocate import Collocate
from ocstrack.utils import convert_longitude

# ---------------------------------------------------------------------------
# 1. Parameters
# ---------------------------------------------------------------------------
SAT_DATA_DIR  = r"Your/Path/to/CoastWatch_satellite_data/"
MODEL_RUN_DIR = r"Your/Path/to/WW3_run_dir/"
OUTPUT_FILE   = r"Your/Path/Here/ww3_coastwatch_collocated.nc"

START_DATE = "2023-01-16"
END_DATE   = "2023-01-31"

LAT_MIN, LAT_MAX =  50.0,  66.0
LON_MIN, LON_MAX = 165.0, -158.0  # crosses the antimeridian

# CoastWatch satellite keys (note the naming differs from CCI):
#   'sentinel3a', 'sentinel3b', 'sentinel6a', 'jason2', 'jason3',
#   'cryosat2', 'saral', 'swot'
SAT_LIST = ['sentinel3a', 'sentinel3b', 'sentinel6a', 'jason2', 'jason3',
            'cryosat2', 'saral', 'swot']
MODEL_VAR = 'HS'

# ---------------------------------------------------------------------------
# 2. Download, crop, and merge CoastWatch data
# ---------------------------------------------------------------------------
# No credentials are needed. The workflow:
#   - Downloads daily merged NetCDF files from the NOAA STAR CoastWatch server
#   - Crops each file to the bounding box
#   - Merges all files per satellite, then across all satellites
print("--- Downloading CoastWatch data ---")
get_multi_sat_coastwatch(
    start_date=START_DATE,
    end_date=END_DATE,
    sat_list=SAT_LIST,
    output_dir=SAT_DATA_DIR,
    lat_min=LAT_MIN,
    lat_max=LAT_MAX,
    lon_min=LON_MIN,
    lon_max=LON_MAX,
    clean_raw=False,
    clean_cropped=False,
)

# ---------------------------------------------------------------------------
# 3. Load the merged CoastWatch satellite data
# ---------------------------------------------------------------------------
# SatelliteData auto-detects the CoastWatch format.
# The merged file path follows the naming convention:
#   <SAT_DATA_DIR>/multisat_cropped_<start>_<end>.nc
sat_file = os.path.join(
    SAT_DATA_DIR, f"multisat_cropped_{START_DATE}_{END_DATE}.nc"
)
print("--- Loading satellite data ---")
sat_data = SatelliteData(sat_file)
print(f"Source      : {sat_data.data_source}")
print(f"N obs       : {len(sat_data.time)}")
print(f"Time range  : {sat_data.ds.time.min().values} to {sat_data.ds.time.max().values}")
print(f"Lat range   : {sat_data.ds.lat.min().values:.2f} to {sat_data.ds.lat.max().values:.2f}")
print(f"Lon range   : {sat_data.ds.lon.min().values:.2f} to {sat_data.ds.lon.max().values:.2f}")
if sat_data.sla is not None:
    print("SLA available.")
if sat_data.source is not None:
    print("Per-observation satellite source available.")

# It's crucial to ensure longitude conventions match between satellite and
# model data. CoastWatch data uses -180 to 180; WW3 here uses 0 to 360, so
# convert with mode=1.
sat_data.lon = convert_longitude(sat_data.lon, mode=1)
print(f"Lon (0-360) : {sat_data.ds.lon.min().values:.2f} to {sat_data.ds.lon.max().values:.2f}")

# ---------------------------------------------------------------------------
# 4. Load WW3 model data
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
# 5. Perform collocation
# ---------------------------------------------------------------------------
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
