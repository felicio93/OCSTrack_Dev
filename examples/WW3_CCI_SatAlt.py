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
CoastWatch or CCI — the collocation workflow is identical for both sources.
"""

import os

import numpy as np

from ocstrack.Model.model import WW3
from ocstrack.Observation.get_sat_cci import get_multi_sat_cci
from ocstrack.Observation.satellite import SatelliteData
from ocstrack.Collocation.collocate import Collocate

# ---------------------------------------------------------------------------
# 1. Credentials (read from environment variables — never hard-code these)
# ---------------------------------------------------------------------------
ftp_user = os.environ.get("CCI_FTP_USER", "")
ftp_pass = os.environ.get("CCI_FTP_PASS", "")

# ---------------------------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------------------------
start_date = "2023-01-16"
end_date   = "2023-01-31"

# Satellites to download. Valid altimeter keys:
#   'jason-3', 'sentinel-3a', 'sentinel-3b', 'sentinel-6a',
#   'cryosat-2', 'saral', 'swot', 'cfosat', 'jason-1', 'jason-2', ...
sat_list = ['jason-3', 'sentinel-3a', 'sentinel-3b']

# Output directories
sat_output_dir = r"Your/Path/to/CCI_satellite_data/"
model_path     = r"Your/Path/to/WW3_run_dir/"
output_path    = r"Your/Path/Here/ww3_cci_collocated.nc"

# Region of interest (Gulf of Mexico example)
lat_min, lat_max = 18.0, 31.0
lon_min, lon_max = -98.0, -80.0

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

print("Downloading and merging CCI satellite data...")
merged_ds = get_multi_sat_cci(
    start_date=start_date,
    end_date=end_date,
    sat_list=sat_list,
    output_dir=sat_output_dir,
    ftp_user=ftp_user,
    ftp_pass=ftp_pass,
    lat_min=lat_min,
    lat_max=lat_max,
    lon_min=lon_min,
    lon_max=lon_max,
    clean_raw=True,      # Delete raw per-pass files after merging
    clean_cropped=True,  # Delete per-satellite cropped files after merging
)
print("Download and merge complete.")

# ---------------------------------------------------------------------------
# 4. Load the merged CCI satellite data
# ---------------------------------------------------------------------------
# SatelliteData auto-detects the CCI format.
# The merged file path follows the naming convention:
#   <sat_output_dir>/multisat_cci_cropped_<start>_<end>.nc
sat_path = os.path.join(
    sat_output_dir,
    f"multisat_cci_cropped_{start_date}_{end_date}.nc"
)

sat_data = SatelliteData(sat_path)
print(f"Detected data source: {sat_data.data_source}")  # 'cci'
print(f"Number of observations: {len(sat_data.time)}")

# Optional: inspect available wave height variables
if sat_data.swh_with_8m_offset_correction is not None:
    print("swh_with_8m_offset_correction is available (no 8 m cap).")
if sat_data.swh_quality_level is not None:
    print("Quality flags available — consider filtering before collocation.")

# ---------------------------------------------------------------------------
# 5. Load WW3 model data
# ---------------------------------------------------------------------------
s_time, e_time = start_date, end_date

model_run = WW3(
    rundir=model_path,
    model_dict={'var': 'hs'},
    start_date=np.datetime64(s_time),
    end_date=np.datetime64(e_time),
)

# ---------------------------------------------------------------------------
# 6. Perform collocation
# ---------------------------------------------------------------------------
# The Collocate engine works identically for CoastWatch and CCI data.
# Collocation is always based on 'swh' as the primary observation variable.
print("Starting collocation...")
coll = Collocate(
    model_run=model_run,
    observation=sat_data,
    n_nearest=3,
    temporal_interp=True,
)
ds_coll = coll.run(output_path=output_path)
print(f"Collocation complete. Results saved to: {output_path}")
