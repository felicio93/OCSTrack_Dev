# Observation Data

This section covers tools for downloading, processing, and handling observational data from satellite altimetry and Argo floats.

## Data Handlers

These classes are used to interact with processed observational data within the collocation workflow.

### Argo Floats

::: ocstrack.Observation.argofloat.ArgoData

### Satellite Altimetry

The `SatelliteData` class handles files from both supported satellite data sources. The format is detected automatically — the collocation workflow is identical regardless of the source.

::: ocstrack.Observation.satellite.SatelliteData

---

## Data Acquisition

These functions are high-level entry points for downloading and pre-processing raw data from public repositories.

### Argo Data Acquisition

Use these functions to download and prepare Argo float data.

!!! tip
    The main function to use here is `get_argo`. It orchestrates the entire download and processing pipeline.

::: ocstrack.Observation.get_argo.get_argo
::: ocstrack.Observation.get_argo.download_argo_data
::: ocstrack.Observation.get_argo.crop_argo_data
::: ocstrack.Observation.get_argo.clean_argo_data
::: ocstrack.Observation.get_argo.generate_monthly_dates
::: ocstrack.Observation.get_argo.crop_by_box_argo

### Satellite Data Acquisition — CoastWatch (NOAA STAR)

Daily merged NetCDF files from the NOAA STAR CoastWatch program. No credentials required. Note that CoastWatch SWH is capped at 8 m and old data may be removed without notice.

!!! tip
    The main functions to use here are `get_per_sat_coastwatch` for a single satellite and `get_multi_sat_coastwatch` for multiple satellites.

::: ocstrack.Observation.get_sat.get_per_sat_coastwatch
::: ocstrack.Observation.get_sat.get_multi_sat_coastwatch
::: ocstrack.Observation.get_sat.download_sat_data
::: ocstrack.Observation.get_sat.crop_sat_data
::: ocstrack.Observation.get_sat.concat_sat_data
::: ocstrack.Observation.get_sat.generate_daily_dates
::: ocstrack.Observation.get_sat.crop_by_box

### Satellite Data Acquisition — ESA CCI Sea State v5 (IFREMER)

Along-track per-pass files from the ESA Climate Change Initiative Sea State project, served from the IFREMER FTP server. No SWH cap; stable long-term archive. **Credentials required** — register at [https://eftp.ifremer.fr](https://eftp.ifremer.fr).

!!! tip
    The main functions to use here are `get_per_sat_cci` for a single satellite and `get_multi_sat_cci` for multiple satellites.

!!! warning
    FTP credentials are required. Never hard-code them in scripts. Use environment variables instead:
    ```bash
    export CCI_FTP_USER="your_username"
    export CCI_FTP_PASS="your_password"
    ```

::: ocstrack.Observation.get_sat_cci.get_per_sat_cci
::: ocstrack.Observation.get_sat_cci.get_multi_sat_cci
::: ocstrack.Observation.get_sat_cci.download_cci_data
::: ocstrack.Observation.get_sat_cci.crop_cci_data
::: ocstrack.Observation.get_sat_cci.crop_cci_data_by_shape
::: ocstrack.Observation.get_sat_cci.concat_cci_data

---

## Data URLs

This module contains the base URLs and FTP configuration for all data sources.

::: ocstrack.Observation.urls

