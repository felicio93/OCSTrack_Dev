# Contributing a New Model or Observation Type

This guide explains how to extend OCSTrack with support for a new ocean/wave
model or a new observational data source. The architecture is built around a
simple adapter pattern: each model class and each observation class exposes a
small, consistent interface that the `Collocate` engine relies on.

---

## Adding a New Model

### 1. Understand the required interface

The `Collocate` class only calls the following on a model object:

| Attribute / method | Type | Description |
|---|---|---|
| `model_dict` | `dict` | Must contain at least `'var'` and `'var_type'`. |
| `mesh_x` | `np.ndarray` | 1-D array of node longitudes (degrees). |
| `mesh_y` | `np.ndarray` | 1-D array of node latitudes (degrees). |
| `mesh_depth` | `np.ndarray` | 1-D array of node depths (metres, positive down). Use `np.nan` if unavailable. |
| `files` | `List[str]` | Ordered list of NetCDF file paths within the requested time range. |
| `time` | `np.ndarray` | Concatenated, sorted `datetime64` array across all selected files. |
| `load_variable(path)` | `xr.DataArray` | Returns the model variable for a given file. Must have a `'time'` dim and a spatial node dim. |
| `load_3d_file_pair(path)` *(3D only)* | `xr.Dataset` | Returns a dataset with the main variable and its z-coordinate, with dims `('time', 'node', ...)`. |

### 2. Create your class

Add a new class to `ocstrack/Model/model.py` (or a new file imported from
there). Use `SCHISM` or `WW3` as a reference template.

```python
class MyModel:
    """
    MyModel interface.
    """
    def __init__(self, rundir: str, model_dict: dict,
                 start_date: np.datetime64, end_date: np.datetime64):
        self.rundir = rundir
        self.model_dict = model_dict
        self.start_date = np.datetime64(start_date)
        self.end_date = np.datetime64(end_date)

        self._validate_model_dict()
        self._files = self._select_model_files()
        self._load_mesh_data()

    def _validate_model_dict(self) -> None:
        required = ['var', 'var_type']
        missing = [k for k in required if k not in self.model_dict]
        if missing:
            raise ValueError(f"Missing keys in model_dict: {missing}")

        valid_types = ['2D', '3D_Surface', '3D_Profile']
        if self.model_dict['var_type'] not in valid_types:
            raise ValueError(f"var_type must be one of {valid_types}")

    def _select_model_files(self) -> list:
        # Return a sorted list of NetCDF paths within start_date..end_date
        ...

    def _load_mesh_data(self) -> None:
        # Populate self._mesh_x, self._mesh_y, self._mesh_depth
        ...

    def load_variable(self, path: str):
        # Open path, return the DataArray for self.model_dict['var']
        # sliced to start_date..end_date
        ...

    @property
    def mesh_x(self): return self._mesh_x

    @property
    def mesh_y(self): return self._mesh_y

    @property
    def mesh_depth(self): return self._mesh_depth

    @property
    def files(self): return self._files

    @property
    def time(self):
        # Return concatenated, sorted datetime64 array across all files
        ...
```

### 3. Expose it from the package

Add your class to `ocstrack/Model/__init__.py` so users can import it cleanly:

```python
from .model import SCHISM, ADCSWAN, WW3, ROMS, MyModel
```

### 4. Write tests

Add a test file `tests/test_mymodel.py` following the pattern in
`tests/test_ww3.py`. Use `pytest` fixtures with `tmpdir_factory` to create
synthetic NetCDF files so no real model output is required.

---

## Adding a New Observation Type

### 1. Understand the required interface

The `Collocate` class requires the following from any observation object:

| Attribute | Type | Description |
|---|---|---|
| `ds` | `xr.Dataset` | The full observation dataset. |
| Time coordinate | `datetime64` dimension | For `SatelliteData` this is `'time'`; for `ArgoData` it is `'JULD'`. The coordinate name is passed via `obs_time_coord` inside `Collocate.__init__`. |
| Spatial coordinates | variables in `ds` | `'lat'`/`'lon'` for satellite; `'LATITUDE'`/`'LONGITUDE'` for Argo. |

`Collocate` also does an `isinstance` check to determine the time coordinate
name and the collocation strategy (2D vs 3D). To plug in a new type:

### 2. Create your class

Add a file under `ocstrack/Observation/`, e.g. `my_obs.py`:

```python
import xarray as xr

class MyObsData:
    """
    MyObsData handler.
    """
    def __init__(self, filepath: str):
        self.ds = xr.open_dataset(filepath)
        self._validate()

    def _validate(self) -> None:
        required = ['time', 'lat', 'lon', 'my_variable']
        missing = [v for v in required if v not in self.ds]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

    def filter_by_time(self, start_date: str, end_date: str) -> None:
        import numpy as np
        self.ds = self.ds.sortby('time').sel(
            time=slice(np.datetime64(start_date), np.datetime64(end_date))
        )
```

### 3. Register it in Collocate

Open `ocstrack/Collocation/collocate.py` and update the two places that
branch on observation type:

**In `__init__`** — add an `elif` to set the time coordinate name:

```python
elif isinstance(self.obs, MyObsData):
    self.obs_time_coord = 'time'
```

**In `run()`** — add an `elif` to dispatch to the correct collocation method:

```python
elif self.collocation_type == '2D':
    if not isinstance(self.obs, (SatelliteData, MyObsData)):
        raise TypeError("2D collocation requires SatelliteData or MyObsData.")
```

### 4. Expose it from the package

```python
# ocstrack/Observation/__init__.py
from .my_obs import MyObsData
```

### 5. Write tests

Add `tests/test_my_obs.py` following the pattern in `tests/test_observations.py`.
Use `tmp_path` fixtures and synthetic NetCDF files.

---

## General tips

- Keep model classes stateless after `__init__` — `load_variable` should open
  and close files rather than holding open file handles.
- Use `xr.open_dataset` as a context manager (`with xr.open_dataset(...) as ds`)
  wherever possible to avoid file handle leaks.
- Catch only specific exceptions (`OSError`, `KeyError`, `ValueError`) rather
  than bare `except Exception`.
- Run the test suite before opening a pull request:

```bash
pytest tests/ -v
```
