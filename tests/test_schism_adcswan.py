"""Tests for SCHISM and ADCSWAN model classes."""

import os
import pytest
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from ocstrack.Model.model import SCHISM, ADCSWAN


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_schism_nc(path, time_val, n_nodes=10, include_3d=False):
    """Write a minimal SCHISM-style NetCDF file."""
    ds = xr.Dataset(
        {
            "sigWaveHeight": (
                ("time", "nSCHISM_hgrid_node"),
                np.random.rand(1, n_nodes).astype(np.float32),
            ),
        },
        coords={
            "time": ("time", np.array([time_val], dtype="datetime64[ns]")),
        },
    )
    if include_3d:
        ds["sigWaveHeight3d"] = xr.DataArray(
            np.random.rand(1, 5, n_nodes).astype(np.float32),
            dims=("time", "nSCHISM_vgrid_layers", "nSCHISM_hgrid_node"),
        )
    ds.to_netcdf(path)


def _make_hgrid(path, n_nodes=10):
    """Write a minimal hgrid.gr3 mesh file."""
    lines = ["test mesh\n", f"0 {n_nodes}\n"]
    for i in range(n_nodes):
        lines.append(f"{i+1} {float(i)} {float(i)} {10.0}\n")
    with open(path, "w") as f:
        f.writelines(lines)


@pytest.fixture(scope="module")
def schism_test_data(tmpdir_factory):
    """Create a minimal SCHISM run directory."""
    rundir = tmpdir_factory.mktemp("schism_run")
    outputs = rundir.mkdir("outputs")
    n_nodes = 10

    _make_hgrid(str(rundir.join("hgrid.gr3")), n_nodes)

    start = datetime(2023, 3, 1)
    for i in range(4):
        t = start + timedelta(hours=i * 6)
        fname = f"out2d_{i+1}.nc"
        _make_schism_nc(str(outputs.join(fname)), t, n_nodes)

    return str(rundir)


@pytest.fixture(scope="module")
def schism_3d_test_data(tmpdir_factory):
    """Create a minimal SCHISM run directory with 3D output."""
    rundir = tmpdir_factory.mktemp("schism_3d_run")
    outputs = rundir.mkdir("outputs")
    n_nodes = 10

    _make_hgrid(str(rundir.join("hgrid.gr3")), n_nodes)

    start = datetime(2023, 3, 1)
    for i in range(2):
        t = start + timedelta(hours=i * 6)
        fname = f"temperature_{i+1}.nc"
        zcor_name = f"zCoordinates_{i+1}.nc"
        # main var file
        ds_main = xr.Dataset(
            {"temperature": (("time", "nSCHISM_vgrid_layers", "nSCHISM_hgrid_node"),
                             np.random.rand(1, 5, n_nodes).astype(np.float32))},
            coords={"time": ("time", np.array([t], dtype="datetime64[ns]"))},
        )
        ds_main.to_netcdf(str(outputs.join(fname)))
        # zcor file
        ds_zcor = xr.Dataset(
            {"zCoordinates": (("time", "nSCHISM_vgrid_layers", "nSCHISM_hgrid_node"),
                              np.random.rand(1, 5, n_nodes).astype(np.float32))},
            coords={"time": ("time", np.array([t], dtype="datetime64[ns]"))},
        )
        ds_zcor.to_netcdf(str(outputs.join(zcor_name)))

    return str(rundir)


# ---------------------------------------------------------------------------
# SCHISM tests
# ---------------------------------------------------------------------------

class TestSCHISMValidation:
    def test_missing_required_key_raises(self, schism_test_data):
        with pytest.raises(ValueError, match="Missing keys"):
            SCHISM(
                rundir=schism_test_data,
                model_dict={"var": "sigWaveHeight"},  # missing 'startswith' and 'var_type'
                start_date=np.datetime64("2023-03-01"),
                end_date=np.datetime64("2023-03-02"),
            )

    def test_invalid_var_type_raises(self, schism_test_data):
        with pytest.raises(ValueError, match="var_type"):
            SCHISM(
                rundir=schism_test_data,
                model_dict={"startswith": "out2d", "var": "sigWaveHeight",
                            "var_type": "INVALID"},
                start_date=np.datetime64("2023-03-01"),
                end_date=np.datetime64("2023-03-02"),
            )

    def test_3d_profile_without_zcor_key_raises(self, schism_test_data):
        with pytest.raises(ValueError, match="zcor"):
            SCHISM(
                rundir=schism_test_data,
                model_dict={"startswith": "out2d", "var": "sigWaveHeight",
                            "var_type": "3D_Profile"},
                start_date=np.datetime64("2023-03-01"),
                end_date=np.datetime64("2023-03-02"),
            )


class TestSCHISMInit:
    def test_init_and_mesh_loaded(self, schism_test_data):
        model = SCHISM(
            rundir=schism_test_data,
            model_dict={"startswith": "out2d", "var": "sigWaveHeight", "var_type": "2D"},
            start_date=np.datetime64("2023-03-01"),
            end_date=np.datetime64("2023-03-02"),
        )
        assert len(model.mesh_x) == 10
        assert len(model.mesh_y) == 10
        assert len(model.mesh_depth) == 10

    def test_files_selected_within_range(self, schism_test_data):
        model = SCHISM(
            rundir=schism_test_data,
            model_dict={"startswith": "out2d", "var": "sigWaveHeight", "var_type": "2D"},
            start_date=np.datetime64("2023-03-01"),
            end_date=np.datetime64("2023-03-01T12"),
        )
        assert len(model.files) >= 1

    def test_no_files_outside_range(self, schism_test_data):
        model = SCHISM(
            rundir=schism_test_data,
            model_dict={"startswith": "out2d", "var": "sigWaveHeight", "var_type": "2D"},
            start_date=np.datetime64("2025-01-01"),
            end_date=np.datetime64("2025-01-02"),
        )
        assert len(model.files) == 0

    def test_load_variable_returns_dataarray(self, schism_test_data):
        model = SCHISM(
            rundir=schism_test_data,
            model_dict={"startswith": "out2d", "var": "sigWaveHeight", "var_type": "2D"},
            start_date=np.datetime64("2023-03-01"),
            end_date=np.datetime64("2023-03-02"),
        )
        assert len(model.files) > 0
        da = model.load_variable(model.files[0])
        assert isinstance(da, xr.DataArray)
        assert da.name == "sigWaveHeight"

    def test_mesh_x_setter_wrong_size_raises(self, schism_test_data):
        model = SCHISM(
            rundir=schism_test_data,
            model_dict={"startswith": "out2d", "var": "sigWaveHeight", "var_type": "2D"},
            start_date=np.datetime64("2023-03-01"),
            end_date=np.datetime64("2023-03-02"),
        )
        with pytest.raises(ValueError, match="must match"):
            model.mesh_x = np.zeros(999)

    def test_time_property_returns_sorted_array(self, schism_test_data):
        model = SCHISM(
            rundir=schism_test_data,
            model_dict={"startswith": "out2d", "var": "sigWaveHeight", "var_type": "2D"},
            start_date=np.datetime64("2023-03-01"),
            end_date=np.datetime64("2023-03-03"),
        )
        t = model.time
        assert len(t) > 0
        assert np.all(np.diff(t) >= np.timedelta64(0))


class TestSCHISM3DProfile:
    def test_load_3d_file_pair(self, schism_3d_test_data):
        model = SCHISM(
            rundir=schism_3d_test_data,
            model_dict={
                "startswith": "temperature",
                "var": "temperature",
                "var_type": "3D_Profile",
                "zcor_var": "zCoordinates",
                "zcor_startswith": "zCoordinates",
            },
            start_date=np.datetime64("2023-03-01"),
            end_date=np.datetime64("2023-03-02"),
        )
        assert len(model.files) > 0
        ds = model.load_3d_file_pair(model.files[0])
        assert "temperature" in ds
        assert "zCoordinates" in ds

    def test_load_3d_file_pair_missing_zcor_raises(self, schism_3d_test_data):
        model = SCHISM(
            rundir=schism_3d_test_data,
            model_dict={
                "startswith": "temperature",
                "var": "temperature",
                "var_type": "3D_Profile",
                "zcor_var": "zCoordinates",
                "zcor_startswith": "NONEXISTENT",
            },
            start_date=np.datetime64("2023-03-01"),
            end_date=np.datetime64("2023-03-02"),
        )
        assert len(model.files) > 0
        with pytest.raises(ValueError, match="Missing zcor file"):
            model.load_3d_file_pair(model.files[0])


# ---------------------------------------------------------------------------
# ADCSWAN tests
# ---------------------------------------------------------------------------

def _make_adcswan_nc(path, time_val, n_nodes=8):
    """Write a minimal ADCSWAN-style NetCDF file."""
    ds = xr.Dataset(
        {
            "x": (("node",), np.linspace(-75, -70, n_nodes)),
            "y": (("node",), np.linspace(34, 39, n_nodes)),
            "depth": (("node",), np.abs(np.random.rand(n_nodes)) * 50),
            "swan_HS": (
                ("time", "node"),
                np.random.rand(1, n_nodes).astype(np.float32),
            ),
        },
        coords={
            "time": ("time", np.array([time_val], dtype="datetime64[ns]")),
            "node": ("node", np.arange(n_nodes)),
        },
    )
    ds.to_netcdf(path)


@pytest.fixture(scope="module")
def adcswan_test_data(tmpdir_factory):
    """Create a minimal ADCSWAN run directory."""
    rundir = tmpdir_factory.mktemp("adcswan_run")
    t = datetime(2023, 5, 10)
    _make_adcswan_nc(str(rundir.join("swan_HS.63.nc")), t)
    return str(rundir)


class TestADCSWANValidation:
    def test_missing_required_key_raises(self, adcswan_test_data):
        with pytest.raises(ValueError, match="Missing keys"):
            ADCSWAN(
                rundir=adcswan_test_data,
                model_dict={"var": "swan_HS"},  # missing 'startswith'
                start_date=np.datetime64("2023-05-10"),
                end_date=np.datetime64("2023-05-11"),
            )


class TestADCSWANInit:
    def test_init_loads_mesh(self, adcswan_test_data):
        model = ADCSWAN(
            rundir=adcswan_test_data,
            model_dict={"startswith": "swan_HS.63", "var": "swan_HS"},
            start_date=np.datetime64("2023-05-10"),
            end_date=np.datetime64("2023-05-11"),
        )
        assert len(model.mesh_x) == 8
        assert len(model.mesh_y) == 8
        assert len(model.mesh_depth) == 8

    def test_init_selects_file(self, adcswan_test_data):
        model = ADCSWAN(
            rundir=adcswan_test_data,
            model_dict={"startswith": "swan_HS.63", "var": "swan_HS"},
            start_date=np.datetime64("2023-05-10"),
            end_date=np.datetime64("2023-05-11"),
        )
        assert len(model.files) == 1

    def test_no_file_outside_range(self, adcswan_test_data):
        model = ADCSWAN(
            rundir=adcswan_test_data,
            model_dict={"startswith": "swan_HS.63", "var": "swan_HS"},
            start_date=np.datetime64("2025-01-01"),
            end_date=np.datetime64("2025-01-02"),
        )
        assert len(model.files) == 0

    def test_load_variable_returns_dataarray(self, adcswan_test_data):
        model = ADCSWAN(
            rundir=adcswan_test_data,
            model_dict={"startswith": "swan_HS.63", "var": "swan_HS"},
            start_date=np.datetime64("2023-05-10"),
            end_date=np.datetime64("2023-05-11"),
        )
        da = model.load_variable(model.files[0])
        assert isinstance(da, xr.DataArray)
        assert da.name == "swan_HS"

    def test_mesh_x_setter_wrong_size_raises(self, adcswan_test_data):
        model = ADCSWAN(
            rundir=adcswan_test_data,
            model_dict={"startswith": "swan_HS.63", "var": "swan_HS"},
            start_date=np.datetime64("2023-05-10"),
            end_date=np.datetime64("2023-05-11"),
        )
        with pytest.raises(ValueError, match="must match"):
            model.mesh_x = np.zeros(999)

    def test_output_dir_points_to_rundir(self, adcswan_test_data):
        model = ADCSWAN(
            rundir=adcswan_test_data,
            model_dict={"startswith": "swan_HS.63", "var": "swan_HS"},
            start_date=np.datetime64("2023-05-10"),
            end_date=np.datetime64("2023-05-11"),
        )
        assert model.output_dir == adcswan_test_data
