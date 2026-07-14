"""Tests for temporal collocation functions."""

import numpy as np
import pytest
import xarray as xr

from ocstrack.Collocation.temporal import temporal_nearest, temporal_interpolated


def _make_obs_ds(times, time_coord='time'):
    """Helper: build a minimal observation dataset."""
    return xr.Dataset(
        {
            'value': (['time'], np.ones(len(times))),
        },
        coords={time_coord: np.array(times, dtype='datetime64[ns]')},
    ).rename({'time': time_coord})


# ---------------------------------------------------------------------------
# temporal_nearest
# ---------------------------------------------------------------------------

class TestTemporalNearest:
    def test_returns_subset_and_indices(self):
        model_times = np.array(
            ['2023-01-01T00', '2023-01-01T06', '2023-01-01T12'],
            dtype='datetime64[ns]',
        )
        obs_times = np.array(
            ['2023-01-01T01', '2023-01-01T07', '2023-01-01T11'],
            dtype='datetime64[ns]',
        )
        buffer = np.timedelta64(3, 'h')
        ds_obs = _make_obs_ds(obs_times)

        obs_sub, nearest_inds, time_deltas = temporal_nearest(
            ds_obs, model_times, buffer
        )

        assert obs_sub.sizes['time'] == 3
        # obs at 01:00 → nearest model at 00:00 (idx 0)
        assert nearest_inds[0] == 0
        # obs at 07:00 → nearest model at 06:00 (idx 1)
        assert nearest_inds[1] == 1
        # obs at 11:00 → nearest model at 12:00 (idx 2)
        assert nearest_inds[2] == 2

    def test_buffer_excludes_obs_outside_window(self):
        model_times = np.array(
            ['2023-01-02T00', '2023-01-02T06'],
            dtype='datetime64[ns]',
        )
        obs_times = np.array(
            ['2023-01-01T00', '2023-01-02T03', '2023-01-03T00'],
            dtype='datetime64[ns]',
        )
        buffer = np.timedelta64(6, 'h')
        ds_obs = _make_obs_ds(obs_times)

        obs_sub, _, _ = temporal_nearest(ds_obs, model_times, buffer)
        # Only the middle observation is within the buffered window
        assert obs_sub.sizes['time'] == 1

    def test_time_deltas_are_in_seconds(self):
        model_times = np.array(['2023-06-01T12:00'], dtype='datetime64[ns]')
        obs_times = np.array(['2023-06-01T12:30'], dtype='datetime64[ns]')  # 30 min later
        buffer = np.timedelta64(2, 'h')
        ds_obs = _make_obs_ds(obs_times)

        _, _, time_deltas = temporal_nearest(ds_obs, model_times, buffer)
        assert time_deltas[0] == 1800  # 30 minutes = 1800 seconds

    def test_empty_result_when_no_obs_in_window(self):
        model_times = np.array(['2023-01-01T00'], dtype='datetime64[ns]')
        obs_times = np.array(['2023-06-01T00'], dtype='datetime64[ns]')
        buffer = np.timedelta64(1, 'h')
        ds_obs = _make_obs_ds(obs_times)

        obs_sub, nearest_inds, time_deltas = temporal_nearest(
            ds_obs, model_times, buffer
        )
        assert obs_sub.sizes['time'] == 0
        assert len(nearest_inds) == 0
        assert len(time_deltas) == 0

    def test_custom_time_coord_name(self):
        model_times = np.array(['2023-01-01T00'], dtype='datetime64[ns]')
        obs_times = np.array(['2023-01-01T01'], dtype='datetime64[ns]')
        buffer = np.timedelta64(3, 'h')
        ds_obs = _make_obs_ds(obs_times, time_coord='JULD')

        obs_sub, _, _ = temporal_nearest(
            ds_obs, model_times, buffer, time_coord_name='JULD'
        )
        assert obs_sub.sizes['JULD'] == 1


# ---------------------------------------------------------------------------
# temporal_interpolated
# ---------------------------------------------------------------------------

class TestTemporalInterpolated:
    def test_returns_correct_weights(self):
        model_times = np.array(
            ['2023-01-01T00', '2023-01-01T12'],
            dtype='datetime64[ns]',
        )
        # obs at 06:00 → weight should be 0.5
        obs_times = np.array(['2023-01-01T06'], dtype='datetime64[ns]')
        buffer = np.timedelta64(1, 'h')
        ds_obs = _make_obs_ds(obs_times)

        obs_sub, ib, ia, weights, _ = temporal_interpolated(
            ds_obs, model_times, buffer
        )
        assert obs_sub.sizes['time'] == 1
        assert ib[0] == 0
        assert ia[0] == 1
        np.testing.assert_allclose(weights[0], 0.5, rtol=1e-6)

    def test_obs_at_model_time_exact(self):
        model_times = np.array(
            ['2023-01-01T00', '2023-01-01T06'],
            dtype='datetime64[ns]',
        )
        # Obs exactly at the first model time — searchsorted gives idx=0,
        # i0=i1 so it's skipped; result should be empty.
        obs_times = np.array(['2023-01-01T00'], dtype='datetime64[ns]')
        buffer = np.timedelta64(1, 'h')
        ds_obs = _make_obs_ds(obs_times)

        obs_sub, ib, ia, weights, _ = temporal_interpolated(
            ds_obs, model_times, buffer
        )
        # At the boundary i0==i1, the point is skipped
        assert obs_sub.sizes['time'] == 0

    def test_weight_near_end(self):
        model_times = np.array(
            ['2023-01-01T00', '2023-01-01T04'],
            dtype='datetime64[ns]',
        )
        # obs at 03:00 → weight = 3/4 = 0.75
        obs_times = np.array(['2023-01-01T03'], dtype='datetime64[ns]')
        buffer = np.timedelta64(2, 'h')
        ds_obs = _make_obs_ds(obs_times)

        _, _, _, weights, _ = temporal_interpolated(
            ds_obs, model_times, buffer
        )
        np.testing.assert_allclose(weights[0], 0.75, rtol=1e-6)

    def test_time_deltas_computed(self):
        model_times = np.array(
            ['2023-03-15T00', '2023-03-15T06'],
            dtype='datetime64[ns]',
        )
        obs_times = np.array(['2023-03-15T01'], dtype='datetime64[ns]')
        buffer = np.timedelta64(3, 'h')
        ds_obs = _make_obs_ds(obs_times)

        _, _, _, _, time_deltas = temporal_interpolated(
            ds_obs, model_times, buffer
        )
        # Nearest model time is 00:00; obs is 01:00 → delta = 3600 s
        assert time_deltas[0] == 3600
