"""Tests for GeocentricSpatialLocator and lat_lon_to_cartesian_vec."""

import numpy as np
import pytest

from ocstrack.Collocation.spatial import (
    GeocentricSpatialLocator,
    lat_lon_to_cartesian_vec,
    inverse_distance_weights,
)


# ---------------------------------------------------------------------------
# lat_lon_to_cartesian_vec
# ---------------------------------------------------------------------------

class TestLatLonToCartesian:
    def test_equator_prime_meridian(self):
        """At (lat=0, lon=0, h=0) X should equal WGS84 semi-major axis."""
        x, y, z = lat_lon_to_cartesian_vec(
            np.array([0.0]), np.array([0.0]), np.array([0.0])
        )
        a = 6378137.0
        np.testing.assert_allclose(x[0], a, rtol=1e-6)
        np.testing.assert_allclose(y[0], 0.0, atol=1.0)
        np.testing.assert_allclose(z[0], 0.0, atol=1.0)

    def test_north_pole(self):
        """At (lat=90, lon=0, h=0) only Z should be large."""
        x, y, z = lat_lon_to_cartesian_vec(
            np.array([90.0]), np.array([0.0]), np.array([0.0])
        )
        np.testing.assert_allclose(x[0], 0.0, atol=1.0)
        np.testing.assert_allclose(y[0], 0.0, atol=1.0)
        assert z[0] > 6_350_000  # close to WGS84 semi-minor axis

    def test_height_increases_radius(self):
        """Adding height should increase the radial distance."""
        x0, y0, z0 = lat_lon_to_cartesian_vec(
            np.array([45.0]), np.array([45.0]), np.array([0.0])
        )
        x1, y1, z1 = lat_lon_to_cartesian_vec(
            np.array([45.0]), np.array([45.0]), np.array([1000.0])
        )
        r0 = np.sqrt(x0**2 + y0**2 + z0**2)
        r1 = np.sqrt(x1**2 + y1**2 + z1**2)
        assert r1[0] > r0[0]

    def test_vectorised(self):
        """Function should work on arrays of multiple points."""
        lats = np.array([0.0, 45.0, 90.0])
        lons = np.array([0.0, 90.0, 0.0])
        heights = np.zeros(3)
        x, y, z = lat_lon_to_cartesian_vec(lats, lons, heights)
        assert x.shape == (3,)
        assert y.shape == (3,)
        assert z.shape == (3,)


# ---------------------------------------------------------------------------
# GeocentricSpatialLocator
# ---------------------------------------------------------------------------

class TestGeocentricSpatialLocator:
    @pytest.fixture
    def simple_locator(self):
        """A locator with 5 model nodes along the equator."""
        model_lon = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        model_lat = np.zeros(5)
        return GeocentricSpatialLocator(model_lon, model_lat)

    def test_query_nearest_returns_correct_shapes(self, simple_locator):
        sat_lon = np.array([0.5, 3.5])
        sat_lat = np.zeros(2)
        sat_h = np.zeros(2)
        dists, inds = simple_locator.query_nearest(sat_lon, sat_lat, sat_h, k=2)
        assert dists.shape == (2, 2)
        assert inds.shape == (2, 2)

    def test_query_nearest_finds_closest_node(self, simple_locator):
        """A query at lon=0.1 should find node 0 as the closest."""
        sat_lon = np.array([0.1])
        sat_lat = np.zeros(1)
        sat_h = np.zeros(1)
        dists, inds = simple_locator.query_nearest(sat_lon, sat_lat, sat_h, k=1)
        assert inds[0, 0] == 0

    def test_query_nearest_exact_match_distance_near_zero(self, simple_locator):
        """Querying at an exact model node location should give a very small distance."""
        sat_lon = np.array([2.0])
        sat_lat = np.zeros(1)
        sat_h = np.zeros(1)
        dists, inds = simple_locator.query_nearest(sat_lon, sat_lat, sat_h, k=1)
        assert inds[0, 0] == 2
        assert dists[0, 0] < 1000  # < 1 km

    def test_query_radius_returns_nodes_within_radius(self, simple_locator):
        """Nodes at lon=0,1,2 are within ~220 km of lon=1; lon=3 is ~222 km."""
        sat_lon = np.array([1.0])
        sat_lat = np.zeros(1)
        sat_h = np.zeros(1)
        # ~115 km per degree at equator; set radius to ~130 km
        radius_m = 130_000
        dists_list, inds_list = simple_locator.query_radius(
            sat_lon, sat_lat, sat_h, radius_m=radius_m
        )
        # Should include node 0, 1, 2 (within ~115km each)
        assert len(inds_list[0]) >= 2

    def test_query_radius_empty_when_no_nodes_nearby(self, simple_locator):
        """A query far from all nodes should return empty lists."""
        sat_lon = np.array([90.0])
        sat_lat = np.array([45.0])
        sat_h = np.zeros(1)
        dists_list, inds_list = simple_locator.query_radius(
            sat_lon, sat_lat, sat_h, radius_m=1.0  # 1 metre
        )
        assert len(inds_list[0]) == 0
        assert len(dists_list[0]) == 0

    def test_default_height_none_is_zero(self):
        """Passing model_height=None should default to sea level."""
        model_lon = np.array([0.0, 1.0])
        model_lat = np.array([0.0, 0.0])
        loc = GeocentricSpatialLocator(model_lon, model_lat, model_height=None)
        # Should not raise and should still find the nearest node
        sat_lon = np.array([0.0])
        sat_lat = np.array([0.0])
        sat_h = np.zeros(1)
        dists, inds = loc.query_nearest(sat_lon, sat_lat, sat_h, k=1)
        assert inds[0, 0] == 0
