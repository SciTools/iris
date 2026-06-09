# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :meth:`iris.mesh.MeshXY.from_coords`."""

import numpy as np
import pytest

from iris.coords import AuxCoord, DimCoord
from iris.mesh import Connectivity, MeshXY, logger
from iris.tests import _shared_utils
from iris.tests.stock import simple_2d_w_multidim_coords


class Test1Dim:
    @pytest.fixture(autouse=True)
    def _setup_1d(self):
        self.lon = DimCoord(
            points=[0.5, 1.5, 2.5],
            bounds=[[0, 1], [1, 2], [2, 3]],
            standard_name="longitude",
            long_name="edge longitudes",
            var_name="lon",
            units="degrees",
            attributes={"test": 1},
        )
        # Should be fine with either a DimCoord or an AuxCoord.
        self.lat = AuxCoord(
            points=[0.5, 2.5, 1.5],
            bounds=[[0, 1], [2, 3], [1, 2]],
            standard_name="latitude",
            long_name="edge_latitudes",
            var_name="lat",
            units="degrees",
            attributes={"test": 1},
        )

    def create(self):
        return MeshXY.from_coords(self.lon, self.lat)

    def test_dimensionality(self):
        mesh = self.create()
        assert 1 == mesh.topology_dimension

        _shared_utils.assert_array_equal(
            [0, 1, 1, 2, 2, 3], mesh.node_coords.node_x.points
        )
        _shared_utils.assert_array_equal(
            [0, 1, 2, 3, 1, 2], mesh.node_coords.node_y.points
        )
        _shared_utils.assert_array_equal(
            [0.5, 1.5, 2.5], mesh.edge_coords.edge_x.points
        )
        _shared_utils.assert_array_equal(
            [0.5, 2.5, 1.5], mesh.edge_coords.edge_y.points
        )
        assert getattr(mesh, "face_coords", None) is None

        for conn_name in Connectivity.UGRID_CF_ROLES:
            conn = getattr(mesh, conn_name, None)
            if conn_name == "edge_node_connectivity":
                _shared_utils.assert_array_equal([[0, 1], [2, 3], [4, 5]], conn.indices)
            else:
                assert conn is None

    def test_node_metadata(self):
        mesh = self.create()
        pairs = [
            (self.lon, mesh.node_coords.node_x),
            (self.lat, mesh.node_coords.node_y),
        ]
        for expected_coord, actual_coord in pairs:
            for attr in ("standard_name", "long_name", "units", "attributes"):
                expected = getattr(expected_coord, attr)
                actual = getattr(actual_coord, attr)
                assert expected == actual
            assert actual_coord.var_name is None

    def test_centre_metadata(self):
        mesh = self.create()
        pairs = [
            (self.lon, mesh.edge_coords.edge_x),
            (self.lat, mesh.edge_coords.edge_y),
        ]
        for expected_coord, actual_coord in pairs:
            for attr in ("standard_name", "long_name", "units", "attributes"):
                expected = getattr(expected_coord, attr)
                actual = getattr(actual_coord, attr)
                assert expected == actual
            assert actual_coord.var_name is None

    def test_mesh_metadata(self):
        # Inappropriate to guess these values from the input coords.
        mesh = self.create()
        for attr in (
            "standard_name",
            "long_name",
            "var_name",
        ):
            assert getattr(mesh, attr) is None
        assert mesh.units.is_unknown()
        assert {} == mesh.attributes

    def test_lazy(self):
        self.lon = AuxCoord.from_coord(self.lon)
        self.lon = self.lon.copy(self.lon.lazy_points(), self.lon.lazy_bounds())
        self.lat = self.lat.copy(self.lat.lazy_points(), self.lat.lazy_bounds())

        mesh = self.create()
        for coord in list(mesh.all_coords):
            if coord is not None:
                assert coord.has_lazy_points()
        for conn in list(mesh.all_connectivities):
            if conn is not None:
                assert conn.has_lazy_indices()

    def test_coord_shape_mismatch(self):
        lat_orig = self.lat.copy(self.lat.points, self.lat.bounds)
        self.lat = lat_orig.copy(
            points=lat_orig.points, bounds=np.tile(lat_orig.bounds, 2)
        )
        with pytest.raises(ValueError, match="bounds shapes are not identical"):
            _ = self.create()

        self.lat = lat_orig.copy(points=lat_orig.points[-1], bounds=lat_orig.bounds[-1])
        with pytest.raises(ValueError, match="points shapes are not identical"):
            _ = self.create()

    def test_reorder(self):
        # Swap the coords.
        self.lat, self.lon = self.lon, self.lat
        mesh = self.create()
        # Confirm that the coords have been swapped back to the 'correct' order.
        assert "longitude" == mesh.node_coords.node_x.standard_name
        assert "latitude" == mesh.node_coords.node_y.standard_name

    def test_non_xy(self, caplog):
        for coord in self.lon, self.lat:
            coord.standard_name = None
        lon_name, lat_name = [coord.long_name for coord in (self.lon, self.lat)]
        # Swap the coords.
        self.lat, self.lon = self.lon, self.lat
        with _shared_utils.assert_logs(
            caplog, logger, "INFO", "Unable to find 'X' and 'Y'"
        ):
            mesh = self.create()
        # Confirm that the coords have not been swapped back.
        assert lat_name == mesh.node_coords.node_x.long_name
        assert lon_name == mesh.node_coords.node_y.long_name


class Test2Dim(Test1Dim):
    @pytest.fixture(autouse=True)
    def _setup_2d(self, _setup_1d):
        self.lon.bounds = [[0, 0.5, 1], [1, 1.5, 2], [2, 2.5, 3]]
        self.lon.long_name = "triangle longitudes"
        self.lat.bounds = [[0, 1, 0], [2, 3, 2], [1, 2, 1]]
        self.lat.long_name = "triangle latitudes"

    def test_dimensionality(self):
        mesh = self.create()
        assert 2 == mesh.topology_dimension

        _shared_utils.assert_array_equal(
            [0, 0.5, 1, 1, 1.5, 2, 2, 2.5, 3], mesh.node_coords.node_x.points
        )
        _shared_utils.assert_array_equal(
            [0, 1, 0, 2, 3, 2, 1, 2, 1], mesh.node_coords.node_y.points
        )
        assert mesh.edge_coords.edge_x is None
        assert mesh.edge_coords.edge_y is None
        _shared_utils.assert_array_equal(
            [0.5, 1.5, 2.5], mesh.face_coords.face_x.points
        )
        _shared_utils.assert_array_equal(
            [0.5, 2.5, 1.5], mesh.face_coords.face_y.points
        )

        for conn_name in Connectivity.UGRID_CF_ROLES:
            conn = getattr(mesh, conn_name, None)
            if conn_name == "face_node_connectivity":
                _shared_utils.assert_array_equal(
                    [[0, 1, 2], [3, 4, 5], [6, 7, 8]], conn.indices
                )
            else:
                assert conn is None

    def test_centre_metadata(self):
        mesh = self.create()
        pairs = [
            (self.lon, mesh.face_coords.face_x),
            (self.lat, mesh.face_coords.face_y),
        ]
        for expected_coord, actual_coord in pairs:
            for attr in ("standard_name", "long_name", "units", "attributes"):
                expected = getattr(expected_coord, attr)
                actual = getattr(actual_coord, attr)
                assert expected == actual
            assert actual_coord.var_name is None

    def test_mixed_shapes(self):
        self.lon = AuxCoord.from_coord(self.lon)
        lon_bounds = np.array([[0, 0, 1, 1], [1, 1, 2, 2], [2, 3, 2.5, 999]])
        self.lon.bounds = np.ma.masked_equal(lon_bounds, 999)

        lat_bounds = np.array([[0, 1, 1, 0], [1, 2, 2, 1], [2, 2, 3, 999]])
        self.lat.bounds = np.ma.masked_equal(lat_bounds, 999)

        mesh = self.create()
        _shared_utils.assert_array_equal(
            mesh.face_node_connectivity.location_lengths(), [4, 4, 3]
        )
        assert mesh.node_coords.node_x.points[-1] == 0.0
        assert mesh.node_coords.node_y.points[-1] == 0.0


class TestInvalidBounds:
    """Invalid bounds not supported."""

    def test_no_bounds(self):
        lon = AuxCoord(points=[0.5, 1.5, 2.5])
        lat = AuxCoord(points=[0, 1, 2])
        with pytest.raises(ValueError, match="bounds missing from"):
            _ = MeshXY.from_coords(lon, lat)

    def test_1_bound(self):
        lon = AuxCoord(points=[0.5, 1.5, 2.5], bounds=[[0], [1], [2]])
        lat = AuxCoord(points=[0, 1, 2], bounds=[[0.5], [1.5], [2.5]])
        with pytest.raises(
            ValueError, match=r"Expected coordinate bounds.shape \(n, >=2\)"
        ):
            _ = MeshXY.from_coords(lon, lat)


class TestInvalidPoints:
    """Only 1D coords supported."""

    def test_2d_coord(self):
        cube = simple_2d_w_multidim_coords()[:3, :3]
        coord_1, coord_2 = cube.coords()
        with pytest.raises(ValueError, match="Expected coordinate ndim == 1"):
            _ = MeshXY.from_coords(coord_1, coord_2)
