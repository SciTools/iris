# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :meth:`iris.experimental.ugrid.mesh.Mesh.from_coords`."""
# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np

from iris.coords import AuxCoord, DimCoord
from iris.experimental.ugrid import logger
from iris.experimental.ugrid.mesh import Connectivity, Mesh
from iris.tests.stock import simple_2d_w_multidim_coords


class Test1Dim(tests.IrisTest):
    def setUp(self):
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
        return Mesh.from_coords(self.lon, self.lat)

    def test_dimensionality(self):
        mesh = self.create()
        self.assertEqual(1, mesh.topology_dimension)

        self.assertArrayEqual([0, 1, 1, 2, 2, 3], mesh.node_coords.node_x.points)
        self.assertArrayEqual([0, 1, 2, 3, 1, 2], mesh.node_coords.node_y.points)
        self.assertArrayEqual([0.5, 1.5, 2.5], mesh.edge_coords.edge_x.points)
        self.assertArrayEqual([0.5, 2.5, 1.5], mesh.edge_coords.edge_y.points)
        self.assertIsNone(getattr(mesh, "face_coords", None))

        for conn_name in Connectivity.UGRID_CF_ROLES:
            conn = getattr(mesh, conn_name, None)
            if conn_name == "edge_node_connectivity":
                self.assertArrayEqual([[0, 1], [2, 3], [4, 5]], conn.indices)
            else:
                self.assertIsNone(conn)

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
                self.assertEqual(expected, actual)
            self.assertIsNone(actual_coord.var_name)

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
                self.assertEqual(expected, actual)
            self.assertIsNone(actual_coord.var_name)

    def test_mesh_metadata(self):
        # Inappropriate to guess these values from the input coords.
        mesh = self.create()
        for attr in (
            "standard_name",
            "long_name",
            "var_name",
        ):
            self.assertIsNone(getattr(mesh, attr))
        self.assertTrue(mesh.units.is_unknown())
        self.assertDictEqual({}, mesh.attributes)

    def test_lazy(self):
        self.lon = AuxCoord.from_coord(self.lon)
        self.lon = self.lon.copy(self.lon.lazy_points(), self.lon.lazy_bounds())
        self.lat = self.lat.copy(self.lat.lazy_points(), self.lat.lazy_bounds())

        mesh = self.create()
        for coord in list(mesh.all_coords):
            if coord is not None:
                self.assertTrue(coord.has_lazy_points())
        for conn in list(mesh.all_connectivities):
            if conn is not None:
                self.assertTrue(conn.has_lazy_indices())

    def test_coord_shape_mismatch(self):
        lat_orig = self.lat.copy(self.lat.points, self.lat.bounds)
        self.lat = lat_orig.copy(
            points=lat_orig.points, bounds=np.tile(lat_orig.bounds, 2)
        )
        with self.assertRaisesRegex(ValueError, "bounds shapes are not identical"):
            _ = self.create()

        self.lat = lat_orig.copy(points=lat_orig.points[-1], bounds=lat_orig.bounds[-1])
        with self.assertRaisesRegex(ValueError, "points shapes are not identical"):
            _ = self.create()

    def test_reorder(self):
        # Swap the coords.
        self.lat, self.lon = self.lon, self.lat
        mesh = self.create()
        # Confirm that the coords have been swapped back to the 'correct' order.
        self.assertEqual("longitude", mesh.node_coords.node_x.standard_name)
        self.assertEqual("latitude", mesh.node_coords.node_y.standard_name)

    def test_non_xy(self):
        for coord in self.lon, self.lat:
            coord.standard_name = None
        lon_name, lat_name = [coord.long_name for coord in (self.lon, self.lat)]
        # Swap the coords.
        self.lat, self.lon = self.lon, self.lat
        with self.assertLogs(logger, "INFO", "Unable to find 'X' and 'Y'"):
            mesh = self.create()
        # Confirm that the coords have not been swapped back.
        self.assertEqual(lat_name, mesh.node_coords.node_x.long_name)
        self.assertEqual(lon_name, mesh.node_coords.node_y.long_name)


class Test2Dim(Test1Dim):
    def setUp(self):
        super().setUp()

        self.lon.bounds = [[0, 0.5, 1], [1, 1.5, 2], [2, 2.5, 3]]
        self.lon.long_name = "triangle longitudes"
        self.lat.bounds = [[0, 1, 0], [2, 3, 2], [1, 2, 1]]
        self.lat.long_name = "triangle latitudes"

    def test_dimensionality(self):
        mesh = self.create()
        self.assertEqual(2, mesh.topology_dimension)

        self.assertArrayEqual(
            [0, 0.5, 1, 1, 1.5, 2, 2, 2.5, 3], mesh.node_coords.node_x.points
        )
        self.assertArrayEqual(
            [0, 1, 0, 2, 3, 2, 1, 2, 1], mesh.node_coords.node_y.points
        )
        self.assertIsNone(mesh.edge_coords.edge_x)
        self.assertIsNone(mesh.edge_coords.edge_y)
        self.assertArrayEqual([0.5, 1.5, 2.5], mesh.face_coords.face_x.points)
        self.assertArrayEqual([0.5, 2.5, 1.5], mesh.face_coords.face_y.points)

        for conn_name in Connectivity.UGRID_CF_ROLES:
            conn = getattr(mesh, conn_name, None)
            if conn_name == "face_node_connectivity":
                self.assertArrayEqual([[0, 1, 2], [3, 4, 5], [6, 7, 8]], conn.indices)
            else:
                self.assertIsNone(conn)

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
                self.assertEqual(expected, actual)
            self.assertIsNone(actual_coord.var_name)

    def test_mixed_shapes(self):
        self.lon = AuxCoord.from_coord(self.lon)
        lon_bounds = np.array([[0, 0, 1, 1], [1, 1, 2, 2], [2, 3, 2.5, 999]])
        self.lon.bounds = np.ma.masked_equal(lon_bounds, 999)

        lat_bounds = np.array([[0, 1, 1, 0], [1, 2, 2, 1], [2, 2, 3, 999]])
        self.lat.bounds = np.ma.masked_equal(lat_bounds, 999)

        mesh = self.create()
        self.assertArrayEqual(mesh.face_node_connectivity.location_lengths(), [4, 4, 3])
        self.assertEqual(mesh.node_coords.node_x.points[-1], 0.0)
        self.assertEqual(mesh.node_coords.node_y.points[-1], 0.0)


class TestInvalidBounds(tests.IrisTest):
    """Invalid bounds not supported."""

    def test_no_bounds(self):
        lon = AuxCoord(points=[0.5, 1.5, 2.5])
        lat = AuxCoord(points=[0, 1, 2])
        with self.assertRaisesRegex(ValueError, "bounds missing from"):
            _ = Mesh.from_coords(lon, lat)

    def test_1_bound(self):
        lon = AuxCoord(points=[0.5, 1.5, 2.5], bounds=[[0], [1], [2]])
        lat = AuxCoord(points=[0, 1, 2], bounds=[[0.5], [1.5], [2.5]])
        with self.assertRaisesRegex(
            ValueError, r"Expected coordinate bounds.shape \(n, >=2\)"
        ):
            _ = Mesh.from_coords(lon, lat)


class TestInvalidPoints(tests.IrisTest):
    """Only 1D coords supported."""

    def test_2d_coord(self):
        cube = simple_2d_w_multidim_coords()[:3, :3]
        coord_1, coord_2 = cube.coords()
        with self.assertRaisesRegex(ValueError, "Expected coordinate ndim == 1"):
            _ = Mesh.from_coords(coord_1, coord_2)
