# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :func:`iris.experimental.ugrid.mesh_from_coords`.

"""
# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from iris.coords import AuxCoord, DimCoord
from iris.experimental.ugrid import Connectivity, mesh_from_coords


class Test1Dim(tests.IrisTest):
    def setUp(self):
        self.lon = DimCoord(
            points=[0.5, 1.5, 2.5],
            bounds=[[0, 1], [1, 2], [2, 3]],
            standard_name="longitude",
            long_name="test lon",
            var_name="lon",
            units="degrees",
            attributes={"test": 1}
            )
        # Should be fine with either a DimCoord or an AuxCoord.
        self.lat = AuxCoord(
            points=[0.5, 2.5, 1.5],
            bounds=[[0, 1], [2, 3], [1, 2]],
            long_name="test lat",
            var_name="lat",
            units="degrees",
            attributes={"test": 1}
        )

    def create(self):
        return mesh_from_coords(self.lon, self.lat)

    def test_dimensionality(self):
        mesh = self.create()
        self.assertEqual(1, mesh.topology_dimension)

        self.assertEqual([0, 1, 1, 2, 2, 3], mesh.node_x)
        self.assertEqual([0, 1, 2, 3, 1, 2], mesh.node_y)
        self.assertEqual([0.5, 1.5, 2.5], mesh.edge_x)
        self.assertEqual([0.5, 2.5, 1.5], mesh.edge_y)
        self.assertIsNotNone(mesh.face_x)
        self.assertIsNotNone(mesh.face_y)

        for conn_name in Connectivity.UGRID_CF_ROLES:
            conn = getattr(mesh, conn_name)
            if conn_name == "edge_node_connectivity":
                self.assertEqual([[0, 1], [2, 3], [4, 5]], conn.indices)
            else:
                self.assertIsNone(conn)

    def test_node_metadata(self):
        mesh = self.create()
        pairs = [(self.lon, mesh.node_x), (self.lat, mesh.node_y)]
        for expected_coord, actual_coord in pairs:
            for attr in ("standard_name", "long_name", "units", "attributes"):
                expected = getattr(expected_coord, attr)
                actual = getattr(actual_coord, attr)
                self.assertEqual(expected, actual)
            self.assertIsNone(actual_coord.var_name)

    def test_mesh_metadata(self):
        # Inappropriate to guess these values from the input coords.
        mesh = self.create()
        for attr in ("standard_name", "long_name", "var_name", "units", "attributes"):
            self.assertIsNone(getattr(mesh, attr))

    def test_not_coord(self):
        self.lon = "not a Coord"
        with self.assertRaisesRegex(ValueError, "Expected a Coord.*"):
            _ = self.create()

    def test_coord_shape_mismatch(self):
        lat_orig = self.lat.copy(self.lat.points, self.lat.bounds)
        self.lat = lat_orig.copy(points=lat_orig.points)
        with self.assertRaisesRegex(ValueError, ".*bounds shape must be"):
            _ = self.create()

        self.lat = lat_orig.copy(points=lat_orig.points[-1])
        with self.assertRaisesRegex(ValueError, ".*points shape must be"):
            _ = self.create()

    def test_coord_invalid_axis(self):
        # TODO: augment guess_axis to accept "X" or "Y".
        # Identical to original lon, but with a non-axis name.
        self.lon = self.lon.__class__(
            points=self.lon.points,
            bounds=self.lon.bounds,
            standard_name="air_temperature",
            long_name="foo",
            var_name="bar",
            units=self.lon.units,
            attributes=self.lon.attributes
        )
        with self.assertRaisesRegex(ValueError, "Unable to guess coord axis.*"):
            _ = self.create()

    def test_coord_axis_mismatch(self):
        # Identical to original lat, but with the same names as lon.
        self.lat = self.lat.__class__(
            points=self.lat.points,
            bounds=self.lat.bounds,
            standard_name=self.lon.standard_name,
            long_name=self.lon.long_name,
            var_name=self.lon.var_name,
            units=self.lat.units,
            attributes=self.lat.units
        )
        with self.assertRaisesRegex(ValueError, ""):
            _ = self.create()


class Test2Dim(Test1Dim):
    def setUp(self):
        pass

    def test_dimensionality(self):
        # topology_dimension, coords, connectivities
        pass


class TestInvalidBounds(tests.IrisTest):
    """Invalid bounds not supported."""
    def test_no_bounds(self):
        pass

    def test_1_bound(self):
        # shape = (n, 1)
        pass
