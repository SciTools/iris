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

import numpy as np

from iris.coords import AuxCoord, DimCoord
from iris.experimental.ugrid import Connectivity, mesh_from_coords
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
        return mesh_from_coords(self.lon, self.lat)

    def test_dimensionality(self):
        mesh = self.create()
        self.assertEqual(1, mesh.topology_dimension)

        self.assertArrayEqual(
            [0, 1, 1, 2, 2, 3], mesh.node_coords.node_x.points
        )
        self.assertArrayEqual(
            [0, 1, 2, 3, 1, 2], mesh.node_coords.node_y.points
        )
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
        self.lon = AuxCoord(
            points=self.lon.lazy_points(),
            bounds=self.lon.lazy_bounds(),
            standard_name="longitude",
        )
        self.lat = AuxCoord(
            points=self.lat.lazy_points(),
            bounds=self.lat.lazy_bounds(),
            standard_name="latitude",
        )

        mesh = self.create()
        for coord in list(mesh.all_coords):
            if coord is not None:
                self.assertTrue(coord.has_lazy_points())
        for conn in list(mesh.all_connectivities):
            if conn is not None:
                self.assertTrue(conn.has_lazy_indices())

    def test_not_coord(self):
        self.lon = "not a Coord"
        with self.assertRaisesRegex(ValueError, "Expected a Coord.*"):
            _ = self.create()

    def test_coord_shape_mismatch(self):
        lat_orig = self.lat.copy(self.lat.points, self.lat.bounds)
        self.lat = lat_orig.copy(
            points=lat_orig.points, bounds=np.tile(lat_orig.bounds, 2)
        )
        with self.assertRaisesRegex(ValueError, ".*bounds shape must be"):
            _ = self.create()

        self.lat = lat_orig.copy(
            points=lat_orig.points[-1], bounds=lat_orig.bounds[-1]
        )
        with self.assertRaisesRegex(ValueError, ".*points shape must be"):
            _ = self.create()

    def test_coords_not_orthogonal(self):
        def replace_standard_name(coord, standard_name):
            return coord.__class__(
                points=coord.points,
                bounds=coord.bounds,
                standard_name=standard_name,
                long_name=coord.long_name,
                var_name=coord.var_name,
                units=coord.units,
                attributes=coord.attributes,
            )

        # Identical to original lat, but with the same names as lon.
        self.lat = replace_standard_name(self.lat, self.lon.standard_name)
        with self.assertRaisesRegex(ValueError, ".*Both axes == X ."):
            _ = self.create()

        new_standard_name = "air_temperature"
        self.lon = replace_standard_name(self.lon, new_standard_name)
        self.lat = replace_standard_name(self.lat, new_standard_name)
        with self.assertRaisesRegex(
            ValueError, f".*Both standard_names == {new_standard_name}"
        ):
            _ = self.create()

        self.lon = replace_standard_name(self.lon, None)
        self.lat = replace_standard_name(self.lon, None)
        with self.assertRaisesRegex(
            ValueError, ".*Both coords' metadata identical."
        ):
            _ = self.create()


class Test2Dim(Test1Dim):
    def setUp(self):
        self.lon = DimCoord(
            points=[0.5, 1.5, 2.5],
            bounds=[[0, 0.5, 1], [1, 1.5, 2], [2, 2.5, 3]],
            standard_name="longitude",
            long_name="triangle longitudes",
            var_name="lon",
            units="degrees",
            attributes={"test": 1},
        )
        # Should be fine with either a DimCoord or an AuxCoord.
        self.lat = AuxCoord(
            points=[0.5, 2.5, 1.5],
            bounds=[[0, 1, 0], [2, 3, 2], [1, 2, 1]],
            standard_name="latitude",
            long_name="triangle latitudes",
            var_name="lat",
            units="degrees",
            attributes={"test": 1},
        )

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
                self.assertArrayEqual(
                    [[0, 1, 2], [3, 4, 5], [6, 7, 8]], conn.indices
                )
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


class TestInvalidBounds(tests.IrisTest):
    """Invalid bounds not supported."""

    def test_no_bounds(self):
        lon = AuxCoord(points=[0.5, 1.5, 2.5], bounds=[[0, 1], [1, 2], [2, 3]])
        lat = AuxCoord(points=[0, 1, 2])
        with self.assertRaisesRegex(
            ValueError, "coord_2 has invalid bounds: None"
        ):
            _ = mesh_from_coords(lon, lat)

    def test_1_bound(self):
        lon = AuxCoord(points=[0.5, 1.5, 2.5], bounds=[[0, 1], [1, 2], [2, 3]])
        lat = AuxCoord(points=[0, 1, 2], bounds=[[0.5], [1.5], [2.5]])
        with self.assertRaisesRegex(
            ValueError, r"coord_2 has invalid bounds: shape.*"
        ):
            _ = mesh_from_coords(lon, lat)


class TestInvalidPoints(tests.IrisTest):
    """Only 1D coords supported."""

    def test_2d_coord(self):
        cube = simple_2d_w_multidim_coords()[:3, :3]
        coord_1, coord_2 = cube.coords()
        with self.assertRaisesRegex(
            ValueError, "Coordinates must have ndim == 1.*"
        ):
            _ = mesh_from_coords(coord_1, coord_2)
