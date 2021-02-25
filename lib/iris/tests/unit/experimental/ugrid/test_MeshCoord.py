# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :class:`iris.experimental.ugrid.MeshCoord`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import unittest.mock as mock

import numpy as np

from iris.cube import Cube
from iris.coords import AuxCoord, Coord
from iris.experimental.ugrid import Connectivity, Mesh

from iris.experimental.ugrid import MeshCoord

# Default test object creation controls
_TEST_N_FACES = 3
_TEST_N_BOUNDS = 4

# Default actual points + bounds.
_TEST_POINTS = np.arange(_TEST_N_FACES)
_TEST_BOUNDS = np.arange(_TEST_N_FACES * _TEST_N_BOUNDS)
_TEST_BOUNDS = _TEST_BOUNDS.reshape((_TEST_N_FACES, _TEST_N_BOUNDS))


def _default_create_args():
    # Produce a minimal set of default constructor args
    kwargs = {
        "standard_name": "longitude",
        "long_name": "long-name",
        "var_name": "var",
        "units": "degrees",
        "attributes": {"a": 1, "b": 2},
        "location": "face",
        "axis": "x",
        "mesh": mock.Mock(spec=Mesh),  # Dummy Mesh for default
        "points": _TEST_POINTS,
        "bounds": _TEST_BOUNDS,
    }
    # NOTE: *don't* include coord_system or climatology.
    # We expect to only set those (non-default) explicitly.
    return kwargs


def _create_test_meshcoord(**override_kwargs):
    kwargs = _default_create_args()
    # Apply requested overrides and additions.
    kwargs.update(override_kwargs)
    # Create and return the test coord.
    result = MeshCoord(**kwargs)
    return result


class Test___init__(tests.IrisTest):
    def setUp(self):
        self.meshcoord = _create_test_meshcoord()

    def test_basic(self):
        kwargs = _default_create_args()
        meshcoord = _create_test_meshcoord(**kwargs)
        for key, val in kwargs.items():
            if hasattr(val, "dtype"):
                self.assertArrayAllClose(getattr(meshcoord, key), val)
            else:
                self.assertEqual(getattr(meshcoord, key), val)
        self.assertIsInstance(meshcoord, MeshCoord)
        self.assertIsInstance(meshcoord, Coord)

    def test_fail_bad_mesh(self):
        with self.assertRaisesRegex(ValueError, "must be a.*Mesh"):
            _create_test_meshcoord(mesh=mock.sentinel.odd)

    def test_valid_locations(self):
        for loc in ("face", "edge", "node"):
            meshcoord = _create_test_meshcoord(location=loc)
            self.assertEqual(meshcoord.location, loc)

    def test_fail_bad_location(self):
        with self.assertRaisesRegex(ValueError, "not a valid Mesh location"):
            _create_test_meshcoord(location="bad")

    def test_valid_axes(self):
        for axis in ("x", "y"):
            meshcoord = _create_test_meshcoord(axis=axis)
            self.assertEqual(meshcoord.axis, axis)

    def test_fail_bad_axis(self):
        with self.assertRaisesRegex(ValueError, "not a valid Mesh axis"):
            _create_test_meshcoord(axis="q")

    def test_fail_bounds_no_points(self):
        # An specific error, *not* a signature mismatch (TypeError).
        # In future, this may actually be possible.
        with self.assertRaisesRegex(ValueError, "Cannot.*without points"):
            _create_test_meshcoord(points=None)


class Test__readonly_properties(tests.IrisTest):
    def setUp(self):
        self.meshcoord = _create_test_meshcoord()

    def test_fixed_metadata(self):
        # Check that you cannot set any of these on an existing MeshCoord.
        meshcoord = self.meshcoord
        props = ("mesh", "location", "axis", "coord_system", "climatological")
        msg = "Cannot set"
        for prop in props:
            with self.assertRaisesRegex(ValueError, msg):
                setattr(meshcoord, prop, mock.sentinel.odd)

    def test_coord_system(self):
        self.assertIsNone(self.meshcoord.coord_system)

    def test_fail_create_with_coord_system(self):
        # Check that you can't specify a non-None coord_system even at creation.
        with self.assertRaisesRegex(ValueError, "Cannot set"):
            _create_test_meshcoord(coord_system=mock.sentinel.odd)

    def test_climatological(self):
        self.assertFalse(self.meshcoord.climatological)

    def test_fail_create_with_climatological(self):
        # Check that you can't set 'climatological', even at creation.
        with self.assertRaisesRegex(ValueError, "Cannot set"):
            _create_test_meshcoord(climatological=True)


class Test__copy(tests.IrisTest):
    def test_basic(self):
        meshcoord = _create_test_meshcoord()
        meshcoord2 = meshcoord.copy()
        self.assertIsNot(meshcoord2, meshcoord)
        self.assertEqual(meshcoord2, meshcoord)
        # In this case, we should *NOT* copy the linked Mesh object.
        self.assertIs(meshcoord2.mesh, meshcoord.mesh)

    def test_points_no_bounds(self):
        meshcoord = _create_test_meshcoord()
        meshcoord2 = meshcoord.copy(points=meshcoord.core_points())
        points1, points2 = meshcoord.core_points(), meshcoord2.core_points()
        self.assertIsNot(points2, points1)
        self.assertArrayAllClose(points2, points1)
        self.assertIsNone(meshcoord2.bounds)
        self.assertIs(meshcoord2.mesh, meshcoord.mesh)

    def test_fail_bounds_no_points(self):
        meshcoord = _create_test_meshcoord()
        with self.assertRaisesRegex(ValueError, "Cannot.*without points"):
            meshcoord.copy(bounds=meshcoord.core_bounds())

    def test_fail_points_bounds_mismatch(self):
        meshcoord = _create_test_meshcoord()
        with self.assertRaisesRegex(ValueError, "shape must be compatible"):
            meshcoord.copy(
                points=meshcoord.core_points(),
                bounds=meshcoord.core_bounds()[:-1],
            )


def _create_test_mesh():
    xco = AuxCoord(np.zeros(_TEST_N_FACES), standard_name="longitude")
    yco = AuxCoord(np.zeros(_TEST_N_FACES), standard_name="latitude")
    face_nodes = Connectivity(
        np.zeros((_TEST_N_FACES, _TEST_N_BOUNDS), dtype=int),
        cf_role="face_node_connectivity",
    )
    mesh = Mesh(
        topology_dimension=2,
        node_coords_and_axes=[(xco, "x"), (yco, "y")],
        connectivities=[face_nodes],
    )
    return mesh


class Test___eq__(tests.IrisTest):
    # We must do this test with *actual* Mesh objects.
    def setUp(self):
        self.mesh = _create_test_mesh()

    def _create_common_mesh(self, **kwargs):
        return _create_test_meshcoord(mesh=self.mesh, **kwargs)

    def test_same_mesh(self):
        meshcoord1 = self._create_common_mesh()
        meshcoord2 = self._create_common_mesh()
        self.assertEqual(meshcoord2, meshcoord1)

    def test_different_identical_mesh(self):
        # For equality, must have the SAME mesh (at present).
        mesh1 = _create_test_mesh()
        mesh2 = _create_test_mesh()  # Presumably identical, but not the same
        meshcoord1 = _create_test_meshcoord(mesh=mesh1)
        meshcoord2 = _create_test_meshcoord(mesh=mesh2)
        # These should NOT compare, because the Meshes are not identical : at
        # present, Mesh equality is not implemented (i.e. limited to identity)
        self.assertNotEqual(
            meshcoord2, meshcoord1
        )  # NOTE: this assumes Mesh equality

    def test_different_location(self):
        meshcoord = self._create_common_mesh()
        meshcoord2 = self._create_common_mesh(location="node")
        self.assertNotEqual(meshcoord2, meshcoord)

    def test_different_axis(self):
        meshcoord = self._create_common_mesh()
        meshcoord2 = self._create_common_mesh(axis="y")
        self.assertNotEqual(meshcoord2, meshcoord)

    def test_different_common_metadata(self):
        # Actually handled by parent class, but maybe worth checking.
        meshcoord = self._create_common_mesh()
        meshcoord2 = self._create_common_mesh(long_name="blurb")
        self.assertNotEqual(meshcoord2, meshcoord)

    def test_different_points(self):
        # Actually handled by parent class, but maybe worth checking.
        meshcoord = self._create_common_mesh()
        points = meshcoord.core_points() + 1.0
        bounds = meshcoord.core_bounds()
        meshcoord2 = self._create_common_mesh(points=points, bounds=bounds)
        self.assertNotEqual(meshcoord2, meshcoord)


class Test_cube_containment(tests.IrisTest):
    # Check that we can put a MeshCoord into a cube, and have it behave just
    # like a regular AuxCoord
    def setUp(self):
        meshcoord = _create_test_meshcoord()
        data_shape = (2,) + _TEST_POINTS.shape
        cube = Cube(np.zeros(data_shape))
        cube.add_aux_coord(meshcoord, 1)
        self.meshcoord = meshcoord
        self.cube = cube

    def test_added_to_cube(self):
        meshcoord = self.meshcoord
        cube = self.cube
        self.assertIn(meshcoord, cube.coords())

    def test_cube_dims(self):
        meshcoord = self.meshcoord
        cube = self.cube
        self.assertEqual(meshcoord.cube_dims(cube), (1,))
        self.assertEqual(cube.coord_dims(meshcoord), (1,))

    def test_find_by_name(self):
        meshcoord = self.meshcoord
        cube = self.cube
        self.assertIs(cube.coord(standard_name="longitude"), meshcoord)
        self.assertIs(cube.coord(long_name="long-name"), meshcoord)
        self.assertIs(cube.coord(var_name="var"), meshcoord)

    def test_find_by_axis(self):
        meshcoord = self.meshcoord
        cube = self.cube
        self.assertIs(cube.coord(axis="x"), meshcoord)
        self.assertEqual(cube.coords(axis="y"), [])

        # NOTE: the meshcoord.axis takes precedence over the older
        # "guessed axis" approach.  So the standard_name does not control it.
        cube.remove_coord(meshcoord)
        meshcoord = _create_test_meshcoord(standard_name="longitude", axis="y")
        cube.add_aux_coord(meshcoord, (1,))
        self.assertIs(cube.coord(axis="y"), meshcoord)
        self.assertEqual(cube.coords(axis="x"), [])


if __name__ == "__main__":
    tests.main()
