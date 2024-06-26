# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for cube arithmetic involving MeshCoords."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np

from iris.analysis.maths import add
from iris.coords import AuxCoord, DimCoord
from iris.tests.stock.mesh import sample_mesh, sample_mesh_cube
from iris.tests.unit.analysis.maths import (
    CubeArithmeticBroadcastingTestMixin,
    CubeArithmeticCoordsTest,
    MathsAddOperationMixin,
)


def _convert_to_meshcube(cube):
    """Convert a cube based on stock.realistic_4d into a "meshcube"."""
    # Replace lat+lon with a small mesh
    cube = cube[..., -1]  # remove final (X) dim
    for name in ("grid_longitude", "grid_latitude"):
        cube.remove_coord(name)
    i_meshdim = len(cube.shape) - 1
    n_meshpoints = cube.shape[i_meshdim]
    mesh = sample_mesh(n_nodes=n_meshpoints, n_faces=n_meshpoints, n_edges=0)
    for co in mesh.to_MeshCoords(location="face"):
        cube.add_aux_coord(co, i_meshdim)
    # also add a dim-coord for the mesh dim, mainly so that
    # the 'xxBroadcastingxx.test_collapse_all_dims' tests can do what they say.
    mesh_dimcoord = DimCoord(np.arange(n_meshpoints), long_name="i_mesh_face")
    cube.add_dim_coord(mesh_dimcoord, i_meshdim)
    return cube


class MeshLocationsMixin:
    # Control allowing us to also include test with derived coordinates.
    use_derived_coords = False

    # Modify the inherited data operation, to test with a mesh-cube.
    # Also, optionally, test with derived coordinates.
    def _base_testcube(self, include_derived=False):
        cube = super()._base_testcube(include_derived=self.use_derived_coords)
        cube = _convert_to_meshcube(cube)
        self.cube_xy_dimcoords = ["i_mesh_face"]
        self.cube = cube
        return self.cube


@tests.skip_data
class TestBroadcastingWithMesh(
    tests.IrisTest,
    MeshLocationsMixin,
    MathsAddOperationMixin,
    CubeArithmeticBroadcastingTestMixin,
):
    """Run all the broadcasting tests on cubes with meshes.

    NOTE: there is a fair amount of special-case code to support this, built
    into the CubeArithmeticBroadcastingTestMixin baseclass.

    """


@tests.skip_data
class TestBroadcastingWithMeshAndDerived(
    tests.IrisTest,
    MeshLocationsMixin,
    MathsAddOperationMixin,
    CubeArithmeticBroadcastingTestMixin,
):
    """Run broadcasting tests with meshes *and* derived coords."""

    use_derived = True


class TestCoordMatchWithMesh(CubeArithmeticCoordsTest):
    """Run the coordinate-mismatch tests with meshcubes."""

    def _convert_to_meshcubes(self, cubes, i_dim):
        """Add a mesh to one dim of the 'normal case' test-cubes."""
        for cube in cubes:
            n_size = cube.shape[i_dim]
            mesh = sample_mesh(n_nodes=n_size, n_faces=n_size, n_edges=0)
            for co in mesh.to_MeshCoords("face"):
                cube.add_aux_coord(co, i_dim)
            assert cube.mesh is not None

    def _check_no_match(self, dim):
        # Duplicate the basic operation, but convert cubes to meshcubes.
        cube1, cube2 = self.SetUpNonMatching()
        self._convert_to_meshcubes([cube1, cube2], dim)
        with self.assertRaises(ValueError):
            add(cube1, cube2)

    def test_no_match_dim0(self):
        self._check_no_match(0)

    def test_no_match_dim1(self):
        self._check_no_match(1)

    def _check_reversed_points(self, dim):
        # Duplicate the basic operation, but convert cubes to meshcubes.
        cube1, cube2 = self.SetUpReversed()
        self._convert_to_meshcubes([cube1, cube2], dim)
        with self.assertRaises(ValueError):
            add(cube1, cube2)

    def test_reversed_points_dim0(self):
        self._check_reversed_points(0)

    def test_reversed_points_dim1(self):
        self._check_reversed_points(1)


class TestBasicMeshOperation(tests.IrisTest):
    """Some very basic standalone tests, in an easier-to-comprehend form."""

    def test_meshcube_same_mesh(self):
        # Two similar cubes on a common mesh add to a third on the same mesh.
        mesh = sample_mesh()
        cube1 = sample_mesh_cube(mesh=mesh)
        cube2 = sample_mesh_cube(mesh=mesh)
        self.assertIs(cube1.mesh, mesh)
        self.assertIs(cube2.mesh, mesh)

        result = cube1 + cube2
        self.assertEqual(result.shape, cube1.shape)
        self.assertIs(result.mesh, mesh)

    def test_meshcube_different_equal_mesh(self):
        # Two similar cubes on identical but different meshes.
        cube1 = sample_mesh_cube()
        cube2 = sample_mesh_cube()
        self.assertEqual(cube1.mesh, cube2.mesh)
        self.assertIsNot(cube1.mesh, cube2.mesh)

        result = cube1 + cube2
        self.assertEqual(result.shape, cube1.shape)
        self.assertEqual(result.mesh, cube1.mesh)
        self.assertTrue(result.mesh is cube1.mesh or result.mesh is cube2.mesh)

    def test_fail_meshcube_nonequal_mesh(self):
        # Cubes on similar but different meshes -- should *not* combine.
        mesh1 = sample_mesh()
        mesh2 = sample_mesh(n_edges=0)
        self.assertNotEqual(mesh1, mesh2)
        cube1 = sample_mesh_cube(mesh=mesh1)
        cube2 = sample_mesh_cube(mesh=mesh2)

        msg = "Mesh coordinate.* does not match"
        with self.assertRaisesRegex(ValueError, msg):
            cube1 + cube2

    def test_meshcube_meshcoord(self):
        # Combining a meshcube and meshcoord.
        cube = sample_mesh_cube()
        cube.coord("latitude").units = "s"
        cube.units = "m"

        # A separately derived, but matching 'latitude' MeshCoord.
        coord = sample_mesh_cube().coord("latitude")
        coord.units = "s"  # N.B. the units **must also match**

        result = cube / coord
        self.assertEqual(result.name(), "unknown")
        self.assertEqual(result.units, "m s-1")

        # Moreover : *cannot* do this with the 'equivalent' AuxCoord
        # cf. https://github.com/SciTools/iris/issues/4671
        coord = AuxCoord.from_coord(coord)
        with self.assertRaises(ValueError):
            cube / coord


if __name__ == "__main__":
    tests.main()
