# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Integration tests for the
:mod:`iris.fileformats.ugrid_cf_reader.UGridCFReader` class.

"""
# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.cube import CubeList
from iris.fileformats.netcdf import load_cubes

from iris.util.ucube_operations import (
    ugrid_plot,
    identify_cubesphere,
    ucube_subset,
    ugrid_subset,
    pseudo_cube,
    PseudoshapedCubeIndexer,
    latlon_extract_faces,
)


def load_unstructured_testcube():
    # Load a standard unstructured cube to work with.
    testfile_path = tests.get_data_path(
        ("NetCDF", "unstructured_grid", "data_C4.nc")
    )

    cubes = CubeList(list(load_cubes(testfile_path)))
    (cube,) = cubes.extract("sample_data")

    return cube


@tests.skip_data
class TestUgridSubset(tests.IrisTest):
    # For now, only testing the 'face' extract functionality.
    def test_faces_subset(self):
        grid = load_unstructured_testcube().ugrid.grid
        selected_face_indices = [12, 3, 7]
        subset_grid = ugrid_subset(grid, selected_face_indices, "face")
        self.assertEqual(subset_grid.mesh_name, "mesh")
        self.assertTrue(np.all(subset_grid.nodes == grid.nodes))
        self.assertEqual(subset_grid.faces.shape, (3, 4))
        self.assertTrue(
            np.all(subset_grid.faces == grid.faces[selected_face_indices])
        )

    def test_faces_subset_boolarray(self):
        grid = load_unstructured_testcube().ugrid.grid
        faces_yesno = np.zeros(96, dtype=bool)
        faces_yesno[[1, 5, 3, 2, 8, 6]] = True
        subset_grid = ugrid_subset(grid, faces_yesno, "face")
        self.assertEqual(subset_grid.mesh_name, "mesh")
        self.assertTrue(np.all(subset_grid.nodes == grid.nodes))
        self.assertTrue(subset_grid.faces.shape == (6, 4))
        self.assertTrue(np.all(subset_grid.faces == grid.faces[faces_yesno]))


@tests.skip_data
class TestUcubeSubset(tests.IrisTest):
    # NOTE: the testdata we're using here has data mapped to faces.
    # For now, test just + only that functionality.
    def test_faces_subset_indices(self):
        cube = load_unstructured_testcube()
        selected_face_indices = [3, 5, 2, 17]
        subset_cube = ucube_subset(cube, selected_face_indices)
        self.assertIsNotNone(subset_cube.ugrid)
        self.assertEqual(subset_cube.ugrid.grid.mesh_name, "mesh")
        self.assertTrue(subset_cube.shape == (4,))
        self.assertTrue(
            np.all(subset_cube.data == cube.data[selected_face_indices])
        )

    def test_faces_subset_boolarray(self):
        cube = load_unstructured_testcube()
        faces_yesno = np.zeros(96, dtype=bool)
        faces_yesno[[1, 5, 3, 2, 8, 6]] = True
        subset_cube = ucube_subset(cube, faces_yesno)
        self.assertIsNotNone(subset_cube.ugrid)
        self.assertEqual(subset_cube.ugrid.grid.mesh_name, "mesh")
        self.assertTrue(subset_cube.shape == (6,))
        self.assertTrue(np.all(subset_cube.data == cube.data[faces_yesno]))


@tests.skip_data
class TestIdentifyCubesphere(tests.IrisTest):
    def test_identify(self):
        cube = load_unstructured_testcube()
        cube_shape = identify_cubesphere(cube.ugrid.grid)
        self.assertEqual(cube_shape, (6, 4, 4))


@tests.skip_data
class TestPlotCubesphere(tests.GraphicsTest):
    def test_plot(self):
        cube = load_unstructured_testcube()
        ugrid_plot(cube)
        self.check_graphic()


@tests.skip_data
class TestPseudoCube(tests.IrisTest):
    def test_pseudocube(self):
        cube = load_unstructured_testcube()
        shape = (6, 4, 4)
        names = ["n_face", "face_y", "face_x"]
        pseudo_cubesphere = pseudo_cube(cube, shape=shape, new_dim_names=names)
        self.assertEqual(pseudo_cubesphere.shape, (6, 4, 4))
        coord_names = [
            co.name() for co in pseudo_cubesphere.coords(dim_coords=True)
        ]
        self.assertEqual(coord_names, names)


@tests.skip_data
class TestPseudoshapedCubeIndexer(tests.IrisTest):
    def test_indexer(self):
        cube = load_unstructured_testcube()
        cube_shape = (6, 4, 4)
        cs_partial_cube = PseudoshapedCubeIndexer(cube, cube_shape)[1, 1:]
        self.assertIsNotNone(cs_partial_cube.ugrid)
        self.assertEqual(cs_partial_cube.ugrid.grid.mesh_name, "mesh")
        self.assertTrue(cs_partial_cube.shape == (12,))
        self.assertTrue(np.all(cs_partial_cube.data == cube.data[20:32]))


@tests.skip_data
class TestlatlonExtract(tests.IrisTest):
    def test_indexer(self):
        cube = load_unstructured_testcube()
        region = [-20, 60, 20, 65]
        region_cube = latlon_extract_faces(cube, region)
        self.assertIsNotNone(region_cube.ugrid)
        self.assertEqual(region_cube.ugrid.grid.mesh_name, "mesh")
        self.assertEqual(region_cube.shape, (7,))
        selected_face_indices = [1, 2, 3, 16, 68, 72, 76]
        self.assertTrue(
            np.all(region_cube.data == cube.data[selected_face_indices])
        )


if __name__ == "__main__":
    tests.main()
