# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for :func:`iris.experimental.ugrid.utils.recombine_submeshes`.

"""
# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import dask.array as da
import numpy as np

from iris.coords import AuxCoord
from iris.cube import CubeList
from iris.experimental.ugrid.utils import recombine_submeshes
from iris.tests.stock.mesh import sample_mesh, sample_mesh_cube


def common_test_setup(self, shape_3d=(0, 2), data_chunks=None):
    # Construct a basic testcase with all-lazy mesh_cube and submesh_cubes
    # full-mesh cube shape is 'shape_3d'
    # data_chunks sets chunking of source cube, (else all-1-chunk)
    n_outer, n_z = shape_3d
    n_mesh = 20
    mesh = sample_mesh(n_nodes=20, n_edges=0, n_faces=n_mesh)
    mesh_cube = sample_mesh_cube(n_z=n_z, mesh=mesh)
    if n_outer:
        # Crudely merge a set of copies to build an outer dimension.
        mesh_cube.add_aux_coord(AuxCoord([0], long_name="outer"))
        meshcubes_2d = []
        for i_outer in range(n_outer):
            cube = mesh_cube.copy()
            cube.coord("outer").points = np.array([i_outer])
            meshcubes_2d.append(cube)
        mesh_cube = CubeList(meshcubes_2d).merge_cube()

    if not data_chunks:
        data_chunks = mesh_cube.shape[:-1] + (-1,)
    mesh_cube.data = da.zeros(mesh_cube.shape, chunks=data_chunks)

    n_regions = 4  # it doesn't divide neatly
    region_len = n_mesh // n_regions
    i_points = np.arange(n_mesh)
    region_inds = [
        np.where((i_points // region_len) == i_region)
        for i_region in range(n_regions)
    ]
    # Disturb slightly to ensure some gaps + some overlaps
    region_inds = [list(indarr[0]) for indarr in region_inds]
    region_inds[2] = region_inds[2][:-2]  # missing points
    region_inds[3] += region_inds[1][:2]  # duplicates
    self.mesh_cube = mesh_cube
    self.region_inds = region_inds
    self.region_cubes = [mesh_cube[..., inds] for inds in region_inds]
    for i_cube, cube in enumerate(self.region_cubes):
        for i_z in range(n_z):
            # Set data='z' ; don't vary over other dimensions.
            cube.data[..., i_z, :] = i_cube + 1000 * i_z + 1
            cube.data = cube.lazy_data()

    # Also construct an array to match the expected result (2d cases only).
    # basic layer showing region allocation (large -ve values for missing)
    expected = np.array(
        [1.0, 1, 1, 1, 1]
        + [4, 4]  # points in #1 overlapped by #3
        + [2, 2, 2]
        + [3, 3, 3]
        + [-99999, -99999]  # missing points
        + [4, 4, 4, 4, 4]
    )
    # second layer should be same but +1000.
    # NOTE: only correct if shape_3d=None; no current need to generalise this.
    expected = np.stack([expected, expected + 1000])
    # convert to masked array with missing points.
    expected = np.ma.masked_less(expected, 0)
    self.expected_result = expected


class TestRecombine__data(tests.IrisTest):
    def setUp(self):
        common_test_setup(self)

    def test_basic(self):
        # Just confirm that all source data is lazy (by default)
        for cube in self.region_cubes + [self.mesh_cube]:
            self.assertTrue(cube.has_lazy_data())
        result = recombine_submeshes(
            self.mesh_cube, self.region_cubes, index_coord_name="i_mesh_face"
        )
        self.assertTrue(result.has_lazy_data())
        self.assertMaskedArrayEqual(result.data, self.expected_result)

    def test_chunking(self):
        # Make non-standard testcube with higher dimensions + specific chunking
        common_test_setup(self, shape_3d=(10, 3), data_chunks=(3, 2, -1))
        self.assertEqual(self.mesh_cube.lazy_data().chunksize, (3, 2, 20))
        result = recombine_submeshes(
            self.mesh_cube, self.region_cubes, index_coord_name="i_mesh_face"
        )
        # Check that the result chunking matches the input.
        self.assertEqual(result.lazy_data().chunksize, (3, 2, 20))

    def test_single_region(self):
        region = self.region_cubes[1]
        result = recombine_submeshes(
            self.mesh_cube, [region], index_coord_name="i_mesh_face"
        )
        # Construct a snapshot of the expected result.
        # basic layer showing region allocation (large -ve values for missing)
        expected = np.ma.masked_array(np.zeros(self.mesh_cube.shape), True)
        inds = region.coord("i_mesh_face").points
        expected[..., inds] = region.data
        self.assertMaskedArrayEqual(result.data, expected)

    def test_region_overlaps(self):
        # generate two identical regions with different values.
        region1 = self.region_cubes[2]
        region1.data[:] = 101.0
        inds = region1.coord("i_mesh_face").points
        region2 = region1.copy()
        region2.data[:] = 202.0
        # check that result values all come from the second.
        result1 = recombine_submeshes(
            self.mesh_cube, [region1, region2], index_coord_name="i_mesh_face"
        )
        result1 = result1[..., inds].data
        self.assertArrayEqual(result1, 202.0)
        # swap the region order, and it should resolve the other way.
        result2 = recombine_submeshes(
            self.mesh_cube, [region2, region1], index_coord_name="i_mesh_face"
        )
        result2 = result2[..., inds].data
        self.assertArrayEqual(result2, 101.0)

    def test_missing_points(self):
        # check results with and without a specific region included.
        region2 = self.region_cubes[2]
        inds = region2.coord("i_mesh_face").points
        # With all regions, no points in reg1 are masked
        result_all = recombine_submeshes(
            self.mesh_cube, self.region_cubes, index_coord_name="i_mesh_face"
        )
        self.assertTrue(np.all(~result_all[..., inds].data.mask))
        # Without region1, all points in reg1 are masked
        regions_not2 = [
            cube for cube in self.region_cubes if cube is not region2
        ]
        result_not2 = recombine_submeshes(
            self.mesh_cube, regions_not2, index_coord_name="i_mesh_face"
        )
        self.assertTrue(np.all(result_not2[..., inds].data.mask))

    def test_transposed(self):
        self.mesh_cube.transpose()
        self.assertEqual(self.mesh_cube.mesh_dim(), 0)
        for cube in self.region_cubes:
            cube.transpose()
        result = recombine_submeshes(
            self.mesh_cube, self.region_cubes, index_coord_name="i_mesh_face"
        )
        self.assertTrue(result.has_lazy_data())
        self.assertEqual(result.mesh_dim(), 0)
        self.assertMaskedArrayEqual(
            result.data.transpose(), self.expected_result
        )

    def test_dtype(self):
        self.assertEqual(self.mesh_cube.dtype, np.float64)
        self.assertTrue(
            all(cube.dtype == np.float64 for cube in self.region_cubes)
        )
        result = recombine_submeshes(
            self.mesh_cube, self.region_cubes, index_coord_name="i_mesh_face"
        )
        self.assertEqual(result.dtype, np.float64)
        region_cubes2 = [
            cube.copy(data=cube.lazy_data().astype(np.int16))
            for cube in self.region_cubes
        ]
        result2 = recombine_submeshes(
            self.mesh_cube, region_cubes2, index_coord_name="i_mesh_face"
        )
        self.assertEqual(result2.dtype, np.int16)

    def test_meshcube_real(self):
        # Real data in reference 'mesh_cube' makes no difference.
        self.mesh_cube.data
        result = recombine_submeshes(
            self.mesh_cube, self.region_cubes, index_coord_name="i_mesh_face"
        )
        self.assertTrue(result.has_lazy_data())
        self.assertMaskedArrayEqual(result.data, self.expected_result)

    def test_regions_real(self):
        # Real data in reference 'mesh_cube' makes no difference.
        for cube in self.region_cubes:
            cube.data
        result = recombine_submeshes(
            self.mesh_cube, self.region_cubes, index_coord_name="i_mesh_face"
        )
        self.assertTrue(result.has_lazy_data())
        self.assertMaskedArrayEqual(result.data, self.expected_result)

    def test_allinput_real(self):
        # Real data in reference AND regions still makes no difference.
        self.mesh_cube.data
        for cube in self.region_cubes:
            cube.data
        result = recombine_submeshes(
            self.mesh_cube, self.region_cubes, index_coord_name="i_mesh_face"
        )
        self.assertTrue(result.has_lazy_data())
        self.assertMaskedArrayEqual(result.data, self.expected_result)

    def test_meshcube_masking(self):
        # Masked points in the reference 'mesh_cube' should make no difference.
        # get real data : copy as default is not writeable
        data = self.mesh_cube.data.copy()
        # mask all
        data[:] = np.ma.masked  # all masked
        # put back
        self.mesh_cube.data = data  # put back real array
        # recast as lazy
        self.mesh_cube.data = self.mesh_cube.lazy_data()  # remake as lazy
        # result should show no difference
        result = recombine_submeshes(
            self.mesh_cube, self.region_cubes, index_coord_name="i_mesh_face"
        )
        self.assertMaskedArrayEqual(result.data, self.expected_result)

    def test_no_missing_results(self):
        # For a result with no missing points, result array is still masked
        # get real data : copy as default is not writeable
        data = self.mesh_cube.data.copy()
        # set all
        data[:] = 7.777
        # put back
        self.mesh_cube.data = data  # put back real array
        # recast as lazy
        self.mesh_cube.data = self.mesh_cube.lazy_data()  # remake as lazy

        # get result including original full-mesh
        region_cubes = [self.mesh_cube] + self.region_cubes
        result = recombine_submeshes(
            self.mesh_cube, region_cubes, index_coord_name="i_mesh_face"
        )
        result = result.data
        expected = self.expected_result
        expected_missing = expected.mask
        expected[expected_missing] = 7.777
        # result is as "normal" expected, except at the usually-missing points.
        self.assertArrayEqual(result, expected)
        # the actual result array is still masked, though with no masked points
        self.assertIsInstance(result, np.ma.MaskedArray)
        self.assertIsInstance(result.mask, np.ndarray)
        self.assertArrayEqual(result.mask, False)

    def test_maskeddata(self):
        # Check that masked points within regions behave like ordinary values.
        # NB use overlap points
        # reg[1][0:2] == reg[3][5:7], but points in reg[3] dominate
        for cube in self.region_cubes:
            cube.data = np.ma.masked_array(cube.data)  # ensure masked arrays
        self.region_cubes[0].data[:, 0] = np.ma.masked  # result-index =5
        self.region_cubes[1].data[:, 0] = np.ma.masked  # result-index =5
        self.region_cubes[3].data[:, 6] = np.ma.masked  # result-index =6
        result = recombine_submeshes(
            self.mesh_cube, self.region_cubes, index_coord_name="i_mesh_face"
        )
        result = result.data
        expected = self.expected_result
        expected[:, 0] = np.ma.masked
        expected[:, 6] = np.ma.masked
        self.assertArrayEqual(result.mask, expected.mask)

    def test_nandata(self):
        # Check that NaN points within regions behave like ordinary values.
        # Duplicate of previous test, replacing masks with NaNs
        self.region_cubes[0].data[:, 0] = np.nan
        self.region_cubes[1].data[:, 0] = np.nan
        self.region_cubes[3].data[:, 6] = np.nan
        result = recombine_submeshes(
            self.mesh_cube, self.region_cubes, index_coord_name="i_mesh_face"
        )
        result = result.data
        expected = self.expected_result
        expected[:, 0] = np.nan
        expected[:, 6] = np.nan
        self.assertArrayEqual(np.isnan(result), np.isnan(expected))


class TestRecombine__checks(tests.IrisTest):
    def setUp(self):
        common_test_setup(self)

    def test_no_regions(self):
        with self.assertRaisesRegex(
            ValueError, "'submesh_cubes' must be non-empty"
        ):
            recombine_submeshes(self.mesh_cube, [])


if __name__ == "__main__":
    # Make it runnable in its own right.
    tests.main()
