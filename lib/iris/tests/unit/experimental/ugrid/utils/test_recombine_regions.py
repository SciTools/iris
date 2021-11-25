# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for :func:`iris.experimental.ugrid.utils.recombine_regions`.

"""
# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np

from iris.experimental.ugrid.utils import recombine_regions
from iris.tests.stock.mesh import sample_mesh, sample_mesh_cube


def common_test_setup(self):
    n_mesh = 20
    mesh = sample_mesh(n_nodes=20, n_edges=0, n_faces=n_mesh)
    mesh_cube = sample_mesh_cube(n_z=2, mesh=mesh)
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
        cube.data[0] = i_cube + 1
        cube.data[1] = i_cube + 1001

    # Also construct an array to match the expected result.
    # basic layer showing region allocation (large -ve values for missing)
    expected = np.array(
        [
            1.0,
            1,
            1,
            1,
            1,
            4,
            4,
            2,
            2,
            2,
            3,
            3,
            3,
            -99999,
            -99999,  # missing points
            4,
            4,
            4,
            4,
            4,
        ]
    )
    # second layer should be same but +1000.
    expected = np.stack([expected, expected + 1000])
    # convert to masked array with missing points.
    expected = np.ma.masked_less(expected, 0)
    self.expected_result = expected


class TestRecombine__data(tests.IrisTest):
    def setUp(self):
        common_test_setup(self)

    def test_basic(self):
        result = recombine_regions(
            self.mesh_cube, self.region_cubes, index_coord_name="i_mesh_face"
        )
        self.assertMaskedArrayEqual(result.data, self.expected_result)

    def test_single_region(self):
        region = self.region_cubes[1]
        result = recombine_regions(
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
        result1 = recombine_regions(
            self.mesh_cube, [region1, region2], index_coord_name="i_mesh_face"
        )
        result1 = result1[..., inds].data
        self.assertArrayEqual(result1, 202.0)
        # swap the region order, and it should resolve the other way.
        result2 = recombine_regions(
            self.mesh_cube, [region2, region1], index_coord_name="i_mesh_face"
        )
        result2 = result2[..., inds].data
        self.assertArrayEqual(result2, 101.0)

    def test_missing_points(self):
        # check results with and without a specific region included.
        region2 = self.region_cubes[2]
        inds = region2.coord("i_mesh_face").points
        # With all regions, no points in reg1 are masked
        result_all = recombine_regions(
            self.mesh_cube, self.region_cubes, index_coord_name="i_mesh_face"
        )
        self.assertTrue(np.all(~result_all[..., inds].data.mask))
        # Without region1, all points in reg1 are masked
        regions_not2 = [
            cube for cube in self.region_cubes if cube is not region2
        ]
        result_not2 = recombine_regions(
            self.mesh_cube, regions_not2, index_coord_name="i_mesh_face"
        )
        self.assertTrue(np.all(result_not2[..., inds].data.mask))


class TestRecombine__checks(tests.IrisTest):
    def setUp(self):
        common_test_setup(self)

    def test_no_regions(self):
        with self.assertRaisesRegex(
            ValueError, "'region_cubes' must be non-empty"
        ):
            recombine_regions(self.mesh_cube, [])


if __name__ == "__main__":
    # Make it runnable in its own right.
    tests.main()
