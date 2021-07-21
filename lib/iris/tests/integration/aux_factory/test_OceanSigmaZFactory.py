# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Integratation tests for the
`iris.aux_factory.OceanSigmaZFactory` class.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import itertools

import numpy as np

from iris._lazy_data import as_lazy_data
from iris.tests.stock import ocean_sigma_z as stock_sample_osz
import iris.util


class Test_sample(tests.IrisTest):
    def setUp(self):
        self.cube = stock_sample_osz()
        # Snapshot result, printed with ...
        #     >>> np.set_printoptions(linewidth=180,
        #            formatter={'float':lambda x:'{:-09.3f}'.format(x)})
        #     >>> print(repr(coord.points))
        self.basic_derived_result = np.array(
            [
                [
                    [
                        [-0000.632, -0000.526, -0000.421, -0000.316],
                        [-0000.789, -0000.684, -0000.579, -0000.474],
                        [-0000.947, -0000.842, -0000.737, -0000.632],
                    ],
                    [
                        [-0014.358, -0014.264, -0014.169, -0014.074],
                        [-0014.501, -0014.406, -0014.311, -0014.216],
                        [-0014.643, -0014.548, -0014.453, -0014.358],
                    ],
                    [
                        [-0082.993, -0082.951, -0082.908, -0082.866],
                        [-0083.056, -0083.014, -0082.972, -0082.929],
                        [-0083.119, -0083.077, -0083.035, -0082.993],
                    ],
                    [
                        [-0368.400, -0368.400, -0368.400, -0368.400],
                        [-0368.400, -0368.400, -0368.400, -0368.400],
                        [-0368.400, -0368.400, -0368.400, -0368.400],
                    ],
                    [
                        [-1495.600, -1495.600, -1495.600, -1495.600],
                        [-1495.600, -1495.600, -1495.600, -1495.600],
                        [-1495.600, -1495.600, -1495.600, -1495.600],
                    ],
                ],
                [
                    [
                        [-0000.842, -0000.737, -0000.632, -0000.526],
                        [-0001.000, -0000.895, -0000.789, -0000.684],
                        [-0001.158, -0001.053, -0000.947, -0000.842],
                    ],
                    [
                        [-0014.548, -0014.453, -0014.358, -0014.264],
                        [-0014.690, -0014.595, -0014.501, -0014.406],
                        [-0014.832, -0014.737, -0014.643, -0014.548],
                    ],
                    [
                        [-0083.077, -0083.035, -0082.993, -0082.951],
                        [-0083.140, -0083.098, -0083.056, -0083.014],
                        [-0083.203, -0083.161, -0083.119, -0083.077],
                    ],
                    [
                        [-0368.400, -0368.400, -0368.400, -0368.400],
                        [-0368.400, -0368.400, -0368.400, -0368.400],
                        [-0368.400, -0368.400, -0368.400, -0368.400],
                    ],
                    [
                        [-1495.600, -1495.600, -1495.600, -1495.600],
                        [-1495.600, -1495.600, -1495.600, -1495.600],
                        [-1495.600, -1495.600, -1495.600, -1495.600],
                    ],
                ],
            ]
        )

        self.derived_coord_name = (
            "sea_surface_height_above_reference_ellipsoid"
        )

    def _check_result(self, cube, expected_result=None, **kwargs):
        if expected_result is None:
            expected_result = self.basic_derived_result
        coord = cube.coord(self.derived_coord_name)
        result = coord.points
        self.assertArrayAllClose(result, expected_result, atol=0.005, **kwargs)

    def test_basic(self):
        self._check_result(self.cube)

    def _lazy_testcube(self):
        cube = self.cube
        for dep_name in ("depth", "layer_depth", "ocean_sigma_z_coordinate"):
            coord = cube.coord(dep_name)
            coord.points = as_lazy_data(coord.points, coord.shape)
        return cube

    def test_nonlazy_cube_has_lazy_derived(self):
        # Check same results when key coords are made lazy.
        cube = self.cube
        self.assertEqual(cube.coord("depth").has_lazy_points(), False)
        self.assertEqual(
            cube.coord(self.derived_coord_name).has_lazy_points(), True
        )

    def test_lazy_cube_same_result(self):
        cube = self._lazy_testcube()
        self.assertEqual(cube.coord("depth").has_lazy_points(), True)
        self.assertEqual(
            cube.coord(self.derived_coord_name).has_lazy_points(), True
        )
        self._check_result(cube)

    def test_transpose(self):
        # Check it works with all possible dimension orders.
        for dims_list in itertools.permutations(range(self.cube.ndim)):
            cube = self.cube.copy()
            cube.transpose(dims_list)
            expected = self.basic_derived_result.transpose(dims_list)
            msg = "Unexpected result when cube transposed by {}"
            msg = msg.format(dims_list)
            self._check_result(cube, expected, err_msg=msg)

    def test_lazy_transpose(self):
        # Check lazy calc works with all possible dimension orders.
        for dims_list in itertools.permutations(range(self.cube.ndim)):
            cube = self._lazy_testcube().copy()
            cube.transpose(dims_list)
            expected = self.basic_derived_result.transpose(dims_list)
            msg = "Unexpected result when cube transposed by {}"
            msg = msg.format(dims_list)
            self._check_result(cube, expected, err_msg=msg)

    def test_extra_dims(self):
        # Insert some extra cube dimensions + check it still works.
        cube = self.cube
        cube = iris.util.new_axis(cube)
        cube = iris.util.new_axis(cube)
        cube = iris.util.new_axis(cube)
        # N.B. shape is now (1, 1, 1, t, z, y, x)
        cube.transpose((0, 3, 1, 4, 5, 2, 6))
        # N.B. shape is now (1, t, 1, z, y, 1, x)
        # Should get same original result, as derived dims are the same.
        self._check_result(cube)

    def test_no_sigma(self):
        # Check it still works when 'sigma' is removed.
        # NOTE: the unit test for this does not cover all cases because it
        # doesn't provide a time dimension.

        # Set all sigma points to zero + snapshot the resulting derived points.
        trial_cube = self.cube.copy()
        trial_cube.coord("ocean_sigma_z_coordinate").points[:] = 0.0
        expected = trial_cube.coord(self.derived_coord_name).points

        # Remove sigma altogether + check the result is the same.
        cube = self.cube
        cube.remove_coord("ocean_sigma_z_coordinate")
        self._check_result(cube, expected)

    def test_no_eta(self):
        # Check it still works when 'eta' is removed.
        # NOTE: the unit test for this does not cover all cases because it
        # doesn't provide a time dimension.

        # Set all sigma points to zero + snapshot the resulting derived points.
        trial_cube = self.cube.copy()
        trial_cube.coord("sea_surface_height").points[:] = 0.0
        expected = trial_cube.coord(self.derived_coord_name).points
        # Check this has no variation between the two timepoints.
        self.assertArrayAllClose(expected[0], expected[1])
        # Take first time, as no sigma --> result *has* no time dimension.
        expected = expected[0]

        # Remove eta altogether + check the result is the same.
        cube = self.cube
        cube.remove_coord("sea_surface_height")
        self._check_result(cube, expected)


if __name__ == "__main__":
    tests.main()
