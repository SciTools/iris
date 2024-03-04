# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the function
:func:`iris.analysis.cartography.rotate_grid_vectors`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest.mock import Mock
from unittest.mock import call as mock_call

import numpy as np

from iris.analysis.cartography import rotate_grid_vectors
from iris.cube import Cube
from iris.tests.stock import sample_2d_latlons


class TestRotateGridVectors(tests.IrisTest):
    def _check_angles_calculation(
        self, angles_in_degrees=True, nan_angles_mask=None
    ):
        # Check basic maths on a 2d latlon grid.
        u_cube = sample_2d_latlons(regional=True, transformed=True)
        u_cube.units = "ms-1"
        u_cube.rename("dx")
        u_cube.data[...] = 0
        v_cube = u_cube.copy()
        v_cube.rename("dy")

        # Define 6 different vectors, repeated in each data row.
        in_vu = np.array([(0, 1), (2, -1), (-1, -1), (-3, 1), (2, 0), (0, 0)])
        in_angs = np.rad2deg(np.arctan2(in_vu[..., 0], in_vu[..., 1]))
        in_mags = np.sqrt(np.sum(in_vu * in_vu, axis=1))
        v_cube.data[...] = in_vu[..., 0]
        u_cube.data[...] = in_vu[..., 1]

        # Define 5 different test rotation angles, one for each data row.
        rotation_angles = np.array([0.0, -45.0, 135, -140.0, 90.0])
        ang_cube_data = np.broadcast_to(rotation_angles[:, None], u_cube.shape)
        ang_cube = u_cube.copy()
        if angles_in_degrees:
            ang_cube.units = "degrees"
        else:
            ang_cube.units = "radians"
            ang_cube_data = np.deg2rad(ang_cube_data)
        ang_cube.data[:] = ang_cube_data

        if nan_angles_mask is not None:
            ang_cube.data[nan_angles_mask] = np.nan

        # Rotate all vectors by all the given angles.
        result = rotate_grid_vectors(u_cube, v_cube, ang_cube)
        out_u, out_v = [cube.data for cube in result]

        # Check that vector magnitudes were unchanged.
        out_mags = np.sqrt(out_u * out_u + out_v * out_v)
        expect_mags = in_mags[None, :]
        self.assertArrayAllClose(out_mags, expect_mags)

        # Check that vector angles are all as expected.
        out_angs = np.rad2deg(np.arctan2(out_v, out_u))
        expect_angs = in_angs[None, :] + rotation_angles[:, None]
        ang_diffs = out_angs - expect_angs
        # Fix for null vectors, and +/-360 differences.
        ang_diffs[np.abs(out_mags) < 0.001] = 0.0
        ang_diffs[np.isclose(np.abs(ang_diffs), 360.0)] = 0.0
        # Check that any differences are very small.
        self.assertArrayAllClose(ang_diffs, 0.0)

        # Check that results are always masked arrays, masked at NaN angles.
        self.assertTrue(np.ma.isMaskedArray(out_u))
        self.assertTrue(np.ma.isMaskedArray(out_v))
        if nan_angles_mask is not None:
            self.assertArrayEqual(out_u.mask, nan_angles_mask)
            self.assertArrayEqual(out_v.mask, nan_angles_mask)

    def test_angles_calculation(self):
        self._check_angles_calculation()

    def test_angles_in_radians(self):
        self._check_angles_calculation(angles_in_degrees=False)

    def test_angles_from_grid(self):
        # Check it will gets angles from 'u_cube', and pass any kwargs on to
        # the angles routine.
        u_cube = sample_2d_latlons(regional=True, transformed=True)
        u_cube = u_cube[:2, :3]
        u_cube.units = "ms-1"
        u_cube.rename("dx")
        u_cube.data[...] = 1.0
        v_cube = u_cube.copy()
        v_cube.rename("dy")
        v_cube.data[...] = 0.0

        # Setup a fake angles result from the inner call to 'gridcell_angles'.
        angles_result_data = np.array(
            [[0.0, 90.0, 180.0], [-180.0, -90.0, 270.0]]
        )
        angles_result_cube = Cube(angles_result_data, units="degrees")
        angles_kwargs = {"this": 2}
        angles_call_patch = self.patch(
            "iris.analysis._grid_angles.gridcell_angles",
            Mock(return_value=angles_result_cube),
        )

        # Call the routine.
        result = rotate_grid_vectors(
            u_cube, v_cube, grid_angles_kwargs=angles_kwargs
        )

        self.assertEqual(
            angles_call_patch.call_args_list, [mock_call(u_cube, this=2)]
        )

        out_u, out_v = [cube.data for cube in result]
        # Records what results should be for the various n*90deg rotations.
        expect_u = np.array([[1.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])
        expect_v = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, -1.0]])
        # Check results are as expected.
        self.assertArrayAllClose(out_u, expect_u)
        self.assertArrayAllClose(out_v, expect_v)

    def test_nan_vectors(self):
        bad_angle_points = np.zeros((5, 6), dtype=bool)
        bad_angle_points[2, 3] = True
        self._check_angles_calculation(nan_angles_mask=bad_angle_points)


if __name__ == "__main__":
    tests.main()
