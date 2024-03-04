# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for :meth:`iris.analysis.trajectory.interpolate`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from collections import namedtuple

import numpy as np
import pytest

from iris.analysis.trajectory import interpolate
from iris.coords import AuxCoord, DimCoord
import iris.tests.stock


class TestFailCases(tests.IrisTest):
    @tests.skip_data
    def test_derived_coord(self):
        cube = iris.tests.stock.realistic_4d()
        sample_pts = [("altitude", [0, 10, 50])]
        msg = "'altitude'.*derived coordinates are not allowed"
        with self.assertRaisesRegex(ValueError, msg):
            interpolate(cube, sample_pts, "nearest")

        # Try to request unknown interpolation method.

    def test_unknown_method(self):
        cube = iris.tests.stock.simple_2d()
        sample_point = [("x", 2.8)]
        msg = "Unhandled interpolation.*linekar"
        with self.assertRaisesRegex(ValueError, msg):
            interpolate(cube, sample_point, method="linekar")


class TestNearest:
    # Test interpolation with 'nearest' method.
    # This is basically a wrapper to the routine:
    #   'analysis._interpolate_private._nearest_neighbour_indices_ndcoords'.
    # That has its own test, so we don't test the basic calculation
    # exhaustively here.  Instead we check the way it handles the source and
    # result cubes (especially coordinates).
    @pytest.fixture
    def src_cube(self):
        cube = iris.tests.stock.simple_3d()
        # Actually, this cube *isn't* terribly realistic, as the lat+lon coords
        # have integer type, which in this case produces some peculiar results.
        # Let's fix that (and not bother to test the peculiar behaviour).
        for coord_name in ("longitude", "latitude"):
            coord = cube.coord(coord_name)
            coord.points = coord.points.astype(float)
        return cube

    @pytest.fixture
    def single_point(self, src_cube):
        # Define coordinates for a single-point testcase.
        y_val, x_val = 0, -90

        # Use slightly-different values to test nearest-neighbour operation.
        sample_point = [
            ("latitude", [y_val + 19.23]),
            ("longitude", [x_val - 17.54]),
        ]

        # Work out cube indices of the testpoint.
        single_point_iy = np.where(src_cube.coord("latitude").points == y_val)[
            0
        ][0]
        single_point_ix = np.where(
            src_cube.coord("longitude").points == x_val
        )[0][0]

        point = namedtuple("point", "ix iy sample_point")
        return point(single_point_ix, single_point_iy, sample_point)

    @pytest.fixture
    def multi_sample_points(self):
        # Use latitude selection to recreate a whole row of the original cube.
        return [
            ("longitude", [-180, -90, 0, 90]),
            ("latitude", [0, 0, 0, 0]),
        ]

    @pytest.fixture
    def expected_multipoint_cube(self, src_cube):
        # The result should be identical to a single latitude section of the
        # original, but with modified coords (latitude has 4 repeated zeros).
        expected = src_cube[:, 1, :]
        # Result 'longitude' is now an aux coord.
        co_x = expected.coord("longitude")
        expected.remove_coord(co_x)
        expected.add_aux_coord(co_x, 1)
        # Result 'latitude' is now an aux coord containing 4*[0].
        expected.remove_coord("latitude")
        co_y = AuxCoord(
            [0, 0, 0, 0], standard_name="latitude", units="degrees"
        )
        expected.add_aux_coord(co_y, 1)

        return expected

    def test_single_point_same_cube(self, src_cube, single_point):
        # Check exact result matching for a single point.
        result = interpolate(
            src_cube, single_point.sample_point, method="nearest"
        )
        # Check that the result is a single trajectory point, exactly equal to
        # the expected part of the original data.
        assert result.shape[-1] == 1
        result = result[..., 0]
        expected = src_cube[:, single_point.iy, single_point.ix]
        assert result == expected

    def test_multi_point_same_cube(
        self, src_cube, multi_sample_points, expected_multipoint_cube
    ):
        # Check an exact result for multiple points.
        result = interpolate(src_cube, multi_sample_points, method="nearest")
        assert result == expected_multipoint_cube

    def test_mask_preserved(
        self, src_cube, multi_sample_points, expected_multipoint_cube
    ):
        mask = np.zeros_like(src_cube.data)
        mask[:, :, 1] = 1
        src_cube.data = np.ma.array(src_cube.data, mask=mask)

        expected_multipoint_cube.data = np.ma.array(
            expected_multipoint_cube.data, mask=mask[:, 0]
        )

        result = interpolate(src_cube, multi_sample_points, method="nearest")
        assert result == expected_multipoint_cube
        assert np.allclose(
            result.data.mask, expected_multipoint_cube.data.mask
        )

    def test_dtype_preserved(
        self, src_cube, multi_sample_points, expected_multipoint_cube
    ):
        src_cube.data = src_cube.data.astype(np.int16)

        result = interpolate(src_cube, multi_sample_points, method="nearest")
        assert result == expected_multipoint_cube
        assert np.allclose(result.data, expected_multipoint_cube.data)
        assert result.data.dtype == np.int16

    def test_aux_coord_noninterpolation_dim(self, src_cube, single_point):
        # Check exact result with an aux-coord mapped to an uninterpolated dim.
        src_cube.add_aux_coord(DimCoord([17, 19], long_name="aux0"), 0)

        # The result cube should exactly equal a single source point.
        result = interpolate(
            src_cube, single_point.sample_point, method="nearest"
        )
        assert result.shape[-1] == 1
        result = result[..., 0]
        expected = src_cube[:, single_point.iy, single_point.ix]
        assert result == expected

    def test_aux_coord_one_interp_dim(self, src_cube, single_point):
        # Check exact result with an aux-coord over one interpolation dims.
        src_cube.add_aux_coord(
            AuxCoord([11, 12, 13, 14], long_name="aux_x"), 2
        )

        # The result cube should exactly equal a single source point.
        result = interpolate(
            src_cube, single_point.sample_point, method="nearest"
        )
        assert result.shape[-1] == 1
        result = result[..., 0]
        expected = src_cube[:, single_point.iy, single_point.ix]
        assert result == expected

    def test_aux_coord_both_interp_dims(self, src_cube, single_point):
        # Check exact result with an aux-coord over both interpolation dims.
        src_cube.add_aux_coord(
            AuxCoord(
                [[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]],
                long_name="aux_xy",
            ),
            (1, 2),
        )

        # The result cube should exactly equal a single source point.
        result = interpolate(
            src_cube, single_point.sample_point, method="nearest"
        )
        assert result.shape[-1] == 1
        result = result[..., 0]
        expected = src_cube[:, single_point.iy, single_point.ix]
        assert result == expected

    def test_aux_coord_fail_mixed_dims(self, src_cube, single_point):
        # Check behaviour with an aux-coord mapped over both interpolation and
        # non-interpolation dims : not supported.
        src_cube.add_aux_coord(
            AuxCoord(
                [[111, 112, 113, 114], [211, 212, 213, 214]],
                long_name="aux_0x",
            ),
            (0, 2),
        )
        msg = (
            "Coord aux_0x at one x-y position has the shape.*"
            "instead of being a single point"
        )
        with pytest.raises(ValueError, match=msg):
            interpolate(src_cube, single_point.sample_point, method="nearest")

    def test_metadata(self, src_cube, single_point):
        # Check exact result matching for a single point, with additional
        # attributes and cell-methods.
        src_cube.attributes["ODD_ATTR"] = "string-value-example"
        src_cube.add_cell_method(iris.coords.CellMethod("mean", "area"))
        result = interpolate(
            src_cube, single_point.sample_point, method="nearest"
        )
        # Check that the result is a single trajectory point, exactly equal to
        # the expected part of the original data.
        assert result.shape[-1] == 1
        result = result[..., 0]
        expected = src_cube[:, single_point.iy, single_point.ix]
        assert result == expected


class TestLinear(tests.IrisTest):
    # Test interpolation with 'linear' method.
    #   This is basically a wrapper to 'analysis._scipy_interpolate''s
    #   _RegulardGridInterpolator. That has its own test, so we don't test the
    #   basic calculation exhaustively here.  Instead we check the way it
    #   handles the source and result cubes (especially coordinates).

    def setUp(self):
        cube = iris.tests.stock.simple_3d()
        # Actually, this cube *isn't* terribly realistic, as the lat+lon coords
        # have integer type, which in this case produces some peculiar results.
        # Let's fix that (and not bother to test the peculiar behaviour).
        for coord_name in ("longitude", "latitude"):
            coord = cube.coord(coord_name)
            coord.points = coord.points.astype(float)
        self.test_cube = cube
        # Set sample point to test single-point linear interpolation operation.
        self.single_sample_point = [
            ("latitude", [9]),
            ("longitude", [-120]),
        ]
        # Set expected results of single-point linear interpolation operation.
        self.single_sample_result = np.array(
            [
                64 / 15,
                244 / 15,
            ]
        )[:, np.newaxis]

    def test_single_point_same_cube(self):
        # Check exact result matching for a single point.
        cube = self.test_cube
        result = interpolate(cube, self.single_sample_point, method="linear")
        # Check that the result is a single trajectory point, exactly equal to
        # the expected part of the original data.
        self.assertEqual(result.shape[-1], 1)
        self.assertArrayAllClose(result.data, self.single_sample_result)

    def test_multi_point_same_cube(self):
        # Check an exact result for multiple points.
        cube = self.test_cube
        # Use latitude selection to recreate a whole row of the original cube.
        sample_points = [
            ("longitude", [-180, -90, 0, 90]),
            ("latitude", [0, 0, 0, 0]),
        ]
        result = interpolate(cube, sample_points, method="linear")

        # The result should be identical to a single latitude section of the
        # original, but with modified coords (latitude has 4 repeated zeros).
        expected = cube[:, 1, :]
        # Result 'longitude' is now an aux coord.
        co_x = expected.coord("longitude")
        expected.remove_coord(co_x)
        expected.add_aux_coord(co_x, 1)
        # Result 'latitude' is now an aux coord containing 4*[0].
        expected.remove_coord("latitude")
        co_y = AuxCoord(
            [0, 0, 0, 0], standard_name="latitude", units="degrees"
        )
        expected.add_aux_coord(co_y, 1)
        self.assertEqual(result, expected)

    def test_aux_coord_noninterpolation_dim(self):
        # Check exact result with an aux-coord mapped to an uninterpolated dim.
        cube = self.test_cube
        cube.add_aux_coord(DimCoord([17, 19], long_name="aux0"), 0)

        # The result cube should exactly equal a single source point.
        result = interpolate(cube, self.single_sample_point, method="linear")
        self.assertEqual(result.shape[-1], 1)
        self.assertArrayAllClose(result.data, self.single_sample_result)

    def test_aux_coord_one_interp_dim(self):
        # Check exact result with an aux-coord over one interpolation dims.
        cube = self.test_cube
        cube.add_aux_coord(AuxCoord([11, 12, 13, 14], long_name="aux_x"), 2)

        # The result cube should exactly equal a single source point.
        result = interpolate(cube, self.single_sample_point, method="linear")
        self.assertEqual(result.shape[-1], 1)
        self.assertArrayAllClose(result.data, self.single_sample_result)

    def test_aux_coord_both_interp_dims(self):
        # Check exact result with an aux-coord over both interpolation dims.
        cube = self.test_cube
        cube.add_aux_coord(
            AuxCoord(
                [[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]],
                long_name="aux_xy",
            ),
            (1, 2),
        )

        # The result cube should exactly equal a single source point.
        result = interpolate(cube, self.single_sample_point, method="linear")
        self.assertEqual(result.shape[-1], 1)
        self.assertArrayAllClose(result.data, self.single_sample_result)

    def test_aux_coord_fail_mixed_dims(self):
        # Check behaviour with an aux-coord mapped over both interpolation and
        # non-interpolation dims : not supported.
        cube = self.test_cube
        cube.add_aux_coord(
            AuxCoord(
                [[111, 112, 113, 114], [211, 212, 213, 214]],
                long_name="aux_0x",
            ),
            (0, 2),
        )
        msg = "Coord aux_0x was expected to have new points of shape .*\\. Found shape of .*\\."
        with self.assertRaisesRegex(ValueError, msg):
            interpolate(cube, self.single_sample_point, method="linear")

    def test_metadata(self):
        # Check exact result matching for a single point, with additional
        # attributes and cell-methods.
        cube = self.test_cube
        cube.attributes["ODD_ATTR"] = "string-value-example"
        cube.add_cell_method(iris.coords.CellMethod("mean", "area"))
        result = interpolate(cube, self.single_sample_point, method="linear")
        # Check that the result is a single trajectory point, exactly equal to
        # the expected part of the original data.
        self.assertEqual(result.shape[-1], 1)
        self.assertArrayAllClose(result.data, self.single_sample_result)


if __name__ == "__main__":
    tests.main()
