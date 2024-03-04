# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Integration tests for loading LBC fieldsfiles."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import numpy as np

import iris


@tests.skip_data
class TestLBC(tests.IrisTest):
    def setUp(self):
        # Load multiple cubes from a test file.
        file_path = tests.get_data_path(("FF", "lbc", "small_lbc"))
        self.all_cubes = iris.load(file_path)
        # Select the second cube for detailed checks (the first is orography).
        self.test_cube = self.all_cubes[1]

    def test_various_cubes_shapes(self):
        # Check a few aspects of the loaded cubes.
        cubes = self.all_cubes
        self.assertEqual(len(cubes), 10)
        self.assertEqual(cubes[0].shape, (16, 16))
        self.assertEqual(cubes[1].shape, (2, 4, 16, 16))
        self.assertEqual(cubes[3].shape, (2, 5, 16, 16))

    def test_cube_coords(self):
        # Check coordinates of one cube.
        cube = self.test_cube
        self.assertEqual(len(cube.coords()), 8)
        for name, shape in [
            ("forecast_reference_time", (1,)),
            ("time", (2,)),
            ("forecast_period", (2,)),
            ("model_level_number", (4,)),
            ("level_height", (1,)),
            ("sigma", (1,)),
            ("grid_latitude", (16,)),
            ("grid_longitude", (16,)),
        ]:
            coords = cube.coords(name)
            self.assertEqual(
                len(coords),
                1,
                "expected one {!r} coord, found {}".format(name, len(coords)),
            )
            (coord,) = coords
            self.assertEqual(
                coord.shape,
                shape,
                "coord {!r} shape is {} instead of {!r}.".format(
                    name, coord.shape, shape
                ),
            )

    def test_cube_data(self):
        # Check just a few points of the data.
        cube = self.test_cube
        self.assertArrayAllClose(
            cube.data[:, ::2, 6, 13],
            np.array([[4.218922, 10.074577], [4.626897, 6.520156]]),
            atol=1.0e-6,
        )

    def test_cube_mask(self):
        # Check the data mask : should be just the centre 6x2 section.
        cube = self.test_cube
        mask = np.zeros((2, 4, 16, 16), dtype=bool)
        mask[:, :, 7:9, 5:11] = True
        self.assertArrayEqual(cube.data.mask, mask)


@tests.skip_data
class TestSkipField(tests.IrisTest):
    def test_missing_lbrel(self):
        infile = tests.get_data_path(("FF", "lbrel_missing"))
        with mock.patch("warnings.warn") as warn_fn:
            fields = iris.load(infile)
        self.assertIn(
            "Input field skipped as PPField creation failed : "
            "error = 'Unsupported header release number: -32768'",
            warn_fn.call_args[0][0],
        )
        self.assertEqual(len(fields), 2)


if __name__ == "__main__":
    tests.main()
