# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for loading LBC fieldsfiles."""

import numpy as np
import pytest

import iris
from iris.tests import _shared_utils


@_shared_utils.skip_data
class TestLBC:
    @pytest.fixture(autouse=True)
    def _setup(self):
        # Load multiple cubes from a test file.
        file_path = _shared_utils.get_data_path(("FF", "lbc", "small_lbc"))
        self.all_cubes = iris.load(file_path)
        # Select the second cube for detailed checks (the first is orography).
        self.test_cube = self.all_cubes[1]

    def test_various_cubes_shapes(self):
        # Check a few aspects of the loaded cubes.
        cubes = self.all_cubes
        assert len(cubes) == 10
        assert cubes[0].shape == (16, 16)
        assert cubes[1].shape == (2, 4, 16, 16)
        assert cubes[3].shape == (2, 5, 16, 16)

    def test_cube_coords(self):
        # Check coordinates of one cube.
        cube = self.test_cube
        assert len(cube.coords()) == 8
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
            assert len(coords) == 1, "expected one {!r} coord, found {}".format(
                name, len(coords)
            )
            (coord,) = coords
            assert coord.shape == shape, (
                "coord {!r} shape is {} instead of {!r}.".format(
                    name, coord.shape, shape
                )
            )

    def test_cube_data(self):
        # Check just a few points of the data.
        cube = self.test_cube
        _shared_utils.assert_array_all_close(
            cube.data[:, ::2, 6, 13],
            np.array([[4.218922, 10.074577], [4.626897, 6.520156]]),
            atol=1.0e-6,
        )

    def test_cube_mask(self):
        # Check the data mask : should be just the centre 6x2 section.
        cube = self.test_cube
        mask = np.zeros((2, 4, 16, 16), dtype=bool)
        mask[:, :, 7:9, 5:11] = True
        _shared_utils.assert_array_equal(cube.data.mask, mask)


@_shared_utils.skip_data
class TestSkipField:
    def test_missing_lbrel(self, mocker):
        infile = _shared_utils.get_data_path(("FF", "lbrel_missing"))
        with mocker.patch("warnings.warn") as warn_fn:
            fields = iris.load(infile)
        assert "Input field skipped as PPField creation failed : ", (
            "error = 'Unsupported header release number: -32768'"
            in warn_fn.call_args[0][0]
        )
        assert len(fields) == 2
