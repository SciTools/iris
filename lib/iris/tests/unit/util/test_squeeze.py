# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.util.squeeze`."""

import pytest

import iris
import iris.tests.stock as stock


class Test:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = stock.simple_2d_w_multidim_and_scalars()

    def test_no_change(self):
        assert self.cube == iris.util.squeeze(self.cube)

    def test_squeeze_one_dim(self):
        cube_3d = iris.util.new_axis(self.cube, scalar_coord="an_other")
        cube_2d = iris.util.squeeze(cube_3d)

        assert self.cube == cube_2d

    def test_squeeze_two_dims(self):
        cube_3d = iris.util.new_axis(self.cube, scalar_coord="an_other")
        cube_4d = iris.util.new_axis(cube_3d, scalar_coord="air_temperature")

        assert self.cube == iris.util.squeeze(cube_4d)

    def test_squeeze_one_anonymous_dim(self):
        cube_3d = iris.util.new_axis(self.cube)
        cube_2d = iris.util.squeeze(cube_3d)

        assert self.cube == cube_2d

    def test_squeeze_to_scalar_cube(self):
        cube_scalar = self.cube[0, 0]
        cube_1d = iris.util.new_axis(cube_scalar)

        assert cube_scalar == iris.util.squeeze(cube_1d)
