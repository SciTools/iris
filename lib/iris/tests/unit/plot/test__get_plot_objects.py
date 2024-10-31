# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.plot._get_plot_objects` function."""

import pytest

import iris.cube
from iris.tests import _shared_utils

if _shared_utils.MPL_AVAILABLE:
    from iris.plot import _get_plot_objects


@_shared_utils.skip_plot
class Test__get_plot_objects:
    def test_scalar(self):
        cube1 = iris.cube.Cube(1)
        cube2 = iris.cube.Cube(1)
        expected = (cube1, cube2, 1, 1, ())
        result = _get_plot_objects((cube1, cube2))
        assert result == expected

    def test_mismatched_size_first_scalar(self):
        cube1 = iris.cube.Cube(1)
        cube2 = iris.cube.Cube([1, 42])
        with pytest.raises(ValueError, match="x and y-axis objects are not compatible"):
            _get_plot_objects((cube1, cube2))

    def test_mismatched_size_second_scalar(self):
        cube1 = iris.cube.Cube(1)
        cube2 = iris.cube.Cube([1, 42])
        with pytest.raises(ValueError, match="x and y-axis objects are not compatible"):
            _get_plot_objects((cube2, cube1))
