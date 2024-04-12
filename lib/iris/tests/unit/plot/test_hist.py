# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.plot.hist` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from unittest import mock

import numpy as np
import pytest

from iris.coords import AncillaryVariable, AuxCoord, CellMeasure, DimCoord
from iris.cube import Cube

if tests.MPL_AVAILABLE:
    import iris.plot as iplt


@tests.skip_plot
class Test:
    @pytest.fixture(autouse=True)
    def create_data(self):
        self.data = np.array([0, 100, 110, 120, 200, 320])

    @pytest.mark.parametrize(
        "x", [AuxCoord, Cube, DimCoord, CellMeasure, AncillaryVariable]
    )
    def test_simple(self, x):
        with mock.patch("matplotlib.pyplot.hist") as mocker:
            iplt.hist(x(self.data))
        # mocker.assert_called_once_with is not working as expected with
        # _DimensionalMetadata objects so we use np.testing array equality
        # checks instead.
        args, kwargs = mocker.call_args
        assert len(args) == 1
        np.testing.assert_array_equal(args[0], self.data)

    def test_kwargs(self):
        cube = Cube(self.data)
        bins = [0, 150, 250, 350]
        with mock.patch("matplotlib.pyplot.hist") as mocker:
            iplt.hist(cube, bins=bins)
        mocker.assert_called_once_with(self.data, bins=bins)

    def test_unsupported_input(self):
        with pytest.raises(TypeError, match="x must be a"):
            iplt.hist(self.data)
