# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.plot.hist` function."""

import numpy as np
import pytest

from iris.coords import AncillaryVariable, AuxCoord, CellMeasure, DimCoord
from iris.cube import Cube
from iris.tests import _shared_utils

if _shared_utils.MPL_AVAILABLE:
    import iris.plot as iplt


@_shared_utils.skip_plot
class Test:
    @pytest.fixture(autouse=True)
    def _create_data(self):
        self.data = np.array([0, 100, 110, 120, 200, 320])

    @pytest.mark.parametrize(
        "x", [AuxCoord, Cube, DimCoord, CellMeasure, AncillaryVariable]
    )
    def test_simple(self, x, mocker):
        mock_patch = mocker.patch("matplotlib.pyplot.hist")
        iplt.hist(x(self.data))
        # mocker.assert_called_once_with is not working as expected with
        # _DimensionalMetadata objects so we use array equality
        # checks instead.
        args, kwargs = mock_patch.call_args
        assert len(args) == 1
        _shared_utils.assert_array_equal(args[0], self.data)

    def test_kwargs(self, mocker):
        cube = Cube(self.data)
        bins = [0, 150, 250, 350]
        mock_patch = mocker.patch("matplotlib.pyplot.hist")
        iplt.hist(cube, bins=bins)
        mock_patch.assert_called_once_with(self.data, bins=bins)

    def test_unsupported_input(self):
        with pytest.raises(TypeError, match="x must be a"):
            iplt.hist(self.data)
