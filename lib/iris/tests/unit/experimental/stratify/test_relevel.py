# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :func:`iris.experimental.stratify.relevel` function."""

from functools import partial

import numpy as np
import pytest

from iris.coords import AuxCoord, DimCoord
from iris.tests import _shared_utils
import iris.tests.stock as stock

try:
    import stratify

    from iris.experimental.stratify import relevel
except ImportError:
    stratify = None


@_shared_utils.skip_stratify
class Test:
    @pytest.fixture(autouse=True)
    def _setup(self):
        cube = stock.simple_3d()[:, :1, :1]
        #: The data from which to get the levels.
        self.src_levels = cube.copy()
        #: The data to interpolate.
        self.cube = cube.copy()
        self.cube.rename("foobar")
        self.cube *= 10
        self.coord = self.src_levels.coord("wibble")
        self.axes = (self.coord, self.coord.name(), None, 0)

    def test_broadcast_fail_src_levels(self):
        emsg = "Cannot broadcast the cube and src_levels"
        data = np.arange(60).reshape(3, 4, 5)
        with pytest.raises(ValueError, match=emsg):
            relevel(self.cube, AuxCoord(data), [1, 2, 3])

    def test_broadcast_fail_tgt_levels(self):
        emsg = "Cannot broadcast the cube and tgt_levels"
        data = np.arange(60).reshape(3, 4, 5)
        with pytest.raises(ValueError, match=emsg):
            relevel(self.cube, self.coord, data)

    def test_standard_input(self):
        for axis in self.axes:
            result = relevel(self.cube, self.src_levels, [-1, 0, 5.5], axis=axis)
            _shared_utils.assert_array_equal(
                result.data.flatten(), np.array([np.nan, 0, 55])
            )
            expected = DimCoord([-1, 0, 5.5], units=1, long_name="thingness")
            assert expected == result.coord("thingness")

    def test_non_monotonic(self):
        for axis in self.axes:
            result = relevel(self.cube, self.src_levels, [2, 3, 2], axis=axis)
            _shared_utils.assert_array_equal(
                result.data.flatten(), np.array([20, 30, np.nan])
            )
            expected = AuxCoord([2, 3, 2], units=1, long_name="thingness")
            assert result.coord("thingness") == expected

    def test_static_level(self):
        for axis in self.axes:
            result = relevel(self.cube, self.src_levels, [2, 2], axis=axis)
            _shared_utils.assert_array_equal(result.data.flatten(), np.array([20, 20]))

    def test_coord_input(self):
        source = AuxCoord(self.src_levels.data)
        metadata = self.src_levels.metadata._asdict()
        metadata["coord_system"] = None
        metadata["climatological"] = None
        source.metadata = metadata

        for axis in self.axes:
            result = relevel(self.cube, source, [0, 12, 13], axis=axis)
            assert result.shape == (3, 1, 1)
            _shared_utils.assert_array_equal(result.data.flatten(), [0, 120, np.nan])

    def test_custom_interpolator(self):
        interpolator = partial(stratify.interpolate, interpolation="nearest")

        for axis in self.axes:
            result = relevel(
                self.cube,
                self.src_levels,
                [-1, 0, 6.5],
                axis=axis,
                interpolator=interpolator,
            )
            _shared_utils.assert_array_equal(
                result.data.flatten(), np.array([np.nan, 0, 120])
            )

    def test_multi_dim_target_levels(self, request):
        interpolator = partial(
            stratify.interpolate,
            interpolation="linear",
            extrapolation="linear",
        )

        for axis in self.axes:
            result = relevel(
                self.cube,
                self.src_levels,
                self.src_levels.data,
                axis=axis,
                interpolator=interpolator,
            )
            _shared_utils.assert_array_equal(result.data.flatten(), np.array([0, 120]))
            _shared_utils.assert_CML(request, result)
