# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Common test code for `iris.plot.pcolor` and `iris.plot.pcolormesh`."""

import numpy as np
import pytest

from iris.tests.stock import simple_2d
from iris.tests.unit.plot import MixinCoords


class MixinStringCoordPlot:
    # Mixin for common string-coord tests on pcolor/pcolormesh.
    # To use, make a class that inherits from this *and*
    # :class:`iris.tests.unit.plot.TestGraphicStringCoord`,
    # and defines "self.blockplot_func()", to return the `iris.plot` function.
    def test_yaxis_labels(self):
        self.blockplot_func()(self.cube, coords=("bar", "str_coord"))
        self.assert_bounds_tick_labels("yaxis")

    def test_xaxis_labels(self):
        self.blockplot_func()(self.cube, coords=("str_coord", "bar"))
        self.assert_bounds_tick_labels("xaxis")

    def test_xaxis_labels_with_axes(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 3)
        self.blockplot_func()(self.cube, coords=("str_coord", "bar"), axes=ax)
        plt.close(fig)
        self.assert_points_tick_labels("xaxis", ax)

    def test_yaxis_labels_with_axes(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylim(0, 3)
        self.blockplot_func()(self.cube, axes=ax, coords=("bar", "str_coord"))
        plt.close(fig)
        self.assert_points_tick_labels("yaxis", ax)

    def test_geoaxes_exception(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        pytest.raises(TypeError, self.blockplot_func(), self.lat_lon_cube, axes=ax)
        plt.close(fig)


class Mixin2dCoordsPlot(MixinCoords):
    # Mixin for common coordinate tests on pcolor/pcolormesh.
    # To use, make a class that inherits from this *and*
    # defines "self.blockplot_func()", to return the `iris.plot` function.
    @pytest.fixture(autouse=True)
    def _blockplot_setup(self, mocker):
        # We have a 2d cube with dimensionality (bar: 3; foo: 4)
        self.cube = simple_2d(with_bounds=True)
        coord = self.cube.coord("foo")
        self.foo = coord.contiguous_bounds()
        self.foo_index = np.arange(coord.points.size + 1)
        coord = self.cube.coord("bar")
        self.bar = coord.contiguous_bounds()
        self.bar_index = np.arange(coord.points.size + 1)
        self.data = self.cube.data
        self.dataT = self.data.T
        self.draw_func = self.blockplot_func()
        patch_target_name = "matplotlib.pyplot." + self.draw_func.__name__
        self.mpl_patch = mocker.patch(patch_target_name)


class Mixin2dCoordsContigTol:
    # Mixin for contiguity tolerance argument to pcolor/pcolormesh.
    # To use, make a class that inherits from this *and*
    # defines "self.blockplot_func()", to return the `iris.plot` function,
    # and defines "self.additional_kwargs" for expected extra call args.
    def test_contig_tol(self, mocker):
        # Patch the inner call to ensure contiguity_tolerance is passed.
        cube_argument = mocker.sentinel.passed_arg
        expected_result = mocker.sentinel.returned_value
        blockplot_patch = mocker.patch(
            "iris.plot._draw_2d_from_bounds",
            mocker.Mock(return_value=expected_result),
        )
        # Make the call
        draw_func = self.blockplot_func()
        other_kwargs = self.additional_kwargs
        result = draw_func(cube_argument, contiguity_tolerance=0.0123)
        drawfunc_name = draw_func.__name__
        # Check details of the call that was made.
        assert blockplot_patch.call_args_list == [
            mocker.call(
                drawfunc_name,
                cube_argument,
                contiguity_tolerance=0.0123,
                **other_kwargs,
            )
        ]
        assert result == expected_result
