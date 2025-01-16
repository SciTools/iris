# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test interaction between :mod:`iris.plot` and
:func:`matplotlib.pyplot.colorbar`.

"""

import numpy as np
import pytest

from iris.coords import AuxCoord
from iris.tests import _shared_utils
import iris.tests.stock

# Run tests in no graphics mode if matplotlib is not available.
if _shared_utils.MPL_AVAILABLE:
    import matplotlib.pyplot as plt

    from iris.plot import contour, contourf, pcolor, pcolormesh, points, scatter


@_shared_utils.skip_plot
class TestColorBarCreation(_shared_utils.GraphicsTest):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.draw_functions = (contour, contourf, pcolormesh, pcolor)
        self.cube = iris.tests.stock.lat_lon_cube()
        self.cube.coord("longitude").guess_bounds()
        self.cube.coord("latitude").guess_bounds()
        self.traj_lon = AuxCoord(
            np.linspace(-180, 180, 50),
            standard_name="longitude",
            units="degrees",
        )
        self.traj_lat = AuxCoord(
            np.sin(np.deg2rad(self.traj_lon.points)) * 30.0,
            standard_name="latitude",
            units="degrees",
        )

    def test_common_draw_functions(self):
        for draw_function in self.draw_functions:
            mappable = draw_function(self.cube)
            cbar = plt.colorbar()
            assert cbar.mappable is mappable, (
                "Problem with draw function iris.plot.{}".format(draw_function.__name__)
            )

    def test_common_draw_functions_specified_mappable(self):
        for draw_function in self.draw_functions:
            mappable_initial = draw_function(self.cube, cmap="cool")
            _ = draw_function(self.cube)
            cbar = plt.colorbar(mappable_initial)
            assert cbar.mappable is mappable_initial, (
                "Problem with draw function iris.plot.{}".format(draw_function.__name__)
            )

    def test_points_with_c_kwarg(self):
        mappable = points(self.cube, c=self.cube.data)
        cbar = plt.colorbar()
        assert cbar.mappable is mappable

    def test_points_with_c_kwarg_specified_mappable(self):
        mappable_initial = points(self.cube, c=self.cube.data, cmap="cool")
        _ = points(self.cube, c=self.cube.data)
        cbar = plt.colorbar(mappable_initial)
        assert cbar.mappable is mappable_initial

    def test_scatter_with_c_kwarg(self):
        mappable = scatter(self.traj_lon, self.traj_lat, c=self.traj_lon.points)
        cbar = plt.colorbar()
        assert cbar.mappable is mappable

    def test_scatter_with_c_kwarg_specified_mappable(self):
        mappable_initial = scatter(self.traj_lon, self.traj_lat, c=self.traj_lon.points)
        _ = scatter(self.traj_lon, self.traj_lat, c=self.traj_lon.points, cmap="cool")
        cbar = plt.colorbar(mappable_initial)
        assert cbar.mappable is mappable_initial
