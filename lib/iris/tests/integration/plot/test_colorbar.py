# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test interaction between :mod:`iris.plot` and
:func:`matplotlib.pyplot.colorbar`

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import numpy as np

from iris.coords import AuxCoord
import iris.tests.stock

# Run tests in no graphics mode if matplotlib is not available.
if tests.MPL_AVAILABLE:
    import matplotlib.pyplot as plt

    from iris.plot import (
        contour,
        contourf,
        pcolor,
        pcolormesh,
        points,
        scatter,
    )


@tests.skip_plot
class TestColorBarCreation(tests.GraphicsTest):
    def setUp(self):
        super().setUp()
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
            self.assertIs(
                cbar.mappable,
                mappable,
                msg="Problem with draw function iris.plot.{}".format(
                    draw_function.__name__
                ),
            )

    def test_common_draw_functions_specified_mappable(self):
        for draw_function in self.draw_functions:
            mappable_initial = draw_function(self.cube, cmap="cool")
            _ = draw_function(self.cube)
            cbar = plt.colorbar(mappable_initial)
            self.assertIs(
                cbar.mappable,
                mappable_initial,
                msg="Problem with draw function iris.plot.{}".format(
                    draw_function.__name__
                ),
            )

    def test_points_with_c_kwarg(self):
        mappable = points(self.cube, c=self.cube.data)
        cbar = plt.colorbar()
        self.assertIs(cbar.mappable, mappable)

    def test_points_with_c_kwarg_specified_mappable(self):
        mappable_initial = points(self.cube, c=self.cube.data, cmap="cool")
        _ = points(self.cube, c=self.cube.data)
        cbar = plt.colorbar(mappable_initial)
        self.assertIs(cbar.mappable, mappable_initial)

    def test_scatter_with_c_kwarg(self):
        mappable = scatter(
            self.traj_lon, self.traj_lat, c=self.traj_lon.points
        )
        cbar = plt.colorbar()
        self.assertIs(cbar.mappable, mappable)

    def test_scatter_with_c_kwarg_specified_mappable(self):
        mappable_initial = scatter(
            self.traj_lon, self.traj_lat, c=self.traj_lon.points
        )
        _ = scatter(
            self.traj_lon, self.traj_lat, c=self.traj_lon.points, cmap="cool"
        )
        cbar = plt.colorbar(mappable_initial)
        self.assertIs(cbar.mappable, mappable_initial)


if __name__ == "__main__":
    tests.main()
