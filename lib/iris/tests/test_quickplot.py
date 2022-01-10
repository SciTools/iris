# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Tests the high-level plotting interface.

"""

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests  # isort:skip
import iris
import iris.tests.test_plot as test_plot

# Run tests in no graphics mode if matplotlib is not available.
if tests.MPL_AVAILABLE:
    import matplotlib.pyplot as plt

    import iris.plot as iplt
    import iris.quickplot as qplt


# Caches _load_theta so subsequent calls are faster
def cache(fn, cache={}):
    def inner(*args, **kwargs):
        key = "result"
        if not cache:
            cache[key] = fn(*args, **kwargs)
        return cache[key]

    return inner


@cache
def _load_theta():
    path = tests.get_data_path(("PP", "COLPEX", "theta_and_orog_subset.pp"))
    theta = iris.load_cube(path, "air_potential_temperature")

    # Improve the unit
    theta.units = "K"

    return theta


@tests.skip_data
@tests.skip_plot
class TestQuickplotCoordinatesGiven(test_plot.TestPlotCoordinatesGiven):
    def setUp(self):
        tests.GraphicsTest.setUp(self)
        filename = tests.get_data_path(
            ("PP", "COLPEX", "theta_and_orog_subset.pp")
        )
        self.cube = test_plot.load_cube_once(
            filename, "air_potential_temperature"
        )

        self.draw_module = iris.quickplot
        self.contourf = test_plot.LambdaStr(
            "iris.quickplot.contourf",
            lambda cube, *args, **kwargs: iris.quickplot.contourf(
                cube, *args, **kwargs
            ),
        )
        self.contour = test_plot.LambdaStr(
            "iris.quickplot.contour",
            lambda cube, *args, **kwargs: iris.quickplot.contour(
                cube, *args, **kwargs
            ),
        )
        self.points = test_plot.LambdaStr(
            "iris.quickplot.points",
            lambda cube, *args, **kwargs: iris.quickplot.points(
                cube, c=cube.data, *args, **kwargs
            ),
        )
        self.plot = test_plot.LambdaStr(
            "iris.quickplot.plot",
            lambda cube, *args, **kwargs: iris.quickplot.plot(
                cube, *args, **kwargs
            ),
        )

        self.results = {
            "yx": (
                [self.contourf, ["grid_latitude", "grid_longitude"]],
                [self.contourf, ["grid_longitude", "grid_latitude"]],
                [self.contour, ["grid_latitude", "grid_longitude"]],
                [self.contour, ["grid_longitude", "grid_latitude"]],
                [self.points, ["grid_latitude", "grid_longitude"]],
                [self.points, ["grid_longitude", "grid_latitude"]],
            ),
            "zx": (
                [self.contourf, ["model_level_number", "grid_longitude"]],
                [self.contourf, ["grid_longitude", "model_level_number"]],
                [self.contour, ["model_level_number", "grid_longitude"]],
                [self.contour, ["grid_longitude", "model_level_number"]],
                [self.points, ["model_level_number", "grid_longitude"]],
                [self.points, ["grid_longitude", "model_level_number"]],
            ),
            "tx": (
                [self.contourf, ["time", "grid_longitude"]],
                [self.contourf, ["grid_longitude", "time"]],
                [self.contour, ["time", "grid_longitude"]],
                [self.contour, ["grid_longitude", "time"]],
                [self.points, ["time", "grid_longitude"]],
                [self.points, ["grid_longitude", "time"]],
            ),
            "x": ([self.plot, ["grid_longitude"]],),
            "y": ([self.plot, ["grid_latitude"]],),
        }


@tests.skip_data
@tests.skip_plot
class TestLabels(tests.GraphicsTest):
    def setUp(self):
        super().setUp()
        self.theta = _load_theta()

    def _slice(self, coords):
        """Returns the first cube containing the requested coordinates."""
        for cube in self.theta.slices(coords):
            break
        return cube

    def _small(self):
        # Use a restricted size so we can make out the detail
        cube = self._slice(["model_level_number", "grid_longitude"])
        return cube[:5, :5]

    def test_contour(self):
        qplt.contour(self._small())
        self.check_graphic()

        qplt.contourf(
            self._small(), coords=["model_level_number", "grid_longitude"]
        )
        self.check_graphic()

    def test_contourf(self):
        qplt.contourf(self._small())

        cube = self._small()
        iplt.orography_at_points(cube)

        self.check_graphic()

        qplt.contourf(
            self._small(), coords=["model_level_number", "grid_longitude"]
        )
        self.check_graphic()

        qplt.contourf(
            self._small(), coords=["grid_longitude", "model_level_number"]
        )
        self.check_graphic()

    def test_contourf_axes_specified(self):
        # Check that the contourf function does not modify the matplotlib
        # pyplot state machine.

        # Create a figure and axes to be used by contourf
        plt.figure()
        axes1 = plt.axes()

        # Create test figure and axes which will be the new results
        # of plt.gcf and plt.gca.
        plt.figure()
        axes2 = plt.axes()

        # Add a title to the test axes.
        plt.title("This should not be changed")
        # Draw the contourf on a specific axes.
        qplt.contourf(self._small(), axes=axes1)

        # Ensure that the correct axes got the appropriate title.
        self.assertEqual(axes2.get_title(), "This should not be changed")
        self.assertEqual(axes1.get_title(), "Air potential temperature")

        # Check that the axes labels were set correctly.
        self.assertEqual(axes1.get_xlabel(), "Grid longitude / degrees")
        self.assertEqual(axes1.get_ylabel(), "Altitude / m")

    def test_contourf_nameless(self):
        cube = self._small()
        cube.standard_name = None
        cube.attributes["STASH"] = ""
        qplt.contourf(cube, coords=["grid_longitude", "model_level_number"])
        self.check_graphic()

    def test_pcolor(self):
        qplt.pcolor(self._small())
        self.check_graphic()

    def test_pcolormesh(self):
        qplt.pcolormesh(self._small())

        # cube = self._small()
        # iplt.orography_at_bounds(cube)

        self.check_graphic()

    def test_pcolormesh_str_symbol(self):
        pcube = self._small().copy()
        pcube.coords("level_height")[0].units = "centimeters"
        qplt.pcolormesh(pcube)

        self.check_graphic()

    def test_map(self):
        cube = self._slice(["grid_latitude", "grid_longitude"])
        qplt.contour(cube)
        self.check_graphic()

        # check that the result of adding 360 to the data is *almost* identically the same result
        lon = cube.coord("grid_longitude")
        lon.points = lon.points + 360
        qplt.contour(cube)
        self.check_graphic()

    def test_alignment(self):
        cube = self._small()
        qplt.contourf(cube)
        # qplt.outline(cube)
        qplt.points(cube)
        self.check_graphic()


@tests.skip_data
@tests.skip_plot
class TestTimeReferenceUnitsLabels(tests.GraphicsTest):
    def setUp(self):
        super().setUp()
        path = tests.get_data_path(("PP", "aPProt1", "rotatedMHtimecube.pp"))
        self.cube = iris.load_cube(path)[:, 0, 0]

    def test_reference_time_units(self):
        # units should not be displayed for a reference time
        qplt.plot(self.cube.coord("time"), self.cube)
        plt.gcf().autofmt_xdate()
        self.check_graphic()

    def test_not_reference_time_units(self):
        # units should be displayed for other time coordinates
        qplt.plot(self.cube.coord("forecast_period"), self.cube)
        self.check_graphic()


if __name__ == "__main__":
    tests.main()
