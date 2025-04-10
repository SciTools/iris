# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

from contextlib import nullcontext

import cf_units
import numpy as np
import pytest

import iris
import iris.analysis
import iris.coords as coords
from iris.tests import _shared_utils
import iris.tests.stock

# Run tests in no graphics mode if matplotlib is not available.
if _shared_utils.MPL_AVAILABLE:
    import matplotlib.pyplot as plt

    import iris.plot as iplt
    import iris.quickplot as qplt
    import iris.symbols


@_shared_utils.skip_data
def simple_cube():
    cube = iris.tests.stock.realistic_4d()
    cube = cube[:, 0, 0, :]
    cube.coord("time").guess_bounds()
    return cube


@_shared_utils.skip_plot
class TestSimple(_shared_utils.GraphicsTest):
    def test_points(self):
        cube = simple_cube()
        qplt.contourf(cube)
        self.check_graphic()

    def test_bounds(self):
        cube = simple_cube()
        qplt.pcolor(cube)
        self.check_graphic()


@_shared_utils.skip_plot
class TestMissingCoord(_shared_utils.GraphicsTest):
    def _check(self, cube):
        qplt.contourf(cube)
        self.check_graphic()

        qplt.pcolor(cube)
        self.check_graphic()

    def test_no_u(self):
        cube = simple_cube()
        cube.remove_coord("grid_longitude")
        self._check(cube)

    def test_no_v(self):
        cube = simple_cube()
        cube.remove_coord("time")
        self._check(cube)

    def test_none(self):
        cube = simple_cube()
        cube.remove_coord("grid_longitude")
        cube.remove_coord("time")
        self._check(cube)


@_shared_utils.skip_data
@_shared_utils.skip_plot
class TestMissingCS(_shared_utils.GraphicsTest):
    @_shared_utils.skip_data
    def test_missing_cs(self):
        cube = iris.tests.stock.simple_pp()
        cube.coord("latitude").coord_system = None
        cube.coord("longitude").coord_system = None
        qplt.contourf(cube)
        qplt.plt.gca().coastlines("110m")
        self.check_graphic()


@_shared_utils.skip_plot
@_shared_utils.skip_data
class TestHybridHeight(_shared_utils.GraphicsTest):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = iris.tests.stock.realistic_4d()[0, :15, 0, :]

    def _check(self, plt_method, test_altitude=True):
        plt_method(self.cube)
        self.check_graphic()

        plt_method(self.cube, coords=["level_height", "grid_longitude"])
        self.check_graphic()

        plt_method(self.cube, coords=["grid_longitude", "level_height"])
        self.check_graphic()

        if test_altitude:
            plt_method(self.cube, coords=["grid_longitude", "altitude"])
            self.check_graphic()

            plt_method(self.cube, coords=["altitude", "grid_longitude"])
            self.check_graphic()

    def test_points(self):
        self._check(qplt.contourf)

    def test_bounds(self):
        self._check(qplt.pcolor, test_altitude=False)

    def test_orography(self):
        qplt.contourf(self.cube)
        iplt.orography_at_points(self.cube)
        iplt.points(self.cube)
        self.check_graphic()

        coords = ["altitude", "grid_longitude"]
        qplt.contourf(self.cube, coords=coords)
        iplt.orography_at_points(self.cube, coords=coords)
        iplt.points(self.cube, coords=coords)
        self.check_graphic()

        # TODO: Test bounds once they are supported.
        qplt.pcolor(self.cube)
        with pytest.raises(NotImplementedError):
            iplt.orography_at_bounds(self.cube)
            # iplt.outline(self.cube)
            # self.check_graphic()


@_shared_utils.skip_plot
@_shared_utils.skip_data
class Test1dPlotMultiArgs(_shared_utils.GraphicsTest):
    # tests for iris.plot using multi-argument calling convention
    @pytest.fixture(autouse=True)
    def _set_draw_method(self):
        self.draw_method = iplt.plot

    @pytest.fixture(autouse=True)
    def _setup(self, load_4d_testcube):
        self.cube1d = load_4d_testcube[0, :, 0, 0]

    def test_cube(self):
        # just plot a cube against its dim coord
        self.draw_method(self.cube1d)  # altitude vs temp
        self.check_graphic()

    def test_coord(self):
        # plot the altitude coordinate
        self.draw_method(self.cube1d.coord("altitude"))
        self.check_graphic()

    def test_coord_cube(self):
        # plot temperature against sigma
        self.draw_method(self.cube1d.coord("sigma"), self.cube1d)
        self.check_graphic()

    def test_cube_coord(self):
        # plot a vertical profile of temperature
        self.draw_method(self.cube1d, self.cube1d.coord("altitude"))
        self.check_graphic()

    def test_coord_coord(self):
        # plot two coordinates that are not mappable
        self.draw_method(self.cube1d.coord("sigma"), self.cube1d.coord("altitude"))
        self.check_graphic()

    def test_coord_coord_map(self):
        # plot lat-lon aux coordinates of a trajectory, which draws a map
        lon = iris.coords.AuxCoord(
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
            standard_name="longitude",
            units="degrees_north",
        )
        lat = iris.coords.AuxCoord(
            [45, 55, 50, 60, 55, 65, 60, 70, 65, 75],
            standard_name="latitude",
            units="degrees_north",
        )
        self.draw_method(lon, lat)
        plt.gca().coastlines("110m")
        self.check_graphic()

    def test_cube_cube(self):
        # plot two phenomena against each other, in this case just dummy data
        cube1 = self.cube1d.copy()
        cube2 = self.cube1d.copy()
        cube1.rename("some phenomenon")
        cube2.rename("some other phenomenon")
        cube1.units = cf_units.Unit("no_unit")
        cube2.units = cf_units.Unit("no_unit")
        cube1.data[:] = np.linspace(0, 1, 7)
        cube2.data[:] = np.exp(cube1.data)
        self.draw_method(cube1, cube2)
        self.check_graphic()

    def test_incompatible_objects(self):
        # incompatible objects (not the same length) should raise an error
        with pytest.raises(ValueError, match="are not compatible"):
            self.draw_method(self.cube1d.coord("time"), (self.cube1d))

    def test_multimidmensional(self, load_4d_testcube):
        # multidimensional cubes are not allowed
        cube = load_4d_testcube[0, :, :, 0]
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            self.draw_method(cube)

    def test_not_cube_or_coord(self):
        # inputs must be cubes or coordinates, otherwise an error should be
        # raised
        xdim = np.arange(self.cube1d.shape[0])
        with pytest.raises(TypeError):
            self.draw_method(xdim, self.cube1d)

    def test_plot_old_coords_kwarg(self):
        # Coords used to be a valid kwarg to plot, but it was deprecated and
        # we are maintaining a reasonable exception, check that it is raised
        # here.
        with pytest.raises(TypeError):
            self.draw_method(self.cube1d, coords=None)


@_shared_utils.skip_plot
class Test1dQuickplotPlotMultiArgs(Test1dPlotMultiArgs):
    # tests for iris.plot using multi-argument calling convention
    @pytest.fixture(autouse=True)
    def _set_draw_method(self):
        self.draw_method = qplt.plot


@_shared_utils.skip_data
@_shared_utils.skip_plot
class Test1dScatter(_shared_utils.GraphicsTest):
    @pytest.fixture(autouse=True)
    def _set_draw_method(self):
        self.draw_method = iplt.scatter

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = iris.load_cube(
            _shared_utils.get_data_path(("NAME", "NAMEIII_trajectory.txt")),
            "Temperature",
        )

    def test_coord_coord(self):
        x = self.cube.coord("longitude")
        y = self.cube.coord("altitude")
        c = self.cube.data
        self.draw_method(x, y, c=c, edgecolor="none")
        self.check_graphic()

    def test_coord_coord_map(self):
        x = self.cube.coord("longitude")
        y = self.cube.coord("latitude")
        c = self.cube.data
        self.draw_method(x, y, c=c, edgecolor="none")
        plt.gca().coastlines("110m")
        self.check_graphic()

    def test_coord_cube(self):
        x = self.cube.coord("latitude")
        y = self.cube
        c = self.cube.coord("Travel Time").points
        self.draw_method(x, y, c=c, edgecolor="none")
        self.check_graphic()

    def test_cube_coord(self):
        x = self.cube
        y = self.cube.coord("altitude")
        c = self.cube.coord("Travel Time").points
        self.draw_method(x, y, c=c, edgecolor="none")
        self.check_graphic()

    def test_cube_cube(self):
        x = iris.load_cube(
            _shared_utils.get_data_path(("NAME", "NAMEIII_trajectory.txt")),
            "Rel Humidity",
        )
        y = self.cube
        c = self.cube.coord("Travel Time").points
        self.draw_method(x, y, c=c, edgecolor="none")
        self.check_graphic()

    def test_incompatible_objects(self):
        # cubes/coordinates of different sizes cannot be plotted
        x = self.cube
        y = self.cube.coord("altitude")[:-1]
        with pytest.raises(ValueError, match="are not compatible"):
            self.draw_method(x, y)

    def test_multidimensional(self, load_4d_testcube):
        # multidimensional cubes/coordinates are not allowed
        x = load_4d_testcube[0, :, :, 0]
        y = x.coord("model_level_number")
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            self.draw_method(x, y)

    def test_not_cube_or_coord(self):
        # inputs must be cubes or coordinates
        x = np.arange(self.cube.shape[0])
        y = self.cube
        with pytest.raises(TypeError):
            self.draw_method(x, y)


@_shared_utils.skip_data
@_shared_utils.skip_plot
class Test1dQuickplotScatter(Test1dScatter):
    @pytest.fixture(autouse=True)
    def _set_draw_method(self):
        self.draw_method = qplt.scatter


@_shared_utils.skip_data
@_shared_utils.skip_plot
class Test2dPoints(_shared_utils.GraphicsTest):
    @pytest.fixture(autouse=True)
    def _setup(self):
        pp_file = _shared_utils.get_data_path(("PP", "globClim1", "u_wind.pp"))
        self.cube = iris.load(pp_file)[0][0]

    def test_circular_changes(self):
        # Circular
        iplt.pcolormesh(self.cube, vmax=50)
        iplt.points(self.cube, s=self.cube.data)
        plt.gca().coastlines()

        self.check_graphic()


@_shared_utils.skip_data
@_shared_utils.skip_plot
class Test1dFillBetween(_shared_utils.GraphicsTest):
    @pytest.fixture(autouse=True)
    def _set_draw_method(self):
        self.draw_method = iplt.fill_between

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = iris.load_cube(
            _shared_utils.get_data_path(("NetCDF", "testing", "small_theta_colpex.nc")),
            "air_potential_temperature",
        )[0, 0]

    def test_coord_coord(self):
        x = self.cube.coord("grid_latitude")
        y1 = self.cube.coord("surface_altitude")[:, 0]
        y2 = self.cube.coord("surface_altitude")[:, 1]
        self.draw_method(x, y1, y2)
        self.check_graphic()

    def test_coord_cube(self):
        x = self.cube.coord("grid_latitude")
        y1 = self.cube.collapsed("grid_longitude", iris.analysis.MIN)
        y2 = self.cube.collapsed("grid_longitude", iris.analysis.MAX)
        self.draw_method(x, y1, y2)
        self.check_graphic()

    def test_cube_coord(self):
        x = self.cube.collapsed("grid_longitude", iris.analysis.MEAN)
        y1 = self.cube.coord("surface_altitude")[:, 0]
        y2 = y1 + 10
        self.draw_method(x, y1, y2)
        self.check_graphic()

    def test_cube_cube(self):
        x = self.cube.collapsed("grid_longitude", iris.analysis.MEAN)
        y1 = self.cube.collapsed("grid_longitude", iris.analysis.MIN)
        y2 = self.cube.collapsed("grid_longitude", iris.analysis.MAX)
        self.draw_method(x, y1, y2)
        self.check_graphic()

    def test_incompatible_objects_x_odd(self):
        # cubes/coordinates of different sizes cannot be plotted
        x = self.cube.coord("grid_latitude")[:-1]
        y1 = self.cube.collapsed("grid_longitude", iris.analysis.MIN)
        y2 = self.cube.collapsed("grid_longitude", iris.analysis.MAX)
        with pytest.raises(ValueError, match="are not all compatible"):
            self.draw_method(x, y1, y2)

    def test_incompatible_objects_y1_odd(self):
        # cubes/coordinates of different sizes cannot be plotted
        x = self.cube.coord("grid_latitude")
        y1 = self.cube.collapsed("grid_longitude", iris.analysis.MIN)[:-1]
        y2 = self.cube.collapsed("grid_longitude", iris.analysis.MAX)
        with pytest.raises(ValueError, match="are not all compatible"):
            self.draw_method(x, y1, y2)

    def test_incompatible_objects_y2_odd(self):
        # cubes/coordinates of different sizes cannot be plotted
        x = self.cube.coord("grid_latitude")
        y1 = self.cube.collapsed("grid_longitude", iris.analysis.MIN)
        y2 = self.cube.collapsed("grid_longitude", iris.analysis.MAX)[:-1]
        with pytest.raises(ValueError, match="are not all compatible"):
            self.draw_method(x, y1, y2)

    def test_incompatible_objects_all_odd(self):
        # cubes/coordinates of different sizes cannot be plotted
        x = self.cube.coord("grid_latitude")
        y1 = self.cube.collapsed("grid_longitude", iris.analysis.MIN)[:-1]
        y2 = self.cube.collapsed("grid_longitude", iris.analysis.MAX)[:-2]
        with pytest.raises(ValueError, match="are not all compatible"):
            self.draw_method(x, y1, y2)

    def test_multidimensional(self):
        # multidimensional cubes/coordinates are not allowed
        x = self.cube.coord("grid_latitude")
        y1 = self.cube
        y2 = self.cube
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            self.draw_method(x, y1, y2)

    def test_not_cube_or_coord(self):
        # inputs must be cubes or coordinates
        x = np.arange(self.cube.shape[0])
        y1 = self.cube.collapsed("grid_longitude", iris.analysis.MIN)
        y2 = self.cube.collapsed("grid_longitude", iris.analysis.MAX)
        with pytest.raises(TypeError):
            self.draw_method(x, y1, y2)


@_shared_utils.skip_data
@_shared_utils.skip_plot
class Test1dQuickplotFillBetween(Test1dFillBetween):
    @pytest.fixture(autouse=True)
    def _set_draw_method(self):
        self.draw_method = qplt.fill_between


@_shared_utils.skip_data
@_shared_utils.skip_plot
class TestAttributePositive(_shared_utils.GraphicsTest):
    def test_1d_positive_up(self):
        path = _shared_utils.get_data_path(("NetCDF", "ORCA2", "votemper.nc"))
        cube = iris.load_cube(path)
        qplt.plot(cube.coord("depth"), cube[0, :, 60, 80])
        self.check_graphic()

    def test_1d_positive_down(self):
        path = _shared_utils.get_data_path(("NetCDF", "ORCA2", "votemper.nc"))
        cube = iris.load_cube(path)
        qplt.plot(cube[0, :, 60, 80], cube.coord("depth"))
        self.check_graphic()

    def test_2d_positive_up(self):
        path = _shared_utils.get_data_path(
            ("NetCDF", "testing", "small_theta_colpex.nc")
        )
        cube = iris.load_cube(path, "air_potential_temperature")[0, :, 42, :]
        qplt.pcolormesh(cube)
        self.check_graphic()

    def test_2d_positive_down(self):
        path = _shared_utils.get_data_path(("NetCDF", "ORCA2", "votemper.nc"))
        cube = iris.load_cube(path)[0, :, 42, :]
        qplt.pcolormesh(cube)
        self.check_graphic()


@_shared_utils.skip_data
@pytest.fixture(scope="module")
def load_4d_testcube():
    """Load the realistic_4d() cube with specific modifications.

    Scoped to only load once - used many times so this is much faster.
    """
    # Load example 4d data (TZYX).
    test_cube = iris.tests.stock.realistic_4d()
    # Replace forecast_period coord with a multi-valued version.
    time_coord = test_cube.coord("time")
    n_times = len(time_coord.points)
    forecast_dims = test_cube.coord_dims(time_coord)
    test_cube.remove_coord("forecast_period")
    # Make up values (including bounds), to roughly match older testdata.
    point_values = np.linspace((1 + 1.0 / 6), 2.0, n_times)
    point_uppers = point_values + (point_values[1] - point_values[0])
    bound_values = np.column_stack([point_values, point_uppers])
    # NOTE: this must be a DimCoord
    #  - an equivalent AuxCoord produces different plots.
    new_forecast_coord = iris.coords.DimCoord(
        points=point_values,
        bounds=bound_values,
        standard_name="forecast_period",
        units=cf_units.Unit("hours"),
    )
    test_cube.add_aux_coord(new_forecast_coord, forecast_dims)
    # Heavily reduce dimensions for faster testing.
    # NOTE: this makes ZYX non-contiguous.  Doesn't seem to matter for now.
    test_cube = test_cube[:, ::10, ::10, ::10]
    return test_cube


@_shared_utils.skip_data
@pytest.fixture(scope="module")
def load_wind_no_bounds():
    """Load a cube representing wind data but with no coordinate bounds.

    Scoped to only load once - used many times so this is much faster.
    """
    # Load the COLPEX data => TZYX
    path = _shared_utils.get_data_path(("PP", "COLPEX", "small_eastward_wind.pp"))
    wind = iris.load_cube(path, "x_wind")

    # Remove bounds from all coords that have them.
    wind.coord("grid_latitude").bounds = None
    wind.coord("grid_longitude").bounds = None
    wind.coord("level_height").bounds = None
    wind.coord("sigma").bounds = None

    return wind[:, :, :50, :50]


def _time_series(src_cube):
    # Until we have plotting support for multiple axes on the same dimension,
    # remove the time coordinate and its axis.
    cube = src_cube.copy()
    cube.remove_coord("time")
    return cube


def _date_series(src_cube):
    # Until we have plotting support for multiple axes on the same dimension,
    # remove the forecast_period coordinate and its axis.
    cube = src_cube.copy()
    cube.remove_coord("forecast_period")
    return cube


@_shared_utils.skip_plot
class SliceMixin:
    """Mixin class providing tests for each 2-dimensional permutation of axes.

    Requires self.draw_method to be the relevant plotting function,
    and self.results to be a dictionary containing the desired test results.
    """

    @pytest.fixture(autouse=True)
    def _set_warnings_stance(self):
        # Defining in a fixture enables inheritance by classes that expect a
        #  warning - setting self.warning_checker to the pytest.warns() context
        #  manager instead.
        self.warning_checker = nullcontext

    def test_yx(self):
        cube = self.wind[0, 0, :, :]
        with self.warning_checker(UserWarning):
            self.draw_method(cube)
        self.check_graphic()

    def test_zx(self):
        cube = self.wind[0, :, 0, :]
        with self.warning_checker(UserWarning):
            self.draw_method(cube)
        self.check_graphic()

    def test_tx(self):
        cube = _time_series(self.wind[:, 0, 0, :])
        with self.warning_checker(UserWarning):
            self.draw_method(cube)
        self.check_graphic()

    def test_zy(self):
        cube = self.wind[0, :, :, 0]
        with self.warning_checker(UserWarning):
            self.draw_method(cube)
        self.check_graphic()

    def test_ty(self):
        cube = _time_series(self.wind[:, 0, :, 0])
        with self.warning_checker(UserWarning):
            self.draw_method(cube)
        self.check_graphic()

    def test_tz(self):
        cube = _time_series(self.wind[:, :, 0, 0])
        with self.warning_checker(UserWarning):
            self.draw_method(cube)
        self.check_graphic()


@_shared_utils.skip_data
class TestContour(_shared_utils.GraphicsTest, SliceMixin):
    """Test the iris.plot.contour routine."""

    @pytest.fixture(autouse=True)
    def _setup(self, load_4d_testcube):
        self.wind = load_4d_testcube
        self.draw_method = iplt.contour


@_shared_utils.skip_data
class TestContourf(_shared_utils.GraphicsTest, SliceMixin):
    """Test the iris.plot.contourf routine."""

    @pytest.fixture(autouse=True)
    def _setup(self, load_4d_testcube):
        self.wind = load_4d_testcube
        self.draw_method = iplt.contourf


@_shared_utils.skip_data
class TestPcolor(_shared_utils.GraphicsTest, SliceMixin):
    """Test the iris.plot.pcolor routine."""

    @pytest.fixture(autouse=True)
    def _setup(self, load_4d_testcube):
        self.wind = load_4d_testcube
        self.draw_method = iplt.pcolor


@_shared_utils.skip_data
class TestPcolormesh(_shared_utils.GraphicsTest, SliceMixin):
    """Test the iris.plot.pcolormesh routine."""

    @pytest.fixture(autouse=True)
    def _setup(self, load_4d_testcube):
        self.wind = load_4d_testcube
        self.draw_method = iplt.pcolormesh


class SliceWarningsMixin(SliceMixin):
    @pytest.fixture(autouse=True)
    def _set_warnings_stance(self):
        self.warning_checker = pytest.warns


@_shared_utils.skip_data
class TestPcolorNoBounds(_shared_utils.GraphicsTest, SliceWarningsMixin):
    """Test the iris.plot.pcolor routine on a cube with coordinates
    that have no bounds.

    """

    @pytest.fixture(autouse=True)
    def _setup(self, load_wind_no_bounds):
        self.wind = load_wind_no_bounds
        self.draw_method = iplt.pcolor


@_shared_utils.skip_data
class TestPcolormeshNoBounds(_shared_utils.GraphicsTest, SliceWarningsMixin):
    """Test the iris.plot.pcolormesh routine on a cube with coordinates
    that have no bounds.

    """

    @pytest.fixture(autouse=True)
    def _setup(self, load_wind_no_bounds):
        self.wind = load_wind_no_bounds
        self.draw_method = iplt.pcolormesh


@_shared_utils.skip_plot
class Slice1dMixin:
    """Mixin class providing tests for each 1-dimensional permutation of axes.

    Requires self.draw_method to be the relevant plotting function,
    and self.results to be a dictionary containing the desired test results.
    """

    def test_x(self):
        cube = self.wind[0, 0, 0, :]
        self.draw_method(cube)
        self.check_graphic()

    def test_y(self):
        cube = self.wind[0, 0, :, 0]
        self.draw_method(cube)
        self.check_graphic()

    def test_z(self):
        cube = self.wind[0, :, 0, 0]
        self.draw_method(cube)
        self.check_graphic()

    def test_t(self):
        cube = _time_series(self.wind[:, 0, 0, 0])
        self.draw_method(cube)
        self.check_graphic()

    def test_t_dates(self):
        cube = _date_series(self.wind[:, 0, 0, 0])
        self.draw_method(cube)
        plt.gcf().autofmt_xdate()
        plt.xlabel("Phenomenon time")

        self.check_graphic()


@_shared_utils.skip_data
class TestPlot(_shared_utils.GraphicsTest, Slice1dMixin):
    """Test the iris.plot.plot routine."""

    @pytest.fixture(autouse=True)
    def _setup(self, load_4d_testcube):
        self.wind = load_4d_testcube
        self.draw_method = iplt.plot


@_shared_utils.skip_data
class TestQuickplotPlot(_shared_utils.GraphicsTest, Slice1dMixin):
    """Test the iris.quickplot.plot routine."""

    @pytest.fixture(autouse=True)
    def _setup(self, load_4d_testcube):
        self.wind = load_4d_testcube
        self.draw_method = qplt.plot


class LambdaStr:
    """Provides a callable function which has a sensible __repr__."""

    def __init__(self, repr, lambda_fn):
        self.repr = repr
        self.lambda_fn = lambda_fn

    def __call__(self, *args, **kwargs):
        return self.lambda_fn(*args, **kwargs)

    def __repr__(self):
        return self.repr


@_shared_utils.skip_data
@_shared_utils.skip_plot
class TestPlotCoordinatesGiven(_shared_utils.GraphicsTest):
    @pytest.fixture(autouse=True, scope="class")
    def _get_cube(self):
        # Class-scoped to avoid wastefully reloading the same Cube repeatedly.
        filename = _shared_utils.get_data_path(
            ("PP", "COLPEX", "theta_and_orog_subset.pp")
        )
        cube = iris.load_cube(filename, "air_potential_temperature")
        if cube.coord_dims("time") != (0,):
            # A quick fix for data which has changed since we support time-varying orography
            cube.transpose((1, 0, 2, 3))
        self.__class__.cube = cube

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.draw_module = iris.plot
        self.contourf = LambdaStr(
            "iris.plot.contourf",
            lambda cube, *args, **kwargs: iris.plot.contourf(cube, *args, **kwargs),
        )
        self.contour = LambdaStr(
            "iris.plot.contour",
            lambda cube, *args, **kwargs: iris.plot.contour(cube, *args, **kwargs),
        )
        self.points = LambdaStr(
            "iris.plot.points",
            lambda cube, *args, **kwargs: iris.plot.points(
                cube, c=cube.data, *args, **kwargs
            ),
        )
        self.plot = LambdaStr(
            "iris.plot.plot",
            lambda cube, *args, **kwargs: iris.plot.plot(cube, *args, **kwargs),
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

    def draw(self, draw_method, *args, **kwargs):
        draw_fn = getattr(self.draw_module, draw_method)
        draw_fn(*args, **kwargs)
        self.check_graphic()

    def run_tests(self, cube, results):
        for draw_method, rcoords in results:
            draw_method(cube, coords=rcoords)
            try:
                self.check_graphic()
            except AssertionError as err:
                err.add_note(
                    f"Draw method {draw_method!r} failed with coords: {rcoords!r}."
                )
                raise

    def run_tests_1d(self, cube, results):
        # there is a different calling convention for 1d plots
        for draw_method, rcoords in results:
            draw_method(cube.coord(rcoords[0]), cube)
            try:
                self.check_graphic()
            except AssertionError as err:
                err.add_note(
                    f"Draw method {draw_method!r} failed with coords: {rcoords!r}."
                )
                raise

    def test_yx(self):
        test_cube = self.cube[0, 0, :, :]
        self.run_tests(test_cube, self.results["yx"])

    def test_zx(self):
        test_cube = self.cube[0, :15, 0, :]
        self.run_tests(test_cube, self.results["zx"])

    def test_tx(self):
        test_cube = self.cube[:, 0, 0, :]
        self.run_tests(test_cube, self.results["tx"])

    def test_x(self):
        test_cube = self.cube[0, 0, 0, :]
        self.run_tests_1d(test_cube, self.results["x"])

    def test_y(self):
        test_cube = self.cube[0, 0, :, 0]
        self.run_tests_1d(test_cube, self.results["y"])

    def test_badcoords(self):
        cube = self.cube[0, 0, :, :]
        draw_fn = getattr(self.draw_module, "contourf")
        pytest.raises(
            ValueError,
            draw_fn,
            cube,
            coords=["grid_longitude", "grid_longitude"],
            match="don't span the 2 data dimensions",
        )
        pytest.raises(
            ValueError,
            draw_fn,
            cube,
            coords=["grid_longitude", "grid_longitude", "grid_latitude"],
            match="should have the same length",
        )
        pytest.raises(
            iris.exceptions.CoordinateNotFoundError,
            draw_fn,
            cube,
            coords=["grid_longitude", "wibble"],
            match="but found none",
        )
        pytest.raises(
            ValueError,
            draw_fn,
            cube,
            coords=[],
            match="should have the same length",
        )
        pytest.raises(
            ValueError,
            draw_fn,
            cube,
            coords=[
                cube.coord("grid_longitude"),
                cube.coord("grid_longitude"),
            ],
            match="don't span the 2 data dimensions",
        )
        pytest.raises(
            ValueError,
            draw_fn,
            cube,
            coords=[
                cube.coord("grid_longitude"),
                cube.coord("grid_longitude"),
                cube.coord("grid_longitude"),
            ],
            match="should have the same length",
        )

    def test_non_cube_coordinate(self):
        cube = self.cube[0, :, :, 0]
        pts = -100 + np.arange(cube.shape[1]) * 13
        x = coords.DimCoord(
            pts,
            standard_name="model_level_number",
            attributes={"positive": "up"},
            units="1",
        )
        self.draw("contourf", cube, coords=["grid_latitude", x])


@_shared_utils.skip_data
@_shared_utils.skip_plot
class TestPlotHist(_shared_utils.GraphicsTest):
    def test_cube(self):
        cube = simple_cube()[0]
        iplt.hist(cube, bins=np.linspace(287.7, 288.2, 11))
        self.check_graphic()


@_shared_utils.skip_data
@_shared_utils.skip_plot
class TestPlotDimAndAuxCoordsKwarg(_shared_utils.GraphicsTest):
    @pytest.fixture(autouse=True)
    def _setup(self):
        filename = _shared_utils.get_data_path(
            ("NetCDF", "rotated", "xy", "rotPole_landAreaFraction.nc")
        )
        self.cube = iris.load_cube(filename)

    def test_default(self):
        iplt.contourf(self.cube)
        plt.gca().coastlines("110m")
        self.check_graphic()

    def test_coords(self):
        # Pass in dimension coords.
        rlat = self.cube.coord("grid_latitude")
        rlon = self.cube.coord("grid_longitude")
        iplt.contourf(self.cube, coords=[rlon, rlat])
        plt.gca().coastlines("110m")
        self.check_graphic()
        # Pass in auxiliary coords.
        lat = self.cube.coord("latitude")
        lon = self.cube.coord("longitude")
        iplt.contourf(self.cube, coords=[lon, lat])
        plt.gca().coastlines("110m")
        self.check_graphic()

    def test_coord_names(self):
        # Pass in names of dimension coords.
        iplt.contourf(self.cube, coords=["grid_longitude", "grid_latitude"])
        plt.gca().coastlines("110m")
        self.check_graphic()
        # Pass in names of auxiliary coords.
        iplt.contourf(self.cube, coords=["longitude", "latitude"])
        plt.gca().coastlines("110m")
        self.check_graphic()

    def test_yx_order(self):
        # Do not attempt to draw coastlines as it is not a map.
        iplt.contourf(self.cube, coords=["grid_latitude", "grid_longitude"])
        self.check_graphic()
        iplt.contourf(self.cube, coords=["latitude", "longitude"])
        self.check_graphic()


@_shared_utils.skip_plot
class TestSymbols(_shared_utils.GraphicsTest):
    def test_cloud_cover(self):
        iplt.symbols(
            list(range(10)),
            [0] * 10,
            [iris.symbols.CLOUD_COVER[i] for i in range(10)],
            0.375,
        )
        iplt.plt.axis("off")
        self.check_graphic()


@_shared_utils.skip_plot
class TestPlottingExceptions:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.bounded_cube = iris.tests.stock.lat_lon_cube()
        self.bounded_cube.coord("latitude").guess_bounds()
        self.bounded_cube.coord("longitude").guess_bounds()

    def test_boundmode_multidim(self):
        # Test exception translation.
        # We can't get contiguous bounded grids from multi-d coords.
        cube = self.bounded_cube
        cube.remove_coord("latitude")
        cube.add_aux_coord(
            coords.AuxCoord(
                points=cube.data, standard_name="latitude", units="degrees"
            ),
            [0, 1],
        )
        with pytest.raises(ValueError, match="Could not get XY grid from bounds"):
            iplt.pcolormesh(cube, coords=["longitude", "latitude"])

    def test_boundmode_4bounds(self):
        # Test exception translation.
        # We can only get contiguous bounded grids with 2 bounds per point.
        cube = self.bounded_cube
        lat = coords.AuxCoord.from_coord(cube.coord("latitude"))
        lat.bounds = np.array(
            [lat.points, lat.points + 1, lat.points + 2, lat.points + 3]
        ).transpose()
        cube.remove_coord("latitude")
        cube.add_aux_coord(lat, 0)
        with pytest.raises(ValueError, match="Could not get XY grid from bounds."):
            iplt.pcolormesh(cube, coords=["longitude", "latitude"])

    def test_different_coord_systems(self):
        cube = self.bounded_cube
        lat = cube.coord("latitude")
        lon = cube.coord("longitude")
        lat.coord_system = iris.coord_systems.GeogCS(7000000)
        lon.coord_system = iris.coord_systems.GeogCS(7000001)
        with pytest.raises(ValueError, match="must have equal coordinate systems"):
            iplt.pcolormesh(cube, coords=["longitude", "latitude"])


@_shared_utils.skip_data
@_shared_utils.skip_plot
class TestPlotOtherCoordSystems(_shared_utils.GraphicsTest):
    def test_plot_tmerc(self):
        filename = _shared_utils.get_data_path(
            ("NetCDF", "transverse_mercator", "tmean_1910_1910.nc")
        )
        self.cube = iris.load_cube(filename)
        iplt.pcolormesh(self.cube[0])
        plt.gca().coastlines("110m")
        self.check_graphic()


@_shared_utils.skip_plot
class TestPlotCitation(_shared_utils.GraphicsTest):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.figure = plt.figure()
        self.axes = self.figure.gca()
        self.text = (
            "Lorem ipsum dolor sit amet, consectetur adipiscing "
            "elit, sed do eiusmod tempor incididunt ut labore et "
            "dolore magna aliqua."
        )

    def test(self):
        iplt.citation(self.text)
        self.check_graphic()

    def test_figure(self):
        iplt.citation(self.text, figure=self.figure)
        self.check_graphic()

    def test_axes(self):
        iplt.citation(self.text, axes=self.axes)
        self.check_graphic()
