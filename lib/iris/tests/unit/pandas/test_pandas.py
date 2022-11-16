# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""All unit tests for the :mod:`iris.pandas` module."""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import copy
import datetime
from termios import IXOFF  # noqa: F401
import warnings

import cf_units
import cftime
import matplotlib.units
import numpy as np
import pytest

import iris
from iris._deprecation import IrisDeprecation

# Importing pandas has the side-effect of messing with the formatters
# used by matplotlib for handling dates.
default_units_registry = copy.copy(matplotlib.units.registry)
try:
    import pandas
except ImportError:
    # Disable all these tests if pandas is not installed.
    pandas = None
matplotlib.units.registry = default_units_registry

skip_pandas = pytest.mark.skipif(
    pandas is None,
    reason='Test(s) require "pandas", ' "which is not available.",
)

if pandas is not None:
    from iris.coords import AncillaryVariable, AuxCoord, CellMeasure, DimCoord
    from iris.cube import Cube, CubeList
    import iris.pandas


@pytest.fixture
def activate_pandas_ndim():
    iris.FUTURE.pandas_ndim = True
    yield None
    iris.FUTURE.pandas_ndim = False


@skip_pandas
@pytest.mark.filterwarnings(
    "ignore:.*as_series has been deprecated.*:iris._deprecation.IrisDeprecation"
)
class TestAsSeries(tests.IrisTest):
    """Test conversion of 1D cubes to Pandas using as_series()"""

    def test_no_dim_coord(self):
        cube = Cube(np.array([0, 1, 2, 3, 4]), long_name="foo")
        series = iris.pandas.as_series(cube)
        expected_index = np.array([0, 1, 2, 3, 4])
        self.assertArrayEqual(series, cube.data)
        self.assertArrayEqual(series.index, expected_index)

    def test_simple(self):
        cube = Cube(np.array([0, 1, 2, 3, 4.4]), long_name="foo")
        dim_coord = DimCoord([5, 6, 7, 8, 9], long_name="bar")
        cube.add_dim_coord(dim_coord, 0)
        expected_index = np.array([5, 6, 7, 8, 9])
        series = iris.pandas.as_series(cube)
        self.assertArrayEqual(series, cube.data)
        self.assertArrayEqual(series.index, expected_index)

    def test_masked(self):
        data = np.ma.MaskedArray([0, 1, 2, 3, 4.4], mask=[0, 1, 0, 1, 0])
        cube = Cube(data, long_name="foo")
        series = iris.pandas.as_series(cube)
        self.assertArrayEqual(series, cube.data.astype("f").filled(np.nan))

    def test_time_standard(self):
        cube = Cube(np.array([0, 1, 2, 3, 4]), long_name="ts")
        time_coord = DimCoord(
            [0, 100.1, 200.2, 300.3, 400.4],
            long_name="time",
            units="days since 2000-01-01 00:00",
        )
        cube.add_dim_coord(time_coord, 0)
        expected_index = [
            datetime.datetime(2000, 1, 1, 0, 0),
            datetime.datetime(2000, 4, 10, 2, 24),
            datetime.datetime(2000, 7, 19, 4, 48),
            datetime.datetime(2000, 10, 27, 7, 12),
            datetime.datetime(2001, 2, 4, 9, 36),
        ]
        series = iris.pandas.as_series(cube)
        self.assertArrayEqual(series, cube.data)
        assert list(series.index) == expected_index

    def test_time_360(self):
        cube = Cube(np.array([0, 1, 2, 3, 4]), long_name="ts")
        time_unit = cf_units.Unit(
            "days since 2000-01-01 00:00", calendar=cf_units.CALENDAR_360_DAY
        )
        time_coord = DimCoord(
            [0, 100.1, 200.2, 300.3, 400.4], long_name="time", units=time_unit
        )
        cube.add_dim_coord(time_coord, 0)
        expected_index = [
            cftime.Datetime360Day(2000, 1, 1, 0, 0),
            cftime.Datetime360Day(2000, 4, 11, 2, 24),
            cftime.Datetime360Day(2000, 7, 21, 4, 48),
            cftime.Datetime360Day(2000, 11, 1, 7, 12),
            cftime.Datetime360Day(2001, 2, 11, 9, 36),
        ]

        series = iris.pandas.as_series(cube)
        self.assertArrayEqual(series, cube.data)
        self.assertArrayEqual(series.index, expected_index)

    def test_copy_true(self):
        cube = Cube(np.array([0, 1, 2, 3, 4]), long_name="foo")
        series = iris.pandas.as_series(cube)
        series[0] = 99
        assert cube.data[0] == 0

    def test_copy_int32_false(self):
        cube = Cube(np.array([0, 1, 2, 3, 4], dtype=np.int32), long_name="foo")
        series = iris.pandas.as_series(cube, copy=False)
        series[0] = 99
        assert cube.data[0] == 99

    def test_copy_int64_false(self):
        cube = Cube(np.array([0, 1, 2, 3, 4], dtype=np.int64), long_name="foo")
        series = iris.pandas.as_series(cube, copy=False)
        series[0] = 99
        assert cube.data[0] == 99

    def test_copy_float_false(self):
        cube = Cube(np.array([0, 1, 2, 3.3, 4]), long_name="foo")
        series = iris.pandas.as_series(cube, copy=False)
        series[0] = 99
        assert cube.data[0] == 99

    def test_copy_masked_true(self):
        data = np.ma.MaskedArray([0, 1, 2, 3, 4], mask=[0, 1, 0, 1, 0])
        cube = Cube(data, long_name="foo")
        series = iris.pandas.as_series(cube)
        series[0] = 99
        assert cube.data[0] == 0

    def test_copy_masked_false(self):
        data = np.ma.MaskedArray([0, 1, 2, 3, 4], mask=[0, 1, 0, 1, 0])
        cube = Cube(data, long_name="foo")
        with pytest.raises(ValueError):
            _ = iris.pandas.as_series(cube, copy=False)


@skip_pandas
@pytest.mark.filterwarnings(
    "ignore:You are using legacy 2-dimensional behaviour.*:FutureWarning"
)
class TestAsDataFrame(tests.IrisTest):
    """Test conversion of 2D cubes to Pandas using as_data_frame()"""

    def test_no_dim_coords(self):
        cube = Cube(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), long_name="foo"
        )
        expected_index = [0, 1]
        expected_columns = [0, 1, 2, 3, 4]
        data_frame = iris.pandas.as_data_frame(cube)
        self.assertArrayEqual(data_frame, cube.data)
        self.assertArrayEqual(data_frame.index, expected_index)
        self.assertArrayEqual(data_frame.columns, expected_columns)

    def test_no_x_coord(self):
        cube = Cube(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), long_name="foo"
        )
        y_coord = DimCoord([10, 11], long_name="bar")
        cube.add_dim_coord(y_coord, 0)
        expected_index = [10, 11]
        expected_columns = [0, 1, 2, 3, 4]
        data_frame = iris.pandas.as_data_frame(cube)
        self.assertArrayEqual(data_frame, cube.data)
        self.assertArrayEqual(data_frame.index, expected_index)
        self.assertArrayEqual(data_frame.columns, expected_columns)

    def test_no_y_coord(self):
        cube = Cube(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), long_name="foo"
        )
        x_coord = DimCoord([10, 11, 12, 13, 14], long_name="bar")
        cube.add_dim_coord(x_coord, 1)
        expected_index = [0, 1]
        expected_columns = [10, 11, 12, 13, 14]
        data_frame = iris.pandas.as_data_frame(cube)
        self.assertArrayEqual(data_frame, cube.data)
        self.assertArrayEqual(data_frame.index, expected_index)
        self.assertArrayEqual(data_frame.columns, expected_columns)

    def test_simple(self):
        cube = Cube(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), long_name="foo"
        )
        x_coord = DimCoord([10, 11, 12, 13, 14], long_name="bar")
        y_coord = DimCoord([15, 16], long_name="milk")
        cube.add_dim_coord(x_coord, 1)
        cube.add_dim_coord(y_coord, 0)
        expected_index = [15, 16]
        expected_columns = [10, 11, 12, 13, 14]
        data_frame = iris.pandas.as_data_frame(cube)
        self.assertArrayEqual(data_frame, cube.data)
        self.assertArrayEqual(data_frame.index, expected_index)
        self.assertArrayEqual(data_frame.columns, expected_columns)

    def test_masked(self):
        data = np.ma.MaskedArray(
            [[0, 1, 2, 3, 4.4], [5, 6, 7, 8, 9]],
            mask=[[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]],
        )
        cube = Cube(data, long_name="foo")
        expected_index = [0, 1]
        expected_columns = [0, 1, 2, 3, 4]
        data_frame = iris.pandas.as_data_frame(cube)
        self.assertArrayEqual(data_frame, cube.data.astype("f").filled(np.nan))
        self.assertArrayEqual(data_frame.index, expected_index)
        self.assertArrayEqual(data_frame.columns, expected_columns)

    def test_time_standard(self):
        cube = Cube(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), long_name="ts"
        )
        day_offsets = [0, 100.1, 200.2, 300.3, 400.4]
        time_coord = DimCoord(
            day_offsets, long_name="time", units="days since 2000-01-01 00:00"
        )
        cube.add_dim_coord(time_coord, 1)
        data_frame = iris.pandas.as_data_frame(cube)
        self.assertArrayEqual(data_frame, cube.data)
        nanoseconds_per_day = 24 * 60 * 60 * 1000000000
        days_to_2000 = 365 * 30 + 7
        # pandas Timestamp class cannot handle floats in pandas <v0.12
        timestamps = [
            pandas.Timestamp(
                int(nanoseconds_per_day * (days_to_2000 + day_offset))
            )
            for day_offset in day_offsets
        ]
        assert all(data_frame.columns == timestamps)
        assert all(data_frame.index == [0, 1])

    def test_time_360(self):
        cube = Cube(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), long_name="ts"
        )
        time_unit = cf_units.Unit(
            "days since 2000-01-01 00:00", calendar=cf_units.CALENDAR_360_DAY
        )
        time_coord = DimCoord(
            [100.1, 200.2], long_name="time", units=time_unit
        )
        cube.add_dim_coord(time_coord, 0)
        expected_index = [
            cftime.Datetime360Day(2000, 4, 11, 2, 24),
            cftime.Datetime360Day(2000, 7, 21, 4, 48),
        ]

        expected_columns = [0, 1, 2, 3, 4]
        data_frame = iris.pandas.as_data_frame(cube)
        self.assertArrayEqual(data_frame, cube.data)
        self.assertArrayEqual(data_frame.index, expected_index)
        self.assertArrayEqual(data_frame.columns, expected_columns)

    def test_copy_true(self):
        cube = Cube(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), long_name="foo"
        )
        data_frame = iris.pandas.as_data_frame(cube)
        data_frame[0][0] = 99
        assert cube.data[0, 0] == 0

    def test_copy_int32_false(self):
        cube = Cube(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=np.int32),
            long_name="foo",
        )
        data_frame = iris.pandas.as_data_frame(cube, copy=False)
        data_frame[0][0] = 99
        assert cube.data[0, 0] == 99

    def test_copy_int64_false(self):
        cube = Cube(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=np.int64),
            long_name="foo",
        )
        data_frame = iris.pandas.as_data_frame(cube, copy=False)
        data_frame[0][0] = 99
        assert cube.data[0, 0] == 99

    def test_copy_float_false(self):
        cube = Cube(
            np.array([[0, 1, 2, 3, 4.4], [5, 6, 7, 8, 9]]), long_name="foo"
        )
        data_frame = iris.pandas.as_data_frame(cube, copy=False)
        data_frame[0][0] = 99
        assert cube.data[0, 0] == 99

    def test_copy_masked_true(self):
        data = np.ma.MaskedArray(
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            mask=[[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]],
        )
        cube = Cube(data, long_name="foo")
        data_frame = iris.pandas.as_data_frame(cube)
        data_frame[0][0] = 99
        assert cube.data[0, 0] == 0

    def test_copy_masked_false(self):
        data = np.ma.MaskedArray(
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            mask=[[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]],
        )
        cube = Cube(data, long_name="foo")
        with pytest.raises(ValueError):
            _ = iris.pandas.as_data_frame(cube, copy=False)

    def test_copy_false_with_cube_view(self):
        data = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        cube = Cube(data[:], long_name="foo")
        data_frame = iris.pandas.as_data_frame(cube, copy=False)
        data_frame[0][0] = 99
        assert cube.data[0, 0] == 99


@skip_pandas
class TestAsDataFrameNDim(tests.IrisTest):
    """Test conversion of n-dimensional cubes to Pandas using as_data_frame()"""

    @pytest.fixture(autouse=True)
    def _activate_pandas_ndim(self, activate_pandas_ndim):
        pass

    def test_no_dim_coords(self):
        cube = Cube(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), long_name="foo"
        )
        expected_dim0 = np.repeat([0, 1], 5)
        expected_dim1 = np.tile([0, 1, 2, 3, 4], 2)
        expected_foo = np.arange(0, 10)
        data_frame = iris.pandas.as_data_frame(cube)
        self.assertArrayEqual(data_frame.foo.values, expected_foo)
        self.assertArrayEqual(
            data_frame.index.get_level_values("dim0"), expected_dim0
        )
        self.assertArrayEqual(
            data_frame.index.get_level_values("dim1"), expected_dim1
        )

    def test_no_x_coord(self):
        cube = Cube(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), long_name="foo"
        )
        dim0 = DimCoord([10, 11], long_name="bar")
        cube.add_dim_coord(dim0, 0)
        expected_bar = np.repeat([10, 11], 5)
        expected_dim1 = np.tile([0, 1, 2, 3, 4], 2)
        expected_foo = np.arange(0, 10)
        data_frame = iris.pandas.as_data_frame(cube)
        self.assertArrayEqual(data_frame.foo, expected_foo)
        self.assertArrayEqual(
            data_frame.index.get_level_values("bar"), expected_bar
        )
        self.assertArrayEqual(
            data_frame.index.get_level_values("dim1"), expected_dim1
        )

    def test_no_y_coord(self):
        cube = Cube(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), long_name="foo"
        )
        dim1 = DimCoord([10, 11, 12, 13, 14], long_name="bar")
        cube.add_dim_coord(dim1, 1)
        expected_dim0 = np.repeat([0, 1], 5)
        expected_bar = np.tile([10, 11, 12, 13, 14], 2)
        expected_foo = np.arange(0, 10)
        data_frame = iris.pandas.as_data_frame(cube)
        self.assertArrayEqual(data_frame.foo, expected_foo.data)
        self.assertArrayEqual(
            data_frame.index.get_level_values("dim0"), expected_dim0
        )
        self.assertArrayEqual(
            data_frame.index.get_level_values("bar"), expected_bar
        )

    def test_simple1D(self):
        cube = Cube(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), long_name="foo")
        dim_coord = DimCoord(
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], long_name="bar"
        )
        cube.add_dim_coord(dim_coord, 0)
        expected_bar = np.arange(10, 20)
        expected_foo = np.arange(0, 10)
        data_frame = iris.pandas.as_data_frame(cube)
        self.assertArrayEqual(data_frame.foo, expected_foo)
        self.assertArrayEqual(
            data_frame.index.get_level_values("bar"), expected_bar
        )

    def test_simple2D(self):
        cube2d = Cube(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), long_name="foo"
        )
        dim0_coord = DimCoord([15, 16], long_name="milk")
        dim1_coord = DimCoord([10, 11, 12, 13, 14], long_name="bar")
        cube2d.add_dim_coord(dim0_coord, 0)
        cube2d.add_dim_coord(dim1_coord, 1)
        expected_milk = np.repeat([15, 16], 5)
        expected_bar = np.tile([10, 11, 12, 13, 14], 2)
        expected_foo = np.arange(0, 10)
        data_frame = iris.pandas.as_data_frame(cube2d)
        self.assertArrayEqual(data_frame.foo, expected_foo)
        self.assertArrayEqual(
            data_frame.index.get_level_values("milk"), expected_milk
        )
        self.assertArrayEqual(
            data_frame.index.get_level_values("bar"), expected_bar
        )

    def test_simple3D(self):
        cube3d = Cube(
            np.array(
                [
                    [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
                    [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
                    [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
                ]
            ),
            long_name="foo",
        )
        dim0_coord = DimCoord([1, 2, 3], long_name="milk")
        dim1_coord = DimCoord([10, 11], long_name="bar")
        dim2_coord = DimCoord([20, 21, 22, 23, 24], long_name="kid")
        cube3d.add_dim_coord(dim0_coord, 0)
        cube3d.add_dim_coord(dim1_coord, 1)
        cube3d.add_dim_coord(dim2_coord, 2)
        expected_milk = np.repeat([1, 2, 3], 10)
        expected_bar = np.tile(np.repeat([10, 11], 5), 3)
        expected_kid = np.tile([20, 21, 22, 23, 24], 6)
        expected_foo = np.arange(0, 30)
        data_frame = iris.pandas.as_data_frame(cube3d)
        self.assertArrayEqual(data_frame.foo, expected_foo)
        self.assertArrayEqual(
            data_frame.index.get_level_values("milk"), expected_milk
        )
        self.assertArrayEqual(
            data_frame.index.get_level_values("bar"), expected_bar
        )
        self.assertArrayEqual(
            data_frame.index.get_level_values("kid"), expected_kid
        )

    def test_copy_false(self):
        cube = Cube(np.array([0, 1, 2, 3, 4]), long_name="foo")
        data_frame = iris.pandas.as_data_frame(cube, copy=False)
        cube.data[2] = 99
        assert cube.data[2] == data_frame.foo[2]

    def test_copy_true(self):
        cube = Cube(np.array([0, 1, 2, 3, 4]), long_name="foo")
        data_frame = iris.pandas.as_data_frame(cube, copy=True)
        cube.data[2] = 99
        assert cube.data[2] != data_frame.foo[2]

    def test_time_standard(self):
        cube = Cube(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), long_name="ts"
        )
        day_offsets = [0, 100.1, 200.2, 300.3, 400.4]
        time_coord = DimCoord(
            day_offsets, long_name="time", units="days since 2000-01-01 00:00"
        )
        cube.add_dim_coord(time_coord, 1)
        expected_ts = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        expected_time = np.array(
            [
                cftime.DatetimeGregorian(
                    2000, 1, 1, 0, 0, 0, 0, has_year_zero=False
                ),
                cftime.DatetimeGregorian(
                    2000, 4, 10, 2, 24, 0, 0, has_year_zero=False
                ),
                cftime.DatetimeGregorian(
                    2000, 7, 19, 4, 48, 0, 0, has_year_zero=False
                ),
                cftime.DatetimeGregorian(
                    2000, 10, 27, 7, 12, 0, 0, has_year_zero=False
                ),
                cftime.DatetimeGregorian(
                    2001, 2, 4, 9, 36, 0, 0, has_year_zero=False
                ),
                cftime.DatetimeGregorian(
                    2000, 1, 1, 0, 0, 0, 0, has_year_zero=False
                ),
                cftime.DatetimeGregorian(
                    2000, 4, 10, 2, 24, 0, 0, has_year_zero=False
                ),
                cftime.DatetimeGregorian(
                    2000, 7, 19, 4, 48, 0, 0, has_year_zero=False
                ),
                cftime.DatetimeGregorian(
                    2000, 10, 27, 7, 12, 0, 0, has_year_zero=False
                ),
                cftime.DatetimeGregorian(
                    2001, 2, 4, 9, 36, 0, 0, has_year_zero=False
                ),
            ],
            dtype=object,
        )
        data_frame = iris.pandas.as_data_frame(cube)
        self.assertArrayEqual(data_frame.ts, expected_ts)
        self.assertArrayEqual(
            data_frame.index.get_level_values("time"), expected_time
        )

    def test_time_360(self):
        cube = Cube(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), long_name="ts"
        )
        time_unit = cf_units.Unit(
            "days since 2000-01-01 00:00", calendar=cf_units.CALENDAR_360_DAY
        )
        time_coord = DimCoord(
            [100.1, 200.2], long_name="time", units=time_unit
        )
        cube.add_dim_coord(time_coord, 0)
        expected_time = np.array(
            [
                cftime.Datetime360Day(
                    2000, 4, 11, 2, 24, 0, 0, has_year_zero=True
                ),
                cftime.Datetime360Day(
                    2000, 4, 11, 2, 24, 0, 0, has_year_zero=True
                ),
                cftime.Datetime360Day(
                    2000, 4, 11, 2, 24, 0, 0, has_year_zero=True
                ),
                cftime.Datetime360Day(
                    2000, 4, 11, 2, 24, 0, 0, has_year_zero=True
                ),
                cftime.Datetime360Day(
                    2000, 4, 11, 2, 24, 0, 0, has_year_zero=True
                ),
                cftime.Datetime360Day(
                    2000, 7, 21, 4, 48, 0, 0, has_year_zero=True
                ),
                cftime.Datetime360Day(
                    2000, 7, 21, 4, 48, 0, 0, has_year_zero=True
                ),
                cftime.Datetime360Day(
                    2000, 7, 21, 4, 48, 0, 0, has_year_zero=True
                ),
                cftime.Datetime360Day(
                    2000, 7, 21, 4, 48, 0, 0, has_year_zero=True
                ),
                cftime.Datetime360Day(
                    2000, 7, 21, 4, 48, 0, 0, has_year_zero=True
                ),
            ],
            dtype=object,
        )
        expected_ts = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        data_frame = iris.pandas.as_data_frame(cube)
        self.assertArrayEqual(data_frame.ts, expected_ts)
        self.assertArrayEqual(
            data_frame.index.get_level_values("time"), expected_time
        )

    def test_aux_coord(self):
        cube = Cube(np.array([[0, 1], [5, 6]]), long_name="foo")
        dim0_coord = DimCoord([15, 16], long_name="milk")
        dim1_coord = DimCoord([10, 11], long_name="bar")
        aux0_coord = AuxCoord(["fiveteen", "sixteen"], long_name="words0")
        aux1_coord = AuxCoord(["ten", "eleven"], long_name="words1")
        cube.add_dim_coord(dim0_coord, 0)
        cube.add_dim_coord(dim1_coord, 1)
        cube.add_aux_coord(aux0_coord, 0)
        cube.add_aux_coord(aux1_coord, 1)
        expected_foo = np.array([0, 1, 5, 6])
        expected_words0 = np.repeat(["fiveteen", "sixteen"], 2)
        expected_words1 = np.tile(["ten", "eleven"], 2)
        data_frame = iris.pandas.as_data_frame(cube, add_aux_coords=True)
        self.assertArrayEqual(data_frame.foo, expected_foo)
        self.assertArrayEqual(data_frame.words0, expected_words0)
        self.assertArrayEqual(data_frame.words1, expected_words1)

    def test_aux_coord2(self):
        cube = Cube(np.array([[0, 1], [5, 6]]), long_name="foo")
        dim0_coord = DimCoord([15, 16], long_name="milk")
        dim1_coord = DimCoord([10, 11], long_name="bar")
        aux0_coord = AuxCoord(["fiveteen0", "sixteen0"], long_name="words0")
        aux1_coord = AuxCoord(["fiveteen1", "sixteen1"], long_name="words1")
        aux2_coord = AuxCoord(["ten", "eleven"], long_name="words2")
        cube.add_dim_coord(dim0_coord, 0)
        cube.add_dim_coord(dim1_coord, 1)
        # Two aux coords associated with dim0
        cube.add_aux_coord(aux0_coord, 0)
        cube.add_aux_coord(aux1_coord, 0)
        # One aux coords associated with dim1
        cube.add_aux_coord(aux2_coord, 1)
        expected_foo = np.array([0, 1, 5, 6])
        expected_words0 = np.repeat(["fiveteen0", "sixteen0"], 2)
        expected_words1 = np.repeat(["fiveteen1", "sixteen1"], 2)
        expected_words2 = np.tile(["ten", "eleven"], 2)
        data_frame = iris.pandas.as_data_frame(cube, add_aux_coords=True)
        self.assertArrayEqual(data_frame.foo, expected_foo)
        self.assertArrayEqual(data_frame.words0, expected_words0)
        self.assertArrayEqual(data_frame.words1, expected_words1)
        self.assertArrayEqual(data_frame.words2, expected_words2)

    def test_multidim_aux(self):
        cube = Cube(
            np.arange(300, 312, 1).reshape([2, 2, 3]),
            long_name="air_temperature",
        )
        dim0_coord = DimCoord([0, 10], long_name="longitude")
        dim1_coord = DimCoord([25, 35], long_name="latitude")
        dim2_coord = DimCoord([0, 100, 200], long_name="height")
        aux0_coord = AuxCoord(
            [[True, False], [False, False]], long_name="in_region"
        )
        cube.add_dim_coord(dim0_coord, 0)
        cube.add_dim_coord(dim1_coord, 1)
        cube.add_dim_coord(dim2_coord, 2)
        cube.add_aux_coord(aux0_coord, data_dims=(0, 1))
        expected_in_region = np.repeat([True, False, False, False], 3)
        data_frame = iris.pandas.as_data_frame(cube, add_aux_coords=True)
        self.assertArrayEqual(data_frame.in_region, expected_in_region)

    def test_add_scalar_coord(self):
        cube = Cube(np.array([[0, 1], [5, 6]]), long_name="foo")
        scalar_coord = iris.coords.AuxCoord(
            1, long_name="scalar_coord", units="no_unit"
        )
        cube.add_aux_coord(scalar_coord)
        expected_scalar_coord = np.repeat(1, 4)
        data_frame = iris.pandas.as_data_frame(cube, add_aux_coords=True)
        self.assertArrayEqual(data_frame.scalar_coord, expected_scalar_coord)

    def test_add_ancillary_variable(self):
        cube = Cube(np.array([[0, 1], [5, 6]]), long_name="foo")
        dim0_coord = DimCoord([0, 10], long_name="bar")
        cube.add_dim_coord(dim0_coord, 0)
        av = AncillaryVariable([10, 100], long_name="ancil_bar")
        av2 = AncillaryVariable(
            [1000], long_name="ancil_bar2"
        )  # Scalar ancillary variable
        cube.add_ancillary_variable(av, 0)
        cube.add_ancillary_variable(av2)
        expected_ancillary_variable = np.repeat([10, 100], 2)
        expected_ancillary_variable2 = np.repeat([1000], 4)
        data_frame = iris.pandas.as_data_frame(
            cube, add_ancillary_variables=True
        )
        self.assertArrayEqual(
            data_frame.ancil_bar, expected_ancillary_variable
        )
        self.assertArrayEqual(
            data_frame.ancil_bar2, expected_ancillary_variable2
        )

    def test_add_cell_measures(self):
        cube = Cube(np.array([[0, 1], [5, 6]]), long_name="foo")
        dim0_coord = DimCoord([0, 10], long_name="bar")
        cube.add_dim_coord(dim0_coord, 0)
        cm = CellMeasure([10, 100], long_name="cell_measure")
        cm2 = CellMeasure(
            1e4, long_name="cell_measure2"
        )  # Scalar cell measure
        cube.add_cell_measure(cm, 0)
        cube.add_cell_measure(cm2)
        expected_cell_measure = np.repeat([10, 100], 2)
        expected_cell_measure2 = np.repeat(1e4, 4)
        data_frame = iris.pandas.as_data_frame(cube, add_cell_measures=True)
        self.assertArrayEqual(data_frame.cell_measure, expected_cell_measure)
        self.assertArrayEqual(data_frame.cell_measure2, expected_cell_measure2)

    def test_instance_error(self):
        with pytest.raises(TypeError):
            _ = iris.pandas.as_data_frame(list())


@skip_pandas
@pytest.mark.filterwarnings(
    "ignore:.*as_cube has been deprecated.*:iris._deprecation.IrisDeprecation"
)
class TestSeriesAsCube(tests.IrisTest):
    def test_series_simple(self):
        series = pandas.Series([0, 1, 2, 3, 4], index=[5, 6, 7, 8, 9])
        self.assertCML(
            iris.pandas.as_cube(series),
            tests.get_result_path(("pandas", "as_cube", "series_simple.cml")),
        )

    def test_series_object(self):
        class Thing:
            def __repr__(self):
                return "A Thing"

        series = pandas.Series(
            [0, 1, 2, 3, 4],
            index=[Thing(), Thing(), Thing(), Thing(), Thing()],
        )
        self.assertCML(
            iris.pandas.as_cube(series),
            tests.get_result_path(("pandas", "as_cube", "series_object.cml")),
        )

    def test_series_masked(self):
        series = pandas.Series(
            [0, float("nan"), 2, np.nan, 4], index=[5, 6, 7, 8, 9]
        )
        self.assertCML(
            iris.pandas.as_cube(series),
            tests.get_result_path(("pandas", "as_cube", "series_masked.cml")),
        )

    def test_series_datetime_standard(self):
        series = pandas.Series(
            [0, 1, 2, 3, 4],
            index=[
                datetime.datetime(2001, 1, 1, 1, 1, 1),
                datetime.datetime(2002, 2, 2, 2, 2, 2),
                datetime.datetime(2003, 3, 3, 3, 3, 3),
                datetime.datetime(2004, 4, 4, 4, 4, 4),
                datetime.datetime(2005, 5, 5, 5, 5, 5),
            ],
        )
        self.assertCML(
            iris.pandas.as_cube(series),
            tests.get_result_path(
                ("pandas", "as_cube", "series_datetime_standard.cml")
            ),
        )

    def test_series_cftime_360(self):
        series = pandas.Series(
            [0, 1, 2, 3, 4],
            index=[
                cftime.datetime(2001, 1, 1, 1, 1, 1),
                cftime.datetime(2002, 2, 2, 2, 2, 2),
                cftime.datetime(2003, 3, 3, 3, 3, 3),
                cftime.datetime(2004, 4, 4, 4, 4, 4),
                cftime.datetime(2005, 5, 5, 5, 5, 5),
            ],
        )
        self.assertCML(
            iris.pandas.as_cube(
                series, calendars={0: cf_units.CALENDAR_360_DAY}
            ),
            tests.get_result_path(
                ("pandas", "as_cube", "series_netcdfimte_360.cml")
            ),
        )

    def test_copy_true(self):
        series = pandas.Series([0, 1, 2, 3, 4], index=[5, 6, 7, 8, 9])
        cube = iris.pandas.as_cube(series)
        cube.data[0] = 99
        assert series[5] == 0

    def test_copy_false(self):
        series = pandas.Series([0, 1, 2, 3, 4], index=[5, 6, 7, 8, 9])
        cube = iris.pandas.as_cube(series, copy=False)
        cube.data[0] = 99
        assert series[5] == 99


@skip_pandas
@pytest.mark.filterwarnings(
    "ignore:.*as_cube has been deprecated.*:iris._deprecation.IrisDeprecation"
)
class TestDataFrameAsCube(tests.IrisTest):
    def test_data_frame_simple(self):
        data_frame = pandas.DataFrame(
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            index=[10, 11],
            columns=[12, 13, 14, 15, 16],
        )
        self.assertCML(
            iris.pandas.as_cube(data_frame),
            tests.get_result_path(
                ("pandas", "as_cube", "data_frame_simple.cml")
            ),
        )

    def test_data_frame_nonotonic(self):
        data_frame = pandas.DataFrame(
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            index=[10, 10],
            columns=[12, 12, 14, 15, 16],
        )
        self.assertCML(
            iris.pandas.as_cube(data_frame),
            tests.get_result_path(
                ("pandas", "as_cube", "data_frame_nonotonic.cml")
            ),
        )

    def test_data_frame_masked(self):
        data_frame = pandas.DataFrame(
            [[0, float("nan"), 2, 3, 4], [5, 6, 7, np.nan, 9]],
            index=[10, 11],
            columns=[12, 13, 14, 15, 16],
        )
        self.assertCML(
            iris.pandas.as_cube(data_frame),
            tests.get_result_path(
                ("pandas", "as_cube", "data_frame_masked.cml")
            ),
        )

    def test_data_frame_multidim(self):
        data_frame = pandas.DataFrame(
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            index=[0, 1],
            columns=["col_1", "col_2", "col_3", "col_4", "col_5"],
        )
        self.assertCML(
            iris.pandas.as_cube(data_frame),
            tests.get_result_path(
                ("pandas", "as_cube", "data_frame_multidim.cml")
            ),
        )

    def test_data_frame_cftime_360(self):
        data_frame = pandas.DataFrame(
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            index=[
                cftime.datetime(2001, 1, 1, 1, 1, 1),
                cftime.datetime(2002, 2, 2, 2, 2, 2),
            ],
            columns=[10, 11, 12, 13, 14],
        )
        self.assertCML(
            iris.pandas.as_cube(
                data_frame, calendars={0: cf_units.CALENDAR_360_DAY}
            ),
            tests.get_result_path(
                ("pandas", "as_cube", "data_frame_netcdftime_360.cml")
            ),
        )

    def test_data_frame_datetime_standard(self):
        data_frame = pandas.DataFrame(
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            index=[
                datetime.datetime(2001, 1, 1, 1, 1, 1),
                datetime.datetime(2002, 2, 2, 2, 2, 2),
            ],
            columns=[10, 11, 12, 13, 14],
        )
        self.assertCML(
            iris.pandas.as_cube(data_frame),
            tests.get_result_path(
                ("pandas", "as_cube", "data_frame_datetime_standard.cml")
            ),
        )

    def test_copy_true(self):
        data_frame = pandas.DataFrame([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        cube = iris.pandas.as_cube(data_frame)
        cube.data[0, 0] = 99
        assert data_frame[0][0] == 0

    def test_copy_false(self):
        data_frame = pandas.DataFrame([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        cube = iris.pandas.as_cube(data_frame, copy=False)
        cube.data[0, 0] = 99
        assert data_frame[0][0] == 99


@skip_pandas
class TestFutureAndDeprecation:
    def test_as_cube_deprecation_warning(self):
        data_frame = pandas.DataFrame([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        with pytest.warns(
            IrisDeprecation, match="as_cube has been deprecated"
        ):
            _ = iris.pandas.as_cube(data_frame)

    def test_as_series_deprecation_warning(self):
        cube = Cube(np.array([0, 1, 2, 3, 4]), long_name="foo")
        with pytest.warns(
            IrisDeprecation, match="as_series has been deprecated"
        ):
            _ = iris.pandas.as_series(cube)

    def test_as_dataframe_future_warning(self):
        cube = Cube(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), long_name="foo"
        )
        with pytest.warns(
            FutureWarning, match="You are using legacy 2-dimensional behaviour"
        ):
            _ = iris.pandas.as_data_frame(cube)

    def test_as_dataframe_no_future_warning(self, activate_pandas_ndim):
        cube = Cube(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), long_name="foo"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            _ = iris.pandas.as_data_frame(cube)


@skip_pandas
class TestPandasAsCubes(tests.IrisTest):
    @staticmethod
    def _create_pandas(index_levels=0, is_series=False):
        index_length = 3

        index_names = [f"index_{i}" for i in range(index_levels)]
        index_values = [
            np.arange(index_length) * 10 * (i + 1) for i in range(index_levels)
        ]

        if index_levels == 1:
            index = pandas.Index(index_values[0], name=index_names[0])
            data_length = index_length
        elif index_levels > 1:
            index = pandas.MultiIndex.from_product(
                index_values, names=index_names
            )
            data_length = index.nunique()
        else:
            index = None
            data_length = index_length

        data = np.arange(data_length) * 10

        if is_series:
            class_ = pandas.Series
        else:
            class_ = pandas.DataFrame

        return class_(data, index=index)

    def test_1d_no_index(self):
        df = self._create_pandas()
        result = iris.pandas.as_cubes(df)

        expected_coord = DimCoord(df.index.values)
        expected_cube = Cube(
            data=df[0].values,
            long_name=str(df[0].name),
            dim_coords_and_dims=[(expected_coord, 0)],
        )
        assert result == [expected_cube]

    def test_1d_with_index(self):
        df = self._create_pandas(index_levels=1)
        result = iris.pandas.as_cubes(df)

        expected_coord = DimCoord(df.index.values, long_name=df.index.name)
        (result_cube,) = result
        assert result_cube.dim_coords == (expected_coord,)

    def test_1d_series_no_index(self):
        series = self._create_pandas(is_series=True)
        result = iris.pandas.as_cubes(series)

        expected_coord = DimCoord(series.index.values)
        expected_cube = Cube(
            data=series.values, dim_coords_and_dims=[(expected_coord, 0)]
        )
        assert result == [expected_cube]

    def test_1d_series_with_index(self):
        series = self._create_pandas(index_levels=1, is_series=True)
        result = iris.pandas.as_cubes(series)

        expected_coord = DimCoord(
            series.index.values, long_name=series.index.name
        )
        (result_cube,) = result
        assert result_cube.dim_coords == (expected_coord,)

    def test_3d(self):
        df = self._create_pandas(index_levels=3)
        result = iris.pandas.as_cubes(df)

        expected_coords = [
            DimCoord(level.values, long_name=level.name)
            for level in df.index.levels
        ]
        (result_cube,) = result
        assert result_cube.dim_coords == tuple(expected_coords)

    def test_3d_series(self):
        series = self._create_pandas(index_levels=3, is_series=True)
        result = iris.pandas.as_cubes(series)

        expected_coords = [
            DimCoord(level.values, long_name=level.name)
            for level in series.index.levels
        ]
        (result_cube,) = result
        assert result_cube.dim_coords == tuple(expected_coords)

    def test_non_unique_index(self):
        df = self._create_pandas(index_levels=1)
        new_index = df.index.values
        new_index[1] = new_index[0]
        df.set_index(new_index)

        with pytest.raises(ValueError, match="not unique per row"):
            _ = iris.pandas.as_cubes(df)

    def test_non_monotonic_index(self):
        df = self._create_pandas(index_levels=1)
        new_index = df.index.values
        new_index[:2] = new_index[1::-1]
        df.set_index(new_index)

        with pytest.raises(ValueError, match="not monotonic"):
            _ = iris.pandas.as_cubes(df)

    def test_missing_rows(self):
        df = self._create_pandas(index_levels=2)
        df = df[:-1]

        with pytest.raises(
            ValueError, match="Not all index values have a corresponding row"
        ):
            _ = iris.pandas.as_cubes(df)

    def test_aux_coord(self):
        df = self._create_pandas()
        coord_name = "foo"
        df[coord_name] = df.index.values
        result = iris.pandas.as_cubes(df, aux_coord_cols=[coord_name])

        expected_aux_coord = AuxCoord(
            df[coord_name].values, long_name=coord_name
        )
        (result_cube,) = result
        assert result_cube.aux_coords == (expected_aux_coord,)

    def test_cell_measure(self):
        df = self._create_pandas()
        coord_name = "foo"
        df[coord_name] = df.index.values
        result = iris.pandas.as_cubes(df, cell_measure_cols=[coord_name])

        expected_cm = CellMeasure(df[coord_name].values, long_name=coord_name)
        (result_cube,) = result
        assert result_cube.cell_measures() == [expected_cm]

    def test_ancillary_variable(self):
        df = self._create_pandas()
        coord_name = "foo"
        df[coord_name] = df.index.values
        result = iris.pandas.as_cubes(df, ancillary_variable_cols=[coord_name])

        expected_av = AncillaryVariable(
            df[coord_name].values, long_name=coord_name
        )
        (result_cube,) = result
        assert result_cube.ancillary_variables() == [expected_av]

    def test_3d_with_2d_coord(self):
        df = self._create_pandas(index_levels=3)
        coord_shape = df.index.levshape[:2]
        coord_values = np.arange(np.product(coord_shape))
        coord_name = "foo"
        df[coord_name] = coord_values.repeat(df.index.levshape[-1])
        result = iris.pandas.as_cubes(df, aux_coord_cols=[coord_name])

        expected_points = coord_values.reshape(coord_shape)
        (result_cube,) = result
        result_coord = result_cube.coord(coord_name)
        self.assertArrayEqual(result_coord.points, expected_points)
        assert result_coord.cube_dims(result_cube) == (0, 1)

    def test_coord_varies_all_indices(self):
        df = self._create_pandas(index_levels=3)
        coord_shape = df.index.levshape
        coord_values = np.arange(np.product(coord_shape))
        coord_name = "foo"
        df[coord_name] = coord_values
        result = iris.pandas.as_cubes(df, aux_coord_cols=[coord_name])

        expected_points = coord_values.reshape(coord_shape)
        (result_cube,) = result
        result_coord = result_cube.coord(coord_name)
        self.assertArrayEqual(result_coord.points, expected_points)
        assert result_coord.cube_dims(result_cube) == (0, 1, 2)

    def test_category_coord(self):
        # Something that varies on a dimension, but doesn't change with every
        #  increment.
        df = self._create_pandas(index_levels=2)
        coord_shape = df.index.levshape
        coord_values = np.arange(np.product(coord_shape))
        coord_name = "foo"

        # Create a repeating value along a dimension.
        step = coord_shape[-1]
        coord_values[1::step] = coord_values[::step]

        df[coord_name] = coord_values
        result = iris.pandas.as_cubes(df, aux_coord_cols=[coord_name])

        expected_points = coord_values.reshape(coord_shape)
        (result_cube,) = result
        result_coord = result_cube.coord(coord_name)
        self.assertArrayEqual(result_coord.points, expected_points)
        assert result_coord.cube_dims(result_cube) == (0, 1)

    def test_scalar_coord(self):
        df = self._create_pandas()
        coord_values = np.ones(len(df))
        coord_name = "foo"
        df[coord_name] = coord_values
        result = iris.pandas.as_cubes(df, aux_coord_cols=[coord_name])

        expected_points = np.unique(coord_values)
        (result_cube,) = result
        result_coord = result_cube.coord(coord_name)
        self.assertArrayEqual(result_coord.points, expected_points)
        assert result_coord.cube_dims(result_cube) == tuple()

    def test_multi_phenom(self):
        df = self._create_pandas()
        new_name = "new_phenom"
        df[new_name] = df[0]
        result = iris.pandas.as_cubes(df)

        # Note the shared coord object between both Cubes.
        expected_coord = DimCoord(df.index.values)
        expected_cube_kwargs = dict(dim_coords_and_dims=[(expected_coord, 0)])

        expected_cube_0 = Cube(
            data=df[0].values,
            long_name=str(df[0].name),
            **expected_cube_kwargs,
        )
        expected_cube_1 = Cube(
            data=df[new_name].values,
            long_name=new_name,
            **expected_cube_kwargs,
        )
        assert result == [expected_cube_0, expected_cube_1]

    def test_empty_series(self):
        series = pandas.Series(dtype=object)
        result = iris.pandas.as_cubes(series)

        assert result == CubeList()

    def test_empty_dataframe(self):
        df = pandas.DataFrame()
        result = iris.pandas.as_cubes(df)

        assert result == CubeList()

    def test_no_phenom(self):
        df = self._create_pandas()
        # Specify the only column as an AuxCoord.
        result = iris.pandas.as_cubes(df, aux_coord_cols=[0])

        assert result == CubeList()

    def test_standard_name_phenom(self):
        # long_name behaviour is tested in test_1d_no_index.
        df = self._create_pandas()
        new_name = "air_temperature"
        df = df.rename(columns={0: new_name})
        result = iris.pandas.as_cubes(df)

        (result_cube,) = result
        assert result_cube.standard_name == new_name

    def test_standard_name_coord(self):
        # long_name behaviour is tested in test_1d_with_index.
        df = self._create_pandas()
        new_name = "longitude"
        df.index.names = [new_name]
        result = iris.pandas.as_cubes(df)

        (result_cube,) = result
        result_coord = result_cube.coord(dim_coords=True)
        assert result_coord.standard_name == new_name

    def test_dtype_preserved_phenom(self):
        df = self._create_pandas()
        df = df.astype("int32")
        result = iris.pandas.as_cubes(df)

        (result_cube,) = result
        assert result_cube.dtype == np.int32

    def test_preserve_dim_order(self):
        new_order = ["index_1", "index_0", "index_2"]

        df = self._create_pandas(index_levels=3)
        df = df.reset_index()
        df = df.set_index(new_order)
        df = df.sort_index()
        result = iris.pandas.as_cubes(df)

        (result_cube,) = result
        dim_order = [c.name() for c in result_cube.dim_coords]
        assert dim_order == new_order

    def test_dtype_preserved_coord(self):
        df = self._create_pandas()
        new_index = df.index.astype("float64")
        df.index = new_index
        result = iris.pandas.as_cubes(df)

        (result_cube,) = result
        result_coord = result_cube.coord(dim_coords=True)
        assert result_coord.dtype == np.float64

    def test_string_phenom(self):
        # Strings can be uniquely troublesome.
        df = self._create_pandas()
        new_values = [str(v) for v in df[0]]
        df[0] = new_values
        result = iris.pandas.as_cubes(df)

        (result_cube,) = result
        self.assertArrayEqual(result_cube.data, new_values)

    def test_string_coord(self):
        # Strings can be uniquely troublesome.
        # Must test using an AuxCoord since strings cannot be DimCoords.
        df = self._create_pandas()
        new_points = [str(v) for v in df.index.values]
        coord_name = "foo"
        df[coord_name] = new_points
        result = iris.pandas.as_cubes(df, aux_coord_cols=[coord_name])

        (result_cube,) = result
        result_coord = result_cube.coord(coord_name)
        self.assertArrayEqual(result_coord.points, new_points)

    def test_series_with_col_args(self):
        series = self._create_pandas(is_series=True)
        with pytest.warns(Warning, match="is a Series; ignoring"):
            _ = iris.pandas.as_cubes(series, aux_coord_cols=["some_column"])

    def test_phenom_view(self):
        df = self._create_pandas()
        result = iris.pandas.as_cubes(df, copy=False)

        # Modify AFTER creating the Cube(s).
        df[0][0] += 1

        (result_cube,) = result
        assert result_cube.data[0] == df[0][0]

    def test_phenom_copy(self):
        df = self._create_pandas()
        result = iris.pandas.as_cubes(df)

        # Modify AFTER creating the Cube(s).
        df[0][0] += 1

        (result_cube,) = result
        assert result_cube.data[0] != df[0][0]

    def test_coord_never_view(self):
        # Using AuxCoord - DimCoords and Pandas indices are immutable.
        df = self._create_pandas()
        coord_name = "foo"
        df[coord_name] = df.index.values
        result = iris.pandas.as_cubes(
            df, copy=False, aux_coord_cols=[coord_name]
        )

        # Modify AFTER creating the Cube(s).
        df[coord_name][0] += 1

        (result_cube,) = result
        result_coord = result_cube.coord(coord_name)
        assert result_coord.points[0] != df[coord_name][0]

    def _test_dates_common(self, mode=None, alt_calendar=False):
        df = self._create_pandas()
        kwargs = dict(pandas_structure=df)
        coord_name = "dates"

        if alt_calendar:
            calendar = cf_units.CALENDAR_360_DAY
            # Only pass this when non-default.
            kwargs["calendars"] = {coord_name: calendar}
            expected_points = [8640, 8641, 8642]
        else:
            calendar = cf_units.CALENDAR_STANDARD
            expected_points = [8760, 8761, 8762]
        expected_units = cf_units.Unit(
            "hours since 1970-01-01 00:00:00", calendar=calendar
        )

        datetime_args = [(1971, 1, 1, i, 0, 0) for i in df.index.values]
        if mode == "index":
            values = [datetime.datetime(*a) for a in datetime_args]
            df.index = pandas.Index(values, name=coord_name)
        elif mode == "numpy":
            values = [datetime.datetime(*a) for a in datetime_args]
            df[coord_name] = values
            kwargs["aux_coord_cols"] = [coord_name]
        elif mode == "cftime":
            values = [
                cftime.datetime(*a, calendar=calendar) for a in datetime_args
            ]
            df[coord_name] = values
            kwargs["aux_coord_cols"] = [coord_name]
        else:
            raise ValueError("mode needs to be set")

        result = iris.pandas.as_cubes(**kwargs)

        (result_cube,) = result
        result_coord = result_cube.coord(coord_name)
        assert result_coord.units == expected_units
        self.assertArrayEqual(result_coord.points, expected_points)

    def test_datetime_index(self):
        self._test_dates_common(mode="index")

    def test_datetime_index_calendar(self):
        self._test_dates_common(mode="index", alt_calendar=True)

    def test_numpy_datetime_coord(self):
        # NumPy format is what happens if a Python datetime is assigned to a
        #  Pandas column.
        self._test_dates_common(mode="numpy")

    def test_numpy_datetime_coord_calendar(self):
        self._test_dates_common(mode="numpy", alt_calendar=True)

    def test_cftime_coord(self):
        self._test_dates_common(mode="cftime")

    def test_cftime_coord_calendar(self):
        self._test_dates_common(mode="cftime", alt_calendar=True)


if __name__ == "__main__":
    tests.main()
