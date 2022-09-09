# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import copy
import datetime
import unittest

import cf_units
import cftime
import matplotlib.units
import numpy as np

# Importing pandas has the side-effect of messing with the formatters
# used by matplotlib for handling dates.
default_units_registry = copy.copy(matplotlib.units.registry)
try:
    import pandas
except ImportError:
    # Disable all these tests if pandas is not installed.
    pandas = None
matplotlib.units.registry = default_units_registry

skip_pandas = unittest.skipIf(
    pandas is None, 'Test(s) require "pandas", ' "which is not available."
)

if pandas is not None:
    from iris.coords import AuxCoord, DimCoord
    from iris.cube import Cube
    import iris.pandas


@skip_pandas
class TestAsDataFrame(tests.IrisTest):
    """Test conversion of 2D cubes to Pandas using as_data_frame()"""

    def test_no_dim_coords(self):
        cube = Cube(
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), long_name="foo"
        )
        expected_dim0 = np.repeat([0, 1], 5)
        expected_dim1 = np.tile([0, 1, 2, 3, 4], 2)
        expected_foo = np.arange(0, 10)
        data_frame = iris.pandas.as_data_frame(cube)
        self.assertArrayEqual(data_frame.foo.values, expected_foo)
        self.assertArrayEqual(data_frame.dim0, expected_dim0)
        self.assertArrayEqual(data_frame.dim1, expected_dim1)

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
        self.assertArrayEqual(data_frame.bar, expected_bar)
        self.assertArrayEqual(data_frame.dim1, expected_dim1)

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
        self.assertArrayEqual(data_frame.dim0, expected_dim0)
        self.assertArrayEqual(data_frame.bar, expected_bar)

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
        self.assertArrayEqual(data_frame.bar, expected_bar)

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
        self.assertArrayEqual(data_frame.milk, expected_milk)
        self.assertArrayEqual(data_frame.bar, expected_bar)

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
        self.assertArrayEqual(data_frame.milk, expected_milk)
        self.assertArrayEqual(data_frame.bar, expected_bar)
        self.assertArrayEqual(data_frame.kid, expected_kid)

    def test_copy_false(self):
        cube = Cube(np.array([0, 1, 2, 3, 4]), long_name="foo")
        data_frame = iris.pandas.as_data_frame(cube, copy=False)
        cube.data[2] = 99
        self.assertEqual(cube.data[2], data_frame.foo[2])

    def test_copy_true(self):
        cube = Cube(np.array([0, 1, 2, 3, 4]), long_name="foo")
        data_frame = iris.pandas.as_data_frame(cube, copy=True)
        cube.data[2] = 99
        self.assertNotEqual(cube.data[2], data_frame.foo[2])

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
        self.assertArrayEqual(data_frame.time, expected_time)

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
        self.assertArrayEqual(data_frame.time, expected_time)

    def test_attributes(self):
        cube = Cube(np.array([0, 1, 2, 3, 4]), long_name="foo")
        cube.attributes["sheep"] = "baa"
        expected_attribute = np.repeat("baa", 5)
        data_frame = iris.pandas.as_data_frame(
            cube, add_global_attributes=["sheep"]
        )
        self.assertArrayEqual(expected_attribute, data_frame.sheep)

    def test_attribute_error(self):
        cube = Cube(np.array([0, 1, 2, 3, 4]), long_name="foo")
        self.assertRaises(
            ValueError,
            iris.pandas.as_data_frame,
            cube,
            add_global_attributes=["sheep"],
        )

    def test_aux_coord(self):
        cube2d = Cube(np.array([[0, 1], [5, 6]]), long_name="foo")
        dim0_coord = DimCoord([15, 16], long_name="milk")
        dim1_coord = DimCoord([10, 11], long_name="bar")
        aux0_coord = AuxCoord(["fiveteen", "sixteen"], long_name="words0")
        aux1_coord = AuxCoord(["ten", "eleven"], long_name="words1")
        cube2d.add_dim_coord(dim0_coord, 0)
        cube2d.add_dim_coord(dim1_coord, 1)
        cube2d.add_aux_coord(aux0_coord, 0)
        cube2d.add_aux_coord(aux1_coord, 1)
        expected_milk = np.repeat([15, 16], 2)
        expected_bar = np.tile([10, 11], 2)
        expected_foo = np.array([0, 1, 5, 6])
        expected_words0 = np.repeat(["fiveteen", "sixteen"], 2)
        expected_words1 = np.tile(["ten", "eleven"], 2)
        data_frame = iris.pandas.as_data_frame(cube2d, add_aux_coord=True)
        self.assertArrayEqual(data_frame.foo, expected_foo)
        self.assertArrayEqual(data_frame.milk, expected_milk)
        self.assertArrayEqual(data_frame.bar, expected_bar)
        self.assertArrayEqual(data_frame.words0, expected_words0)
        self.assertArrayEqual(data_frame.words1, expected_words1)

    def test_multidim_aux(self):
        cube = Cube(
            np.arange(300, 312, 1).reshape(2, 2, 3),
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
        data_frame = iris.pandas.as_data_frame(cube, add_aux_coord=True)
        self.assertArrayEqual(data_frame.in_region, expected_in_region)


@skip_pandas
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
        self.assertEqual(series[5], 0)

    def test_copy_false(self):
        series = pandas.Series([0, 1, 2, 3, 4], index=[5, 6, 7, 8, 9])
        cube = iris.pandas.as_cube(series, copy=False)
        cube.data[0] = 99
        self.assertEqual(series[5], 99)


@skip_pandas
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
        self.assertEqual(data_frame[0][0], 0)

    def test_copy_false(self):
        data_frame = pandas.DataFrame([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        cube = iris.pandas.as_cube(data_frame, copy=False)
        cube.data[0, 0] = 99
        self.assertEqual(data_frame[0][0], 99)


if __name__ == "__main__":
    tests.main()
