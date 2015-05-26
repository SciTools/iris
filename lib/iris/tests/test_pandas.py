# (C) British Crown Copyright 2013 - 2015, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import copy
import datetime
import unittest

import matplotlib.units
import netcdftime
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

skip_pandas = unittest.skipIf(pandas is None,
                              'Test(s) require "pandas", '
                              'which is not available.')

if pandas is not None:
    from iris.coords import DimCoord
    from iris.cube import Cube
    import iris.pandas
    import iris.unit


@skip_pandas
class TestAsSeries(tests.IrisTest):
    """Test conversion of 1D cubes to Pandas using as_series()"""

    def test_no_dim_coord(self):
        cube = Cube(np.array([0, 1, 2, 3, 4]), long_name="foo")
        series = iris.pandas.as_series(cube)
        self.assertArrayEqual(series, cube.data)
        self.assertString(
            str(series),
            tests.get_result_path(('pandas', 'as_series',
                                   'no_dim_coord.txt')))

    def test_simple(self):
        cube = Cube(np.array([0, 1, 2, 3, 4.4]), long_name="foo")
        dim_coord = DimCoord([5, 6, 7, 8, 9], long_name="bar")
        cube.add_dim_coord(dim_coord, 0)
        series = iris.pandas.as_series(cube)
        self.assertArrayEqual(series, cube.data)
        self.assertString(
            str(series),
            tests.get_result_path(('pandas', 'as_series', 'simple.txt')))

    def test_masked(self):
        data = np.ma.MaskedArray([0, 1, 2, 3, 4.4], mask=[0, 1, 0, 1, 0])
        cube = Cube(data, long_name="foo")
        series = iris.pandas.as_series(cube)
        self.assertArrayEqual(series, cube.data.astype('f').filled(np.nan))
        self.assertString(
            str(series),
            tests.get_result_path(('pandas', 'as_series', 'masked.txt')))

    def test_time_gregorian(self):
        cube = Cube(np.array([0, 1, 2, 3, 4]), long_name="ts")
        time_coord = DimCoord([0, 100.1, 200.2, 300.3, 400.4],
                              long_name="time",
                              units="days since 2000-01-01 00:00")
        cube.add_dim_coord(time_coord, 0)
        series = iris.pandas.as_series(cube)
        self.assertArrayEqual(series, cube.data)
        self.assertString(
            str(series),
            tests.get_result_path(('pandas', 'as_series',
                                  'time_gregorian.txt')))

    def test_time_360(self):
        cube = Cube(np.array([0, 1, 2, 3, 4]), long_name="ts")
        time_unit = iris.unit.Unit("days since 2000-01-01 00:00",
                                   calendar=iris.unit.CALENDAR_360_DAY)
        time_coord = DimCoord([0, 100.1, 200.2, 300.3, 400.4],
                              long_name="time", units=time_unit)
        cube.add_dim_coord(time_coord, 0)
        series = iris.pandas.as_series(cube)
        self.assertArrayEqual(series, cube.data)
        self.assertString(
            str(series),
            tests.get_result_path(('pandas', 'as_series',
                                   'time_360.txt')))

    def test_copy_true(self):
        cube = Cube(np.array([0, 1, 2, 3, 4]), long_name="foo")
        series = iris.pandas.as_series(cube)
        series[0] = 99
        self.assertEqual(cube.data[0], 0)

    def test_copy_int32_false(self):
        cube = Cube(np.array([0, 1, 2, 3, 4], dtype=np.int32), long_name="foo")
        series = iris.pandas.as_series(cube, copy=False)
        series[0] = 99
        self.assertEqual(cube.data[0], 99)

    def test_copy_int64_false(self):
        cube = Cube(np.array([0, 1, 2, 3, 4], dtype=np.int32), long_name="foo")
        series = iris.pandas.as_series(cube, copy=False)
        series[0] = 99
        self.assertEqual(cube.data[0], 99)

    def test_copy_float_false(self):
        cube = Cube(np.array([0, 1, 2, 3.3, 4]), long_name="foo")
        series = iris.pandas.as_series(cube, copy=False)
        series[0] = 99
        self.assertEqual(cube.data[0], 99)

    def test_copy_masked_true(self):
        data = np.ma.MaskedArray([0, 1, 2, 3, 4], mask=[0, 1, 0, 1, 0])
        cube = Cube(data, long_name="foo")
        series = iris.pandas.as_series(cube)
        series[0] = 99
        self.assertEqual(cube.data[0], 0)

    def test_copy_masked_false(self):
        data = np.ma.MaskedArray([0, 1, 2, 3, 4], mask=[0, 1, 0, 1, 0])
        cube = Cube(data, long_name="foo")
        with self.assertRaises(ValueError):
            series = iris.pandas.as_series(cube, copy=False)


@skip_pandas
class TestAsDataFrame(tests.IrisTest):
    """Test conversion of 2D cubes to Pandas using as_data_frame()"""

    def test_no_dim_coords(self):
        cube = Cube(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                    long_name="foo")
        data_frame = iris.pandas.as_data_frame(cube)
        self.assertArrayEqual(data_frame, cube.data)
        self.assertString(
            str(data_frame),
            tests.get_result_path(('pandas', 'as_dataframe',
                                   'no_dim_coords.txt')))

    def test_no_x_coord(self):
        cube = Cube(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                    long_name="foo")
        y_coord = DimCoord([10, 11], long_name="bar")
        cube.add_dim_coord(y_coord, 0)
        data_frame = iris.pandas.as_data_frame(cube)
        self.assertArrayEqual(data_frame, cube.data)
        self.assertString(
            str(data_frame),
            tests.get_result_path(('pandas', 'as_dataframe',
                                   'no_x_coord.txt')))

    def test_no_y_coord(self):
        cube = Cube(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                    long_name="foo")
        x_coord = DimCoord([10, 11, 12, 13, 14], long_name="bar")
        cube.add_dim_coord(x_coord, 1)
        data_frame = iris.pandas.as_data_frame(cube)
        self.assertArrayEqual(data_frame, cube.data)
        self.assertString(
            str(data_frame),
            tests.get_result_path(('pandas', 'as_dataframe',
                                   'no_y_coord.txt')))

    def test_simple(self):
        cube = Cube(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                    long_name="foo")
        x_coord = DimCoord([10, 11, 12, 13, 14], long_name="bar")
        y_coord = DimCoord([15, 16], long_name="milk")
        cube.add_dim_coord(x_coord, 1)
        cube.add_dim_coord(y_coord, 0)
        data_frame = iris.pandas.as_data_frame(cube)
        self.assertArrayEqual(data_frame, cube.data)
        self.assertString(
            str(data_frame),
            tests.get_result_path(('pandas', 'as_dataframe',
                                   'simple.txt')))

    def test_masked(self):
        data = np.ma.MaskedArray([[0, 1, 2, 3, 4.4], [5, 6, 7, 8, 9]],
                                 mask=[[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])
        cube = Cube(data, long_name="foo")
        data_frame = iris.pandas.as_data_frame(cube)
        self.assertArrayEqual(data_frame, cube.data.astype('f').filled(np.nan))
        self.assertString(
            str(data_frame),
            tests.get_result_path(('pandas', 'as_dataframe',
                                   'masked.txt')))

    def test_time_gregorian(self):
        cube = Cube(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                    long_name="ts")
        day_offsets = [0, 100.1, 200.2, 300.3, 400.4]
        time_coord = DimCoord(day_offsets, long_name="time",
                              units="days since 2000-01-01 00:00")
        cube.add_dim_coord(time_coord, 1)
        data_frame = iris.pandas.as_data_frame(cube)
        self.assertArrayEqual(data_frame, cube.data)
        nanoseconds_per_day = 24 * 60 * 60 * 1000000000
        days_to_2000 = 365 * 30 + 7
        # pandas Timestamp class cannot handle floats in pandas <v0.12
        timestamps = [pandas.Timestamp(int(nanoseconds_per_day *
                                       (days_to_2000 + day_offset)))
                      for day_offset in day_offsets]
        self.assertTrue(all(data_frame.columns == timestamps))
        self.assertTrue(all(data_frame.index == [0, 1]))

    def test_time_360(self):
        cube = Cube(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                    long_name="ts")
        time_unit = iris.unit.Unit("days since 2000-01-01 00:00",
                                   calendar=iris.unit.CALENDAR_360_DAY)
        time_coord = DimCoord([100.1, 200.2], long_name="time",
                              units=time_unit)
        cube.add_dim_coord(time_coord, 0)
        data_frame = iris.pandas.as_data_frame(cube)
        self.assertArrayEqual(data_frame, cube.data)
        self.assertString(
            str(data_frame),
            tests.get_result_path(('pandas', 'as_dataframe',
                                   'time_360.txt')))

    def test_copy_true(self):
        cube = Cube(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
                    long_name="foo")
        data_frame = iris.pandas.as_data_frame(cube)
        data_frame[0][0] = 99
        self.assertEqual(cube.data[0, 0], 0)

    def test_copy_int32_false(self):
        cube = Cube(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
                             dtype=np.int32), long_name="foo")
        data_frame = iris.pandas.as_data_frame(cube, copy=False)
        data_frame[0][0] = 99
        self.assertEqual(cube.data[0, 0], 99)

    def test_copy_int64_false(self):
        cube = Cube(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
                             dtype=np.int64), long_name="foo")
        data_frame = iris.pandas.as_data_frame(cube, copy=False)
        data_frame[0][0] = 99
        self.assertEqual(cube.data[0, 0], 99)

    def test_copy_float_false(self):
        cube = Cube(np.array([[0, 1, 2, 3, 4.4], [5, 6, 7, 8, 9]]),
                    long_name="foo")
        data_frame = iris.pandas.as_data_frame(cube, copy=False)
        data_frame[0][0] = 99
        self.assertEqual(cube.data[0, 0], 99)

    def test_copy_masked_true(self):
        data = np.ma.MaskedArray([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
                                 mask=[[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])
        cube = Cube(data, long_name="foo")
        data_frame = iris.pandas.as_data_frame(cube)
        data_frame[0][0] = 99
        self.assertEqual(cube.data[0, 0], 0)

    def test_copy_masked_false(self):
        data = np.ma.MaskedArray([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
                                 mask=[[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])
        cube = Cube(data, long_name="foo")
        with self.assertRaises(ValueError):
            data_frame = iris.pandas.as_data_frame(cube, copy=False)


@skip_pandas
class TestSeriesAsCube(tests.IrisTest):

    def test_series_simple(self):
        series = pandas.Series([0, 1, 2, 3, 4], index=[5, 6, 7, 8, 9])
        self.assertCML(
            iris.pandas.as_cube(series),
            tests.get_result_path(('pandas', 'as_cube', 'series_simple.cml')))

    def test_series_object(self):
        class Thing(object):
            def __repr__(self):
                return "A Thing"
        series = pandas.Series(
            [0, 1, 2, 3, 4],
            index=[Thing(), Thing(), Thing(), Thing(), Thing()])
        self.assertCML(
            iris.pandas.as_cube(series),
            tests.get_result_path(('pandas', 'as_cube', 'series_object.cml')))

    def test_series_masked(self):
        series = pandas.Series([0, float('nan'), 2, np.nan, 4],
                               index=[5, 6, 7, 8, 9])
        self.assertCML(
            iris.pandas.as_cube(series),
            tests.get_result_path(('pandas', 'as_cube', 'series_masked.cml')))

    def test_series_datetime_gregorian(self):
        series = pandas.Series(
            [0, 1, 2, 3, 4],
            index=[datetime.datetime(2001, 1, 1, 1, 1, 1),
                   datetime.datetime(2002, 2, 2, 2, 2, 2),
                   datetime.datetime(2003, 3, 3, 3, 3, 3),
                   datetime.datetime(2004, 4, 4, 4, 4, 4),
                   datetime.datetime(2005, 5, 5, 5, 5, 5)])
        self.assertCML(
            iris.pandas.as_cube(series),
            tests.get_result_path(('pandas', 'as_cube',
                                   'series_datetime_gregorian.cml')))

    def test_series_netcdftime_360(self):
        series = pandas.Series(
            [0, 1, 2, 3, 4],
            index=[netcdftime.datetime(2001, 1, 1, 1, 1, 1),
                   netcdftime.datetime(2002, 2, 2, 2, 2, 2),
                   netcdftime.datetime(2003, 3, 3, 3, 3, 3),
                   netcdftime.datetime(2004, 4, 4, 4, 4, 4),
                   netcdftime.datetime(2005, 5, 5, 5, 5, 5)])
        self.assertCML(
            iris.pandas.as_cube(series,
                                calendars={0: iris.unit.CALENDAR_360_DAY}),
            tests.get_result_path(('pandas', 'as_cube',
                                   'series_netcdfimte_360.cml')))

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
        data_frame = pandas.DataFrame([[0, 1, 2, 3, 4],
                                       [5, 6, 7, 8, 9]],
                                      index=[10, 11],
                                      columns=[12, 13, 14, 15, 16])
        self.assertCML(
            iris.pandas.as_cube(data_frame),
            tests.get_result_path(('pandas', 'as_cube',
                                   'data_frame_simple.cml')))

    def test_data_frame_nonotonic(self):
        data_frame = pandas.DataFrame([[0, 1, 2, 3, 4],
                                       [5, 6, 7, 8, 9]],
                                      index=[10, 10],
                                      columns=[12, 12, 14, 15, 16])
        self.assertCML(
            iris.pandas.as_cube(data_frame),
            tests.get_result_path(('pandas', 'as_cube',
                                   'data_frame_nonotonic.cml')))

    def test_data_frame_masked(self):
        data_frame = pandas.DataFrame([[0, float('nan'), 2, 3, 4],
                                       [5, 6, 7, np.nan, 9]],
                                      index=[10, 11],
                                      columns=[12, 13, 14, 15, 16])
        self.assertCML(
            iris.pandas.as_cube(data_frame),
            tests.get_result_path(('pandas', 'as_cube',
                                   'data_frame_masked.cml')))

    def test_data_frame_netcdftime_360(self):
        data_frame = pandas.DataFrame(
            [[0, 1, 2, 3, 4],
             [5, 6, 7, 8, 9]],
            index=[netcdftime.datetime(2001, 1, 1, 1, 1, 1),
                   netcdftime.datetime(2002, 2, 2, 2, 2, 2)],
            columns=[10, 11, 12, 13, 14])
        self.assertCML(
            iris.pandas.as_cube(
                data_frame,
                calendars={0: iris.unit.CALENDAR_360_DAY}),
            tests.get_result_path(('pandas', 'as_cube',
                                   'data_frame_netcdftime_360.cml')))

    def test_data_frame_datetime_gregorian(self):
        data_frame = pandas.DataFrame(
            [[0, 1, 2, 3, 4],
             [5, 6, 7, 8, 9]],
            index=[datetime.datetime(2001, 1, 1, 1, 1, 1),
                   datetime.datetime(2002, 2, 2, 2, 2, 2)],
            columns=[10, 11, 12, 13, 14])
        self.assertCML(
            iris.pandas.as_cube(data_frame),
            tests.get_result_path(('pandas', 'as_cube',
                                   'data_frame_datetime_gregorian.cml')))

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
