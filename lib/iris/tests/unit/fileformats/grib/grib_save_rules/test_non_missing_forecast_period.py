# (C) British Crown Copyright 2013, Met Office
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
"""Unit tests for module-level functions."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import gribapi
import mock
import numpy as np

from iris.fileformats.grib.grib_save_rules import _non_missing_forecast_period
from iris.tests.test_grib_load import TestGribSimple
import iris.unit


class Test(TestGribSimple):
    def _t_coord(self, bounds=False):
        t_coord = mock.Mock()
        t_coord.points = [15]
        t_coord.units = iris.unit.Unit('hours')
        if bounds:
            t_coord.has_bounds = mock.Mock(return_value=True)
            t_coord.bounds = np.array([[10, 20]])
        else:
            t_coord.has_bounds = mock.Mock(return_value=False)
        return t_coord

    def _mock_cube(self, t_bounds=False):
        coords = {'time': self._t_coord(t_bounds),
                  'forecast_period': mock.Mock()}
        coords['time'].copy = mock.Mock(return_value=self._t_coord(t_bounds))
        coords['forecast_period'].points = [10]
        coords['forecast_period'].units = iris.unit.Unit('hours')
        coords['forecast_period'].has_bounds = mock.Mock(return_value=False)

        cube = mock.Mock()
        cube.coord = coords.get
        return cube

    def test_time_point(self):
        cube = self._mock_cube()
        rt, rt_meaning, fp, fp_meaning = _non_missing_forecast_period(cube)
        self.assertEqual((rt_meaning, fp, fp_meaning), (1, 10, 1))

    def test_time_bounds(self):
        cube = self._mock_cube(t_bounds=True)
        rt, rt_meaning, fp, fp_meaning = _non_missing_forecast_period(cube)
        self.assertEqual((rt_meaning, fp, fp_meaning), (1, 5, 1))


if __name__ == "__main__":
    tests.main()
