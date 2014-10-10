# (C) British Crown Copyright 2013 - 2014, Met Office
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
import numpy as np

from iris.fileformats.grib.grib_save_rules import _non_missing_forecast_period
import iris.unit


class Test(tests.IrisTest):
    def _cube(self, t_bounds=False):
        time_coord = iris.coords.DimCoord(15, standard_name='time',
                                          units='hours since epoch')
        fp_coord = iris.coords.DimCoord(10, standard_name='forecast_period',
                                        units='hours')
        if t_bounds:
            time_coord.bounds = [[8, 100]]
            fp_coord.bounds = [[3, 95]]
        cube = iris.cube.Cube([23])
        cube.add_dim_coord(time_coord, 0)
        cube.add_aux_coord(fp_coord, 0)
        return cube

    def test_time_point(self):
        cube = self._cube()
        rt, rt_meaning, fp, fp_meaning = _non_missing_forecast_period(cube)
        self.assertEqual((rt_meaning, fp, fp_meaning), (1, 10, 1))

    def test_time_bounds(self):
        cube = self._cube(t_bounds=True)
        rt, rt_meaning, fp, fp_meaning = _non_missing_forecast_period(cube)
        self.assertEqual((rt_meaning, fp, fp_meaning), (1, 3, 1))

    def test_time_bounds_in_minutes(self):
        cube = self._cube(t_bounds=True)
        cube.coord('forecast_period').convert_units('minutes')
        rt, rt_meaning, fp, fp_meaning = _non_missing_forecast_period(cube)
        self.assertEqual((rt_meaning, fp, fp_meaning), (1, 180, 0))


if __name__ == "__main__":
    tests.main()
