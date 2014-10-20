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

import mock
import numpy as np

from iris.fileformats.grib.grib_save_rules import _missing_forecast_period
from iris.tests.test_grib_load import TestGribSimple


class Test(TestGribSimple):
    def test_point(self):
        t_coord = mock.Mock()
        t_coord.has_bounds = mock.Mock(return_value=False)
        t_coord.points = [15]

        cube = mock.Mock()
        cube.coord = mock.Mock(return_value=t_coord)
        rt, rt_meaning, fp, fp_meaning = _missing_forecast_period(cube)

        t_coord.units.assert_has_call(mock.call.num2date(15))
        self.assertEqual((rt_meaning, fp, fp_meaning), (2, 0, 1))

    def test_bounds(self):
        t_coord = mock.Mock()
        t_coord.has_bounds = mock.Mock(return_value=True)
        t_coord.points = [15]
        t_coord.bounds = np.array([[10, 20]])

        cube = mock.Mock()
        cube.coord = mock.Mock(return_value=t_coord)
        rt, rt_meaning, fp, fp_meaning = _missing_forecast_period(cube)

        t_coord.units.assert_has_call(mock.call.num2date(10))
        self.assertEqual((rt_meaning, fp, fp_meaning), (2, 0, 1))


if __name__ == "__main__":
    tests.main()
