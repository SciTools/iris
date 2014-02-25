# (C) British Crown Copyright 2014, Met Office
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
"""Integration tests for PP/GRIB interoperability."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import iris


class TestBoundedTime(tests.IrisTest):
    @iris.tests.skip_data
    def test_time_and_forecast_period_round_trip(self):
        pp_path = tests.get_data_path(('PP', 'meanMaxMin',
                                       '200806081200__qwpb.T24.pp'))
        # Choose the first time-bounded Cube in the PP dataset.
        original = [cube for cube in iris.load(pp_path) if
                    cube.coord('time').has_bounds()][0]
        # Save it to GRIB2 and re-load.
        with self.temp_filename('.grib2') as grib_path:
            iris.save(original, grib_path)
            from_grib = iris.load_cube(grib_path)
            # Avoid the downcasting warning when saving to PP.
            from_grib.data = from_grib.data.astype('f4')
        # Re-save to PP and re-load.
        with self.temp_filename('.pp') as pp_path:
            iris.save(from_grib, pp_path)
            from_pp = iris.load_cube(pp_path)
        self.assertEqual(original.coord('time'), from_grib.coord('time'))
        self.assertEqual(original.coord('forecast_period'),
                         from_grib.coord('forecast_period'))
        self.assertEqual(original.coord('time'), from_pp.coord('time'))
        self.assertEqual(original.coord('forecast_period'),
                         from_pp.coord('forecast_period'))


if __name__ == "__main__":
    tests.main()
