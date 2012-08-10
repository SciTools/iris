# (C) British Crown Copyright 2010 - 2012, Met Office
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


# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import os

import iris
import iris.analysis
import iris.fileformats.pp as pp
from iris.tests.test_pp_module import IrisPPTest
import iris.util


@iris.tests.skip_data
class TestTrui(IrisPPTest):
    def test_mean_save(self):
        files = ['200812011200', '200812021200', '200812031200', '200812041200',
                 '200812051200', '200812061200', '200812071200', '200812081200']
        files = [tests.get_data_path(('PP', 'trui', 'air_temp_T24', f + '__qwqg12ff.T24.pp')) for f in files]

        air_temp_cube = iris.load_strict(files)
        self.assertCML(air_temp_cube, ['trui', 'air_temp_T24_subset.cml'])

        mean = air_temp_cube.collapsed("time", iris.analysis.MEAN)
        self.assertCML(mean, ['trui', 'air_temp_T24_subset_mean.cml'])
        
        temp_filename = iris.util.create_temp_filename(".pp")
        iris.io.save(mean, temp_filename)
        
        r = list(pp.load(temp_filename))
        self.check_pp(r[0:1], ('trui', 'air_temp_T24_subset_mean.pp.txt'))

        os.remove(temp_filename)

    def test_meanTrial_diff(self):
        air_temp_T00_cube = iris.load_strict(tests.get_data_path(('PP', 'trui', 'air_temp_init', '*.pp')))
        self.assertCML(air_temp_T00_cube, ['trui', 'air_temp_T00.cml'])
        
        air_temp_T24_cube = iris.load_strict(tests.get_data_path(('PP', 'trui', 'air_temp_T24', '*.T24.pp')))
        
        self.assertCML(air_temp_T24_cube, ['trui', 'air_temp_T24.cml'])

        air_temp_T00_cube, air_temp_T24_cube = iris.analysis.maths.intersection_of_cubes(air_temp_T00_cube, air_temp_T24_cube)
        
        delta_cube = air_temp_T00_cube - air_temp_T24_cube
        self.assertCML(delta_cube, ('trui', 'air_temp_trial_diff_T00_to_T24.cml'))
        
        mean_delta_cube = delta_cube.collapsed("time", iris.analysis.MEAN)
        self.assertCML(mean_delta_cube, ('trui', 'mean_air_temp_trial_diff_T00_to_T24.cml'))
        
if __name__ == "__main__":
    tests.main()
