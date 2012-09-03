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
"""
Test the file loading mechanism.

"""

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests
import iris


@iris.tests.skip_data
class TestFileload_strict(tests.IrisTest):
    def _test_file(self, src_path, reference_filename):
        """
        Checks the result of loading the given file spec, or creates the
        reference file if it doesn't exist.

        NB. The direct use of :func:`iris._load_common` bypasses the cube merge process.
        
        """
        cubes = iris._load_common(tests.get_data_path(src_path), constraints=None, strict=False, unique=True, merge=False)
        self.assertCML(cubes, ['file_load', reference_filename])

    def test_no_file(self):
        # Test an IOError is recieved when a filename is given which doesnt match any files
        real_file = ['PP', 'globClim1', 'theta.pp']
        non_existant_file = ['PP', 'globClim1', 'no_such_file*']

        self.assertRaises(IOError, iris.load, tests.get_data_path(non_existant_file))
        self.assertRaises(IOError, iris.load, [tests.get_data_path(non_existant_file), tests.get_data_path(real_file)])
        self.assertRaises(IOError, iris.load, [tests.get_data_path(real_file), tests.get_data_path(non_existant_file)])

    def test_single_file(self):
        src_path = ['PP', 'globClim1', 'theta.pp']
        self._test_file(src_path, 'theta_levels.cml')

    def test_star_wildcard(self):
        src_path = ['PP', 'globClim1', '*_wind.pp']
        self._test_file(src_path, 'wind_levels.cml')

    def test_query_wildcard(self):
        src_path = ['PP', 'globClim1', '?_wind.pp']
        self._test_file(src_path, 'wind_levels.cml')

    def test_charset_wildcard(self):
        src_path = ['PP', 'globClim1', '[rstu]_wind.pp']
        self._test_file(src_path, 'u_wind_levels.cml')

    def test_negative_charset_wildcard(self):
        src_path = ['PP', 'globClim1', '[!rstu]_wind.pp']
        self._test_file(src_path, 'v_wind_levels.cml')


if __name__ == "__main__":
    tests.main()
