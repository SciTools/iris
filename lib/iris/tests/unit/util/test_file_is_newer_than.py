# (C) British Crown Copyright 2010 - 2013, Met Office
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
Test function :meth:`iris.util.test_file_is_newer`.

"""
# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import os
import os.path
import shutil
import tempfile
import unittest

from iris.util import file_is_newer_than


class TestFileIsNewer(tests.IrisTest):
    """Test the :meth:`iris.util.file_is_newer_than` function."""

    def _name2path(self, filename):
        """Add the temporary dirpath to a filename to make a full path."""
        return os.path.join(self.temp_dir, filename)

    def setUp(self):
        # make a temporary directory with testfiles of known timestamp order.
        self.temp_dir = tempfile.mkdtemp('_testfiles_tempdir')
        # define the names of some files to create
        create_file_names = ['older_source_1', 'older_source_2',
                             'example_result',
                             'newer_source_1', 'newer_source_2']
        # create testfiles + ensure distinct 'mtime's in the required order.
        mtime_offset = 5.0
        mtime_offset_increment = 10.0
        for file_name in create_file_names:
            file_path = self._name2path(file_name)
            with open(file_path, 'w') as test_file:
                test_file.write('..content..')
            # Ensure 'mtime's are adequately separated
            mtime = os.stat(self._name2path(create_file_names[0])).st_mtime
            mtime += mtime_offset
            mtime_offset += mtime_offset_increment
            os.utime(file_path, (mtime, mtime))

    def tearDown(self):
        # destroy whole contents of temporary directory
        shutil.rmtree(self.temp_dir)

    def _test(self, boolean_result, result_name, source_names):
        """Test expected result of executing with given args."""
        # Make args into full paths
        result_path = self._name2path(result_name)
        if isinstance(source_names, basestring):
            source_paths = self._name2path(source_names)
        else:
            source_paths = [self._name2path(name)
                            for name in source_names]
        # Check result is as expected.
        self.assertEqual(
            boolean_result,
            file_is_newer_than(result_path, source_paths))

    def test_no_sources(self):
        self._test(True, 'example_result', [])

    def test_string_ok(self):
        self._test(True, 'example_result', 'older_source_1')

    def test_string_fail(self):
        self._test(False, 'example_result', 'newer_source_1')

    def test_self_result(self):
        # This fails, because same-timestamp is *not* acceptable.
        self._test(False, 'example_result', 'example_result')

    def test_single_ok(self):
        self._test(True, 'example_result', ['older_source_2'])

    def test_single_fail(self):
        self._test(False, 'example_result', ['newer_source_2'])

    def test_multiple_ok(self):
        self._test(True, 'example_result', ['older_source_1',
                                            'older_source_2'])

    def test_multiple_fail(self):
        self._test(False, 'example_result', ['older_source_1',
                                             'older_source_2',
                                             'newer_source_1'])

    def test_wild_ok(self):
        self._test(True, 'example_result', ['older_sour*_*'])

    def test_wild_fail(self):
        self._test(False, 'example_result', ['older_sour*', 'newer_sour*'])

    def test_error_missing_result(self):
        with self.assertRaises(OSError) as error_trap:
            self._test(False, 'non_exist', ['older_sour*'])
        error = error_trap.exception
        self.assertEqual(error.strerror, 'No such file or directory')
        self.assertEqual(error.filename, self._name2path('non_exist'))

    def test_error_missing_source(self):
        with self.assertRaises(IOError) as error_trap:
            self._test(False, 'example_result', ['older_sour*', 'non_exist'])
        self.assertTrue(error_trap.exception.message.startswith(
            'One or more of the files specified did not exist'))

    def test_error_missing_wild(self):
        with self.assertRaises(IOError) as error_trap:
            self._test(False, 'example_result', ['older_sour*', 'unknown_*'])
        self.assertTrue(error_trap.exception.message.startswith(
            'One or more of the files specified did not exist'))


if __name__ == '__main__':
    unittest.main()
