# (C) British Crown Copyright 2016 - 2018, Met Office
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
"""Unit tests for :func:`iris.sample_data_path` class."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import os
import os.path
import shutil
import tempfile

from iris import sample_data_path
from iris.tests import mock


def _temp_file(sample_dir):
    # Return the full path to a new genuine file within our
    # temporary directory.
    sample_handle, sample_path = tempfile.mkstemp(dir=sample_dir)
    os.close(sample_handle)
    return sample_path


@tests.skip_sample_data
class TestIrisSampleData_path(tests.IrisTest):
    def setUp(self):
        self.sample_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.sample_dir)

    def test_path(self):
        with mock.patch('iris_sample_data.path', self.sample_dir):
            import iris_sample_data
            self.assertEqual(iris_sample_data.path, self.sample_dir)

    def test_call(self):
        sample_file = _temp_file(self.sample_dir)
        with mock.patch('iris_sample_data.path', self.sample_dir):
            result = sample_data_path(os.path.basename(sample_file))
            self.assertEqual(result, sample_file)

    def test_file_not_found(self):
        with mock.patch('iris_sample_data.path', self.sample_dir):
            with self.assertRaisesRegexp(ValueError,
                                         'Sample data .* not found'):
                sample_data_path('foo')

    def test_file_absolute(self):
        with mock.patch('iris_sample_data.path', self.sample_dir):
            with self.assertRaisesRegexp(ValueError, 'Absolute path'):
                sample_data_path(os.path.abspath('foo'))

    def test_glob_ok(self):
        sample_path = _temp_file(self.sample_dir)
        sample_glob = '?' + os.path.basename(sample_path)[1:]
        with mock.patch('iris_sample_data.path', self.sample_dir):
            result = sample_data_path(sample_glob)
            self.assertEqual(result, os.path.join(self.sample_dir,
                                                  sample_glob))

    def test_glob_not_found(self):
        with mock.patch('iris_sample_data.path', self.sample_dir):
            with self.assertRaisesRegexp(ValueError,
                                         'Sample data .* not found'):
                sample_data_path('foo.*')

    def test_glob_absolute(self):
        with mock.patch('iris_sample_data.path', self.sample_dir):
            with self.assertRaisesRegexp(ValueError, 'Absolute path'):
                sample_data_path(os.path.abspath('foo.*'))


class TestIrisSampleDataMissing(tests.IrisTest):
    def test_no_iris_sample_data(self):
        self.patch('iris.iris_sample_data', None)
        with self.assertRaisesRegexp(ImportError, 'Please install'):
            sample_data_path('')


if __name__ == '__main__':
    tests.main()
