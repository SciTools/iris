# (C) British Crown Copyright 2016, Met Office
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

import mock

from iris import sample_data_path
from iris._deprecation import IrisDeprecation


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


class TestConfig(tests.IrisTest):
    def setUp(self):
        # Force iris_sample_data to be unavailable.
        self.patch('iris.iris_sample_data', None)
        # All of our tests are going to run with SAMPLE_DATA_DIR
        # redirected to a temporary directory.
        self.sample_dir = tempfile.mkdtemp()
        patcher = mock.patch('iris.config.SAMPLE_DATA_DIR', self.sample_dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        shutil.rmtree(self.sample_dir)

    def test_file_ok(self):
        sample_path = _temp_file(self.sample_dir)
        result = sample_data_path(os.path.basename(sample_path))
        self.assertEqual(result, sample_path)

    def test_file_not_found(self):
        with self.assertRaisesRegexp(ValueError, 'Sample data .* not found'):
            sample_data_path('foo')

    def test_file_absolute(self):
        with self.assertRaisesRegexp(ValueError, 'Absolute path'):
            sample_data_path(os.path.abspath('foo'))

    def test_glob_ok(self):
        sample_path = _temp_file(self.sample_dir)
        sample_glob = '?' + os.path.basename(sample_path)[1:]
        result = sample_data_path(sample_glob)
        self.assertEqual(result, os.path.join(self.sample_dir, sample_glob))

    def test_glob_not_found(self):
        with self.assertRaisesRegexp(ValueError, 'Sample data .* not found'):
            sample_data_path('foo.*')

    def test_glob_absolute(self):
        with self.assertRaisesRegexp(ValueError, 'Absolute path'):
            sample_data_path(os.path.abspath('foo.*'))

    def test_warn_deprecated(self):
        sample_path = _temp_file(self.sample_dir)
        with mock.patch('warnings.warn') as warn:
            sample_data_path(os.path.basename(sample_path))
            self.assertEqual(warn.call_count, 1)
            (warn_msg, warn_exception), _ = warn.call_args
            msg = 'iris.config.SAMPLE_DATA_DIR was deprecated'
            self.assertTrue(warn_msg.startswith(msg))
            self.assertEqual(warn_exception, IrisDeprecation)


if __name__ == '__main__':
    tests.main()
