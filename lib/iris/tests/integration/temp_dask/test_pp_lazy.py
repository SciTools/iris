# (C) British Crown Copyright 2017, Met Office
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
Test lazy data handlingin iris.fileformats.pp.

Note: probably belongs in "tests/unit/fileformats/pp", if a separate test is
actually required.

"""
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from dask.array.core import Array as DaskArray
import numpy as np

import iris


@tests.skip_data
class TestLazyLoad(tests.IrisTest):
    def setUp(self):
        path = tests.get_data_path(('PP', 'aPPglob1', 'global.pp'))
        self.cube, = iris.load_raw(path)
        # This is the same as iris.tests.stock.global_pp(), but avoids the
        # merge, which is presently not working.

    def test_load(self):
        # Check that a simple load gets us lazy data.
        cube = self.cube
        raw_data = cube._my_data
        self.assertIsInstance(raw_data, DaskArray)

    def test_data(self):
        # Check that data access realises.
        cube = self.cube
        raw_data = cube._my_data
        data = cube.data
        self.assertIsInstance(data, np.ndarray)
        self.assertArrayAllClose(data, raw_data.compute())

    def test_has_lazy(self):
        # Check cube.has_lazy_data().
        cube = self.cube
        self.assertTrue(cube.has_lazy_data())
        cube.data
        self.assertFalse(cube.has_lazy_data())

    def test_lazy_data(self):
        # Check cube.lazy_data().
        cube = self.cube
        raw_data = cube._my_data
        lazy = cube.lazy_data()
        self.assertIs(cube.lazy_data(), raw_data)
        cube.data
        lazy = cube.lazy_data()
        self.assertIsNot(cube.lazy_data(), raw_data)
        self.assertArrayAllClose(lazy.compute(), raw_data.compute())

    def test_lazy_data__set(self):
        # Check cube.lazy_data(<new-data>).
        cube = self.cube
        raw_data = cube._my_data
        cube.lazy_data(raw_data + 100.0)
        real_data = raw_data.compute()
        self.assertArrayAllClose(cube.lazy_data(),
                                 real_data + 100.0)

    def test_lazy_data__fail_set_bad_shape(self):
        # Check cube.lazy_data(<new-data>).
        cube = self.cube
        raw_data = cube.lazy_data()
        msg = 'cube data with shape \(73, 96\), got \(72, 96\)'
        with self.assertRaisesRegexp(ValueError, msg):
            cube.lazy_data(raw_data[1:])

    def test_lazy_data__fail_set_not_lazy(self):
        # Check cube.lazy_data(<new-data>).
        cube = self.cube
        raw_data = cube.lazy_data()
        with self.assertRaisesRegexp(TypeError, 'must be a lazy array'):
            cube.lazy_data(np.zeros(raw_data.shape))


if __name__ == '__main__':
    tests.main()
