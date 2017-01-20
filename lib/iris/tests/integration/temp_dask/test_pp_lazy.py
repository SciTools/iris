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
Test lazy data handling in :mod:`iris.fileformats.pp`.

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


class MixinLazyCubeLoad(object):
    def setUp(self):
        path = tests.get_data_path(('PP', 'aPPglob1', 'global.pp'))
        self.cube, = iris.load_raw(path)
        # This is the same as iris.tests.stock.global_pp(), but avoids the
        # merge, which is presently not working.


@tests.skip_data
class TestLazyCubeLoad(MixinLazyCubeLoad, tests.IrisTest):
    def setUp(self):
        path = tests.get_data_path(('PP', 'aPPglob1', 'global.pp'))
        self.cube, = iris.load_raw(path)
        # This is the same as iris.tests.stock.global_pp(), but avoids the
        # merge, which is presently not working.

    def test_load(self):
        # Check that a simple load results in a cube with a lazy data array.
        cube = self.cube
        raw_data = cube._my_data
        # It has loaded as a dask array.
        self.assertIsInstance(raw_data, DaskArray)

    def test_data(self):
        # Check that .data returns a realised array with the expected values.
        cube = self.cube
        raw_data = cube._my_data
        data = cube.data
        # "normal" .data is a numpy array.
        self.assertIsInstance(data, np.ndarray)
        # values match the lazy original.
        self.assertArrayAllClose(data, raw_data.compute())


@tests.skip_data
class Test_has_lazy_data(MixinLazyCubeLoad, tests.IrisTest):
    def test(self):
        # Check result before and after touching the data.
        cube = self.cube
        # normal load yields lazy data.
        self.assertTrue(cube.has_lazy_data())
        # touch data.
        cube.data
        # cube has real data after .data access.
        self.assertFalse(cube.has_lazy_data())


@tests.skip_data
class Test_lazy_data(MixinLazyCubeLoad, tests.IrisTest):
    def test__before_and_after_realise(self):
        # Check return values from cube.lazy_data().
        cube = self.cube
        raw_data = cube._my_data
        self.assertIsInstance(raw_data, DaskArray)
        # before touching .data, lazy_data() returns the original raw data.
        lazy_before = cube.lazy_data()
        self.assertIs(lazy_before, raw_data)
        # touch data.
        cube.data
        # after touching .data, lazy_data() is not the original raw data, but
        # it computes the same result.
        lazy_after = cube.lazy_data()
        self.assertIsInstance(lazy_after, DaskArray)
        self.assertIsNot(lazy_after, lazy_before)
        self.assertArrayAllClose(lazy_after.compute(),
                                 lazy_before.compute())

    def test__newdata(self):
        # Check cube.lazy_data(<new-data>).
        cube = self.cube
        raw_data = cube._my_data
        real_data = raw_data.compute()
        # set new lazy value.
        cube.lazy_data(raw_data + 100.0)
        # check that results are as expected.
        self.assertArrayAllClose(cube.lazy_data().compute(),
                                 real_data + 100.0)

    def test__newdata_fail_bad_shape(self):
        # Check cube.lazy_data(<new-data>) with bad shape.
        cube = self.cube
        raw_data = cube.lazy_data()
        msg = 'cube data with shape \(73, 96\), got \(72, 96\)'
        with self.assertRaisesRegexp(ValueError, msg):
            cube.lazy_data(raw_data[1:])

    def test__newdata_fail_not_lazy(self):
        # Check cube.lazy_data(<new-data>) with non-lazy argument.
        cube = self.cube
        raw_data = cube.lazy_data()
        with self.assertRaisesRegexp(TypeError, 'must be a lazy array'):
            cube.lazy_data(np.zeros(raw_data.shape))


if __name__ == '__main__':
    tests.main()
