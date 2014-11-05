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
"""
Unit tests for the `iris.fileformats.grib.GribWrapper` class.

"""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from biggus import NumpyArrayAdapter
import mock
import numpy as np

from iris.fileformats.grib import GribWrapper, GribDataProxy

_message_length = 1000


def _mock_grib_get_long(grib_message, key):
    lookup = dict(totalLength=_message_length,
                  numberOfValues=200,
                  jPointsAreConsecutive=0,
                  Ni=20,
                  Nj=10)
    try:
        result = lookup[key]
    except KeyError:
        msg = 'Mock grib_get_long unknown key: {!r}'.format(key)
        raise AttributeError(msg)
    return result


def _mock_grib_get_string(grib_message, key):
    return grib_message


def _mock_grib_get_native_type(grib_message, key):
    result = int
    if key == 'gridType':
        result = str
    return result


class Test_deferred(tests.IrisTest):
    def setUp(self):
        confirm_patch = mock.patch(
            'iris.fileformats.grib.GribWrapper._confirm_in_scope')
        compute_patch = mock.patch(
            'iris.fileformats.grib.GribWrapper._compute_extra_keys')
        long_patch = mock.patch('gribapi.grib_get_long', _mock_grib_get_long)
        string_patch = mock.patch('gribapi.grib_get_string',
                                  _mock_grib_get_string)
        native_patch = mock.patch('gribapi.grib_get_native_type',
                                  _mock_grib_get_native_type)
        confirm_patch.start()
        compute_patch.start()
        long_patch.start()
        string_patch.start()
        native_patch.start()
        self.addCleanup(confirm_patch.stop)
        self.addCleanup(compute_patch.stop)
        self.addCleanup(long_patch.stop)
        self.addCleanup(string_patch.stop)
        self.addCleanup(native_patch.stop)

    def test_regular_sequential(self):
        tell_tale = np.arange(1, 5) * _message_length
        grib_fh = mock.Mock(tell=mock.Mock(side_effect=tell_tale))
        auto_regularise = False
        grib_message = 'regular_ll'
        for i, _ in enumerate(tell_tale):
            gw = GribWrapper(grib_message, grib_fh, auto_regularise)
            self.assertIsInstance(gw.lazy_data(), NumpyArrayAdapter)
            proxy = gw.lazy_data().concrete
            self.assertIsInstance(proxy, GribDataProxy)
            self.assertEqual(proxy.shape, (10, 20))
            self.assertEqual(proxy.dtype, np.float)
            self.assertIs(proxy.fill_value, np.nan)
            self.assertEqual(proxy.path, grib_fh.name)
            self.assertEqual(proxy.offset, _message_length * i)
            self.assertEqual(proxy.regularise, auto_regularise)

    def test_regular_mixed(self):
        tell_tale = np.arange(1, 5) * _message_length
        expected = tell_tale - _message_length
        grib_fh = mock.Mock(tell=mock.Mock(side_effect=tell_tale))
        auto_regularise = False
        grib_message = 'regular_ll'
        for offset in expected:
            gw = GribWrapper(grib_message, grib_fh, auto_regularise)
            self.assertIsInstance(gw.lazy_data(), NumpyArrayAdapter)
            proxy = gw.lazy_data().concrete
            self.assertIsInstance(proxy, GribDataProxy)
            self.assertEqual(proxy.shape, (10, 20))
            self.assertEqual(proxy.dtype, np.float)
            self.assertIs(proxy.fill_value, np.nan)
            self.assertEqual(proxy.path, grib_fh.name)
            self.assertEqual(proxy.offset, offset)
            self.assertEqual(proxy.regularise, auto_regularise)

    def test_reduced_sequential(self):
        tell_tale = np.arange(1, 5) * _message_length
        grib_fh = mock.Mock(tell=mock.Mock(side_effect=tell_tale))
        auto_regularise = False
        grib_message = 'reduced_gg'
        for i, _ in enumerate(tell_tale):
            gw = GribWrapper(grib_message, grib_fh, auto_regularise)
            self.assertIsInstance(gw.lazy_data(), NumpyArrayAdapter)
            proxy = gw.lazy_data().concrete
            self.assertIsInstance(proxy, GribDataProxy)
            self.assertEqual(proxy.shape, (200,))
            self.assertEqual(proxy.dtype, np.float)
            self.assertIs(proxy.fill_value, np.nan)
            self.assertEqual(proxy.path, grib_fh.name)
            self.assertEqual(proxy.offset, _message_length * i)
            self.assertEqual(proxy.regularise, auto_regularise)

    def test_reduced_mixed(self):
        tell_tale = np.arange(1, 5) * _message_length
        expected = tell_tale - _message_length
        grib_fh = mock.Mock(tell=mock.Mock(side_effect=tell_tale))
        auto_regularise = False
        grib_message = 'reduced_gg'
        for offset in expected:
            gw = GribWrapper(grib_message, grib_fh, auto_regularise)
            self.assertIsInstance(gw.lazy_data(), NumpyArrayAdapter)
            proxy = gw.lazy_data().concrete
            self.assertIsInstance(proxy, GribDataProxy)
            self.assertEqual(proxy.shape, (200,))
            self.assertEqual(proxy.dtype, np.float)
            self.assertIs(proxy.fill_value, np.nan)
            self.assertEqual(proxy.path, grib_fh.name)
            self.assertEqual(proxy.offset, offset)
            self.assertEqual(proxy.regularise, auto_regularise)


if __name__ == '__main__':
    tests.main()
