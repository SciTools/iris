# (C) British Crown Copyright 2014 - 2015, Met Office
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
"""Unit tests for :class:`iris.analysis.Linear`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

from iris.analysis import Linear


def create_scheme(mode=None):
    kwargs = {}
    if mode is not None:
        kwargs['extrapolation_mode'] = mode
    return Linear(**kwargs)


class Test_extrapolation_mode(tests.IrisTest):
    def check_mode(self, mode):
        linear = create_scheme(mode)
        self.assertEqual(linear.extrapolation_mode, mode)

    def test_default(self):
        linear = Linear()
        self.assertEqual(linear.extrapolation_mode, 'linear')

    def test_extrapolate(self):
        self.check_mode('extrapolate')

    def test_linear(self):
        self.check_mode('linear')

    def test_nan(self):
        self.check_mode('nan')

    def test_error(self):
        self.check_mode('error')

    def test_mask(self):
        self.check_mode('mask')

    def test_nanmask(self):
        self.check_mode('nanmask')

    def test_invalid(self):
        with self.assertRaisesRegexp(ValueError, 'Extrapolation mode'):
            Linear('bogus')


class Test_interpolator(tests.IrisTest):
    def check_mode(self, mode=None):
        linear = create_scheme(mode)

        # Check that calling `linear.interpolator(...)` returns an
        # instance of RectilinearInterpolator which has been created
        # using the correct arguments.
        with mock.patch('iris.analysis.RectilinearInterpolator',
                        return_value=mock.sentinel.interpolator) as ri:
            interpolator = linear.interpolator(mock.sentinel.cube,
                                               mock.sentinel.coords)
        if mode is None or mode == 'linear':
            expected_mode = 'extrapolate'
        else:
            expected_mode = mode
        ri.assert_called_once_with(mock.sentinel.cube, mock.sentinel.coords,
                                   'linear', expected_mode)
        self.assertIs(interpolator, mock.sentinel.interpolator)

    def test_default(self):
        self.check_mode()

    def test_extrapolate(self):
        self.check_mode('extrapolate')

    def test_linear(self):
        self.check_mode('linear')

    def test_nan(self):
        self.check_mode('nan')

    def test_error(self):
        self.check_mode('error')

    def test_mask(self):
        self.check_mode('mask')

    def test_nanmask(self):
        self.check_mode('nanmask')


class Test_regridder(tests.IrisTest):
    def check_mode(self, mode=None):
        linear = create_scheme(mode)

        # Check that calling `linear.regridder(...)` returns an instance
        # of RectilinearRegridder which has been created using the correct
        # arguments.
        with mock.patch('iris.analysis.RectilinearRegridder',
                        return_value=mock.sentinel.regridder) as lr:
            regridder = linear.regridder(mock.sentinel.src,
                                         mock.sentinel.target)
        if mode is None or mode == 'linear':
            expected_mode = 'extrapolate'
        else:
            expected_mode = mode
        lr.assert_called_once_with(mock.sentinel.src, mock.sentinel.target,
                                   'linear', expected_mode)
        self.assertIs(regridder, mock.sentinel.regridder)

    def test_default(self):
        self.check_mode()

    def test_extrapolate(self):
        self.check_mode('extrapolate')

    def test_linear(self):
        self.check_mode('linear')

    def test_nan(self):
        self.check_mode('nan')

    def test_error(self):
        self.check_mode('error')

    def test_mask(self):
        self.check_mode('mask')

    def test_nanmask(self):
        self.check_mode('nanmask')


if __name__ == '__main__':
    tests.main()
