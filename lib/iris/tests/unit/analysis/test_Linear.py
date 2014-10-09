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
"""Unit tests for :class:`iris.analysis.Linear`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

from iris.analysis import Linear


class TestModes(tests.IrisTest):
    def check_mode(self, mode=None):
        kwargs = {}
        if mode is not None:
            kwargs['extrapolation_mode'] = mode
        linear = Linear(**kwargs)
        if mode is None:
            mode = 'linear'
        self.assertEqual(linear.extrapolation_mode, mode)

        # To avoid duplicating tests, just check that creating an
        # interpolator defers to the LinearInterpolator with the
        # correct arguments (and honouring the return value!)
        with mock.patch('iris.analysis.RegularInterpolator',
                        return_value=mock.sentinel.interpolator) as ri:
            interpolator = linear.interpolator(mock.sentinel.cube,
                                               mock.sentinel.coords)
        expected_mode = mode
        if mode == 'linear':
            expected_mode = 'extrapolate'
        ri.assert_called_once_with(mock.sentinel.cube, mock.sentinel.coords,
                                   'linear', expected_mode)
        self.assertIs(interpolator, mock.sentinel.interpolator)

        # As above, check method defers to LinearRegridder.
        with mock.patch('iris.analysis.LinearRegridder',
                        return_value=mock.sentinel.regridder) as lr:
            regridder = linear.regridder(mock.sentinel.src,
                                         mock.sentinel.target)
        lr.assert_called_once_with(mock.sentinel.src, mock.sentinel.target,
                                   expected_mode)
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

    def test_invalid(self):
        with self.assertRaisesRegexp(ValueError, 'Extrapolation mode'):
            Linear('bogus')


if __name__ == '__main__':
    tests.main()
