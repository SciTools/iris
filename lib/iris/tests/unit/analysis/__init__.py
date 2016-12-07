# (C) British Crown Copyright 2013 - 2015, Met Office
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
"""Unit tests for the :mod:`iris.analysis` package."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.analysis import Linear
from iris.tests import mock


class Test_Linear(tests.IrisTest):
    def setUp(self):
        self.extrap = 'some extrapolation'

    def test___init__(self):
        linear = Linear(extrapolation_mode=self.extrap)
        self.assertEqual(getattr(linear, 'extrapolation_mode', None),
                         self.extrap)

    @mock.patch('iris.analysis.LinearInterpolator', name='LinearInterpolator')
    def test_interpolator(self, linear_interp_patch):
        mock_interpolator = mock.Mock(name='mocked linear interpolator')
        linear_interp_patch.return_value = mock_interpolator

        linear = Linear(self.extrap)
        cube = mock.Mock(name='cube')
        coords = mock.Mock(name='coords')

        interpolator = linear.interpolator(cube, coords)

        self.assertIs(interpolator, mock_interpolator)
        linear_interp_patch.assert_called_once_with(
            cube, coords, extrapolation_mode=self.extrap)


if __name__ == "__main__":
    tests.main()
