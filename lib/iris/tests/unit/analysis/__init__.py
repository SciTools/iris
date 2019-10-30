# Copyright Iris Contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full licensing details.
"""Unit tests for the :mod:`iris.analysis` package."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from unittest import mock

from iris.analysis import Linear


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
