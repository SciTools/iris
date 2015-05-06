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
"""Unit tests for :class:`iris.analysis.AreaWeighted`."""

from __future__ import (absolute_import, division, print_function)

import six

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

from iris.analysis import AreaWeighted


class Test(tests.IrisTest):
    def check_call(self, mdtol=None):
        # Check that `iris.analysis.AreaWeighted` correctly calls an
        # `iris.analysis._area_weighted.AreaWeightedRegridder` object.
        if mdtol is None:
            area_weighted = AreaWeighted()
            mdtol = 1
        else:
            area_weighted = AreaWeighted(mdtol=mdtol)
        self.assertEqual(area_weighted.mdtol, mdtol)

        with mock.patch('iris.analysis.AreaWeightedRegridder',
                        return_value=mock.sentinel.regridder) as awr:
            regridder = area_weighted.regridder(mock.sentinel.src,
                                                mock.sentinel.target)

        awr.assert_called_once_with(mock.sentinel.src,
                                    mock.sentinel.target,
                                    mdtol=mdtol)
        self.assertIs(regridder, mock.sentinel.regridder)

    def test_default(self):
        self.check_call()

    def test_specified_mdtol(self):
        self.check_call(0.5)

    def test_invalid_high_mdtol(self):
        msg = 'mdtol must be in range 0 - 1'
        with self.assertRaisesRegexp(ValueError, msg):
            AreaWeighted(mdtol=1.2)

    def test_invalid_low_mdtol(self):
        msg = 'mdtol must be in range 0 - 1'
        with self.assertRaisesRegexp(ValueError, msg):
            AreaWeighted(mdtol=-0.2)


if __name__ == '__main__':
    tests.main()
