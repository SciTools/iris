# (C) British Crown Copyright 2015, Met Office
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
"""Unit tests for :class:`iris.experimental.regrid.PointInCell`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

from iris.experimental.regrid import PointInCell


class Test_regridder(tests.IrisTest):
    def test(self):
        point_in_cell = PointInCell(mock.sentinel.weights)

        with mock.patch('iris.experimental.regrid._CurvilinearRegridder',
                        return_value=mock.sentinel.regridder) as ecr:
            regridder = point_in_cell.regridder(mock.sentinel.src,
                                                mock.sentinel.target)

        ecr.assert_called_once_with(mock.sentinel.src,
                                    mock.sentinel.target,
                                    mock.sentinel.weights)
        self.assertIs(regridder, mock.sentinel.regridder)


if __name__ == '__main__':
    tests.main()
