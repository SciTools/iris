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
"""Unit tests for :class:`iris.coords.AuxCoord`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.coords import AuxCoord
import iris


class Test_copy(tests.IrisTest):
    def test_share_data_default(self):
        original = AuxCoord(np.arange(4))
        copy = original.copy()
        original.points[1] = 999
        self.assertArrayEqual(copy.points, [0, 1, 2, 3])

    def test_share_data_false(self):
        original = AuxCoord(np.arange(4))
        with iris.FUTURE.context(share_data=False):
            copy = original.copy()
        original.points[1] = 999
        self.assertArrayEqual(copy.points, [0, 1, 2, 3])

    def test_share_data_true(self):
        original = AuxCoord(np.arange(4))
        with iris.FUTURE.context(share_data=True):
            copy = original.copy()
        original.points[1] = 999
        self.assertArrayEqual(copy.points, [0, 999, 2, 3])


if __name__ == '__main__':
    tests.main()
