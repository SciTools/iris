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
"""
Unit tests for :meth:`iris.analysis.trajectory.interpolate`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import iris.tests.stock

from iris.analysis.trajectory import interpolate


@tests.skip_data
class TestFailCases(tests.IrisTest):
    def test_derived_coord(self):
        cube = iris.tests.stock.realistic_4d()
        sample_pts = [('altitude', [0, 10, 50])]
        msg = "'altitude'.*derived coordinates are not allowed"
        with self.assertRaisesRegexp(ValueError, msg):
            interpolate(cube, sample_pts, 'nearest')


if __name__ == "__main__":
    tests.main()
