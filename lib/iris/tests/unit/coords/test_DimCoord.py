# (C) British Crown Copyright 2013, Met Office
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
"""Unit tests for the `iris.coords.DimCoord` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.coords import DimCoord


class Test_from_regular(tests.IrisTest):
    def test_awkward_values(self):
        # Check that a coord with specific "awkward" point values is regular.
        sample_coord = DimCoord.from_regular(zeroth=355.626,
                                             step=0.0135,
                                             count=3)
        steps = np.diff(sample_coord.points)
        steps_mean = np.mean(steps)
        steps_max = np.max(steps)
        steps_min = np.min(steps)
        steps_range_relative = 0.5 * (steps_max - steps_min) / steps_mean
        self.assertLess(steps_range_relative, 1.0e-10)


if __name__ == '__main__':
    tests.main()
