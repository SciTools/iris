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
"""Unit tests for the :class:`iris.coord_systems.GeogCS` class."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.coord_systems import GeogCS


class Test___eq__(tests.IrisTest):
    def test_derived_quantities_eq(self):
        # Ensure that derived quantities are equal.
        crs1 = GeogCS(semi_major_axis=6377563.396, semi_minor_axis=6356256.909)
        crs2 = GeogCS(semi_major_axis=6377563.396,
                      inverse_flattening=299.324)
        self.assertEqual(crs1, crs2)

    def test_32bit_derived_quantities_eq(self):
        # Ensure that 32-bit coerced quantities don't fail equality.
        crs1 = GeogCS(semi_major_axis=6377563.396, semi_minor_axis=6356256.909)
        crs2 = GeogCS(semi_major_axis=np.float32(6377563.396),
                      inverse_flattening=np.float32(299.324))
        self.assertEqual(crs1, crs2)

    def test_derived_quantities_neq(self):
        # Ensure that derived quantities are not equal.
        crs1 = GeogCS(semi_major_axis=6377563.396, semi_minor_axis=6356256.909)
        crs2 = GeogCS(semi_major_axis=6377563.396,
                      inverse_flattening=299.33)
        self.assertFalse(crs1 == crs2)

    def test_straight_eq(self):
        # No derived quantities.
        crs1 = GeogCS(6377560)
        crs2 = GeogCS(6377570)
        self.assertEqual(crs1, crs2)

    def test_straight_neq(self):
        # No derived quantities.
        crs1 = GeogCS(6377500)
        crs2 = GeogCS(6377600)
        self.assertFalse(crs1 == crs2)

    def test_straight_32bit_eq(self):
        # No derived quantities.
        crs1 = GeogCS(6377560)
        crs2 = GeogCS(np.float32(6377570))
        self.assertEqual(crs1, crs2)

    def test_straight_32bit_neq(self):
        # No derived quantities.
        crs1 = GeogCS(6377500)
        crs2 = GeogCS(np.float32(6377600))
        self.assertFalse(crs1 == crs2)


if __name__ == '__main__':
    tests.main()
