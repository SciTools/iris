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
"""Unit tests for the `iris.fileformats.pp.as_pairs` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

from iris.coords import DimCoord
from iris.fileformats._ff_cross_references import STASH_TRANS
import iris.fileformats.pp as pp
import iris.tests.stock as stock


class TestAsFields(tests.IrisTest):
    def setUp(self):
        self.cube = stock.realistic_3d()

    def test_cube_only(self):
        slices_and_fields = pp.as_pairs(self.cube)
        for aslice, field in slices_and_fields:
            self.assertEqual(aslice.shape, (9, 11))
            self.assertEqual(field.lbcode, 101)

    def test_field_coords(self):
        slices_and_fields = pp.as_pairs(self.cube,
                                        field_coords=['grid_longitude',
                                                      'grid_latitude'])
        for aslice, field in slices_and_fields:
            self.assertEqual(aslice.shape, (11, 9))
            self.assertEqual(field.lbcode, 101)


if __name__ == "__main__":
    tests.main()
