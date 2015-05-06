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
"""
Unit tests for
:func:`iris.fileformats.pp_rules._convert_pseudo_level_coords`.

"""

from __future__ import (absolute_import, division, print_function)

import six

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.coords import DimCoord
from iris.tests.unit.fileformats import TestField

from iris.fileformats.pp_rules import _convert_scalar_pseudo_level_coords


class Test(TestField):
    def test_valid(self):
        coords_and_dims = _convert_scalar_pseudo_level_coords(lbuser5=21)
        self.assertEqual(coords_and_dims,
                         [(DimCoord([21], long_name='pseudo_level'), None)])

    def test_missing_indicator(self):
        coords_and_dims = _convert_scalar_pseudo_level_coords(lbuser5=0)
        self.assertEqual(coords_and_dims, [])


if __name__ == "__main__":
    tests.main()
