# (C) British Crown Copyright 2014, Met Office
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
"""Unit tests for :func:`iris.fileformats.pp_rules._dim_or_aux`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.coords import DimCoord, AuxCoord
from iris.fileformats.pp_rules import _dim_or_aux


class Test(tests.IrisTest):
    def setUp(self):
        self.mono = range(5)
        self.non_mono = [0, 1, 3, 2, 4]
        self.std_name = 'depth'
        self.units = 'm'
        self.attr = {'positive': 'up',
                     'wibble': 'wobble'}

    def test_dim_monotonic(self):
        result = _dim_or_aux(self.mono, standard_name=self.std_name,
                             units=self.units, attributes=self.attr.copy())
        expected = DimCoord(self.mono, standard_name=self.std_name,
                            units=self.units, attributes=self.attr)
        self.assertEqual(result, expected)

    def test_dim_non_monotonic(self):
        result = _dim_or_aux(self.non_mono, standard_name=self.std_name,
                             units=self.units, attributes=self.attr.copy())
        attr = self.attr.copy()
        del attr['positive']
        expected = AuxCoord(self.non_mono, standard_name=self.std_name,
                            units=self.units, attributes=attr)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    tests.main()
