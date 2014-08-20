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
"""Unit tests for um_cf mappings in
:func:`iris.fileformats.pp_rules._all_other_rules`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

from iris.fileformats.pp_rules import convert
from iris.fileformats.pp import STASH
import iris.tests.unit.fileformats


class Test_STASH_CF(iris.tests.unit.fileformats.TestField):
    def test_stash_cf_air_temp(self):
        lbuser = [1, 0, 0, 16203, 0, 0, 1]
        lbfc = 16
        stash = STASH(lbuser[6], lbuser[3] / 1000, lbuser[3] % 1000)
        field = mock.MagicMock(lbuser=lbuser, lbfc=lbfc, stash=stash)
        (factories, references, standard_name, long_name, units,
         attributes, cell_methods, dim_coords_and_dims,
         aux_coords_and_dims) = convert(field)
        self.assertEqual(standard_name, 'air_temperature')
        self.assertEqual(units, 'K')

    def test_no_std_name(self):
        lbuser = [1, 0, 0, 0, 0, 0, 0]
        lbfc = 0
        stash = STASH(lbuser[6], lbuser[3] / 1000, lbuser[3] % 1000)
        field = mock.MagicMock(lbuser=lbuser, lbfc=lbfc, stash=stash)
        (factories, references, standard_name, long_name, units,
         attributes, cell_methods, dim_coords_and_dims,
         aux_coords_and_dims) = convert(field)
        self.assertIsNone(standard_name)
        self.assertIsNone(units)


class Test_LBFC_CF(iris.tests.unit.fileformats.TestField):
    def test_fc_cf_air_temp(self):
        lbuser = [1, 0, 0, 0, 0, 0, 0]
        lbfc = 16
        stash = STASH(lbuser[6], lbuser[3] / 1000, lbuser[3] % 1000)
        field = mock.MagicMock(lbuser=lbuser, lbfc=lbfc, stash=stash)
        (factories, references, standard_name, long_name, units,
         attributes, cell_methods, dim_coords_and_dims,
         aux_coords_and_dims) = convert(field)
        self.assertEqual(standard_name, 'air_temperature')
        self.assertEqual(units, 'K')


if __name__ == "__main__":
    tests.main()
