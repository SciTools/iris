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
"""Unit tests for the `iris.fileformats.pp.as_fields` function."""

from __future__ import (absolute_import, division, print_function)

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
        self.cube = stock.lat_lon_cube()

    def test_pseudo_level(self):
        pseudo_level = 123
        coord = DimCoord(pseudo_level, long_name='pseudo_level', units='1')
        self.cube.add_aux_coord(coord)
        fields = pp.as_fields(self.cube)
        for field in fields:
            self.assertEqual(pseudo_level, field.lbuser[4])


class TestLbfcProduction(tests.IrisTest):
    def setUp(self):
        self.cube = stock.lat_lon_cube()

    def check_cube_stash_yields_lbfc(self, stash, lbfc_expected):
        if stash:
            self.cube.attributes['STASH'] = stash
        fields = pp.as_fields(self.cube)
        for field in fields:
            self.assertEqual(field.lbfc, lbfc_expected)

    def test_known_stash(self):
        stashcode_str = 'm04s07i002'
        self.assertIn(stashcode_str, STASH_TRANS)
        self.check_cube_stash_yields_lbfc(stashcode_str, 359)

    def test_unknown_stash(self):
        stashcode_str = 'm99s99i999'
        self.assertNotIn(stashcode_str, STASH_TRANS)
        self.check_cube_stash_yields_lbfc(stashcode_str, 0)

    def test_no_stash(self):
        self.assertNotIn('STASH', self.cube.attributes)
        self.check_cube_stash_yields_lbfc(None, 0)

    def check_cube_name_units_yields_lbfc(self, name, units, lbfc_expected):
        self.cube.rename(name)
        self.cube.units = units
        fields = pp.as_fields(self.cube)
        for field in fields:
            self.assertEqual(field.lbfc, lbfc_expected,
                             'Lbfc for ({!r} / {!r}) should be {:d}, '
                             'got {:d}'.format(
                                 name, units, lbfc_expected, field.lbfc))

    def test_name_units_to_lbfc(self):
        # Check LBFC value produced from name and units.
        self.check_cube_name_units_yields_lbfc(
            'sea_ice_temperature', 'K', 209)

    def test_bad_name_units_to_lbfc_0(self):
        # Check that badly-formed / unrecognised cases yield LBFC == 0.
        self.check_cube_name_units_yields_lbfc('sea_ice_temperature', 'degC',
                                               0)
        self.check_cube_name_units_yields_lbfc('Junk_Name', 'K',
                                               0)


class TestLbsrceProduction(tests.IrisTest):
    def setUp(self):
        self.cube = stock.lat_lon_cube()

    def check_cube_um_source_yields_lbsrce(
            self, source_str=None, um_version_str=None, lbsrce_expected=None):
        if source_str is not None:
            self.cube.attributes['source'] = source_str
        if um_version_str is not None:
            self.cube.attributes['um_version'] = um_version_str
        fields = pp.as_fields(self.cube)
        for field in fields:
            self.assertEqual(field.lbsrce, lbsrce_expected)

    def test_none(self):
        self.check_cube_um_source_yields_lbsrce(
            None, None, 0)

    def test_source_only_no_version(self):
        self.check_cube_um_source_yields_lbsrce(
            'Data from Met Office Unified Model', None, 1111)

    def test_source_only_with_version(self):
        self.check_cube_um_source_yields_lbsrce(
            'Data from Met Office Unified Model 12.17', None, 12171111)

    def test_um_version(self):
        self.check_cube_um_source_yields_lbsrce(
            'Data from Met Office Unified Model 12.17', '25.36', 25361111)


if __name__ == "__main__":
    tests.main()
