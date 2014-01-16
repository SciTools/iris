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
"""Unit tests for the `iris.fileformats.pp.save` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

from iris.fileformats._ff_cross_references import STASH_TRANS
import iris.fileformats.pp as pp
import iris.tests.stock as stock


def _pp_save_ppfield_values(cube):
    """
    Emulate saving a cube as PP, and capture the resulting PP field values.

    """
    # Create a test object to stand in for a real PPField.
    pp_field = mock.MagicMock(spec=pp.PPField3)
    # Add minimal content required by the pp.save operation.
    pp_field.HEADER_DEFN = pp.PPField3.HEADER_DEFN
    # Save cube to a dummy file, mocking the internally created PPField
    with mock.patch('iris.fileformats.pp.PPField3',
                    return_value=pp_field):
        target_filelike = mock.Mock(name='target')
        target_filelike.mode = ('b')
        pp.save(cube, target_filelike)
    # Return pp-field mock with all the written properties
    return pp_field


class TestLbfcProduction(tests.IrisTest):
    def setUp(self):
        self.cube = stock.lat_lon_cube()

    def check_cube_stash_yields_lbfc(self, stash, lbfc_expected):
        if stash:
            self.cube.attributes['STASH'] = stash
        lbfc_produced = _pp_save_ppfield_values(self.cube).lbfc
        self.assertEqual(lbfc_produced, lbfc_expected)

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


if __name__ == "__main__":
    tests.main()
