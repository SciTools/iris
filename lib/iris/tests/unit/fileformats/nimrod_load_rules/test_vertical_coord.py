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
Unit tests for the `iris.fileformats.nimrod_load_rules.vertical_coord`
function.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.fileformats.nimrod_load_rules import (vertical_coord,
                                                NIMROD_DEFAULT,
                                                TranslationWarning)
from iris.fileformats.nimrod import NimrodField
from iris.tests import mock


class Test(tests.IrisTest):
    NIMROD_LOCATION = 'iris.fileformats.nimrod_load_rules'

    def setUp(self):
        self.field = mock.Mock(vertical_coord_type=NIMROD_DEFAULT,
                               int_mdi=mock.sentinel.int_mdi,
                               field_code=mock.sentinel.field_code,
                               spec=NimrodField)
        self.cube = mock.Mock()

    def _call_vertical_coord(self, vertical_coord_type):
        self.field.vertical_coord_type = vertical_coord_type
        vertical_coord(self.cube, self.field)

    def test_unhandled(self):
        with mock.patch('warnings.warn') as warn:
            self._call_vertical_coord(-1)
        warn.assert_called_once_with("Vertical coord -1 not yet handled",
                                     TranslationWarning)

    def test_orography(self):
        name = 'orography_vertical_coord'
        with mock.patch(self.NIMROD_LOCATION + '.' + name) as orog:
            self.field.field_code = 73
            self._call_vertical_coord(None)
        orog.assert_called_once_with(self.cube, self.field)

    def test_height(self):
        name = 'height_vertical_coord'
        with mock.patch(self.NIMROD_LOCATION + '.' + name) as height:
            self._call_vertical_coord(0)
        height.assert_called_once_with(self.cube, self.field)

    def test_null(self):
        with mock.patch('warnings.warn') as warn:
            self._call_vertical_coord(NIMROD_DEFAULT)
            self._call_vertical_coord(self.field.int_mdi)
        self.assertEqual(warn.call_count, 0)


if __name__ == "__main__":
    tests.main()
