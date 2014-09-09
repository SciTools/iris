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
"""
Unit tests for the `iris.fileformats.grib.GribMessage` class.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

import gribapi

from iris.fileformats.grib._grib_message import GribMessage


class Test(tests.IrisTest):
    def setUp(self):
        self.filename = tests.get_data_path(('GRIB', 'uk_t', 'uk_t.grib2'))
        with open(self.filename, 'rb') as grib_fh:
            grib_id = gribapi.grib_new_from_file(grib_fh)
            self.message = GribMessage(grib_id)

    def test_sections__set(self):
        # Test that sections writes into the _sections attribute.
        res = self.message.sections
        self.assertNotEqual(self.message._sections, None)

    def test_sections__indexing(self):
        res = self.message.sections[3]['scanningMode']
        expected = 64
        self.assertEqual(expected, res)

    def test__get_message_sections__section_numbers(self):
        res = self.message.sections.keys()
        self.assertEqual(res, range(9))

    def test_sections__numberOfSection_value(self):
        # The key `numberOfSection` is repeated in every section meaning that
        # if requested using gribapi it always defaults to its last value (7).
        # This tests that the `GribMessage._get_message_sections` override is
        # functioning.
        section_number = 4
        res = self.message.sections[section_number]['numberOfSection']
        self.assertEqual(res, section_number)


if __name__ == '__main__':
    tests.main()
