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

from iris.fileformats.grib.grib_message import GribMessage


class Test(tests.IrisTest):
    def setUp(self):
        self.sample_keys = ['totalLength', 'section1Length', 'scanningMode',
                            'section4Length', 'numberOfSection',
                            'section7Length', 'codedValues', '7777']
        self.sample_values = [mock.sentinel for i in
                              range(len(self.sample_keys))]
        method = 'iris.fileformats.grib.grib_message.GribMessage.{}'
        keys_patch = mock.patch(method.format('_get_message_keys'),
                                return_value=self.sample_keys)
        values_patch = mock.patch(method.format('_get_key_value'),
                                  side_effect=self.sample_values)
        keys_patch.start()
        values_patch.start()
        self.message = GribMessage(1)
        self.addCleanup(keys_patch.stop)
        self.addCleanup(values_patch.stop)

    def test_sections__set(self):
        # Test that sections writes into the _sections attribute.
        res = self.message.sections
        self.assertNotEqual(self.message._sections, None)

    def test_sections__indexing(self):
        key = 'scanningMode'
        #from nose.tools import set_trace; set_trace()
        res = self.message.sections[1][key]
        index = self.sample_keys.index(key)
        expected = self.sample_values[index]
        self.assertEqual(expected, res)

    def test__get_message_sections__section_numbers(self):
        res = self.message.sections.keys()
        self.assertEqual(res, [0, 1, 4, 7, 8])

    def test_get_containing_section(self):
        res = self.message.get_containing_section('scanningMode')
        self.assertEqual(res, 1)

    def test_get_containing_section_bad_key(self):
        with self.assertRaisesRegexp(KeyError, 'was not found in message'):
            self.message.get_containing_section('foo')

    def test_get_containing_section__numberOfSection_value(self):
        # The key `numberOfSection` is repeated in every section meaning that
        # if requested using gribapi it always defaults to its last value (7).
        # This tests that the `GribMessage._get_message_sections` override is
        # functioning.
        section_number = 4
        res = self.message.sections[section_number]['numberOfSection']
        self.assertEqual(res, section_number)

if __name__ == '__main__':
    tests.main()
