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
Unit tests for the `iris.fileformats.grib.message._GribMessage` class.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import biggus
import mock
import numpy as np

from iris.exceptions import TranslationError
from iris.fileformats.grib._message import _GribMessage
from iris.tests.unit.fileformats.grib import _make_test_message


@tests.skip_data
class Test_messages_from_filename(tests.IrisTest):
    def test(self):
        filename = tests.get_data_path(('GRIB', '3_layer_viz',
                                        '3_layer.grib2'))
        messages = list(_GribMessage.messages_from_filename(filename))
        self.assertEqual(len(messages), 3)


class Test_sections(tests.IrisTest):
    def test(self):
        # Check that the `sections` attribute defers to the `sections`
        # attribute on the underlying _RawGribMessage.
        message = _make_test_message(mock.sentinel.SECTIONS)
        self.assertIs(message.sections, mock.sentinel.SECTIONS)


class Test_data__unsupported(tests.IrisTest):
    def test_unsupported_grid_definition(self):
        message = _make_test_message({3: {'sourceOfGridDefinition': 1}})
        with self.assertRaisesRegexp(TranslationError, 'source'):
            message.data

    def test_unsupported_quasi_regular__number_of_octets(self):
        message = _make_test_message(
            {3: {'sourceOfGridDefinition': 0,
                 'numberOfOctectsForNumberOfPoints': 1}})
        with self.assertRaisesRegexp(TranslationError, 'quasi-regular'):
            message.data

    def test_unsupported_quasi_regular__interpretation(self):
        message = _make_test_message(
            {3: {'sourceOfGridDefinition': 0,
                 'numberOfOctectsForNumberOfPoints': 0,
                 'interpretationOfNumberOfPoints': 1}})
        with self.assertRaisesRegexp(TranslationError, 'quasi-regular'):
            message.data

    def test_unsupported_template(self):
        message = _make_test_message(
            {3: {'sourceOfGridDefinition': 0,
                 'numberOfOctectsForNumberOfPoints': 0,
                 'interpretationOfNumberOfPoints': 0,
                 'gridDefinitionTemplateNumber': 1}})
        with self.assertRaisesRegexp(TranslationError, 'template'):
            message.data


class Test_data__grid_template_0(tests.IrisTest):
    def test_unsupported_scanning_mode(self):
        message = _make_test_message(
            {3: {'sourceOfGridDefinition': 0,
                 'numberOfOctectsForNumberOfPoints': 0,
                 'interpretationOfNumberOfPoints': 0,
                 'gridDefinitionTemplateNumber': 0,
                 'scanningMode': 1}})
        with self.assertRaisesRegexp(TranslationError, 'scanning mode'):
            message.data

    def _test(self, scanning_mode):
        def make_raw_message():
            sections = {3: {'sourceOfGridDefinition': 0,
                            'numberOfOctectsForNumberOfPoints': 0,
                            'interpretationOfNumberOfPoints': 0,
                            'gridDefinitionTemplateNumber': 0,
                            'scanningMode': scanning_mode,
                            'Nj': 3,
                            'Ni': 4},
                        7: {'codedValues': np.arange(12)}}
            raw_message = mock.Mock(sections=sections)
            return raw_message
        message = _GribMessage(make_raw_message(), make_raw_message, False)
        data = message.data
        self.assertIsInstance(data, biggus.Array)
        self.assertEqual(data.shape, (3, 4))
        self.assertEqual(data.dtype, np.floating)
        self.assertIs(data.fill_value, np.nan)
        self.assertArrayEqual(data.ndarray(), np.arange(12).reshape(3, 4))

    def test_regular_mode_0(self):
        self._test(0)

    def test_regular_mode_64(self):
        self._test(64)

    def test_regular_mode_128(self):
        self._test(128)

    def test_regular_mode_64_128(self):
        self._test(64 | 128)


if __name__ == '__main__':
    tests.main()
