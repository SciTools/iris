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
Unit tests for the `iris.fileformats.grib._GribMessage` class.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import numpy as np

from iris.exceptions import TranslationError
from iris.fileformats.grib._message import _GribMessage


class TestSections(tests.IrisTest):
    def test(self):
        # Check that the `sections` attribute defers to the `sections`
        # attribute on the underlying _RawGribMessage.
        message = _GribMessage(mock.Mock(sections=mock.sentinel.SECTIONS))
        self.assertIs(message.sections, mock.sentinel.SECTIONS)


def _message(sections):
    return _GribMessage(mock.Mock(sections=sections))


class TestData(tests.IrisTest):
    def test_unsupported_grid_definition(self):
        message = _message({3: {'sourceOfGridDefinition': 1}})
        with self.assertRaisesRegexp(TranslationError, 'source'):
            message.data

    def test_unsupported_quasi_regular__number_of_octets(self):
        message = _message({3: {'sourceOfGridDefinition': 0,
                                'numberOfOctectsForNumberOfPoints': 1}})
        with self.assertRaisesRegexp(TranslationError, 'quasi-regular'):
            message.data

    def test_unsupported_quasi_regular__interpretation(self):
        message = _message({3: {'sourceOfGridDefinition': 0,
                                'numberOfOctectsForNumberOfPoints': 0,
                                'interpretationOfNumberOfPoints': 1}})
        with self.assertRaisesRegexp(TranslationError, 'quasi-regular'):
            message.data

    def test_unsupported_template(self):
        message = _message({3: {'sourceOfGridDefinition': 0,
                                'numberOfOctectsForNumberOfPoints': 0,
                                'interpretationOfNumberOfPoints': 0,
                                'gridDefinitionTemplateNumber': 1}})
        with self.assertRaisesRegexp(TranslationError, 'template'):
            message.data

    def test_regular_data(self):
        message = _message({3: {'sourceOfGridDefinition': 0,
                                'numberOfOctectsForNumberOfPoints': 0,
                                'interpretationOfNumberOfPoints': 0,
                                'gridDefinitionTemplateNumber': 0,
                                'Nj': 3,
                                'Ni': 4},
                            7: {'codedValues': np.arange(12)}})
        self.assertArrayEqual(message.data, np.arange(12).reshape(3, 4))


if __name__ == '__main__':
    tests.main()
