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
"""
Test function
:func:`iris.fileformats.grib._load_convert.grid_definition_template_0_and_1`.

"""

from __future__ import (absolute_import, division, print_function)

import six

# Import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from iris.exceptions import TranslationError
from iris.fileformats.grib._load_convert import \
    grid_definition_template_0_and_1


class Test(tests.IrisTest):

    def test_unsupported_quasi_regular__number_of_octets(self):
        section = {'numberOfOctectsForNumberOfPoints': 1}
        cs = None
        metadata = None
        with self.assertRaisesRegexp(TranslationError, 'quasi-regular'):
            grid_definition_template_0_and_1(section,
                                             metadata,
                                             'latitude',
                                             'longitude',
                                             cs)

    def test_unsupported_quasi_regular__interpretation(self):
        section = {'numberOfOctectsForNumberOfPoints': 1,
                   'interpretationOfNumberOfPoints': 1}
        cs = None
        metadata = None
        with self.assertRaisesRegexp(TranslationError, 'quasi-regular'):
            grid_definition_template_0_and_1(section,
                                             metadata,
                                             'latitude',
                                             'longitude',
                                             cs)


if __name__ == '__main__':
    tests.main()
