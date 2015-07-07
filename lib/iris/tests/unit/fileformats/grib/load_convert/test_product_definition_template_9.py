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
Tests for function
:func:`iris.fileformats.grib._load_convert.product_definition_template_9`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import mock

from iris.exceptions import TranslationError

from iris.fileformats.grib._load_convert import product_definition_template_9
from iris.fileformats.grib._load_convert import Probability, _MDI


class Test(tests.IrisTest):
    def setUp(self):
        # Create patches for called routines
        module = 'iris.fileformats.grib._load_convert'
        self.patch_pdt8_call = self.patch(
            module + '.product_definition_template_8')
        # Construct dummy call arguments
        self.section = {}
        self.section['probabilityType'] = 1
        self.section['scaledValueOfUpperLimit'] = 53
        self.section['scaleFactorOfUpperLimit'] = 1
        self.frt_coord = mock.sentinel.frt_coord
        self.metadata = {'cell_methods': [mock.sentinel.cell_method],
                         'aux_coords_and_dims': []}

    def test_basic(self):
        result = product_definition_template_9(
            self.section, self.metadata, self.frt_coord)
        # Check expected function was called.
        self.assertEqual(
            self.patch_pdt8_call.call_args_list,
            [mock.call(self.section, self.metadata, self.frt_coord)])
        # Check metadata content (N.B. cell_method has been removed!).
        self.assertEqual(self.metadata, {'cell_methods': [],
                                         'aux_coords_and_dims': []})
        # Check result.
        self.assertEqual(result, Probability('above_threshold', 5.3))

    def test_fail_bad_probability_type(self):
        self.section['probabilityType'] = 17
        with self.assertRaisesRegexp(TranslationError,
                                     'unsupported probability type'):
            product_definition_template_9(
                self.section, self.metadata, self.frt_coord)

    def test_fail_bad_threshold_value(self):
        self.section['scaledValueOfUpperLimit'] = _MDI
        with self.assertRaisesRegexp(TranslationError,
                                     'missing scaled value'):
            product_definition_template_9(
                self.section, self.metadata, self.frt_coord)

    def test_fail_bad_threshold_scalefactor(self):
        self.section['scaleFactorOfUpperLimit'] = _MDI
        with self.assertRaisesRegexp(TranslationError,
                                     'missing scale factor'):
            product_definition_template_9(
                self.section, self.metadata, self.frt_coord)


if __name__ == '__main__':
    tests.main()
