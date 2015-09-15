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
:func:`iris.fileformats.grib._load_convert.product_definition_template_8`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from iris.fileformats.grib._load_convert import product_definition_template_8
from iris.tests import mock


class Test(tests.IrisTest):
    def setUp(self):
        module = 'iris.fileformats.grib._load_convert'
        self.module = module
        # Create patches for called routines
        self.patch_generating_process = self.patch(
            module + '.generating_process')
        self.patch_data_cutoff = self.patch(module + '.data_cutoff')
        self.patch_statistical_cell_method = self.patch(
            module + '.statistical_cell_method',
            return_value=mock.sentinel.dummy_cell_method)
        self.patch_statistical_fp_coord = self.patch(
            module + '.statistical_forecast_period_coord',
            return_value=mock.sentinel.dummy_fp_coord)
        self.patch_time_coord = self.patch(
            module + '.validity_time_coord',
            return_value=mock.sentinel.dummy_time_coord)
        self.patch_vertical_coords = self.patch(module + '.vertical_coords')
        # Construct dummy call arguments
        self.section = {}
        self.section['hoursAfterDataCutoff'] = mock.sentinel.cutoff_hours
        self.section['minutesAfterDataCutoff'] = mock.sentinel.cutoff_mins
        self.frt_coord = mock.sentinel.frt_coord
        self.metadata = {'cell_methods': [], 'aux_coords_and_dims': []}

    def test_basic(self):
        product_definition_template_8(
            self.section, self.metadata, self.frt_coord)
        # Check all expected functions were called just once.
        self.assertEqual(self.patch_generating_process.call_count, 1)
        self.assertEqual(self.patch_data_cutoff.call_count, 1)
        self.assertEqual(self.patch_statistical_cell_method.call_count, 1)
        self.assertEqual(self.patch_statistical_fp_coord.call_count, 1)
        self.assertEqual(self.patch_time_coord.call_count, 1)
        self.assertEqual(self.patch_vertical_coords.call_count, 1)
        # Check metadata content.
        self.assertEqual(sorted(self.metadata.keys()),
                         ['aux_coords_and_dims', 'cell_methods'])
        self.assertEqual(self.metadata['cell_methods'],
                         [mock.sentinel.dummy_cell_method])
        six.assertCountEqual(self, self.metadata['aux_coords_and_dims'],
                             [(self.frt_coord, None),
                              (mock.sentinel.dummy_fp_coord, None),
                              (mock.sentinel.dummy_time_coord, None)])


if __name__ == '__main__':
    tests.main()
