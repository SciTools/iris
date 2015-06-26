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
"""Test function :func:`iris.fileformats.grib._load_convert.grib2_convert`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import copy

import iris
from iris.fileformats.grib._load_convert import grib2_convert
from iris.tests import mock
from iris.tests.unit.fileformats.grib import _make_test_message


class Test(tests.IrisTest):
    def setUp(self):
        this = 'iris.fileformats.grib._load_convert'
        self.patch('{}.reference_time_coord'.format(this), return_value=None)
        self.patch('{}.grid_definition_section'.format(this))
        self.patch('{}.product_definition_section'.format(this))
        self.patch('{}.data_representation_section'.format(this))
        self.patch('{}.bitmap_section'.format(this))

    def test(self):
        sections = [{'discipline': mock.sentinel.discipline},       # section 0
                    {'centre': 'ecmf',                              # section 1
                     'tablesVersion': mock.sentinel.tablesVersion},
                    None,                                           # section 2
                    mock.sentinel.grid_definition_section,          # section 3
                    mock.sentinel.product_definition_section,       # section 4
                    mock.sentinel.data_representation_section,      # section 5
                    mock.sentinel.bitmap_section]                   # section 6
        field = _make_test_message(sections)
        metadata = {'factories': [], 'references': [],
                    'standard_name': None,
                    'long_name': None, 'units': None, 'attributes': {},
                    'cell_methods': [], 'dim_coords_and_dims': [],
                    'aux_coords_and_dims': []}
        expected = copy.deepcopy(metadata)
        centre = 'European Centre for Medium Range Weather Forecasts'
        expected['attributes'] = {'centre': centre}
        # The call being tested.
        grib2_convert(field, metadata)
        self.assertEqual(metadata, expected)
        this = iris.fileformats.grib._load_convert
        this.reference_time_coord.assert_called_with(sections[1])
        this.grid_definition_section.assert_called_with(sections[3],
                                                        expected)
        args = (sections[4], expected, sections[0]['discipline'],
                sections[1]['tablesVersion'], None)
        this.product_definition_section.assert_called_with(*args)


if __name__ == '__main__':
    tests.main()
