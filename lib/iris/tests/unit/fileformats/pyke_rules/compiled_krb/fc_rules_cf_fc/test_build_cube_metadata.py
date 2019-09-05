# (C) British Crown Copyright 2014 - 2019, Met Office
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
Test function :func:`iris.fileformats._pyke_rules.compiled_krb.\
fc_rules_cf_fc.build_cube_metadata`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np

from iris.cube import Cube
from iris.fileformats._pyke_rules.compiled_krb.fc_rules_cf_fc import \
    build_cube_metadata
from iris.tests import mock


def _make_engine(global_attributes=None, standard_name=None, long_name=None):
    if global_attributes is None:
        global_attributes = {}

    cf_group = mock.Mock(global_attributes=global_attributes)

    cf_var = mock.Mock(
        cf_name='wibble',
        standard_name=standard_name,
        long_name=long_name,
        units='m',
        dtype=np.float64,
        cell_methods=None,
        cf_group=cf_group)

    engine = mock.Mock(
        cube=Cube([23]),
        cf_var=cf_var)

    return engine


class TestInvalidGlobalAttributes(tests.IrisTest):
    def test_valid(self):
        global_attributes = {'Conventions': 'CF-1.5',
                             'comment': 'Mocked test object'}
        engine = _make_engine(global_attributes)
        build_cube_metadata(engine)
        expected = global_attributes
        self.assertEqual(engine.cube.attributes, expected)

    def test_invalid(self):
        global_attributes = {'Conventions': 'CF-1.5',
                             'comment': 'Mocked test object',
                             'calendar': 'standard'}
        engine = _make_engine(global_attributes)
        with mock.patch('warnings.warn') as warn:
            build_cube_metadata(engine)
        # Check for a warning.
        self.assertEqual(warn.call_count, 1)
        self.assertIn("Skipping global attribute 'calendar'",
                      warn.call_args[0][0])
        # Check resulting attributes. The invalid entry 'calendar'
        # should be filtered out.
        global_attributes.pop('calendar')
        expected = global_attributes
        self.assertEqual(engine.cube.attributes, expected)


class TestCubeName(tests.IrisTest):
    def check_cube_names(self, inputs, expected):
        # Inputs - attributes on the fake CF Variable.
        standard_name, long_name = inputs
        # Expected - The expected cube attributes.
        exp_standard_name, exp_long_name = expected

        engine = _make_engine(standard_name=standard_name, long_name=long_name)
        build_cube_metadata(engine)

        # Check the cube's standard name and long name are as expected.
        self.assertEqual(engine.cube.standard_name, exp_standard_name)
        self.assertEqual(engine.cube.long_name, exp_long_name)

    def test_standard_name_none_long_name_none(self):
        inputs = (None, None)
        expected = (None, None)
        self.check_cube_names(inputs, expected)

    def test_standard_name_none_long_name_set(self):
        inputs = (None, 'ice_thickness_long_name')
        expected = (None, 'ice_thickness_long_name')
        self.check_cube_names(inputs, expected)

    def test_standard_name_valid_long_name_none(self):
        inputs = ('sea_ice_thickness', None)
        expected = ('sea_ice_thickness', None)
        self.check_cube_names(inputs, expected)

    def test_standard_name_valid_long_name_set(self):
        inputs = ('sea_ice_thickness', 'ice_thickness_long_name')
        expected = ('sea_ice_thickness', 'ice_thickness_long_name')
        self.check_cube_names(inputs, expected)

    def test_standard_name_invalid_long_name_none(self):
        inputs = ('not_a_standard_name', None)
        expected = (None, 'not_a_standard_name',)
        self.check_cube_names(inputs, expected)

    def test_standard_name_invalid_long_name_set(self):
        inputs = ('not_a_standard_name', 'ice_thickness_long_name')
        expected = (None, 'ice_thickness_long_name')
        self.check_cube_names(inputs, expected)


if __name__ == "__main__":
    tests.main()
