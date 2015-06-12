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
Test function :func:`iris.fileformats._pyke_rules.compiled_krb.\
fc_rules_cf_fc.build_cube_metadata`.

"""

from __future__ import (absolute_import, division, print_function)

import six

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np
import mock

from iris.cube import Cube
from iris.fileformats._pyke_rules.compiled_krb.fc_rules_cf_fc import \
    build_cube_metadata


class TestInvalidGlobalAttributes(tests.IrisTest):
    @staticmethod
    def _make_engine(global_attributes=None):
        if global_attributes is None:
            global_attributes = {}

        cf_group = mock.Mock(global_attributes=global_attributes)

        cf_var = mock.Mock(
            cf_name='wibble',
            standard_name=None,
            long_name=None,
            units='m',
            dtype=np.float64,
            cell_methods=None,
            cf_group=cf_group)

        engine = mock.Mock(
            cube=Cube([23]),
            cf_var=cf_var)

        return engine

    def test_valid(self):
        global_attributes = {'Conventions': 'CF-1.5',
                             'comment': 'Mocked test object'}
        engine = self._make_engine(global_attributes)
        build_cube_metadata(engine)
        expected = global_attributes
        self.assertEqual(engine.cube.attributes, expected)

    def test_invalid(self):
        global_attributes = {'Conventions': 'CF-1.5',
                             'comment': 'Mocked test object',
                             'calendar': 'standard'}
        engine = self._make_engine(global_attributes)
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


if __name__ == "__main__":
    tests.main()
