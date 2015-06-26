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
Test function :func:`iris.fileformats._pyke_rules.compiled_krb.\
fc_rules_cf_fc._parse_cell_methods`.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

from iris.coords import CellMethod
from iris.fileformats._pyke_rules.compiled_krb.fc_rules_cf_fc import \
    _parse_cell_methods
from iris.tests import mock


class Test(tests.IrisTest):
    def test_simple(self):
        cell_method_str = 'time: mean'
        expected = (CellMethod(method='mean', coords='time'),)
        res = _parse_cell_methods('test_var', cell_method_str)
        self.assertEqual(res, expected)

    def test_with_interval(self):
        cell_method_str = 'time: variance (interval: 1 hr)'
        expected = (CellMethod(method='variance', coords='time',
                               intervals='1 hr'),)
        res = _parse_cell_methods('test_var', cell_method_str)
        self.assertEqual(res, expected)

    def test_multiple(self):
        cell_method_str = 'time: maximum (interval: 1 hr) ' \
                          'time: mean (interval: 1 day)'
        expected = (CellMethod(method='maximum', coords='time',
                               intervals='1 hr'),
                    CellMethod(method='mean', coords='time',
                               intervals='1 day'))
        res = _parse_cell_methods('test_var', cell_method_str)
        self.assertEqual(res, expected)

    def test_comment(self):
        cell_method_str = 'time: maximum (interval: 1 hr comment: first bit) ' \
                          'time: mean (interval: 1 day comment: second bit)'
        expected = (CellMethod(method='maximum', coords='time',
                               intervals='1 hr', comments='first bit'),
                    CellMethod(method='mean', coords='time',
                               intervals='1 day', comments='second bit'))
        res = _parse_cell_methods('test_var', cell_method_str)
        self.assertEqual(res, expected)

    def test_portions_of_cells(self):
        cell_method_str = 'area: mean where sea_ice over sea'
        expected = (CellMethod(method='mean where sea_ice over sea',
                               coords='area'),)
        res = _parse_cell_methods('test_var', cell_method_str)
        self.assertEqual(res, expected)

    def test_climatology(self):
        cell_method_str = 'time: minimum within days time: mean over days'
        expected = (CellMethod(method='minimum within days', coords='time'),
                    CellMethod(method='mean over days', coords='time'))
        res = _parse_cell_methods('test_var', cell_method_str)
        self.assertEqual(res, expected)

    def test_climatology_with_unknown_method(self):
        cell_method_str = 'time: min within days time: mean over days'
        expected = (CellMethod(method='min within days', coords='time'),
                    CellMethod(method='mean over days', coords='time'))
        with mock.patch('warnings.warn') as warn:
            res = _parse_cell_methods('test_var', cell_method_str)
        self.assertIn("NetCDF variable 'test_var' contains unknown "
                      "cell method 'min'",
                      warn.call_args[0][0])
        self.assertEqual(res, expected)


if __name__ == "__main__":
    tests.main()
