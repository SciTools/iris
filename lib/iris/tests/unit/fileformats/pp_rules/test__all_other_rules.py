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
"""Unit tests for the `iris.fileformats.pp._all_other_rules` function."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

import iris
from iris.fileformats.pp_rules import _all_other_rules
from iris.coords import CellMethod


class TestCellMethods(tests.IrisTest):
    def test_time_mean(self):
        # lbproc = 128 -> mean
        # lbtim.ib = 2 -> simple t1 to t2 interval.
        field = mock.MagicMock(lbproc=128,
                               lbtim=mock.Mock(ia=0, ib=2, ic=3))
        res = _all_other_rules(field)[5]
        expected = [CellMethod('mean', 'time')]
        self.assertEqual(res, expected)

    def test_hourly_mean(self):
        # lbtim.ia = 1 -> hourly
        field = mock.MagicMock(lbproc=128,
                               lbtim=mock.Mock(ia=1, ib=2, ic=3))
        res = _all_other_rules(field)[5]
        expected = [CellMethod('mean', 'time', '1 hour')]
        self.assertEqual(res, expected)

    def test_daily_mean(self):
        # lbtim.ia = 24 -> daily
        field = mock.MagicMock(lbproc=128,
                               lbtim=mock.Mock(ia=24, ib=2, ic=3))
        res = _all_other_rules(field)[5]
        expected = [CellMethod('mean', 'time', '24 hour')]
        self.assertEqual(res, expected)

    def test_custom_max(self):
        field = mock.MagicMock(lbproc=8192,
                               lbtim=mock.Mock(ia=47, ib=2, ic=3))
        res = _all_other_rules(field)[5]
        expected = [CellMethod('maximum', 'time', '47 hour')]
        self.assertEqual(res, expected)

    def test_daily_min(self):
        # lbproc = 4096 -> min
        field = mock.MagicMock(lbproc=4096,
                               lbtim=mock.Mock(ia=24, ib=2, ic=3))
        res = _all_other_rules(field)[5]
        expected = [CellMethod('minimum', 'time', '24 hour')]
        self.assertEqual(res, expected)

    def test_time_mean_over_multiple_years(self):
        # lbtim.ib = 3 -> interval within a year, over multiple years.
        field = mock.MagicMock(lbproc=128,
                               lbtim=mock.Mock(ia=0, ib=3, ic=3))
        res = _all_other_rules(field)[5]
        expected = [CellMethod('mean within years', 'time'),
                    CellMethod('mean over years', 'time')]
        self.assertEqual(res, expected)

    def test_hourly_mean_over_multiple_years(self):
        field = mock.MagicMock(lbproc=128,
                               lbtim=mock.Mock(ia=1, ib=3, ic=3))
        res = _all_other_rules(field)[5]
        expected = [CellMethod('mean within years', 'time', '1 hour'),
                    CellMethod('mean over years', 'time')]
        self.assertEqual(res, expected)

    def test_climatology_max(self):
        field = mock.MagicMock(lbproc=8192,
                               lbtim=mock.Mock(ia=24, ib=3, ic=3))
        res = _all_other_rules(field)[5]
        expected = [CellMethod('maximum', 'time')]
        self.assertEqual(res, expected)

    def test_climatology_max(self):
        field = mock.MagicMock(lbproc=4096,
                               lbtim=mock.Mock(ia=24, ib=3, ic=3))
        res = _all_other_rules(field)[5]
        expected = [CellMethod('minimum', 'time')]
        self.assertEqual(res, expected)

    def test_other_lbtim_ib(self):
        # lbtim.ib = 5 -> non-specific aggregation
        field = mock.MagicMock(lbproc=4096,
                               lbtim=mock.Mock(ia=24, ib=5, ic=3))
        res = _all_other_rules(field)[5]
        expected = [CellMethod('minimum', 'time')]
        self.assertEqual(res, expected)


if __name__ == "__main__":
    tests.main()
