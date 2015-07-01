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
"""Tests for NAME loading."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests
import iris


@tests.skip_data
class TestLoad(tests.IrisTest):
    def test_NAMEIII_field(self):
        cubes = iris.load(tests.get_data_path(('NAME', 'NAMEIII_field.txt')))
        self.assertCMLApproxData(cubes, ('name', 'NAMEIII_field.cml'))

    def test_NAMEII_field(self):
        cubes = iris.load(tests.get_data_path(('NAME', 'NAMEII_field.txt')))
        self.assertCMLApproxData(cubes, ('name', 'NAMEII_field.cml'))

    def test_NAMEIII_timeseries(self):
        cubes = iris.load(tests.get_data_path(('NAME',
                                               'NAMEIII_timeseries.txt')))
        self.assertCMLApproxData(cubes, ('name', 'NAMEIII_timeseries.cml'))

    def test_NAMEII_timeseries(self):
        cubes = iris.load(tests.get_data_path(('NAME',
                                               'NAMEII_timeseries.txt')))
        self.assertCMLApproxData(cubes, ('name', 'NAMEII_timeseries.cml'))

    def test_NAMEII_trajectory(self):
        cubes = iris.load(tests.get_data_path(('NAME',
                                              'NAMEIII_trajectory.txt')))
        self.assertCML(cubes[0], ('name', 'NAMEIII_trajectory0.cml'))
        self.assertCML(cubes, ('name', 'NAMEIII_trajectory.cml'),
                       checksum=False)


if __name__ == "__main__":
    tests.main()
