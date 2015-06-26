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
"""Unit tests for the `iris.fileformats.grib.as_messages` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import iris.tests as tests

import gribapi

import iris
from iris.coords import DimCoord
import iris.fileformats.grib as grib
from iris.tests import mock
import iris.tests.stock as stock


class TestAsMessages(tests.IrisTest):
    def setUp(self):
        self.cube = stock.realistic_3d()

    def test_as_messages(self):
        realization = 2
        type_of_process = 4
        coord = DimCoord(realization, standard_name='realization', units='1')
        self.cube.add_aux_coord(coord)
        messages = grib.as_messages(self.cube)
        for message in messages:
            self.assertEqual(gribapi.grib_get_long(message,
                                                   'typeOfProcessedData'),
                             type_of_process)


if __name__ == "__main__":
    tests.main()
