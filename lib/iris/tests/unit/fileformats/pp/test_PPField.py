# (C) British Crown Copyright 2013 - 2014, Met Office
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
"""Unit tests for the `iris.fileformats.pp.PPField` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import warnings

import mock
import numpy as np

from iris.fileformats.pp import PPField
from iris.fileformats.pp import SplittableInt

# The PPField class is abstract, so to test we define a minimal,
# concrete subclass with the `t1` and `t2` properties.
#
# NB. We define dummy header items to allow us to zero the unused header
# items when written to disk and get consistent results.


class TestPPField(PPField):

    HEADER_DEFN = [
        ('dummy1', (0, 13)),
        ('lblrec', (14,)),
        ('dummy2', (15, 18)),
        ('lbext',  (19,)),
        ('lbpack', (20,)),
        ('dummy3', (21, 37)),
        ('lbuser', (38, 39, 40, 41, 42, 43, 44,)),
        ('dummy4', (45, 63)),
    ]

    @property
    def t1(self):
        return netcdftime.datetime(2013, 10, 14, 10, 4)

    @property
    def t2(self):
        return netcdftime.datetime(2013, 10, 14, 10, 5)


class Test_save(tests.IrisTest):
    def test_float64(self):
        # Tests down-casting of >f8 data to >f4.

        def field_checksum(data):
            field = TestPPField()
            field.dummy1 = 0
            field.dummy2 = 0
            field.dummy3 = 0
            field.dummy4 = 0
            field.lblrec = 0
            field.lbext = 0
            field.lbpack = 0
            field.lbuser = 0
            field.data = data
            with self.temp_filename('.pp') as temp_filename:
                with open(temp_filename, 'wb') as pp_file:
                    field.save(pp_file)
                checksum = self.file_checksum(temp_filename)
            return checksum

        data_64 = np.linspace(0, 1, num=10, endpoint=False).reshape(2, 5)
        checksum_32 = field_checksum(data_64.astype('>f4'))
        with mock.patch('warnings.warn') as warn:
            checksum_64 = field_checksum(data_64.astype('>f8'))

        self.assertEquals(checksum_32, checksum_64)
        warn.assert_called()


class Test_calendar(tests.IrisTest):
    def test_greg(self):
        field = TestPPField()
        field.lbtim = SplittableInt(1, {'ia': 2, 'ib': 1, 'ic': 0})
        self.assertEqual(field.calendar, 'gregorian')

    def test_360(self):
        field = TestPPField()
        field.lbtim = SplittableInt(2, {'ia': 2, 'ib': 1, 'ic': 0})
        self.assertEqual(field.calendar, '360_day')

    def test_365(self):
        field = TestPPField()
        field.lbtim = SplittableInt(4, {'ia': 2, 'ib': 1, 'ic': 0})
        self.assertEqual(field.calendar, '365_day')


if __name__ == "__main__":
    tests.main()
