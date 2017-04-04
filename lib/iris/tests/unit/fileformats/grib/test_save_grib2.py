# (C) British Crown Copyright 2016 - 2017, Met Office
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
"""Unit tests for the `iris.fileformats.grib.save_grib2` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

import iris.fileformats.grib


class TestSaveGrib2(tests.IrisGribTest):
    def setUp(self):
        self.cube = mock.sentinel.cube
        self.target = mock.sentinel.target
        func = 'iris.fileformats.grib.save_pairs_from_cube'
        self.messages = list(range(10))
        slices = self.messages
        side_effect = [zip(slices, self.messages)]
        self.save_pairs_from_cube = self.patch(func, side_effect=side_effect)
        func = 'iris.fileformats.grib.save_messages'
        self.save_messages = self.patch(func)

    def _check(self, append=False):
        iris.fileformats.grib.save_grib2(self.cube, self.target, append=append)
        self.save_pairs_from_cube.called_once_with(self.cube)
        args, kwargs = self.save_messages.call_args
        self.assertEqual(len(args), 2)
        messages, target = args
        self.assertEqual(list(messages), self.messages)
        self.assertEqual(target, self.target)
        self.assertEqual(kwargs, dict(append=append))

    def test_save_no_append(self):
        self._check()

    def test_save_append(self):
        self._check(append=True)


if __name__ == "__main__":
    tests.main()
