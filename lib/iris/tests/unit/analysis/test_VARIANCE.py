# (C) British Crown Copyright 2013, Met Office
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
"""Unit tests for the :data:`iris.analysis.VARIANCE` aggregator."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

from iris.analysis import VARIANCE


class Test_units_func(tests.IrisTest):
    def test(self):
        self.assertIsNotNone(VARIANCE.units_func)
        mul = mock.Mock(return_value=mock.sentinel.new_unit)
        units = mock.Mock(__mul__=mul)
        new_units = VARIANCE.units_func(units)
        # Make sure the VARIANCE units_func tries to square the units.
        mul.assert_called_once_with(units)
        self.assertEqual(new_units, mock.sentinel.new_unit)


if __name__ == "__main__":
    tests.main()
