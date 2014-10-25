# (C) British Crown Copyright 2010 - 2014, Met Office
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

from __future__ import (absolute_import, division, print_function)

import unittest

from iris.std_names import STD_NAMES


class TestStandardNames(unittest.TestCase):
    """
    standard_names.py is a machine generated file which contains a single dictionary
    called STD_NAMES
    """

    longMessage = True

    def test_standard_names(self):
        # Check we have a dict
        self.assertIsInstance(STD_NAMES, dict)

        keyset = set(STD_NAMES)

        # Check for some known standard names
        valid_nameset = set(["air_density", "northward_wind", "wind_speed"])
        self.assertTrue(valid_nameset.issubset(keyset), "Known standard name missing from STD_NAMES")

        # Check for some invalid standard names
        invalid_nameset = set(["invalid_air_density", "invalid_northward_wind",
                               "invalid_wind_speed",
                               "stratiform_snowfall_rate"])
        self.assertSetEqual(invalid_nameset - keyset, invalid_nameset,
                            "\nInvalid standard name(s) present in STD_NAMES")


if __name__ == "__main__":
    unittest.main()
