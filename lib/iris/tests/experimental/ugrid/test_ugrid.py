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
"""
Test the :func:`iris.experimental.ugrid.ugrid` function.

"""

from __future__ import (absolute_import, division, print_function)

import iris.tests as tests

import unittest

# Import pyugrid if installed, else fail quietly + disable all the tests.
try:
    import pyugrid
    # Check it *is* the real module, and not an iris.proxy FakeModule.
    pyugrid.ugrid
except (ImportError, AttributeError):
    pyugrid = None
skip_pyugrid = unittest.skipIf(
    condition=pyugrid is None,
    reason='Requires pyugrid, which is not available.')

import iris.experimental.ugrid


data_path = ("NetCDF", "ugrid", )
file21 = "21_triangle_example.nc"
long_name = "volume flux between cells"


@skip_pyugrid
@tests.skip_data
class TestUgrid(tests.IrisTest):
    def test_ugrid(self):
        path = tests.get_data_path(data_path + (file21, ))
        cube = iris.experimental.ugrid.ugrid(path, long_name)
        self.assertTrue(hasattr(cube, 'mesh'))


if __name__ == "__main__":
    tests.main()
