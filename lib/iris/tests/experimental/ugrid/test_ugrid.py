# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test the :func:`iris.experimental.ugrid.ugrid` function.

"""

import iris.tests as tests  # isort:skip

import unittest

# Import pyugrid if installed, else fail quietly + disable all the tests.
try:
    import pyugrid
except (ImportError, AttributeError):
    pyugrid = None
skip_pyugrid = unittest.skipIf(
    condition=pyugrid is None,
    reason="Requires pyugrid, which is not available.",
)

import iris.experimental.ugrid

data_path = (
    "NetCDF",
    "ugrid",
)
file21 = "21_triangle_example.nc"
long_name = "volume flux between cells"


@skip_pyugrid
@tests.skip_data
class TestUgrid(tests.IrisTest):
    def test_ugrid(self):
        path = tests.get_data_path(data_path + (file21,))
        cube = iris.experimental.ugrid.ugrid(path, long_name)
        self.assertTrue(hasattr(cube, "mesh"))


if __name__ == "__main__":
    tests.main()
