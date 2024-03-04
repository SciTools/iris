# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Integration tests for :func:`iris.util.new_axis`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import iris
from iris.util import new_axis


class Test(tests.IrisTest):
    @tests.skip_data
    def test_lazy_data(self):
        filename = tests.get_data_path(("PP", "globClim1", "theta.pp"))
        cube = iris.load_cube(filename)
        new_cube = new_axis(cube, "time")
        self.assertTrue(cube.has_lazy_data())
        self.assertTrue(new_cube.has_lazy_data())
        self.assertEqual(new_cube.shape, (1,) + cube.shape)


if __name__ == "__main__":
    tests.main()
