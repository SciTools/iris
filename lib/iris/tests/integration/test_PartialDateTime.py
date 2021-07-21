# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Integration tests for :class:`iris.time.PartialDateTime`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import iris
from iris.time import PartialDateTime


class Test(tests.IrisTest):
    @tests.skip_data
    def test_cftime_interface(self):
        # The `netcdf4` Python module introduced new calendar classes by v1.2.7
        # This test is primarily of this interface, so the
        # final test assertion is simple.
        filename = tests.get_data_path(("PP", "structured", "small.pp"))
        cube = iris.load_cube(filename)
        pdt = PartialDateTime(year=1992, month=10, day=1, hour=2)
        time_constraint = iris.Constraint(time=lambda cell: cell < pdt)
        sub_cube = cube.extract(time_constraint)
        self.assertEqual(sub_cube.coord("time").points.shape, (1,))


if __name__ == "__main__":
    tests.main()
