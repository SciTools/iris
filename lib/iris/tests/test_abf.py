# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import numpy as np

import iris
import iris.fileformats.abf


@tests.skip_data
class TestAbfLoad(tests.IrisTest):
    def setUp(self):
        self.path = tests.get_data_path(("abf", "AVHRRBUVI01.1985apra.abf"))

    def test_load(self):
        cubes = iris.load(self.path)
        # On a 32-bit platform the time coordinate will have 32-bit integers.
        # We force them to 64-bit to ensure consistent test results.
        time_coord = cubes[0].coord("time")
        time_coord.points = np.array(time_coord.points, dtype=np.int64)
        time_coord.bounds = np.array(time_coord.bounds, dtype=np.int64)
        # Normalise the different array orders returned by version 1.6
        # and 1.7 of NumPy.
        cubes[0].data = cubes[0].data.copy(order="C")
        self.assertCML(cubes, ("abf", "load.cml"))

    def test_fill_value(self):
        field = iris.fileformats.abf.ABFField(self.path)
        # Make sure the fill value is appropriate. It must avoid the
        # data range (0 to 100 inclusive) but still fit within the dtype
        # range (0 to 255 inclusive).
        self.assertGreater(field.data.fill_value, 100)
        self.assertLess(field.data.fill_value, 256)


if __name__ == "__main__":
    tests.main()
