# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Integration tests for pickling things."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import pickle

import iris


class Common:
    def pickle_cube(self, protocol):
        # Ensure that data proxies are pickleable.
        cube = iris.load(self.path)[0]
        with self.temp_filename(".pkl") as filename:
            with open(filename, "wb") as f:
                pickle.dump(cube, f, protocol)
            with open(filename, "rb") as f:
                ncube = pickle.load(f)
        self.assertEqual(ncube, cube)

    def test_protocol_0(self):
        self.pickle_cube(0)

    def test_protocol_1(self):
        self.pickle_cube(1)

    def test_protocol_2(self):
        self.pickle_cube(2)


@tests.skip_data
class test_netcdf(Common, tests.IrisTest):
    def setUp(self):
        self.path = tests.get_data_path(
            ("NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc")
        )


@tests.skip_data
class test_pp(Common, tests.IrisTest):
    def setUp(self):
        self.path = tests.get_data_path(("PP", "aPPglob1", "global.pp"))


@tests.skip_data
class test_ff(Common, tests.IrisTest):
    def setUp(self):
        self.path = tests.get_data_path(("FF", "n48_multi_field"))


if __name__ == "__main__":
    tests.main()
