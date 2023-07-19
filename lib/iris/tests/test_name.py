# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Tests for NAME loading."""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import tempfile

import iris


@tests.skip_data
class TestLoad(tests.IrisTest):
    def test_NAMEIII_field(self):
        cubes = iris.load(tests.get_data_path(("NAME", "NAMEIII_field.txt")))
        self.assertCMLApproxData(cubes, ("name", "NAMEIII_field.cml"))

    def test_NAMEII_field(self):
        cubes = iris.load(tests.get_data_path(("NAME", "NAMEII_field.txt")))
        self.assertCMLApproxData(cubes, ("name", "NAMEII_field.cml"))

    def test_NAMEIII_timeseries(self):
        cubes = iris.load(
            tests.get_data_path(("NAME", "NAMEIII_timeseries.txt"))
        )
        self.assertCMLApproxData(cubes, ("name", "NAMEIII_timeseries.cml"))

    def test_NAMEII_timeseries(self):
        cubes = iris.load(
            tests.get_data_path(("NAME", "NAMEII_timeseries.txt"))
        )
        self.assertCMLApproxData(cubes, ("name", "NAMEII_timeseries.cml"))

    def test_NAMEIII_version2(self):
        cubes = iris.load(
            tests.get_data_path(("NAME", "NAMEIII_version2.txt"))
        )
        self.assertCMLApproxData(cubes, ("name", "NAMEIII_version2.cml"))

    def test_NAMEIII_trajectory(self):
        cubes = iris.load(
            tests.get_data_path(("NAME", "NAMEIII_trajectory.txt"))
        )
        self.assertCML(cubes[0], ("name", "NAMEIII_trajectory0.cml"))
        self.assertCML(
            cubes, ("name", "NAMEIII_trajectory.cml"), checksum=False
        )

    def test_NAMEII__no_time_averaging(self):
        cubes = iris.load(
            tests.get_data_path(("NAME", "NAMEII_no_time_averaging.txt"))
        )

        # Also check that it saves without error.
        # This was previously failing, see https://github.com/SciTools/iris/issues/3288
        with tempfile.TemporaryDirectory() as temp_dirpath:
            iris.save(cubes, temp_dirpath + "/tmp.nc")

        self.assertCML(
            cubes[0],
            (
                "name",
                "NAMEII_field__no_time_averaging_0.cml",
            ),
        )
        self.assertCML(
            cubes,
            (
                "name",
                "NAMEII_field__no_time_averaging.cml",
            ),
            checksum=False,
        )


if __name__ == "__main__":
    tests.main()
