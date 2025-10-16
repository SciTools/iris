# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for NAME loading."""

import iris
from iris.tests import _shared_utils


@_shared_utils.skip_data
class TestLoad:
    def test_NAMEIII_field(self, request):
        cubes = iris.load(_shared_utils.get_data_path(("NAME", "NAMEIII_field.txt")))
        _shared_utils.assert_CML(
            request, cubes, ("name", "NAMEIII_field.cml"), approx_data=True
        )

    def test_NAMEII_field(self, request):
        cubes = iris.load(_shared_utils.get_data_path(("NAME", "NAMEII_field.txt")))
        _shared_utils.assert_CML(
            request, cubes, ("name", "NAMEII_field.cml"), approx_data=True
        )

    def test_NAMEIII_timeseries(self, request):
        cubes = iris.load(
            _shared_utils.get_data_path(("NAME", "NAMEIII_timeseries.txt"))
        )
        _shared_utils.assert_CML(
            request, cubes, ("name", "NAMEIII_timeseries.cml"), approx_data=True
        )

    def test_NAMEII_timeseries(self, request):
        cubes = iris.load(
            _shared_utils.get_data_path(("NAME", "NAMEII_timeseries.txt"))
        )
        _shared_utils.assert_CML(
            request, cubes, ("name", "NAMEII_timeseries.cml"), approx_data=True
        )

    def test_NAMEIII_version2(self, request):
        cubes = iris.load(_shared_utils.get_data_path(("NAME", "NAMEIII_version2.txt")))
        _shared_utils.assert_CML(
            request, cubes, ("name", "NAMEIII_version2.cml"), approx_data=True
        )

    def test_NAMEIII_trajectory(self, request):
        cubes = iris.load(
            _shared_utils.get_data_path(("NAME", "NAMEIII_trajectory.txt"))
        )
        _shared_utils.assert_CML(request, cubes[0], ("name", "NAMEIII_trajectory0.cml"))
        _shared_utils.assert_CML(
            request, cubes, ("name", "NAMEIII_trajectory.cml"), checksum=False
        )

    def test_NAMEII__no_time_averaging(self, request, tmp_path):
        cubes = iris.load(
            _shared_utils.get_data_path(("NAME", "NAMEII_no_time_averaging.txt"))
        )

        # Also check that it saves without error.
        # This was previously failing, see https://github.com/SciTools/iris/issues/3288
        iris.save(cubes, tmp_path / "tmp.nc")

        _shared_utils.assert_CML(
            request,
            cubes[0],
            (
                "name",
                "NAMEII_field__no_time_averaging_0.cml",
            ),
        )
        _shared_utils.assert_CML(
            request,
            cubes,
            (
                "name",
                "NAMEII_field__no_time_averaging.cml",
            ),
            checksum=False,
        )
