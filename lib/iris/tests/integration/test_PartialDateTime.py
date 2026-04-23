# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for :class:`iris.time.PartialDateTime`."""

import iris
from iris.tests import _shared_utils
from iris.time import PartialDateTime


class Test:
    @_shared_utils.skip_data
    def test_cftime_interface(self):
        # The `netcdf4` Python module introduced new calendar classes by v1.2.7
        # This test is primarily of this interface, so the
        # final test assertion is simple.
        filename = _shared_utils.get_data_path(("PP", "structured", "small.pp"))
        cube = iris.load_cube(filename)
        pdt = PartialDateTime(year=1992, month=10, day=1, hour=2)
        time_constraint = iris.Constraint(time=lambda cell: cell < pdt)
        sub_cube = cube.extract(time_constraint)
        assert sub_cube.coord("time").points.shape == (1,)
