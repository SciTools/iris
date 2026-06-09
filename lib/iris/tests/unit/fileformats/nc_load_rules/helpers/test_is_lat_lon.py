# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.fileformats._nc_load_rules.helpers._is_lat_lon`."""

from iris.fileformats._nc_load_rules import helpers
from iris.fileformats.cf import CFCoordinateVariable


def test_non_string_units(mocker):
    cf_var = mocker.MagicMock(spec=CFCoordinateVariable, units=1.0)
    is_lat_lon = helpers._is_lat_lon(cf_var, [], "latitude", "grid_latitude", "y", [])
    assert is_lat_lon is False
