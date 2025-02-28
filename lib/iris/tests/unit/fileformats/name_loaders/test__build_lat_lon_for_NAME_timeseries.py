# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris.analysis.name_loaders._build_lat_lon_for_NAME_timeseries`."""

from iris.fileformats.name_loaders import NAMECoord, _build_lat_lon_for_NAME_timeseries
from iris.tests._shared_utils import assert_array_equal


class TestCellMethods:
    def test_float(self):
        column_headings = {
            "X": ["X = -.100 Lat-Long", "X = -1.600 Lat-Long"],
            "Y": ["Y = 52.450 Lat-Long", "Y = 51. Lat-Long"],
        }
        lat, lon = _build_lat_lon_for_NAME_timeseries(column_headings)
        assert isinstance(lat, NAMECoord)
        assert isinstance(lon, NAMECoord)
        assert lat.name == "latitude"
        assert lon.name == "longitude"
        assert lat.dimension is None
        assert lon.dimension is None
        assert_array_equal(lat.values, [52.45, 51.0])
        assert_array_equal(lon.values, [-0.1, -1.6])

    def test_int(self):
        column_headings = {
            "X": ["X = -1 Lat-Long", "X = -2 Lat-Long"],
            "Y": ["Y = 52 Lat-Long", "Y = 51 Lat-Long"],
        }
        lat, lon = _build_lat_lon_for_NAME_timeseries(column_headings)
        assert isinstance(lat, NAMECoord)
        assert isinstance(lon, NAMECoord)
        assert lat.name == "latitude"
        assert lon.name == "longitude"
        assert lat.dimension is None
        assert lon.dimension is None
        assert_array_equal(lat.values, [52.0, 51.0])
        assert_array_equal(lon.values, [-1.0, -2.0])
        assert isinstance(lat.values[0], float)
        assert isinstance(lon.values[0], float)
