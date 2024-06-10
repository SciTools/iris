# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris.analysis.name_loaders._build_lat_lon_for_NAME_timeseries`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from iris.fileformats.name_loaders import NAMECoord, _build_lat_lon_for_NAME_timeseries


class TestCellMethods(tests.IrisTest):
    def test_float(self):
        column_headings = {
            "X": ["X = -.100 Lat-Long", "X = -1.600 Lat-Long"],
            "Y": ["Y = 52.450 Lat-Long", "Y = 51. Lat-Long"],
        }
        lat, lon = _build_lat_lon_for_NAME_timeseries(column_headings)
        self.assertIsInstance(lat, NAMECoord)
        self.assertIsInstance(lon, NAMECoord)
        self.assertEqual(lat.name, "latitude")
        self.assertEqual(lon.name, "longitude")
        self.assertIsNone(lat.dimension)
        self.assertIsNone(lon.dimension)
        self.assertArrayEqual(lat.values, [52.45, 51.0])
        self.assertArrayEqual(lon.values, [-0.1, -1.6])

    def test_int(self):
        column_headings = {
            "X": ["X = -1 Lat-Long", "X = -2 Lat-Long"],
            "Y": ["Y = 52 Lat-Long", "Y = 51 Lat-Long"],
        }
        lat, lon = _build_lat_lon_for_NAME_timeseries(column_headings)
        self.assertIsInstance(lat, NAMECoord)
        self.assertIsInstance(lon, NAMECoord)
        self.assertEqual(lat.name, "latitude")
        self.assertEqual(lon.name, "longitude")
        self.assertIsNone(lat.dimension)
        self.assertIsNone(lon.dimension)
        self.assertArrayEqual(lat.values, [52.0, 51.0])
        self.assertArrayEqual(lon.values, [-1.0, -2.0])
        self.assertIsInstance(lat.values[0], float)
        self.assertIsInstance(lon.values[0], float)
