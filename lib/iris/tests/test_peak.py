# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import numpy as np
import numpy.ma as ma

import iris.analysis
import iris.tests.stock


class TestPeakAggregator(tests.IrisTest):
    def test_peak_coord_length_1(self):
        # Coordinate contains a single point.
        latitude = iris.coords.DimCoord(
            np.array([0]), standard_name="latitude", units="degrees"
        )
        cube = iris.cube.Cube(
            np.array([1]), standard_name="air_temperature", units="kelvin"
        )
        cube.add_dim_coord(latitude, 0)

        collapsed_cube = cube.collapsed("latitude", iris.analysis.PEAK)
        self.assertArrayAlmostEqual(
            collapsed_cube.data, np.array([1], dtype=np.float32)
        )

    def test_peak_coord_length_2(self):
        # Coordinate contains 2 points.
        latitude = iris.coords.DimCoord(
            np.arange(0, 2, 1), standard_name="latitude", units="degrees"
        )
        cube = iris.cube.Cube(
            np.array([1, 2]), standard_name="air_temperature", units="kelvin"
        )
        cube.add_dim_coord(latitude, 0)

        collapsed_cube = cube.collapsed("latitude", iris.analysis.PEAK)
        self.assertArrayAlmostEqual(
            collapsed_cube.data, np.array([2], dtype=np.float32)
        )

    def test_peak_coord_length_3(self):
        # Coordinate contains 3 points.
        latitude = iris.coords.DimCoord(
            np.arange(0, 3, 1), standard_name="latitude", units="degrees"
        )
        cube = iris.cube.Cube(
            np.array([1, 2, 1]),
            standard_name="air_temperature",
            units="kelvin",
        )
        cube.add_dim_coord(latitude, 0)

        collapsed_cube = cube.collapsed("latitude", iris.analysis.PEAK)
        self.assertArrayAlmostEqual(
            collapsed_cube.data, np.array([2], dtype=np.float32)
        )

    def test_peak_1d(self):
        # Collapse a 1d cube.
        latitude = iris.coords.DimCoord(
            np.arange(0, 11, 1), standard_name="latitude", units="degrees"
        )
        cube = iris.cube.Cube(
            np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]),
            standard_name="air_temperature",
            units="kelvin",
        )
        cube.add_dim_coord(latitude, 0)

        collapsed_cube = cube.collapsed("latitude", iris.analysis.PEAK)
        self.assertArrayAlmostEqual(
            collapsed_cube.data, np.array([6], dtype=np.float32)
        )

    def test_peak_duplicate_coords(self):
        # Collapse cube along 2 coordinates (both the same).
        latitude = iris.coords.DimCoord(
            np.arange(0, 4, 1), standard_name="latitude", units="degrees"
        )
        cube = iris.cube.Cube(
            np.array([1, 2, 3, 1]),
            standard_name="air_temperature",
            units="kelvin",
        )
        cube.add_dim_coord(latitude, 0)

        collapsed_cube = cube.collapsed("latitude", iris.analysis.PEAK)
        self.assertArrayAlmostEqual(
            collapsed_cube.data, np.array([3], dtype=np.float32)
        )

        collapsed_cube = cube.collapsed(
            ("latitude", "latitude"), iris.analysis.PEAK
        )
        self.assertArrayAlmostEqual(
            collapsed_cube.data, np.array([3], dtype=np.float32)
        )

    def test_peak_2d(self):
        # Collapse a 2d cube.
        longitude = iris.coords.DimCoord(
            np.arange(0, 4, 1), standard_name="longitude", units="degrees"
        )
        latitude = iris.coords.DimCoord(
            np.arange(0, 3, 1), standard_name="latitude", units="degrees"
        )
        cube = iris.cube.Cube(
            np.array([[1, 2, 3, 1], [4, 5, 7, 4], [2, 3, 4, 2]]),
            standard_name="air_temperature",
            units="kelvin",
        )

        cube.add_dim_coord(latitude, 0)
        cube.add_dim_coord(longitude, 1)

        collapsed_cube = cube.collapsed("longitude", iris.analysis.PEAK)
        self.assertArrayAlmostEqual(
            collapsed_cube.data, np.array([3, 7.024054, 4], dtype=np.float32)
        )

        collapsed_cube = cube.collapsed("latitude", iris.analysis.PEAK)
        self.assertArrayAlmostEqual(
            collapsed_cube.data,
            np.array(
                [4.024977, 5.024977, 7.017852, 4.024977], dtype=np.float32
            ),
        )

        collapsed_cube = cube.collapsed(
            ("longitude", "latitude"), iris.analysis.PEAK
        )
        self.assertArrayAlmostEqual(
            collapsed_cube.data, np.array([7.041787], dtype=np.float32)
        )

        collapsed_cube = cube.collapsed(
            ("latitude", "longitude"), iris.analysis.PEAK
        )
        self.assertArrayAlmostEqual(
            collapsed_cube.data, np.array([7.041629], dtype=np.float32)
        )

    def test_peak_without_peak_value(self):
        # No peak in column (values equal).
        latitude = iris.coords.DimCoord(
            np.arange(0, 4, 1), standard_name="latitude", units="degrees"
        )
        cube = iris.cube.Cube(
            np.array([1, 1, 1, 1]),
            standard_name="air_temperature",
            units="kelvin",
        )
        cube.add_dim_coord(latitude, 0)

        collapsed_cube = cube.collapsed("latitude", iris.analysis.PEAK)
        self.assertArrayAlmostEqual(
            collapsed_cube.data, np.array([1], dtype=np.float32)
        )

    def test_peak_with_nan(self):
        # Single nan in column.
        latitude = iris.coords.DimCoord(
            np.arange(0, 5, 1), standard_name="latitude", units="degrees"
        )
        cube = iris.cube.Cube(
            np.array([1, 4, 2, 3, 1], dtype=np.float32),
            standard_name="air_temperature",
            units="kelvin",
        )
        cube.add_dim_coord(latitude, 0)

        cube.data[3] = np.nan

        collapsed_cube = cube.collapsed("latitude", iris.analysis.PEAK)
        self.assertArrayAlmostEqual(
            collapsed_cube.data, np.array([4.024977], dtype=np.float32)
        )
        self.assertEqual(collapsed_cube.data.shape, (1,))

        # Only nans in column.
        cube.data[:] = np.nan

        collapsed_cube = cube.collapsed("latitude", iris.analysis.PEAK)
        self.assertTrue(np.isnan(collapsed_cube.data).all())
        self.assertEqual(collapsed_cube.data.shape, (1,))

    def test_peak_with_mask(self):
        # Single value in column masked.
        latitude = iris.coords.DimCoord(
            np.arange(0, 5, 1), standard_name="latitude", units="degrees"
        )
        cube = iris.cube.Cube(
            ma.array([1, 4, 2, 3, 2], dtype=np.float32),
            standard_name="air_temperature",
            units="kelvin",
        )
        cube.add_dim_coord(latitude, 0)

        cube.data[3] = ma.masked

        collapsed_cube = cube.collapsed("latitude", iris.analysis.PEAK)
        self.assertArrayAlmostEqual(
            collapsed_cube.data, np.array([4.024977], dtype=np.float32)
        )
        self.assertTrue(ma.isMaskedArray(collapsed_cube.data))
        self.assertEqual(collapsed_cube.data.shape, (1,))

        # Whole column masked.
        cube.data[:] = ma.masked

        collapsed_cube = cube.collapsed("latitude", iris.analysis.PEAK)
        masked_array = ma.array(ma.masked)
        self.assertTrue(ma.allequal(collapsed_cube.data, masked_array))
        self.assertTrue(ma.isMaskedArray(collapsed_cube.data))
        self.assertEqual(collapsed_cube.data.shape, (1,))

    def test_peak_with_nan_and_mask(self):
        # Single nan in column with single value masked.
        latitude = iris.coords.DimCoord(
            np.arange(0, 5, 1), standard_name="latitude", units="degrees"
        )
        cube = iris.cube.Cube(
            ma.array([1, 4, 2, 3, 1], dtype=np.float32),
            standard_name="air_temperature",
            units="kelvin",
        )
        cube.add_dim_coord(latitude, 0)

        cube.data[3] = np.nan
        cube.data[4] = ma.masked

        collapsed_cube = cube.collapsed("latitude", iris.analysis.PEAK)
        self.assertArrayAlmostEqual(
            collapsed_cube.data, np.array([4.024977], dtype=np.float32)
        )
        self.assertTrue(ma.isMaskedArray(collapsed_cube.data))
        self.assertEqual(collapsed_cube.data.shape, (1,))

        # Only nans in column where values not masked.
        cube.data[0:3] = np.nan

        collapsed_cube = cube.collapsed("latitude", iris.analysis.PEAK)
        self.assertTrue(np.isnan(collapsed_cube.data).all())
        self.assertTrue(ma.isMaskedArray(collapsed_cube.data))
        self.assertEqual(collapsed_cube.data.shape, (1,))

    def test_peak_against_max(self):
        # Cube with data that infers a peak value greater than the column max.
        latitude = iris.coords.DimCoord(
            np.arange(0, 7, 1), standard_name="latitude", units="degrees"
        )
        cube = iris.cube.Cube(
            np.array([0, 1, 3, 7, 7, 4, 2], dtype=np.float32),
            standard_name="air_temperature",
            units="kelvin",
        )
        cube.add_dim_coord(latitude, 0)

        collapsed_cube = cube.collapsed("latitude", iris.analysis.PEAK)
        self.assertArrayAlmostEqual(
            collapsed_cube.data, np.array([7.630991], dtype=np.float32)
        )

        collapsed_cube = cube.collapsed("latitude", iris.analysis.MAX)
        self.assertArrayAlmostEqual(
            collapsed_cube.data, np.array([7], dtype=np.float32)
        )


if __name__ == "__main__":
    tests.main()
