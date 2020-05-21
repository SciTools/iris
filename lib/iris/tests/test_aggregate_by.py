# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np
import numpy.ma as ma
import unittest

import iris
import iris.analysis
import iris.coord_systems
import iris.coords


class TestAggregateBy(tests.IrisTest):
    def setUp(self):
        #
        # common
        #
        cs_latlon = iris.coord_systems.GeogCS(6371229)
        points = np.arange(3, dtype=np.float32) * 3
        coord_lat = iris.coords.DimCoord(
            points, "latitude", units="degrees", coord_system=cs_latlon
        )
        coord_lon = iris.coords.DimCoord(
            points, "longitude", units="degrees", coord_system=cs_latlon
        )

        #
        # single coordinate aggregate-by
        #
        a = np.arange(9, dtype=np.int32).reshape(3, 3) + 1
        b = np.arange(36, dtype=np.int32).reshape(36, 1, 1)
        data = b * a

        self.cube_single = iris.cube.Cube(
            data, long_name="temperature", units="kelvin"
        )

        z_points = np.array(
            [
                1,
                2,
                2,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
                5,
                5,
                5,
                5,
                5,
                6,
                6,
                6,
                6,
                6,
                6,
                7,
                7,
                7,
                7,
                7,
                7,
                7,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
            ],
            dtype=np.int32,
        )
        self.coord_z_single = iris.coords.AuxCoord(
            z_points, long_name="height", units="m"
        )

        model_level = iris.coords.DimCoord(
            np.arange(z_points.size),
            standard_name="model_level_number",
            units="1",
        )

        self.cube_single.add_aux_coord(self.coord_z_single, 0)
        self.cube_single.add_dim_coord(model_level, 0)
        self.cube_single.add_dim_coord(coord_lon, 1)
        self.cube_single.add_dim_coord(coord_lat, 2)

        #
        # multi coordinate aggregate-by
        #
        a = np.arange(9, dtype=np.int32).reshape(3, 3) + 1
        b = np.arange(20, dtype=np.int32).reshape(20, 1, 1)
        data = b * a

        self.cube_multi = iris.cube.Cube(
            data, long_name="temperature", units="kelvin"
        )

        z1_points = np.array(
            [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4, 1, 5, 5, 2, 2],
            dtype=np.int32,
        )
        self.coord_z1_multi = iris.coords.AuxCoord(
            z1_points, long_name="height", units="m"
        )
        z2_points = np.array(
            [1, 1, 1, 3, 3, 3, 5, 5, 5, 7, 7, 7, 9, 9, 9, 11, 11, 11, 3, 3],
            dtype=np.int32,
        )
        self.coord_z2_multi = iris.coords.AuxCoord(
            z2_points, long_name="level", units="1"
        )

        model_level = iris.coords.DimCoord(
            np.arange(z1_points.size),
            standard_name="model_level_number",
            units="1",
        )

        self.cube_multi.add_aux_coord(self.coord_z1_multi, 0)
        self.cube_multi.add_aux_coord(self.coord_z2_multi, 0)
        self.cube_multi.add_dim_coord(model_level, 0)
        self.cube_multi.add_dim_coord(coord_lon.copy(), 1)
        self.cube_multi.add_dim_coord(coord_lat.copy(), 2)

        #
        # expected data results
        #
        self.single_expected = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.5, 3.0, 4.5], [6.0, 7.5, 9.0], [10.5, 12.0, 13.5]],
                [[4.0, 8.0, 12.0], [16.0, 20.0, 24.0], [28.0, 32.0, 36.0]],
                [[7.5, 15.0, 22.5], [30.0, 37.5, 45.0], [52.5, 60.0, 67.5]],
                [[12.0, 24.0, 36.0], [48.0, 60.0, 72.0], [84.0, 96.0, 108.0]],
                [
                    [17.5, 35.0, 52.5],
                    [70.0, 87.5, 105.0],
                    [122.5, 140.0, 157.5],
                ],
                [
                    [24.0, 48.0, 72.0],
                    [96.0, 120.0, 144.0],
                    [168.0, 192.0, 216.0],
                ],
                [
                    [31.5, 63.0, 94.5],
                    [126.0, 157.5, 189.0],
                    [220.5, 252.0, 283.5],
                ],
            ],
            dtype=np.float64,
        )

        row1 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        row2 = [
            list(np.sqrt([2.5, 10.0, 22.5])),
            list(np.sqrt([40.0, 62.5, 90.0])),
            list(np.sqrt([122.5, 160.0, 202.5])),
        ]
        row3 = [
            list(np.sqrt([16.66666667, 66.66666667, 150.0])),
            list(np.sqrt([266.66666667, 416.66666667, 600.0])),
            list(np.sqrt([816.66666667, 1066.66666667, 1350.0])),
        ]
        row4 = [
            list(np.sqrt([57.5, 230.0, 517.5])),
            list(np.sqrt([920.0, 1437.5, 2070.0])),
            list(np.sqrt([2817.5, 3680.0, 4657.5])),
        ]
        row5 = [
            list(np.sqrt([146.0, 584.0, 1314.0])),
            list(np.sqrt([2336.0, 3650.0, 5256.0])),
            list(np.sqrt([7154.0, 9344.0, 11826.0])),
        ]
        row6 = [
            list(np.sqrt([309.16666667, 1236.66666667, 2782.5])),
            list(np.sqrt([4946.66666667, 7729.16666667, 11130.0])),
            list(np.sqrt([15149.16666667, 19786.66666667, 25042.5])),
        ]
        row7 = [
            list(np.sqrt([580.0, 2320.0, 5220.0])),
            list(np.sqrt([9280.0, 14500.0, 20880.0])),
            list(np.sqrt([28420.0, 37120.0, 46980.0])),
        ]
        row8 = [
            list(np.sqrt([997.5, 3990.0, 8977.5])),
            list(np.sqrt([15960.0, 24937.5, 35910.0])),
            list(np.sqrt([48877.5, 63840.0, 80797.5])),
        ]
        self.single_rms_expected = np.array(
            [row1, row2, row3, row4, row5, row6, row7, row8], dtype=np.float64
        )

        self.multi_expected = np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[3.5, 7.0, 10.5], [14.0, 17.5, 21.0], [24.5, 28.0, 31.5]],
                [[14.0, 28.0, 42.0], [56.0, 70.0, 84.0], [98.0, 112.0, 126.0]],
                [[7.0, 14.0, 21.0], [28.0, 35.0, 42.0], [49.0, 56.0, 63.0]],
                [[9.0, 18.0, 27.0], [36.0, 45.0, 54.0], [63.0, 72.0, 81.0]],
                [[10.5, 21.0, 31.5], [42.0, 52.5, 63.0], [73.5, 84.0, 94.5]],
                [[13.0, 26.0, 39.0], [52.0, 65.0, 78.0], [91.0, 104.0, 117.0]],
                [
                    [15.0, 30.0, 45.0],
                    [60.0, 75.0, 90.0],
                    [105.0, 120.0, 135.0],
                ],
                [
                    [16.5, 33.0, 49.5],
                    [66.0, 82.5, 99.0],
                    [115.5, 132.0, 148.5],
                ],
            ],
            dtype=np.float64,
        )

    def test_single(self):
        # mean group-by with single coordinate name.
        aggregateby_cube = self.cube_single.aggregated_by(
            "height", iris.analysis.MEAN
        )
        self.assertCML(
            aggregateby_cube, ("analysis", "aggregated_by", "single.cml")
        )

        # mean group-by with single coordinate.
        aggregateby_cube = self.cube_single.aggregated_by(
            self.coord_z_single, iris.analysis.MEAN
        )
        self.assertCML(
            aggregateby_cube, ("analysis", "aggregated_by", "single.cml")
        )

        np.testing.assert_almost_equal(
            aggregateby_cube.data, self.single_expected
        )

        # rms group-by with single coordinate name.
        aggregateby_cube = self.cube_single.aggregated_by(
            "height", iris.analysis.RMS
        )
        self.assertCML(
            aggregateby_cube, ("analysis", "aggregated_by", "single_rms.cml")
        )

        # rms group-by with single coordinate.
        aggregateby_cube = self.cube_single.aggregated_by(
            self.coord_z_single, iris.analysis.RMS
        )
        self.assertCML(
            aggregateby_cube, ("analysis", "aggregated_by", "single_rms.cml")
        )

        np.testing.assert_almost_equal(
            aggregateby_cube.data, self.single_rms_expected
        )

    def test_single_shared(self):
        z2_points = np.arange(36, dtype=np.int32)
        coord_z2 = iris.coords.AuxCoord(
            z2_points, long_name="wibble", units="1"
        )
        self.cube_single.add_aux_coord(coord_z2, 0)

        # group-by with single coordinate name on shared axis.
        aggregateby_cube = self.cube_single.aggregated_by(
            "height", iris.analysis.MEAN
        )
        self.assertCML(
            aggregateby_cube,
            ("analysis", "aggregated_by", "single_shared.cml"),
        )

        # group-by with single coordinate on shared axis.
        aggregateby_cube = self.cube_single.aggregated_by(
            self.coord_z_single, iris.analysis.MEAN
        )
        self.assertCML(
            aggregateby_cube,
            ("analysis", "aggregated_by", "single_shared.cml"),
        )

        np.testing.assert_almost_equal(
            aggregateby_cube.data, self.single_expected
        )

    def test_single_shared_circular(self):
        points = np.arange(36) * 10.0
        circ_coord = iris.coords.DimCoord(
            points, long_name="circ_height", units="degrees", circular=True
        )
        self.cube_single.add_aux_coord(circ_coord, 0)

        # group-by with single coordinate name on shared axis.
        aggregateby_cube = self.cube_single.aggregated_by(
            "height", iris.analysis.MEAN
        )
        self.assertCML(
            aggregateby_cube,
            ("analysis", "aggregated_by", "single_shared_circular.cml"),
        )

        # group-by with single coordinate on shared axis.
        coord = self.cube_single.coords("height")
        aggregateby_cube = self.cube_single.aggregated_by(
            coord, iris.analysis.MEAN
        )
        self.assertCML(
            aggregateby_cube,
            ("analysis", "aggregated_by", "single_shared_circular.cml"),
        )
        np.testing.assert_almost_equal(
            aggregateby_cube.data, self.single_expected
        )

    def test_multi(self):
        # group-by with multiple coordinate names.
        aggregateby_cube = self.cube_multi.aggregated_by(
            ["height", "level"], iris.analysis.MEAN
        )
        self.assertCML(
            aggregateby_cube, ("analysis", "aggregated_by", "multi.cml")
        )

        # group-by with multiple coordinate names (different order).
        aggregateby_cube = self.cube_multi.aggregated_by(
            ["level", "height"], iris.analysis.MEAN
        )
        self.assertCML(
            aggregateby_cube, ("analysis", "aggregated_by", "multi.cml")
        )

        # group-by with multiple coordinates.
        aggregateby_cube = self.cube_multi.aggregated_by(
            [self.coord_z1_multi, self.coord_z2_multi], iris.analysis.MEAN
        )
        self.assertCML(
            aggregateby_cube, ("analysis", "aggregated_by", "multi.cml")
        )

        # group-by with multiple coordinates (different order).
        aggregateby_cube = self.cube_multi.aggregated_by(
            [self.coord_z2_multi, self.coord_z1_multi], iris.analysis.MEAN
        )
        self.assertCML(
            aggregateby_cube, ("analysis", "aggregated_by", "multi.cml")
        )

        np.testing.assert_almost_equal(
            aggregateby_cube.data, self.multi_expected
        )

    def test_multi_shared(self):
        z3_points = np.arange(20, dtype=np.int32)
        coord_z3 = iris.coords.AuxCoord(
            z3_points, long_name="sigma", units="1"
        )
        z4_points = np.arange(19, -1, -1, dtype=np.int32)
        coord_z4 = iris.coords.AuxCoord(
            z4_points, long_name="gamma", units="1"
        )

        self.cube_multi.add_aux_coord(coord_z3, 0)
        self.cube_multi.add_aux_coord(coord_z4, 0)

        # group-by with multiple coordinate names on shared axis.
        aggregateby_cube = self.cube_multi.aggregated_by(
            ["height", "level"], iris.analysis.MEAN
        )
        self.assertCML(
            aggregateby_cube, ("analysis", "aggregated_by", "multi_shared.cml")
        )

        # group-by with multiple coordinate names on shared axis (different
        # order).
        aggregateby_cube = self.cube_multi.aggregated_by(
            ["level", "height"], iris.analysis.MEAN
        )
        self.assertCML(
            aggregateby_cube, ("analysis", "aggregated_by", "multi_shared.cml")
        )

        # group-by with multiple coordinates on shared axis.
        aggregateby_cube = self.cube_multi.aggregated_by(
            [self.coord_z1_multi, self.coord_z2_multi], iris.analysis.MEAN
        )
        self.assertCML(
            aggregateby_cube, ("analysis", "aggregated_by", "multi_shared.cml")
        )

        # group-by with multiple coordinates on shared axis (different order).
        aggregateby_cube = self.cube_multi.aggregated_by(
            [self.coord_z2_multi, self.coord_z1_multi], iris.analysis.MEAN
        )
        self.assertCML(
            aggregateby_cube, ("analysis", "aggregated_by", "multi_shared.cml")
        )

        np.testing.assert_almost_equal(
            aggregateby_cube.data, self.multi_expected
        )

    def test_easy(self):
        data = np.array(
            [[6, 10, 12, 18], [8, 12, 14, 20], [18, 12, 10, 6]],
            dtype=np.float32,
        )
        cube = iris.cube.Cube(data, long_name="temperature", units="kelvin")

        llcs = iris.coord_systems.GeogCS(6371229)
        cube.add_aux_coord(
            iris.coords.AuxCoord(
                np.array([0, 0, 10], dtype=np.float32),
                "latitude",
                units="degrees",
                coord_system=llcs,
            ),
            0,
        )
        cube.add_aux_coord(
            iris.coords.AuxCoord(
                np.array([0, 0, 10, 10], dtype=np.float32),
                "longitude",
                units="degrees",
                coord_system=llcs,
            ),
            1,
        )

        #
        # Easy mean aggregate test by each coordinate.
        #
        aggregateby_cube = cube.aggregated_by("longitude", iris.analysis.MEAN)
        np.testing.assert_almost_equal(
            aggregateby_cube.data,
            np.array(
                [[8.0, 15.0], [10.0, 17.0], [15.0, 8.0]], dtype=np.float32
            ),
        )
        self.assertCML(
            aggregateby_cube,
            ("analysis", "aggregated_by", "easy.cml"),
            checksum=False,
        )

        aggregateby_cube = cube.aggregated_by("latitude", iris.analysis.MEAN)
        np.testing.assert_almost_equal(
            aggregateby_cube.data,
            np.array(
                [[7.0, 11.0, 13.0, 19.0], [18.0, 12.0, 10.0, 6.0]],
                dtype=np.float32,
            ),
        )

        #
        # Easy max aggregate test by each coordinate.
        #
        aggregateby_cube = cube.aggregated_by("longitude", iris.analysis.MAX)
        np.testing.assert_almost_equal(
            aggregateby_cube.data,
            np.array(
                [[10.0, 18.0], [12.0, 20.0], [18.0, 10.0]], dtype=np.float32
            ),
        )

        aggregateby_cube = cube.aggregated_by("latitude", iris.analysis.MAX)
        np.testing.assert_almost_equal(
            aggregateby_cube.data,
            np.array(
                [[8.0, 12.0, 14.0, 20.0], [18.0, 12.0, 10.0, 6.0]],
                dtype=np.float32,
            ),
        )

        #
        # Easy sum aggregate test by each coordinate.
        #
        aggregateby_cube = cube.aggregated_by("longitude", iris.analysis.SUM)
        np.testing.assert_almost_equal(
            aggregateby_cube.data,
            np.array(
                [[16.0, 30.0], [20.0, 34.0], [30.0, 16.0]], dtype=np.float32
            ),
        )

        aggregateby_cube = cube.aggregated_by("latitude", iris.analysis.SUM)
        np.testing.assert_almost_equal(
            aggregateby_cube.data,
            np.array(
                [[14.0, 22.0, 26.0, 38.0], [18.0, 12.0, 10.0, 6.0]],
                dtype=np.float32,
            ),
        )

        #
        # Easy percentile aggregate test by each coordinate.
        #
        aggregateby_cube = cube.aggregated_by(
            "longitude", iris.analysis.PERCENTILE, percent=25
        )
        np.testing.assert_almost_equal(
            aggregateby_cube.data,
            np.array(
                [[7.0, 13.5], [9.0, 15.5], [13.5, 7.0]], dtype=np.float32
            ),
        )

        aggregateby_cube = cube.aggregated_by(
            "latitude", iris.analysis.PERCENTILE, percent=25
        )
        np.testing.assert_almost_equal(
            aggregateby_cube.data,
            np.array(
                [[6.5, 10.5, 12.5, 18.5], [18.0, 12.0, 10.0, 6.0]],
                dtype=np.float32,
            ),
        )

        #
        # Easy root mean square aggregate test by each coordinate.
        #
        aggregateby_cube = cube.aggregated_by("longitude", iris.analysis.RMS)
        row = [
            list(np.sqrt([68.0, 234.0])),
            list(np.sqrt([104.0, 298.0])),
            list(np.sqrt([234.0, 68.0])),
        ]
        np.testing.assert_almost_equal(
            aggregateby_cube.data, np.array(row, dtype=np.float32)
        )

        aggregateby_cube = cube.aggregated_by("latitude", iris.analysis.RMS)
        row = [
            list(np.sqrt([50.0, 122.0, 170.0, 362.0])),
            [18.0, 12.0, 10.0, 6.0],
        ]
        np.testing.assert_almost_equal(
            aggregateby_cube.data, np.array(row, dtype=np.float32)
        )

    def test_single_missing(self):
        # aggregation correctly handles masked data
        mask = np.vstack(
            (
                np.array([[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]).repeat(
                    26, axis=0
                ),
                np.zeros([10, 3, 3]),
            )
        )
        self.cube_single.data = ma.array(self.cube_single.data, mask=mask)
        single_expected = ma.masked_invalid(
            [
                [
                    [0.0, np.nan, 0.0],
                    [np.nan, 0.0, np.nan],
                    [0.0, np.nan, 0.0],
                ],
                [
                    [1.5, np.nan, 4.5],
                    [np.nan, 7.5, np.nan],
                    [10.5, np.nan, 13.5],
                ],
                [
                    [4.0, np.nan, 12.0],
                    [np.nan, 20.0, np.nan],
                    [28.0, np.nan, 36.0],
                ],
                [
                    [7.5, np.nan, 22.5],
                    [np.nan, 37.5, np.nan],
                    [52.5, np.nan, 67.5],
                ],
                [
                    [12.0, np.nan, 36.0],
                    [np.nan, 60.0, np.nan],
                    [84.0, np.nan, 108.0],
                ],
                [
                    [17.5, np.nan, 52.5],
                    [np.nan, 87.5, np.nan],
                    [122.5, np.nan, 157.5],
                ],
                [
                    [24.0, 53.0, 72.0],
                    [106.0, 120.0, 159.0],
                    [168.0, 212.0, 216.0],
                ],
                [
                    [31.5, 63.0, 94.5],
                    [126.0, 157.5, 189.0],
                    [220.5, 252.0, 283.5],
                ],
            ]
        )
        aggregateby_cube = self.cube_single.aggregated_by(
            "height", iris.analysis.MEAN
        )
        self.assertCML(
            aggregateby_cube,
            ("analysis", "aggregated_by", "single_missing.cml"),
            checksum=False,
        )
        self.assertMaskedArrayAlmostEqual(
            aggregateby_cube.data, single_expected
        )

    def test_multi_missing(self):
        # aggregation correctly handles masked data
        mask = np.vstack(
            (
                np.array([[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]).repeat(
                    16, axis=0
                ),
                np.ones([2, 3, 3]),
                np.zeros([2, 3, 3]),
            )
        )
        self.cube_multi.data = ma.array(self.cube_multi.data, mask=mask)
        multi_expected = ma.masked_invalid(
            [
                [
                    [1.0, np.nan, 3.0],
                    [np.nan, 5.0, np.nan],
                    [7.0, np.nan, 9.0],
                ],
                [
                    [3.5, np.nan, 10.5],
                    [np.nan, 17.5, np.nan],
                    [24.5, np.nan, 31.5],
                ],
                [
                    [14.0, 37.0, 42.0],
                    [74.0, 70.0, 111.0],
                    [98.0, 148.0, 126.0],
                ],
                [
                    [7.0, np.nan, 21.0],
                    [np.nan, 35.0, np.nan],
                    [49.0, np.nan, 63.0],
                ],
                [
                    [9.0, np.nan, 27.0],
                    [np.nan, 45.0, np.nan],
                    [63.0, np.nan, 81.0],
                ],
                [
                    [10.5, np.nan, 31.5],
                    [np.nan, 52.5, np.nan],
                    [73.5, np.nan, 94.5],
                ],
                [
                    [13.0, np.nan, 39.0],
                    [np.nan, 65.0, np.nan],
                    [91.0, np.nan, 117.0],
                ],
                [
                    [15.0, np.nan, 45.0],
                    [np.nan, 75.0, np.nan],
                    [105.0, np.nan, 135.0],
                ],
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
            ]
        )
        aggregateby_cube = self.cube_multi.aggregated_by(
            ["height", "level"], iris.analysis.MEAN
        )
        self.assertCML(
            aggregateby_cube,
            ("analysis", "aggregated_by", "multi_missing.cml"),
            checksum=False,
        )
        self.assertMaskedArrayAlmostEqual(
            aggregateby_cube.data, multi_expected
        )

    def test_returned_weights(self):
        self.assertRaises(
            ValueError,
            self.cube_single.aggregated_by,
            "height",
            iris.analysis.MEAN,
            returned=True,
        )
        self.assertRaises(
            ValueError,
            self.cube_single.aggregated_by,
            "height",
            iris.analysis.MEAN,
            weights=[1, 2, 3, 4, 5],
        )


if __name__ == "__main__":
    unittest.main()
