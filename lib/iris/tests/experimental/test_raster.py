# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

import iris.tests as tests  # isort:skip
import PIL.Image
import numpy as np

import iris


@tests.skip_gdal
@tests.skip_data
class TestGeoTiffExport(tests.IrisTest):
    def check_tiff_header(self, tiff_filename, expect_keys, expect_entries):
        """
        Checks the given tiff file's metadata contains the expected keys,
        and some matching values (not all).

        """
        with open(tiff_filename, "rb") as fh:
            im = PIL.Image.open(fh)
            file_keys = im.tag.keys()

            missing_keys = sorted(set(expect_keys) - set(file_keys))
            msg_nokeys = "Tiff header has missing keys : {}."
            self.assertEqual(missing_keys, [], msg_nokeys.format(missing_keys))

            extra_keys = sorted(set(file_keys) - set(expect_keys))
            msg_extrakeys = "Tiff header has extra unexpected keys : {}."
            self.assertEqual(extra_keys, [], msg_extrakeys.format(extra_keys))

            msg_badval = "Tiff header entry {} has value {} != {}."
            for key, value in expect_entries.items():
                content = im.tag[key]
                self.assertEqual(
                    content, value, msg_badval.format(key, content, value)
                )

    def check_tiff(self, cube, header_keys, header_items):
        # Check that the cube saves correctly to TIFF :
        #   * the header contains expected keys and (some) values
        #   * the data array retrives correctly
        import iris.experimental.raster

        with self.temp_filename(".tif") as temp_filename:
            iris.experimental.raster.export_geotiff(cube, temp_filename)

            # Check the metadata is correct.
            self.check_tiff_header(temp_filename, header_keys, header_items)

            # Ensure that north is at the top then check the data is correct.
            coord_y = cube.coord(axis="Y", dim_coords=True)
            data = cube.data
            if np.diff(coord_y.bounds[0]) > 0:
                data = cube.data[::-1, :]
            im = PIL.Image.open(temp_filename)
            im_data = np.array(im)
            # Currently we only support writing 32-bit tiff, when comparing
            # the data ensure that it is also 32-bit
            np.testing.assert_array_equal(im_data, data.astype(np.float32))

    def _check_tiff_export(self, masked, inverted=False):
        tif_header_keys = [
            256,
            257,
            258,
            259,
            262,
            273,
            277,
            278,
            279,
            284,
            339,
            33550,
            33922,
            42113,
            # Don't add a check entry for this, as coding changed between gdal
            # version 1 and 2.
            # tif_header_entries[42113] = (u'1e+20',)
        ]
        tif_header_entries = {
            256: (160,),
            257: (159,),
            258: (32,),
            259: (1,),
            262: (1,),
            # Skip this one: behaviour is not consistent across gdal versions.
            # 273: (354, 8034, 15714, 23394, 31074, 38754, 46434,
            #       54114, 61794, 69474, 77154, 84834, 92514, 100194),
            277: (1,),
            278: (12,),
            279: (
                7680,
                7680,
                7680,
                7680,
                7680,
                7680,
                7680,
                7680,
                7680,
                7680,
                7680,
                7680,
                7680,
                1920,
            ),
            284: (1,),
            339: (3,),
            33550: (1.125, 1.125, 0.0),
            33922: (0.0, 0.0, 0.0, -0.5625, 89.4375, 0.0),
        }
        fin = tests.get_data_path(
            ("NetCDF", "global", "xyt", "SMALL_total_column_co2.nc")
        )
        cube = iris.load_cube(fin)[0]
        # PIL doesn't support float64
        cube.data = cube.data.astype("f4")

        # Ensure longitude values are continuous and monotonically increasing,
        # and discard the 'half cells' at the top and bottom of the UM output
        # by extracting a subset.
        east = iris.Constraint(longitude=lambda cell: cell < 180)
        non_edge = iris.Constraint(latitude=lambda cell: -90 < cell < 90)
        cube = cube.extract(east & non_edge)
        cube.coord("longitude").guess_bounds()
        cube.coord("latitude").guess_bounds()

        if masked:
            # Mask some of the data + expect a slightly different header...
            cube.data = np.ma.masked_where(cube.data <= 380, cube.data)

        if inverted:
            # Check with the latitude coordinate (and the corresponding
            # cube.data) inverted.
            # The output should be exactly the same.
            coord = cube.coord("latitude")
            coord.points = coord.points[::-1]
            coord.bounds = None
            coord.guess_bounds()
            cube.data = cube.data[::-1, :]

        self.check_tiff(cube, tif_header_keys, tif_header_entries)

    def test_unmasked(self):
        self._check_tiff_export(masked=False)

    def test_masked(self):
        self._check_tiff_export(masked=True)

    def test_inverted(self):
        self._check_tiff_export(masked=False, inverted=True)


if __name__ == "__main__":
    tests.main()
